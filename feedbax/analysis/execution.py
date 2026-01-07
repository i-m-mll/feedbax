import copy
import inspect
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from math import e
from pathlib import Path
from types import ModuleType
from typing import Any, List, Literal, Optional, Union

import dill as pickle
import equinox as eqx
import feedbax
import jax
import jax.tree as jt
import jax_cookbook.tree as jtree
import optax
import plotly
import plotly.graph_objects as go
from equinox import Module
from jax_cookbook import is_module, is_none, is_type
from jax_cookbook._func import wrap_to_accept_var_kwargs
from jax_cookbook.progress import piter
from jaxtyping import PyTree
from rich.prompt import Prompt
from ruamel.yaml import YAML
from sqlalchemy.orm import Session

import feedbax_experiments
from feedbax.analysis._dependencies import compute_dependency_results
from feedbax.analysis.analysis import (
    AbstractAnalysis,
    get_validation_trial_specs,
    logger,
)
from feedbax.colors import COMMON_COLOR_SPECS, setup_colors

# Access project paths and string constants
from feedbax.config import PATHS
from feedbax.config.yaml import get_yaml_loader
from feedbax.constants import REPLICATE_CRITERION

# `record_to_dict` converts SQLAlchemy records to plain dicts
# Added utilities for unflattening record hyperparameters into namespaces
from feedbax.database import (
    EvaluationRecord,
    ModelRecord,
    add_evaluation,
    check_model_files,
    db_session,
    fill_hps_with_train_params,
    get_db_session,
    get_model_record,
)

# `cast_hps` is needed to convert dictionaries (e.g. `where`) back into the expected objects
from feedbax.hyperparams import (
    config_to_hps,
    flatten_hps,
    use_train_hps_when_none,
)
from feedbax.misc import delete_all_files_in_dir, log_version_info
from feedbax.plugins import EXPERIMENT_REGISTRY
from feedbax.plugins.registry import ExperimentRegistry
from feedbax.setup_utils import query_and_load_model
from feedbax.training.post_training import process_model_post_training
from feedbax.tree_utils import tree_level_labels
from feedbax.types import (
    AnalysisInputData,
    LDict,
    TreeNamespace,
    namespace_to_dict,
)

STATES_CACHE_SUBDIR = "states"


@dataclass
class FigDumpManager:
    """Helper for batch-aware figure organization."""

    root: Path = PATHS.figures_dump
    clear_existing: Literal["none", "module"] = "module"
    _counters: dict[str, int] = field(default_factory=dict)

    def module_dir(self, module_key: str) -> Path:
        return self.root / Path(module_key.replace(".", "/"))

    def prepare_module_dir(self, module_key: str, module_config: dict) -> Path:
        d = self.module_dir(module_key)
        if self.clear_existing in ("module", "all") and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        yaml = get_yaml_loader(typ="safe")
        with open(d / "module.yml", "w") as f:
            yaml.dump(module_config, f)
        return d

    def prepare_run_dir(self, module_key: str, run_params: dict) -> Path:
        i = self._counters.get(module_key, 0) + 1
        self._counters[module_key] = i
        module_dir = self.module_dir(module_key)
        i_str = f"{i:03d}"
        dump_dir = module_dir / i_str
        dump_dir.mkdir(parents=True, exist_ok=True)
        yaml = get_yaml_loader(typ="safe")
        with open(module_dir / f"{i_str}.yml", "w") as f:
            yaml.dump(run_params, f)
        return dump_dir

    def clear_all_figures(self) -> None:
        """Clear all figures in the root dump directory."""
        if self.root.exists():
            shutil.rmtree(self.root)
            self.root.mkdir(parents=True, exist_ok=True)
            logger.info(f"Deleted all existing dump figures in {self.root}")


# Pre-setup can be either granular dict or combined function
PreSetupSpec = Union[
    dict[str, Optional[Callable]],  # Runtime validation will check keys are 'task'/'models'
    Callable[[Module, LDict[float, Module]], tuple[Module, LDict[float, Module]]],
]

# Post-eval can be either granular dict or combined function
PostEvalSpec = Union[
    # Runtime validation will check keys are 'models'/'tasks'/'states'
    dict[str, Optional[Callable]],
    Callable[[LDict[float, Module], LDict, PyTree], tuple[LDict[float, Module], LDict, PyTree]],
]


class AnalysisModuleTransformSpec(Module):
    """Specifies transformations to apply at different stages of analysis execution.

    Attributes
    ----------
    pre_setup : Optional[PreSetupSpec]
        Transform applied to task_base and models_base before setup_eval_tasks_and_models.
        Can be either:
        - dict with 'task' and/or 'models' keys mapping to transform functions
        - single function taking (task, models) and returning (task, models)
        Useful for operations like selecting best replicates to save computation.
    post_eval : Optional[PostEvalSpec]
        Transform applied to models, tasks, and/or states after evaluation completes.
        Can be either:
        - dict with 'models', 'tasks', and/or 'states' keys mapping to transform functions
        - single function taking (models, tasks, states) and returning (models, tasks, states)
        Useful for post-processing results while preserving intermediate data.
    """

    pre_setup: Optional[PreSetupSpec] = None
    post_eval: Optional[PostEvalSpec] = None


# Also allow plain dict for convenience
TransformsSpec = Union[
    AnalysisModuleTransformSpec,  # Structured (better static typing)
    dict[str, Any],  # Dict (convenience) - runtime validation will check keys
]


def validate_and_convert_transforms(
    transforms_spec: Optional[TransformsSpec],
) -> AnalysisModuleTransformSpec:
    """Convert and validate transforms specification to canonical form."""
    if transforms_spec is None:
        return AnalysisModuleTransformSpec()

    if isinstance(transforms_spec, dict):
        # Validate top-level keys
        valid_keys = {"pre_setup", "post_eval"}
        invalid_keys = set(transforms_spec.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid transform keys: {invalid_keys}. Valid keys are: {valid_keys}"
            )

        pre_setup = transforms_spec.get("pre_setup")
        post_eval = transforms_spec.get("post_eval")

        # Validate pre_setup if present
        if pre_setup is not None and isinstance(pre_setup, dict):
            valid_pre_keys = {"task", "models"}
            invalid_pre_keys = set(pre_setup.keys()) - valid_pre_keys
            if invalid_pre_keys:
                raise ValueError(
                    f"Invalid pre_setup keys: {invalid_pre_keys}. Valid keys are: {valid_pre_keys}"
                )

        # Validate post_eval if present
        if post_eval is not None and isinstance(post_eval, dict):
            valid_post_keys = {"models", "tasks", "states"}
            invalid_post_keys = set(post_eval.keys()) - valid_post_keys
            if invalid_post_keys:
                raise ValueError(
                    f"Invalid post_eval keys: {invalid_post_keys}. "
                    f"Valid keys are: {valid_post_keys}"
                )

        return AnalysisModuleTransformSpec(pre_setup=pre_setup, post_eval=post_eval)

    elif isinstance(transforms_spec, AnalysisModuleTransformSpec):
        return transforms_spec

    else:
        raise ValueError(
            f"transforms_spec must be dict or AnalysisModuleTransformSpec, "
            f"got {type(transforms_spec)}"
        )


def load_trained_models_and_aux_objects(
    training_module_name: str, hps: TreeNamespace, db_session: Session, registry
):
    """Given the analysis config, load the trained models and related objects (e.g. train tasks)."""

    # Load training module from registered packages
    training_module = registry.get_training_module(training_module_name)

    pairs, model_info, replicate_info, n_replicates_included, hps_train_dict = jtree.unzip(
        #! Should this structure be hardcoded here?
        #! At least for this project, we typically load spreads of trained models,
        #! and those spreads are always over the training perturbation std.
        LDict.of("train__pert__std")(
            {
                train_pert_std: query_and_load_model(
                    db_session,
                    training_module.setup_task_model_pair,
                    params_query=namespace_to_dict(flatten_hps(hps.train))
                    | dict(expt_name=training_module_name, pert__std=train_pert_std),
                    noise_stds=dict(
                        feedback=hps.model.feedback_noise_std,
                        motor=hps.model.motor_noise_std,
                    ),
                    surgeries={
                        # Change
                        ("n_steps",): hps.task.n_steps,
                    },
                    exclude_underperformers_by=REPLICATE_CRITERION,
                    return_task=True,
                )
                for train_pert_std in hps.train.pert.std
            }
        )
    )

    tasks_train, models = jtree.unzip(pairs)

    return models, model_info, replicate_info, tasks_train, n_replicates_included, hps_train_dict


def setup_eval_for_module(
    module_key: str,
    hps: TreeNamespace,
    db_session: Session,
    version_info: dict[str, str],
):
    """Given the analysis module, set up the evaluation(s).

    1. Construct the task-model pairs to evaluate, to produce the state needed for the analyses.
    2. Add an evaluation record to the database.
    """

    # Load analysis module from registered packages
    analysis_module: ModuleType = EXPERIMENT_REGISTRY.get_analysis_module(module_key)

    #! For this project, assume only a single-level mapping between training experiments and analysis subpackages;
    #! thus `analysis.modules.part1` corresponds to `training.modules.part1`.
    #! In general, it might be better to go back to explicitly specifying the `expt_name` for the
    #! training experiment, in each analysis config file.
    training_module_name = module_key.split(".")[0]

    models_base, model_info, replicate_info, tasks_train, n_replicates_included, hps_train_dict = (
        load_trained_models_and_aux_objects(
            training_module_name, hps, db_session, EXPERIMENT_REGISTRY
        )
    )

    # Fill-in any missing training hyper-parameters **out-of-place** and switch to
    # the enriched version from here onward. Use training hps from database records.
    # Take the first entry from hps_train_dict as representative training hyperparameters
    hps_train_representative = jt.leaves(hps_train_dict, is_leaf=is_type(TreeNamespace))[0]
    hps_filled = fill_hps_with_train_params(hps, hps_train_representative)
    hps = hps_filled  #  use the enriched version for everything below

    # For this project, the training task should not vary with the train field std
    # so we just keep a single one of them.
    # TODO: In future, could keep the full `tasks_base`, and update `get_task_variant`/`setup_fn`
    task_base = jt.leaves(tasks_train, is_leaf=is_module)[0]

    #! TODO: This should be generalized once we move `n_steps` to `task` (in YAML config)
    task_base = eqx.tree_at(
        lambda task: task.n_steps,
        task_base,
        hps.task.n_steps,
    )

    # Load and validate transforms once
    transforms_raw = getattr(analysis_module, "TRANSFORMS", None)
    transforms = validate_and_convert_transforms(transforms_raw)

    # If there is no system noise (i.e. the stds are zero), set the number of evaluations per condition to zero.
    # (Is there any other reason than the noise samples, why evaluations might differ?)
    # TODO: Make this optional?
    # ? What is the point of using `jt.leaves` here?
    any_system_noise = any(
        jt.leaves(
            (
                hps.model.feedback_noise_std,
                hps.model.motor_noise_std,
            )
        )
    )
    if not any_system_noise:
        hps.task.eval_n = 1

    # Get indices for taking important subsets of replicates
    # best_replicate, included_replicates = jtree.unzip(LDict.of("train__pert__std")({
    #     std: (
    #         replicate_info[std]['best_replicates'][REPLICATE_CRITERION],
    #         replicate_info[std]['included_replicates'][REPLICATE_CRITERION],
    #     )
    #     # Assumes that `train.pert.std` is given as a sequence
    #     for std in hps.train.pert.std
    # }))

    # Add evaluation record to the database
    eval_info = add_evaluation(
        db_session,
        expt_name=module_key,
        models=model_info,
        # ? Could move the flattening/conversion to `database`?
        #! TODO: Could exclude train parameters, since
        eval_parameters=namespace_to_dict(flatten_hps(hps)),
        version_info=version_info,
    )

    # Construct common inputs needed by transforms and analyses
    # Note: We construct trial_specs for all task variants here
    #! Note that modifying `n_steps` may not work here yet, because of the way that `n_steps` is
    #! baked into the model's `Iterator`. Here, only the task will be modified.
    task_variants = LDict.of("task_variant")(
        {
            variant_key: eqx.tree_at(
                lambda task: [getattr(task, name) for name in variant_params.keys()],
                task_base,
                [value for value in variant_params.values()],
            )
            for variant_key, variant_params in namespace_to_dict(hps.task.variants).items()
        }
    )

    trial_specs = jt.map(get_validation_trial_specs, task_variants, is_leaf=is_module)

    colors, colorscales = setup_colors(hps, COMMON_COLOR_SPECS | analysis_module.COLOR_FNS)

    common_inputs = dict(
        hps_common=hps,
        colors=colors,
        colorscales=colorscales,
        replicate_info=replicate_info,
        trial_specs=trial_specs,
    )

    def setup_per_task_variant(task, models_base, hps, variant_key, **kwargs):
        # Apply pre-setup transformations if present
        if transforms.pre_setup is not None:
            if isinstance(transforms.pre_setup, dict):
                # Granular transforms - apply to task and/or models separately
                if "task" in transforms.pre_setup and transforms.pre_setup["task"] is not None:
                    func = wrap_to_accept_var_kwargs(transforms.pre_setup["task"])
                    task = func(task, common_inputs)
                if "models" in transforms.pre_setup and transforms.pre_setup["models"] is not None:
                    func = wrap_to_accept_var_kwargs(transforms.pre_setup["models"])
                    models_base = func(models_base, common_inputs)
            else:
                # Combined function - pass task and models as tuple
                func = wrap_to_accept_var_kwargs(transforms.pre_setup)
                task, models_base = func((task, models_base), common_inputs)

        tasks, models, hps, extras = analysis_module.setup_eval_tasks_and_models(
            task, models_base, hps
        )

        def _promote_variant_params(hps_task, variant_key):
            hps_variant = hps_task | getattr(hps_task.variants, variant_key)
            del hps_variant.variants
            return hps_variant

        hps = jt.map(
            lambda hps: eqx.tree_at(
                lambda hps: hps.task,
                hps,
                _promote_variant_params(hps.task, variant_key),
            ),
            hps,
            is_leaf=is_type(TreeNamespace),
        )

        return tasks, models, hps, extras

    # Outer level is task variants, inner is the structure returned by `setup_fn`
    # i.e. "task variants" are a way to evaluate different sets of conditions
    all_tasks, all_models, all_hps, all_extras = jtree.unzip(
        LDict.of("task_variant")(
            {
                variant_key: setup_per_task_variant(
                    task_variant,
                    models_base,
                    hps,
                    variant_key,
                )
                for variant_key, task_variant in task_variants.items()
            }
        )
    )

    data = AnalysisInputData(
        hps=all_hps,
        tasks=all_tasks,
        models=all_models,
        states=None,  # Empty prior to evaluation
        extras=all_extras,
    )

    return (
        analysis_module,
        data,
        common_inputs,  # includes replicate_info, trial_specs, colors, etc.
        transforms,  # validated transforms
        model_info,
        eval_info,
    )


def perform_all_analyses(
    db_session: Session,
    analyses: dict[str, AbstractAnalysis],
    data: AnalysisInputData,
    model_info: ModelRecord,
    eval_info: EvaluationRecord,
    *,
    fig_dump_path: Optional[Path] = None,
    fig_dump_formats: List[str] = ["html"],
    custom_dependencies: Optional[dict[str, AbstractAnalysis]] = None,
    **kwargs,
) -> tuple[PyTree[AbstractAnalysis], PyTree[Any], PyTree[go.Figure]]:
    """Given a list or dict of instances of `AbstractAnalysis`, perform all analyses and save any figures."""

    if not analyses:
        logger.warning("No analyses given to perform; nothing returned")
        return None, None, None

    if not all(isinstance(value, AbstractAnalysis) for value in analyses.values()):
        raise ValueError(
            "All analyses defined in given analysis module must be instances of `AbstractAnalysis`"
        )

    if not any(analyses):
        raise ValueError("No analyses given to perform")

    # Phase 1: Compute all analysis nodes (dependencies + leaves)
    logger.info("Computing results for analyses and their dependencies")
    all_dependency_results = compute_dependency_results(
        analyses, data, custom_dependencies, **kwargs
    )

    # Phase 2: Keep results & generate figures for leaf analyses only
    def finish_analysis(analysis_key: str, analysis: AbstractAnalysis, inputs: dict):
        # Get the computed result for this analysis (computed in phase 1)
        result = inputs.pop("result")

        # Check if make_figs is overridden before attempting figure generation
        if "make_figs" not in analysis.__class__.__dict__:
            logger.debug(
                f"Skipping figure generation for {analysis_key} (no make_figs implementation)"
            )
            figs = None
        else:
            logger.debug(f"Making figures: {analysis_key}")
            figs = analysis._make_figs_with_ops(data, result, **inputs)

            analysis.save_figs(
                db_session,
                eval_info,
                result,
                figs,
                data.hps,
                model_info,
                dump_path=fig_dump_path,
                dump_formats=fig_dump_formats,
                label=analysis_key,
                **inputs,
            )

        return analysis, result, figs

    logger.info("Making and saving figures for analyses")
    all_analyses, all_results, all_figs = jtree.unzip(
        {
            analysis_key: finish_analysis(analysis_key, analysis, dependencies)
            for (analysis_key, analysis), dependencies in piter(
                zip(analyses.items(), all_dependency_results),
                total=len(analyses),
                description="Making and saving figures",
                right=lambda p, i: p[0][0],  # analysis key on rhs of progress bar
                eta_halflife=10.0,
            )
        }
    )

    return all_analyses, all_results, all_figs


def check_records_for_analysis(
    module_key: str,
    module_config: dict,
):
    hps = config_to_hps(module_config, config_type="analysis")

    #! TODO: Along with `setup_eval_for_module`: reconcile this with `ExperimentRegistry` logic
    training_module_name = module_key.split(".")[0]

    with db_session(autocommit=False) as db:
        for train_pert_std in hps.train.pert.std:
            params_query = namespace_to_dict(flatten_hps(hps.train)) | dict(
                expt_name=training_module_name, pert__std=train_pert_std
            )
            model_info = get_model_record(
                db,
                postprocessed=True,
                exclude_defunct=True,
                explain_on_miss=True,
                **params_query,
            )
            # if model_info.is_path_defunct:
            #     try:
            #         raw_model_info = get_model_record(
            #             db,
            #             postprocessed=False,
            #             exclude_defunct=True,
            #             **params_query,
            #         )
            #         if raw_model_info is not None:
            #             model_info = process_model_post_training(
            #                 db,
            #                 raw_model_info,
            #                 2,
            #                 process_all=True,
            #                 save_figures=False,
            #             )
            #     except Exception as e:
            #         raise e
            if model_info is None:
                raise ValueError(
                    f"No trained model found for {module_key} with parameters {params_query}"
                )
    return True


def run_analysis_module(
    module_key: str,
    module_config: dict,
    *,
    fig_dump_dir: str | Path | None = None,
    fig_dump_formats: list[str] = ["html", "webp", "svg"],
    no_pickle: bool = False,
    states_pkl_dir: Path | None = PATHS.cache / "states",
    eval_only: bool = False,
    memory_warn_gb: float = 24.0,
    key,
):
    """Run a single analysis module using provided hyperparameters.

    - `module_key` is the analysis module key (e.g., 'part2.plant_perts').
    - `module_config` is the resolved config returned by config.load_config(...).
    - `fig_dump_dir`, if provided, is passed to analysis.save_figs(..., dump_path=...).
    """
    version_info = log_version_info(
        jax,
        eqx,
        optax,
        plotly,
        git_modules=(feedbax, feedbax_experiments),
    )

    if fig_dump_dir is None:
        fig_dump_dir = PATHS.figures_dump
    assert fig_dump_dir is not None

    # Ensure the directory for state pickles exists
    if states_pkl_dir is None:
        states_pkl_dir = PATHS.cache / STATES_CACHE_SUBDIR
    assert states_pkl_dir is not None
    states_pkl_dir.mkdir(parents=True, exist_ok=True)

    # Start a database session for loading trained models, and saving evaluation/figure records
    #! TODO: Switch to context manager
    db_session = get_db_session()
    check_model_files(db_session)  # Ensure we don't try to load any models whose files don't exist

    # Use the provided config (hyperparameters) for the analysis module
    hps = config_to_hps(module_config, config_type="analysis")
    # Establish a provisional common namespace (needed for querying the DB); this
    # will be superseded after we load the model records and fill in any missing
    # hyper-parameters.
    provisional_hps_common = use_train_hps_when_none(hps)

    (
        analysis_module,
        data,
        common_inputs,
        transforms,
        model_info,
        eval_info,
    ) = setup_eval_for_module(
        module_key,
        provisional_hps_common,
        db_session,
        version_info,
    )

    def evaluate_all_states(all_tasks, all_models, all_hps):
        return jt.map(  # Map over the task-base model subtree pairs generated by `schedule_intervenor` for each base task
            lambda task, models, hps: jt.map(
                lambda model: analysis_module.eval_fn(key, hps, model, task),
                models,
                is_leaf=is_module,
            ),
            all_tasks,
            all_models,
            all_hps,
            is_leaf=is_module,
        )

    # Helper function to compute states from scratch
    def _compute_states_and_log_memory_estimate():
        states_shapes = eqx.filter_eval_shape(
            evaluate_all_states, data.tasks, data.models, data.hps
        )
        logger.info(
            f"Evaluated states PyTree structure: {tree_level_labels(states_shapes, is_leaf=is_module)}"
        )

        memory_estimate_gb = jtree.struct_bytes(states_shapes) / 1e9
        logger.info(f"{memory_estimate_gb:.2f} GB of memory estimated to store all states.")

        if memory_estimate_gb > memory_warn_gb:
            logger.warning(
                f"Estimated memory usage ({memory_estimate_gb:.2f} GB) exceeds the warning threshold "
                f"({memory_warn_gb:.2f} GB). Consider reducing the number of evaluations or model sizes. "
                f"State shapes PyTree written to {PATHS.logs / 'states_shapes.txt'}."
            )

            with open(PATHS.logs / "states_shapes.txt", "w") as f:
                f.write(eqx.tree_pformat(states_shapes))

            print("\n")
            while True:
                proceed = (
                    Prompt.ask("Proceed with the evaluation and analysis? (y/n): ").strip().lower()
                )

                if proceed == "y":
                    print("\n")
                    break
                elif proceed == "n":
                    logger.info("Evaluation cancelled by user.")
                    sys.exit()
                else:
                    print("Invalid input; please enter 'y' or 'n'.")

        computed_states = evaluate_all_states(data.tasks, data.models, data.hps)
        logger.info("All states evaluated.")
        return computed_states

    # Create a filename based on the evaluation hash
    states_pickle_path = states_pkl_dir / f"{eval_info.hash}.pkl"

    loaded_from_pickle = False
    # If --no-pickle is set, we won't try to load from or save to pickle
    if not no_pickle and states_pickle_path.exists():
        # Try to load from pickle
        logger.info(f"Loading states from {states_pickle_path}...")
        try:
            with open(states_pickle_path, "rb") as f:
                states = pickle.load(f)
            logger.debug(
                "Loaded pickled states with PyTree structure: "
                f"{tree_level_labels(states, is_leaf=is_module)}"
            )
            loaded_from_pickle = True
        except Exception as e:
            logger.error(f"Failed to load pickled states: {e}")
            logger.info("Computing states from scratch instead...")
            states = _compute_states_and_log_memory_estimate()
    else:
        if no_pickle and states_pickle_path.exists():
            logger.info(f"Ignoring pickle file at {states_pickle_path} due to --no-pickle flag.")

        # Compute from scratch
        states = _compute_states_and_log_memory_estimate()

    # Save states if we didn't use --no-pickle and we didn't successfully load from pickle
    if not no_pickle and not loaded_from_pickle:
        # def _test(tree):
        #     return jt.map(lambda x: x, tree)

        with open(states_pickle_path, "wb") as f:
            pickle.dump(states, f)
        logger.info(f"Saved evaluated states to {states_pickle_path}")

    # Apply post-eval transformations if present
    if transforms.post_eval is not None:
        if isinstance(transforms.post_eval, dict):
            # Granular transforms - apply to models, tasks, and/or states separately
            if "models" in transforms.post_eval and transforms.post_eval["models"] is not None:
                func = wrap_to_accept_var_kwargs(transforms.post_eval["models"])
                data = eqx.tree_at(
                    lambda data: data.models,
                    data,
                    func(data.models, common_inputs),
                    is_leaf=is_none,
                )
            if "tasks" in transforms.post_eval and transforms.post_eval["tasks"] is not None:
                func = wrap_to_accept_var_kwargs(transforms.post_eval["tasks"])
                data = eqx.tree_at(
                    lambda data: data.tasks,
                    data,
                    func(data.tasks, common_inputs),
                    is_leaf=is_none,
                )
            if "states" in transforms.post_eval and transforms.post_eval["states"] is not None:
                func = wrap_to_accept_var_kwargs(transforms.post_eval["states"])
                states = func(states, common_inputs)
        else:
            func = wrap_to_accept_var_kwargs(transforms.post_eval)
            # Combined function - pass models, tasks, states as tuple
            transformed_models, transformed_tasks, transformed_states = func(
                (data.models, data.tasks, states), common_inputs
            )
            data = eqx.tree_at(lambda data: data.models, data, transformed_models, is_leaf=is_none)
            data = eqx.tree_at(lambda data: data.tasks, data, transformed_tasks, is_leaf=is_none)
            states = transformed_states

    data = eqx.tree_at(lambda data: data.states, data, states, is_leaf=is_none)

    if eval_only:
        logger.info("Eval-only requested; skipping analyses and returning.")
        return data, common_inputs, None, None, None

    # jax.profiler.stop_trace()

    all_analyses, all_results, all_figs = perform_all_analyses(
        db_session,
        analysis_module.ANALYSES,
        data,
        model_info,
        eval_info,
        fig_dump_path=Path(fig_dump_dir),
        fig_dump_formats=fig_dump_formats,
        custom_dependencies=getattr(analysis_module, "DEPENDENCIES", {}),
        **common_inputs,
    )

    db_session.close()

    return data, common_inputs, all_analyses, all_results, all_figs
