#
#! TODO: Could use this to refactor the loops in `analysis.py` and `train.py`

from typing import Any, Callable, Literal, Mapping, Sequence

from jax_cookbook.misc import deep_merge

from feedbax._experiments.config.batch import load_batch_config
from feedbax._experiments.config.config import load_config

# Called once per *module* (outer loop). Returns an opaque module_ctx passed to later hooks.
BeforeModule = Callable[[str, Any], Any]  # (module_key, base_config) -> module_ctx

# Called once per *run* (inner loop). Returns kwargs to forward into `runner(...)`.
BeforeRun = Callable[
    [str, Any, Any, int], Mapping[str, Any]
]  # (module_key, cfg, module_ctx, run_index) -> extra_kwargs

# Optional tear-downs
AfterRun = Callable[
    [str, Any, Any, Any, int], None
]  # (module_key, cfg, result, module_ctx, run_index) -> None
AfterModule = Callable[[str, Any, Any], None]  # (module_key, base_config, module_ctx) -> None


def run_single_or_batched(
    args,  # argparse.Namespace with .single or .batched/.batch
    *,
    domain: Literal["training", "analysis"],  # "training" | "analysis" (passed to your loaders)
    runner: Callable[..., Any],  # e.g. partial(train_and_save_models, ...)
    before_module: BeforeModule | None = None,
    before_run: BeforeRun | None = None,
    after_run: AfterRun | None = None,
    after_module: AfterModule | None = None,
) -> None:
    """Unified single/batch orchestration with explicit module/run hooks."""
    single_key = getattr(args, "single", None)
    if single_key is not None:
        base_cfg = load_config(single_key, config_type=domain)
        module_ctx = before_module(single_key, base_cfg) if before_module else None
        extra = before_run(single_key, base_cfg, module_ctx, 0) if before_run else {}
        result = runner(module_key=single_key, config=base_cfg, **dict(extra))
        if after_run:
            after_run(single_key, base_cfg, result, module_ctx, 0)
        if after_module:
            after_module(single_key, base_cfg, module_ctx)
        return

    batch_key = getattr(args, "batched", None)
    if batch_key is not None:
        batch_spec = load_batch_config(domain=domain, config_key=batch_key)

        # Expecting: batch_spec: dict[module_key, Sequence[Mapping[str, Any]]]
        for module_key, run_param_list in batch_spec.items():
            base_cfg = load_config(module_key, config_type=domain)
            module_ctx = before_module(module_key, base_cfg) if before_module else None

            # tolerate list/tuple/generator
            assert isinstance(run_param_list, Sequence), (
                "batch spec must yield a sequence of run-param mappings"
            )
            for idx, run_params in enumerate(run_param_list):
                cfg_i = deep_merge(base_cfg, run_params)
                extra = {}
                if before_run is not None:
                    extra = before_run(module_key, cfg_i, module_ctx, idx)
                result = runner(module_key=module_key, config=cfg_i, **dict(extra))
                if after_run:
                    after_run(module_key, cfg_i, result, module_ctx, idx)

            if after_module:
                after_module(module_key, base_cfg, module_ctx)
