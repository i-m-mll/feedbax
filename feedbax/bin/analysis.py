#!/usr/bin/env python
"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

# NOTE: JAX arrays are not directly picklable if they contain device memory references.
# Since we're using pickle to cache states which may contain JAX arrays, we rely on JAX's
# implicit handling of arrays during pickling (it should work for CPU arrays and most
# host-accessible device arrays).
import jax
import jax.random as jr
import plotly.io as pio
import rich
from jax_cookbook.progress import progress_session

from feedbax.analysis.execution import (
    FigDumpManager,
    check_records_for_analysis,
    run_analysis_module,
)
from feedbax.analysis.bundles import (
    dry_run_staged_analysis_bundle,
    execute_analysis_bundle,
    execute_staged_analysis_bundle,
    load_analysis_bundle,
)
from feedbax.analysis.exact_parents import StagedExactParents
from feedbax.analysis.execution_context import (
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
    StagedManifestRootBinding,
)
from feedbax.analysis.specs import execute_analysis_run_spec
from feedbax.config import (
    PATHS,
    PLOTLY_CONFIG,
    PRNG_CONFIG,
    load_batch_config,
    load_config,
)
from feedbax.config.utils import deep_merge
from feedbax.config.yaml import get_yaml_loader
from feedbax.contracts.staged_execution import StagedExecutionDescriptor
from feedbax.plugins import EXPERIMENT_REGISTRY

logger = logging.getLogger(os.path.basename(__file__))

RUN_SUBCOMMAND = "run"


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"JSON document contains duplicate key {key!r}")
        payload[key] = value
    return payload


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} document must be a JSON object")
    return payload


def _load_spec_document(path: Path, *, label: str) -> dict[str, object]:
    """Load a serialized spec document from a JSON or YAML file."""
    if path.suffix.lower() in {".yml", ".yaml"}:
        payload = get_yaml_loader().load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{label} document must be a YAML mapping")
        return dict(payload)
    return _load_json_object(path, label=label)


def _binding_parts(value: str, *, option: str) -> tuple[str, str]:
    name, separator, root = value.partition("=")
    if not separator or not name or not root:
        raise ValueError(f"{option} must use NAME=ROOT syntax")
    return name, root


def _apply_plotly_template_default(requested: str | None = None) -> None:
    """Apply the project Plotly template default shared by every CLI execution path.

    Args:
        requested: Explicit template name from a CLI flag, or `None` to use the
            project default from `PLOTLY_CONFIG`.
    """
    pio.templates.default = requested or PLOTLY_CONFIG.templates.default


@contextmanager
def _bundle_human_output_to_stderr():
    """Keep stdout reserved for bundle JSON while progress/noisy output renders on stderr."""
    console = rich.get_console()
    previous_console_state = console.__dict__.copy()
    rich.reconfigure(file=sys.stderr)
    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        console.__dict__ = previous_console_state


def build_run_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feedbax-analysis run",
        description="Execute an AnalysisRunSpec JSON/YAML file and write its manifest.",
    )
    parser.add_argument(
        "spec",
        type=Path,
        help="Path to an AnalysisRunSpec or AnalysisRunDeltaSpec JSON/YAML file.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Manifest root to write the AnalysisRunManifest under.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root for resolving a delta spec's content-pinned parent references.",
    )
    parser.add_argument(
        "--execution-descriptor",
        type=Path,
        default=None,
        help="Versioned staged execution descriptor for explicit runtime bindings.",
    )
    parser.add_argument(
        "--artifact-provider",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="Bind one authenticated-manifest artifact provider root.",
    )
    parser.add_argument(
        "--checkpoint-custody",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="Bind one named checkpoint custody root.",
    )
    parser.add_argument(
        "--manifest-root",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="Bind one retained Feedbax manifest-store root.",
    )
    return parser


def run_analysis_run_spec_file(argv: list[str]) -> None:
    """Execute one serialized ``AnalysisRunSpec`` file and print its manifest summary."""
    args = build_run_arg_parser().parse_args(argv)
    _apply_plotly_template_default()
    spec = _load_spec_document(args.spec, label="AnalysisRunSpec")
    if (args.artifact_provider or args.manifest_root or args.checkpoint_custody) and (
        args.execution_descriptor is None
    ):
        raise ValueError(
            "--artifact-provider, --manifest-root, and --checkpoint-custody "
            "require --execution-descriptor"
        )
    execution_descriptor = None
    if args.execution_descriptor is not None:
        execution_descriptor = StagedExecutionDescriptor.model_validate(
            _load_json_object(
                args.execution_descriptor,
                label="--execution-descriptor",
            )
        )
    artifact_provider_bindings = [
        StagedArtifactProviderRootBinding(*_binding_parts(value, option="--artifact-provider"))
        for value in args.artifact_provider
    ]
    manifest_root_bindings = [
        StagedManifestRootBinding(*_binding_parts(value, option="--manifest-root"))
        for value in args.manifest_root
    ]
    checkpoint_custody_bindings = [
        StagedCheckpointCustodyRootBinding(*_binding_parts(value, option="--checkpoint-custody"))
        for value in args.checkpoint_custody
    ]
    with _bundle_human_output_to_stderr():
        manifest, path = execute_analysis_run_spec(
            spec,
            root=args.root,
            repo_root=args.repo_root,
            execution_descriptor=execution_descriptor,
            artifact_provider_bindings=artifact_provider_bindings,
            manifest_root_bindings=manifest_root_bindings,
            checkpoint_custody_bindings=checkpoint_custody_bindings,
        )
    payload = {
        "manifest_id": manifest.id,
        "manifest_path": str(path),
        "status": manifest.status,
        "artifacts": [
            artifact.model_dump(mode="json", exclude_none=True) for artifact in manifest.artifacts
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run analysis modules on trained models.",
        epilog=(
            f"Subcommand: '{RUN_SUBCOMMAND} <spec.json> [--root ROOT]' executes a serialized "
            "AnalysisRunSpec file directly."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--single",
        metavar="MODULE_KEY",
        help="Run single analysis module (e.g., part2.plant_perts)",
    )
    mode.add_argument(
        "--batched",
        metavar="BATCH_NAME",
        help="Run batch analysis (e.g., part2/plant_perts_sweeps)",
    )
    mode.add_argument(
        "--bundle",
        metavar="BUNDLE_NAME",
        help="Run a manifest-canonical analysis bundle (e.g., rlrmp/standard_matrix)",
    )
    parser.add_argument(
        "--fig-dump-dir",
        type=str,
        default=Path(PATHS.figures_dump) / "analysis",
        help="Directory to dump figures.",
    )
    parser.add_argument(
        "--fig-dump-formats",
        type=str,
        default="html,webp,svg",
        help="Format(s) to dump figures in, comma-separated (e.g., 'html,png,pdf')",
    )
    parser.add_argument(
        "--no-pickle",
        action="store_true",
        help="Do not use pickle for states (don't load existing or save new).",
    )
    parser.add_argument(
        "--states-pkl-dir",
        type=str,
        default=None,
        help="Alternative directory for state pickle files (default: PATHS.cache/'states')",
    )
    parser.add_argument(
        "--clear-fig-dumps",
        type=str,
        choices=("none", "module", "all"),
        default="module",
        help="Clear existing figure dumps: 'none', 'module' (default), or 'all'.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the analysis.")
    parser.add_argument(
        "--plotly-template",
        type=str,
        default=None,
        help="Plotly template to use for figures (default: 'simple_white').",
    )
    parser.add_argument(
        "--show-duplicate-warnings",
        action="store_true",
        help="If set, all occurrences of each distinct warning message are shown.",
    )
    parser.add_argument(
        "--memory-warn-gb",
        type=float,
        default=30.0,
        help="Warn if estimated memory usage exceeds this value (in GB).",
    )
    parser.add_argument(
        "--manifest-root",
        type=str,
        default=None,
        help="Manifest root used by --bundle and manifest-canonical analysis outputs.",
    )
    parent_selection = parser.add_mutually_exclusive_group()
    parent_selection.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated manifest IDs to constrain --bundle selection.",
    )
    parent_selection.add_argument(
        "--exact-parents",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Versioned exact-parent JSON document for staged --bundle execution; "
            "requires --manifest-root."
        ),
    )
    parser.add_argument(
        "--issue",
        action="append",
        default=[],
        help="Issue ID to record on AnalysisRunManifest provenance for --bundle.",
    )
    parser.add_argument(
        "--execution-descriptor",
        type=str,
        default=None,
        metavar="PATH",
        help="Versioned staged execution descriptor JSON for explicit runtime bindings.",
    )
    parser.add_argument(
        "--artifact-provider",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="Bind one logical immutable artifact provider to an absolute runtime root.",
    )
    parser.add_argument(
        "--checkpoint-custody",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="Bind one logical checkpoint custody authority to an absolute runtime root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight a staged bundle without recipe, cache, output, or manifest effects.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args_in = list(argv or sys.argv[1:])
    if args_in and args_in[0] == RUN_SUBCOMMAND:
        run_analysis_run_spec_file(args_in[1:])
        return

    parser = build_arg_parser()
    args = parser.parse_args(args_in)

    if args.exact_parents is not None and args.bundle is None:
        parser.error("--exact-parents is only valid with --bundle")
    if args.exact_parents is not None and args.manifest_root is None:
        parser.error("--exact-parents requires --manifest-root")
    if args.execution_descriptor is not None and args.bundle is None:
        parser.error("--execution-descriptor is only valid with --bundle")
    if args.dry_run and args.bundle is None:
        parser.error("--dry-run is only valid with --bundle")
    if (args.artifact_provider or args.checkpoint_custody) and (args.execution_descriptor is None):
        parser.error("--artifact-provider and --checkpoint-custody require --execution-descriptor")

    _apply_plotly_template_default(args.plotly_template)

    if args.seed is None:
        key = jr.PRNGKey(PRNG_CONFIG.seed)
    else:
        key = jr.PRNGKey(args.seed)

    # Parse the figure dump formats
    fig_dump_formats = args.fig_dump_formats.split(",")

    if args.bundle:
        bundle = load_analysis_bundle(args.bundle, registry=EXPERIMENT_REGISTRY)
        run_ids = (
            [item.strip() for item in args.runs.split(",") if item.strip()]
            if args.runs is not None
            else None
        )
        execution_kwargs = {
            "root": Path(args.manifest_root) if args.manifest_root else None,
            "run_ids": run_ids,
            "issues": list(args.issue),
            "fig_dump_path": Path(args.fig_dump_dir),
            "fig_dump_formats": fig_dump_formats,
        }
        if bundle.templates and not bundle.stages:
            if args.exact_parents is not None:
                raise ValueError("--exact-parents is only valid for staged analysis bundles")
            if args.execution_descriptor is not None:
                raise ValueError("--execution-descriptor is only valid for staged analysis bundles")
            if args.dry_run:
                raise ValueError("--dry-run is only valid for staged analysis bundles")
            with _bundle_human_output_to_stderr():
                outputs = execute_analysis_bundle(bundle, **execution_kwargs)
            payload = [
                {
                    "bundle": expansion.bundle_name,
                    "template": expansion.template_name,
                    "mode": expansion.mode,
                    "matched_run_ids": list(expansion.matched_run_ids),
                    "manifest_id": manifest.id,
                    "manifest_path": str(path),
                }
                for expansion, manifest, path in outputs
            ]
        elif bundle.stages and not bundle.templates:
            exact_parents = None
            if args.exact_parents is not None:
                exact_path = Path(args.exact_parents)
                exact_payload = _load_json_object(exact_path, label="--exact-parents")
                missing_schema = [
                    field_name
                    for field_name in ("schema_id", "schema_version")
                    if field_name not in exact_payload
                ]
                if missing_schema:
                    raise ValueError(
                        "--exact-parents document requires explicit schema_id and "
                        f"schema_version; missing {', '.join(missing_schema)}"
                    )
                exact_parents = StagedExactParents.model_validate(exact_payload)
            execution_descriptor = None
            if args.execution_descriptor is not None:
                descriptor_payload = _load_json_object(
                    Path(args.execution_descriptor),
                    label="--execution-descriptor",
                )
                execution_descriptor = StagedExecutionDescriptor.model_validate(descriptor_payload)
            artifact_provider_bindings = [
                StagedArtifactProviderRootBinding(
                    *_binding_parts(value, option="--artifact-provider")
                )
                for value in args.artifact_provider
            ]
            checkpoint_custody_bindings = [
                StagedCheckpointCustodyRootBinding(
                    *_binding_parts(value, option="--checkpoint-custody")
                )
                for value in args.checkpoint_custody
            ]
            if args.dry_run:
                with _bundle_human_output_to_stderr():
                    dry_run = dry_run_staged_analysis_bundle(
                        bundle,
                        root=execution_kwargs["root"],
                        run_ids=run_ids,
                        exact_parents=exact_parents,
                        execution_descriptor=execution_descriptor,
                        artifact_provider_bindings=artifact_provider_bindings,
                        checkpoint_custody_bindings=checkpoint_custody_bindings,
                    )
                payload = dry_run.model_dump(mode="json", exclude_none=True)
            else:
                with _bundle_human_output_to_stderr():
                    execution = execute_staged_analysis_bundle(
                        bundle,
                        exact_parents=exact_parents,
                        execution_descriptor=execution_descriptor,
                        artifact_provider_bindings=artifact_provider_bindings,
                        checkpoint_custody_bindings=checkpoint_custody_bindings,
                        **execution_kwargs,
                    )
                payload = execution.model_dump(mode="json", exclude_none=True)
        else:
            raise ValueError(
                f"Analysis bundle {bundle.name!r} must define exactly one non-empty "
                "execution shape: templates or stages"
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    # Set states pickle directory
    states_pkl_dir = Path(args.states_pkl_dir) if args.states_pkl_dir else PATHS.cache / "states"

    fig_dump_manager = FigDumpManager(
        root=Path(args.fig_dump_dir),
        clear_existing=args.clear_fig_dumps if args.clear_fig_dumps != "all" else "module",
    )

    # Clear existing figures if not retaining past dumps
    if args.clear_fig_dumps == "all":
        fig_dump_manager.clear_all_figures()

    run_analysis_fn = partial(
        run_analysis_module,
        fig_dump_formats=fig_dump_formats,
        no_pickle=args.no_pickle if args.single else True,  #! For now, don't pickle in batched mode
        states_pkl_dir=states_pkl_dir,
        memory_warn_gb=args.memory_warn_gb,
        key=key,
    )

    with progress_session():  # Keep the global `rich` progress region alive
        if args.single:
            module_key = args.single
            module_config = load_config(
                module_key, config_type="analysis", registry=EXPERIMENT_REGISTRY
            )
            fig_dump_dir = fig_dump_manager.prepare_module_dir(module_key, module_config)

            data, common_inputs, all_analyses, all_results, all_figs = run_analysis_fn(
                module_key=module_key,
                module_config=module_config,
                fig_dump_dir=fig_dump_dir,
            )
        else:
            batched_spec = load_batch_config(domain="analysis", config_key=args.batched)

            module_configs_base = {
                module_key: load_config(
                    module_key, config_type="analysis", registry=EXPERIMENT_REGISTRY
                )
                for module_key in batched_spec.keys()
            }

            module_configs = {
                module_key: [
                    deep_merge(module_configs_base[module_key], run_params)
                    for run_params in run_params_list
                ]
                for module_key, run_params_list in batched_spec.items()
            }

            # First: Make sure all the required models are actually in the db, so we don't
            # raise an exception later
            for module_key in batched_spec:
                for module_config in module_configs[module_key]:
                    check_records_for_analysis(module_key, module_config)

            for module_key in batched_spec:
                fig_dump_manager.prepare_module_dir(module_key, module_configs_base[module_key])
                for i, module_config in enumerate(module_configs[module_key]):
                    run_params_list = batched_spec[module_key]

                    logger.info(
                        f"Running analysis module {module_key}, run {i + 1} of {len(run_params_list)}"
                    )

                    fig_dump_dir = fig_dump_manager.prepare_run_dir(module_key, run_params_list[i])

                    # Do not keep results in memory across runs
                    run_analysis_fn(
                        module_key=module_key,
                        module_config=module_config,
                        fig_dump_dir=fig_dump_dir,
                    )

                    jax.clear_caches()

    logger.info("All analyses complete. Exiting.")


if __name__ == "__main__":
    main()
