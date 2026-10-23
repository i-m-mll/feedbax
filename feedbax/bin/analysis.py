#!/usr/bin/env python
"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import argparse
import asyncio
from contextlib import contextmanager, redirect_stdout
import json
import logging
import os
import sys
from pathlib import Path

# NOTE: JAX arrays are not directly picklable if they contain device memory references.
# Since we're using pickle to cache states which may contain JAX arrays, we rely on JAX's
# implicit handling of arrays during pickling (it should work for CPU arrays and most
# host-accessible device arrays).
import plotly.io as pio
import rich

from feedbax.analysis.bundles import (
    AnalysisBundleDeltaSpec,
    AnalysisBundleSpec,
    authored_analysis_bundle_from_payload,
    dry_run_staged_analysis_bundle,
    execute_analysis_bundle,
    execute_staged_analysis_bundle,
    load_analysis_bundle,
    resolve_analysis_bundle_authoring,
)
from feedbax.analysis.exact_parents import migrate_staged_exact_parents
from feedbax.analysis.execution_context import (
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
    StagedManifestRootBinding,
)
from feedbax.analysis.evaluation import (
    EvaluationBatchExecution,
    execute_evaluation_run_matrix,
)
from feedbax.analysis.specs import execute_analysis_run_spec
from feedbax.analysis.reports import (
    ReportRecipeExecutionError,
    execute_authored_report_spec,
)
from feedbax.config import (
    PATHS,
    PLOTLY_CONFIG,
)
from feedbax.config.yaml import get_yaml_loader
from feedbax.contracts.staged_execution import StagedExecutionDescriptor
from feedbax.contracts.run_aliases import RunAliasCatalog
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.plugins.composition import compose_application
from feedbax.bin._setup import setup_application_package
from feedbax.bin.staged_inputs import binding_parts, load_json_object

logger = logging.getLogger(os.path.basename(__file__))

RUN_SUBCOMMAND = "run"
EVALUATE_SUBCOMMAND = "evaluate"
REPORT_SUBCOMMAND = "report"
BUNDLE_SUBCOMMAND = "bundle"


def _load_spec_document(path: Path, *, label: str) -> dict[str, object]:
    """Load a serialized spec document from a JSON or YAML file."""
    if path.suffix.lower() in {".yml", ".yaml"}:
        payload = get_yaml_loader().load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{label} document must be a YAML mapping")
        return dict(payload)
    return load_json_object(path, label=label)


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
        "--run-aliases",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help=(
            "Explicit versioned run-alias catalog; may be repeated. "
            "Aliases expand to authenticated manifest pins before execution."
        ),
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


def run_analysis_run_spec_file(argv: list[str], state: BootstrapState) -> None:
    """Execute one serialized ``AnalysisRunSpec`` file and print its manifest summary."""
    args = build_run_arg_parser().parse_args(argv)
    _apply_plotly_template_default()
    spec = _load_spec_document(args.spec, label="AnalysisRunSpec")
    run_alias_catalogs = [
        RunAliasCatalog.model_validate(_load_spec_document(path, label="run alias catalog"))
        for path in args.run_aliases
    ]
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
            load_json_object(
                args.execution_descriptor,
                label="--execution-descriptor",
            )
        )
    artifact_provider_bindings = [
        StagedArtifactProviderRootBinding(*binding_parts(value, option="--artifact-provider"))
        for value in args.artifact_provider
    ]
    manifest_root_bindings = [
        StagedManifestRootBinding(*binding_parts(value, option="--manifest-root"))
        for value in args.manifest_root
    ]
    checkpoint_custody_bindings = [
        StagedCheckpointCustodyRootBinding(*binding_parts(value, option="--checkpoint-custody"))
        for value in args.checkpoint_custody
    ]
    with _bundle_human_output_to_stderr():
        manifest, path = execute_analysis_run_spec(
            spec,
            registry=state.bundle.analysis_recipes,
            evaluation_registry=state.bundle.evaluation_recipes,
            experiment_registry=state.bundle.experiment_packages,
            root=args.root,
            repo_root=args.repo_root,
            run_alias_catalogs=run_alias_catalogs,
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


def build_evaluate_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feedbax-analysis evaluate",
        description=(
            "Execute a compiled evaluation document — an EvaluationRunMatrixSpec, an "
            "EvaluationRunMatrixDeltaSpec, or a flat EvaluationRunSpec escape hatch — and "
            "write its evaluation-run manifests."
        ),
    )
    parser.add_argument(
        "spec",
        type=Path,
        help=(
            "Path to an EvaluationRunMatrixSpec, EvaluationRunMatrixDeltaSpec, or "
            "EvaluationRunSpec JSON/YAML file."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Manifest root to write the evaluation-run manifests under.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root for resolving content-pinned bases and delta parents.",
    )
    parser.add_argument(
        "--parent-manifest-root",
        type=Path,
        default=None,
        help="Retained manifest root the matrix's authenticated staged parents resolve against.",
    )
    parser.add_argument(
        "--rows",
        type=str,
        default=None,
        help=(
            "Comma-separated ordered subset of matrix row ids to execute through the "
            "governed batch path; the evaluation type must have a registered batch recipe."
        ),
    )
    parser.add_argument(
        "--escape-hatch-reason",
        type=str,
        default=None,
        help="Stated reason required to execute a flat EvaluationRunSpec without a matrix.",
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
    return parser


def run_evaluation_spec_file(argv: list[str], state: BootstrapState) -> None:
    """Execute one serialized evaluation document and print its manifest summary."""
    args = build_evaluate_arg_parser().parse_args(argv)
    _apply_plotly_template_default()
    spec = _load_spec_document(args.spec, label="EvaluationRunMatrixSpec")
    if (args.artifact_provider or args.checkpoint_custody) and (args.execution_descriptor is None):
        raise ValueError(
            "--artifact-provider and --checkpoint-custody require --execution-descriptor"
        )
    execution_descriptor = None
    if args.execution_descriptor is not None:
        execution_descriptor = StagedExecutionDescriptor.model_validate(
            load_json_object(args.execution_descriptor, label="--execution-descriptor")
        )
    artifact_provider_bindings = [
        StagedArtifactProviderRootBinding(*binding_parts(value, option="--artifact-provider"))
        for value in args.artifact_provider
    ]
    checkpoint_custody_bindings = [
        StagedCheckpointCustodyRootBinding(*binding_parts(value, option="--checkpoint-custody"))
        for value in args.checkpoint_custody
    ]
    batch = None
    if args.rows is not None:
        ordered_row_ids = [item.strip() for item in args.rows.split(",") if item.strip()]
        if not ordered_row_ids:
            raise ValueError("--rows requires at least one non-empty row id")
        batch = EvaluationBatchExecution(ordered_row_ids=ordered_row_ids)
    with _bundle_human_output_to_stderr():
        result = execute_evaluation_run_matrix(
            spec,
            registry=state.bundle.evaluation_recipes,
            root=args.root,
            repo_root=args.repo_root,
            escape_hatch_reason=args.escape_hatch_reason,
            parent_manifest_root=args.parent_manifest_root,
            execution_descriptor=execution_descriptor,
            artifact_provider_bindings=artifact_provider_bindings,
            checkpoint_custody_bindings=checkpoint_custody_bindings,
            batch=batch,
        )
    payload = {
        "note": result.note,
        "escape_hatch_reason": result.escape_hatch_reason,
        "rows": [
            {
                "row_id": row.row_id,
                "manifest_id": getattr(row.result, "id", None),
                "manifest_path": str(row.manifest_path) if row.manifest_path else None,
                "status": getattr(row.result, "status", None),
                "artifacts": [
                    artifact.model_dump(mode="json", exclude_none=True)
                    for artifact in row.artifacts
                ],
            }
            for row in result.rows
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_report_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feedbax-analysis report",
        description=("Execute an authored ReportSpec JSON/YAML file against exact staged parents."),
    )
    parser.add_argument(
        "spec",
        type=Path,
        help="Path to an authored ReportSpec JSON/YAML file.",
    )
    parser.add_argument(
        "--exact-parents",
        type=Path,
        required=True,
        help="Versioned authoritative StagedExactParents JSON document.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Explicit retained parent and report manifest/output root.",
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
    return parser


def _report_manifest_payload(manifest, path: Path) -> dict[str, object]:
    return {
        "manifest_id": manifest.id,
        "manifest_path": str(path),
        "status": manifest.status,
        "artifacts": [
            artifact.model_dump(mode="json", exclude_none=True) for artifact in manifest.artifacts
        ],
    }


def run_authored_report_spec_file(argv: list[str], state: BootstrapState) -> None:
    """Execute one serialized authored ``ReportSpec`` with exact staged parents."""
    args = build_report_arg_parser().parse_args(argv)
    _apply_plotly_template_default()
    spec = _load_spec_document(args.spec, label="ReportSpec")
    exact_payload = load_json_object(args.exact_parents, label="--exact-parents")
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
    exact_parents = migrate_staged_exact_parents(exact_payload)
    if (args.artifact_provider or args.checkpoint_custody) and (args.execution_descriptor is None):
        raise ValueError(
            "--artifact-provider and --checkpoint-custody require --execution-descriptor"
        )
    execution_descriptor = None
    if args.execution_descriptor is not None:
        execution_descriptor = StagedExecutionDescriptor.model_validate(
            load_json_object(
                args.execution_descriptor,
                label="--execution-descriptor",
            )
        )
    artifact_provider_bindings = [
        StagedArtifactProviderRootBinding(*binding_parts(value, option="--artifact-provider"))
        for value in args.artifact_provider
    ]
    checkpoint_custody_bindings = [
        StagedCheckpointCustodyRootBinding(*binding_parts(value, option="--checkpoint-custody"))
        for value in args.checkpoint_custody
    ]
    try:
        with _bundle_human_output_to_stderr():
            manifest, path = execute_authored_report_spec(
                spec,
                registry=state.bundle.report_recipes,
                exact_parents=exact_parents,
                root=args.root,
                execution_descriptor=execution_descriptor,
                artifact_provider_bindings=artifact_provider_bindings,
                checkpoint_custody_bindings=checkpoint_custody_bindings,
            )
    except ReportRecipeExecutionError as exc:
        print(
            json.dumps(_report_manifest_payload(exc.manifest, exc.path), indent=2, sort_keys=True)
        )
        raise
    print(json.dumps(_report_manifest_payload(manifest, path), indent=2, sort_keys=True))


def _execute_authored_analysis_bundle(
    authored_bundle: AnalysisBundleSpec | AnalysisBundleDeltaSpec,
    *,
    root: Path | None,
    repo_root: Path | None,
    runs: str | None,
    issues: list[str],
    fig_dump_path: Path,
    fig_dump_formats: list[str],
    exact_parents_path: Path | None,
    execution_descriptor_path: Path | None,
    artifact_provider: list[str],
    checkpoint_custody: list[str],
    dry_run: bool,
    state: BootstrapState,
) -> object:
    """Dispatch one authored bundle on its execution shape and return its JSON payload.

    Shared by the registry-key form (`--bundle`) and the path form (`bundle <spec>`),
    so both reach exactly one executor per bundle shape with identical bindings.
    """
    bundle = authored_bundle
    if isinstance(authored_bundle, AnalysisBundleDeltaSpec):
        bundle, _flattening = resolve_analysis_bundle_authoring(
            authored_bundle,
            repo_root=repo_root,
        )
    run_ids = (
        [item.strip() for item in runs.split(",") if item.strip()] if runs is not None else None
    )
    execution_descriptor = None
    if execution_descriptor_path is not None:
        execution_descriptor = StagedExecutionDescriptor.model_validate(
            load_json_object(execution_descriptor_path, label="--execution-descriptor")
        )
    artifact_provider_bindings = [
        StagedArtifactProviderRootBinding(*binding_parts(value, option="--artifact-provider"))
        for value in artifact_provider
    ]
    checkpoint_custody_bindings = [
        StagedCheckpointCustodyRootBinding(*binding_parts(value, option="--checkpoint-custody"))
        for value in checkpoint_custody
    ]
    execution_kwargs = {
        "root": root,
        "repo_root": repo_root,
        "run_ids": run_ids,
        "issues": issues,
        "fig_dump_path": fig_dump_path,
        "fig_dump_formats": fig_dump_formats,
    }
    binding_kwargs = {
        "execution_descriptor": execution_descriptor,
        "artifact_provider_bindings": artifact_provider_bindings,
        "checkpoint_custody_bindings": checkpoint_custody_bindings,
    }
    if bundle.templates and not bundle.stages:
        if exact_parents_path is not None:
            raise ValueError("--exact-parents is only valid for staged analysis bundles")
        if dry_run:
            raise ValueError("--dry-run is only valid for staged analysis bundles")
        with _bundle_human_output_to_stderr():
            outputs = execute_analysis_bundle(
                authored_bundle,
                registries=state.bundle,
                **execution_kwargs,
                **binding_kwargs,
            )
        return [
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
    if bundle.stages and not bundle.templates:
        exact_parents = None
        if exact_parents_path is not None:
            exact_payload = load_json_object(exact_parents_path, label="--exact-parents")
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
            exact_parents = migrate_staged_exact_parents(exact_payload)
        if dry_run:
            with _bundle_human_output_to_stderr():
                preflight = dry_run_staged_analysis_bundle(
                    authored_bundle,
                    root=execution_kwargs["root"],
                    repo_root=execution_kwargs["repo_root"],
                    run_ids=run_ids,
                    exact_parents=exact_parents,
                    **binding_kwargs,
                )
            return preflight.model_dump(mode="json", exclude_none=True)
        with _bundle_human_output_to_stderr():
            execution = execute_staged_analysis_bundle(
                authored_bundle,
                registries=state.bundle,
                exact_parents=exact_parents,
                **binding_kwargs,
                **execution_kwargs,
            )
        return execution.model_dump(mode="json", exclude_none=True)
    raise ValueError(
        f"Analysis bundle {bundle.name!r} must define exactly one non-empty "
        "execution shape: templates or stages"
    )


def build_bundle_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feedbax-analysis bundle",
        description=(
            "Execute a file-authored AnalysisBundleSpec JSON/YAML document without "
            "requiring a registered experiment package."
        ),
    )
    parser.add_argument(
        "spec",
        type=Path,
        help="Path to an AnalysisBundleSpec or AnalysisBundleDeltaSpec JSON/YAML file.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Manifest root to select parents from and write manifests under.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root for resolving a delta spec's content-pinned parent references.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated manifest IDs to constrain bundle selection.",
    )
    parser.add_argument(
        "--issue",
        action="append",
        default=[],
        help="Issue ID to record on the executed manifests' provenance.",
    )
    parser.add_argument(
        "--fig-dump-dir",
        type=Path,
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
        "--exact-parents",
        type=Path,
        default=None,
        metavar="PATH",
        help=("Versioned exact-parent JSON document for staged bundle execution; requires --root."),
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
        "--dry-run",
        action="store_true",
        help="Preflight a staged bundle without recipe, cache, output, or manifest effects.",
    )
    return parser


def run_analysis_bundle_spec_file(argv: list[str], state: BootstrapState) -> None:
    """Execute one serialized ``AnalysisBundleSpec`` file and print its JSON payload."""
    args = build_bundle_arg_parser().parse_args(argv)
    _apply_plotly_template_default()
    if (args.artifact_provider or args.checkpoint_custody) and (args.execution_descriptor is None):
        raise ValueError(
            "--artifact-provider and --checkpoint-custody require --execution-descriptor"
        )
    if args.exact_parents is not None and args.root is None:
        raise ValueError("--exact-parents requires --root")
    authored_bundle = authored_analysis_bundle_from_payload(
        _load_spec_document(args.spec, label="AnalysisBundleSpec")
    )
    result = _execute_authored_analysis_bundle(
        authored_bundle,
        root=args.root,
        repo_root=args.repo_root,
        runs=args.runs,
        issues=list(args.issue),
        fig_dump_path=args.fig_dump_dir,
        fig_dump_formats=args.fig_dump_formats.split(","),
        exact_parents_path=args.exact_parents,
        execution_descriptor_path=args.execution_descriptor,
        artifact_provider=args.artifact_provider,
        checkpoint_custody=args.checkpoint_custody,
        dry_run=args.dry_run,
        state=state,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run analysis modules on trained models.",
        epilog=(
            f"Subcommands: '{RUN_SUBCOMMAND} <spec.json> [--root ROOT]' executes a serialized "
            f"AnalysisRunSpec file directly; '{EVALUATE_SUBCOMMAND} <spec.json> --root ROOT' "
            "executes a compiled EvaluationRunMatrixSpec/delta/flat EvaluationRunSpec; "
            f"'{REPORT_SUBCOMMAND} <spec.json> --exact-parents PATH --root ROOT' executes an "
            f"authored ReportSpec; '{BUNDLE_SUBCOMMAND} <spec.json> [--root ROOT]' "
            "executes a file-authored AnalysisBundleSpec without a registered package."
        ),
    )
    parser.add_argument(
        "--bundle",
        metavar="BUNDLE_NAME",
        required=True,
        help="Run a manifest-canonical analysis bundle (e.g., rlrmp/standard_matrix)",
    )
    parser.add_argument(
        "--repo-root",
        help="Repository root for resolving a bundle delta's content-pinned parents.",
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
        "--plotly-template",
        type=str,
        default=None,
        help="Plotly template to use for figures (default: 'simple_white').",
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


def main(
    argv: list[str] | None = None,
    *,
    bootstrap_state: BootstrapState | None = None,
    application_setup: bool = True,
) -> None:
    args_in = list(argv or sys.argv[1:])
    state = bootstrap_state or asyncio.run(compose_application())
    if application_setup:
        setup_application_package(
            args_in,
            domain="analysis",
            registry=state.bundle.experiment_packages,
        )
    if args_in and args_in[0] == RUN_SUBCOMMAND:
        run_analysis_run_spec_file(args_in[1:], state)
        return
    if args_in and args_in[0] == EVALUATE_SUBCOMMAND:
        run_evaluation_spec_file(args_in[1:], state)
        return
    if args_in and args_in[0] == REPORT_SUBCOMMAND:
        run_authored_report_spec_file(args_in[1:], state)
        return
    if args_in and args_in[0] == BUNDLE_SUBCOMMAND:
        run_analysis_bundle_spec_file(args_in[1:], state)
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

    fig_dump_formats = args.fig_dump_formats.split(",")
    authored_bundle = load_analysis_bundle(args.bundle, registry=state.bundle.experiment_packages)
    payload = _execute_authored_analysis_bundle(
        authored_bundle,
        root=Path(args.manifest_root) if args.manifest_root else None,
        repo_root=Path(args.repo_root) if args.repo_root else None,
        runs=args.runs,
        issues=list(args.issue),
        fig_dump_path=Path(args.fig_dump_dir),
        fig_dump_formats=fig_dump_formats,
        exact_parents_path=(Path(args.exact_parents) if args.exact_parents is not None else None),
        execution_descriptor_path=(
            Path(args.execution_descriptor) if args.execution_descriptor is not None else None
        ),
        artifact_provider=args.artifact_provider,
        checkpoint_custody=args.checkpoint_custody,
        dry_run=args.dry_run,
        state=state,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
