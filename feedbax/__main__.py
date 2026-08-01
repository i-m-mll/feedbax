"""Command-line entrypoint for ``python -m feedbax``."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import TypeAdapter

from feedbax.contracts.checkpoints import CheckpointForkPlan
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeCompilerError,
    ExperimentEnvelopeRejection,
    dispatch_experiment_envelope,
    missing_outputs,
)
from feedbax.contracts.project_extension import (
    ProjectExtensionDeclarationError,
    resolve_authored_extension_labels,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.run_matrix import ExecutionDependency
from feedbax.contracts.training import (
    TrainingMethodRegistry,
    TrainingRunSpec,
    resolve_training_run_spec,
    validate_training_run_spec_semantics,
)
from feedbax.plugins.composition import compose_application
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.orchestration.events import RunEventEmitter
from feedbax.training.executor import (
    execute_training_run_spec,
    prepare_training_manifest_metadata_projection,
)
from feedbax.training.interruption import RunInterruptionController
from feedbax.training.preparation import (
    ExecutionPreparationPlan,
    ExecutionPreparationRequest,
    lower_zero_level_preparation_plan,
    materialize_execution_preparation,
    require_execution_preparation_provider,
)
from feedbax.training.worker_validation import resolve_execution_mapping
from feedbax.training.manifest_preflight import (
    preflight_training_run_manifest_payloads,
    validate_training_run_spec,
)
from feedbax.training.checkpoint_custody import (
    CheckpointForkPlanBindings,
    RunContractProjectionEvidence,
    classify_checkpoint_fork_plan_derived_digests,
    fork_checkpoint_plan,
    fork_checkpoint_transaction,
    relock_checkpoint_fork_plan_file,
    validate_checkpoint_fork_execution_dependencies,
)
from feedbax.training.legacy_checkpoint_adoption import (
    ManifestDumpRequest,
    PathMappingRule,
    adopt_legacy_checkpoint,
    dump_leaf_manifests_via_worktrees,
    load_leaf_manifest,
    load_path_mapping_registry,
)


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_pickle(path: str) -> Any:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def _load_checkpoint_fork_plan_bindings(
    path: str,
    training_method_registry: TrainingMethodRegistry,
) -> CheckpointForkPlanBindings:
    manifest_path = Path(path).resolve()
    payload = _read_json(str(manifest_path))
    expected = "feedbax.runtime.checkpoint_fork_plan_bindings"
    if (payload.get("schema_id"), payload.get("schema_version")) != (expected, f"{expected}.v1"):
        raise ValueError("checkpoint fork binding manifest has unsupported schema identity")
    base = manifest_path.parent

    def resolved(raw: str) -> Path:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else base / candidate

    def loaded(name: str, loader: Any) -> dict[str, Any]:
        return {key: loader(str(resolved(value))) for key, value in payload.get(name, {}).items()}

    run_specs = {
        key: validate_training_run_spec_semantics(
            TrainingRunSpec.model_validate(value),
            training_method_registry,
        )
        for key, value in loaded("run_specs", _read_json).items()
    }
    return CheckpointForkPlanBindings(
        checkpoint_roots={
            key: resolved(value) for key, value in payload["checkpoint_roots"].items()
        },
        run_specs=run_specs,
        slot_templates=loaded("slot_templates", _read_pickle),
        segment_history_templates=loaded("segment_history_templates", _read_pickle),
        population_member_ids=loaded("population_member_ids", _read_json),
    )


def _load_run_contract_historical_evidence(
    path: str | None,
) -> dict[str, RunContractProjectionEvidence]:
    """Load a JSON mapping of target_id to authenticated historical evidence."""
    if path is None:
        return {}
    payload = _read_json(path)
    entries = payload.get("targets", payload) if isinstance(payload, Mapping) else {}
    evidence: dict[str, RunContractProjectionEvidence] = {}
    for target_id, item in entries.items():
        evidence[target_id] = RunContractProjectionEvidence(
            canonical_projection=item["canonical_projection"],
            recorded_sha256=item["recorded_sha256"],
            recorded_algorithm_version=item["recorded_algorithm_version"],
            training_run_spec_schema_id=item["training_run_spec_schema_id"],
            training_run_spec_schema_version=item["training_run_spec_schema_version"],
            evidence_refs=tuple(item.get("evidence_refs", ())),
        )
    return evidence


def _load_path_mapping(
    path: str | None,
) -> tuple[tuple[PathMappingRule, ...], tuple[PathMappingRule, ...]]:
    if path is None:
        return (), ()
    payload = _read_json(path)
    if isinstance(payload, list):
        rules = tuple(PathMappingRule(**item) for item in payload)
        return rules, rules
    if "schema_id" in payload or "rules" in payload:
        registry = load_path_mapping_registry(path)
        return registry.rules_for("model"), registry.rules_for("optimizer")
    model_rules = tuple(PathMappingRule(**item) for item in payload.get("model", ()))
    optimizer_rules = tuple(PathMappingRule(**item) for item in payload.get("optimizer", ()))
    return model_rules, optimizer_rules


def _load_callable(ref: str | None):
    if ref is None:
        return None
    module_name, _, function_name = ref.partition(":")
    if not module_name or not function_name:
        raise ValueError("callable references must be module:function")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _load_slot_transforms(refs: Sequence[str] | None) -> dict[str, Any]:
    transforms: dict[str, Any] = {}
    for ref in refs or ():
        slot, sep, callable_ref = ref.partition("=")
        if not sep or not slot or not callable_ref:
            raise ValueError("--slot-transform entries must use SLOT=module:function")
        transforms[slot] = _load_callable(callable_ref)
    return transforms


def _checkpoint_fork_targets(args: argparse.Namespace) -> list[str]:
    targets: list[str] = []
    targets.extend(args.target or ())
    targets.extend(args.targets or ())
    if not targets:
        raise ValueError("checkpoint fork requires at least one --target or --targets entry")
    return targets


def _parse_checkpoint_fork_target(raw: str) -> tuple[Path, Path]:
    spec, sep, root = raw.partition(":")
    if not sep or not spec or not root:
        raise ValueError("checkpoint fork targets must use '<run-spec-json>:<checkpoint-root>'")
    return Path(spec), Path(root)


def _progress_loss(metrics: Mapping[str, Any]) -> float | None:
    value = metrics.get("train_loss")
    if value is None:
        for key, candidate in metrics.items():
            if str(key).endswith("loss"):
                value = candidate
                break
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _console_progress_printer(started_at: float):
    def print_progress(event: Mapping[str, Any]) -> None:
        coordinate = event.get("coordinate", {})
        coordinate = coordinate if isinstance(coordinate, Mapping) else {}
        metrics = event.get("metrics", {})
        metrics = metrics if isinstance(metrics, Mapping) else {}
        batch = int(coordinate.get("program_step") or 0)
        phase = str(coordinate.get("phase") or "unknown")
        loss = _progress_loss(metrics)
        loss_text = "nan" if loss is None else f"{loss:.6g}"
        elapsed = time.perf_counter() - started_at
        print(
            f"phase={phase} batch={batch} loss={loss_text} elapsed={elapsed:.2f}s",
            file=sys.stderr,
            flush=True,
        )

    return print_progress


def _dump_manifest_requests(
    specs: Sequence[str],
    *,
    commit: str,
    output: str | None,
    batch: Sequence[str] | None = None,
) -> list[ManifestDumpRequest]:
    if not specs and not batch:
        raise ValueError("dump-manifest requires at least one --spec")
    if specs and not commit:
        raise ValueError("dump-manifest requires --commit")
    if output is not None and len(specs) != 1:
        raise ValueError("--output may only be used with one --spec")
    requests: list[ManifestDumpRequest] = []
    for spec in specs:
        spec_path = Path(spec)
        output_path = (
            Path(output)
            if output is not None
            else spec_path.with_suffix(spec_path.suffix + ".leaf_manifest.json")
        )
        requests.append(
            ManifestDumpRequest(commit=commit, spec_path=spec_path, output_path=output_path)
        )
    for raw_batch in batch or ():
        payload = json.loads(raw_batch)
        requests.append(
            ManifestDumpRequest(
                commit=str(payload["commit"]),
                spec_path=Path(payload["spec"]),
                output_path=Path(payload["output"]),
            )
        )
    return requests


def _preflight_experiment_envelope(args: argparse.Namespace, registries: Any) -> int:
    """Route one authored envelope to its registered downstream compiler.

    Exit codes are the documented authoring contract: 0 accepted, 2 rejected
    with an actionable diagnostic on stderr, 1 infrastructure failure.
    """
    envelope_path = Path(args.envelope)
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    try:
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read experiment envelope {envelope_path}: {exc}", file=sys.stderr)
        return 1
    try:
        # Extension labels resolve against the declaring project before the
        # compiler runs, so an unresolved label rejects before any output.
        resolve_authored_extension_labels(envelope, registries.project_extensions)
        result = dispatch_experiment_envelope(
            envelope,
            registries.experiment_envelope_compilers,
            envelope_path=envelope_path,
            repo_root=repo_root,
            out_dir=out_dir,
        )
    except ExperimentEnvelopeRejection as rejection:
        print(rejection.render(), file=sys.stderr)
        return 2
    except ProjectExtensionDeclarationError as exc:
        print(f"project extension declaration failed: {exc}", file=sys.stderr)
        return 1
    except ExperimentEnvelopeCompilerError as exc:
        print(f"experiment envelope dispatch failed: {exc}", file=sys.stderr)
        return 1
    absent = missing_outputs(result, out_dir)
    if absent:
        print(
            f"compiler reported outputs it did not write: {list(absent)}",
            file=sys.stderr,
        )
        return 1
    json.dump(
        result.model_dump(mode="json", exclude_none=True),
        fp=sys.stdout,
        indent=2,
        sort_keys=True,
    )
    print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    command_started_at = time.perf_counter()
    parser = argparse.ArgumentParser(prog="python -m feedbax")
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser(
        "execute-training-run-spec",
        help="Execute a TrainingRunSpec and emit a native TrainingRunManifest.",
    )
    execute_parser.add_argument("spec", help="TrainingRunSpec JSON path")
    execute_parser.add_argument("--manifest-root", help="Manifest root override")
    execute_parser.add_argument("--checkpoint-root", help="Checkpoint custody root override")
    execute_parser.add_argument("--run-id", help="Stable run id override")
    execute_parser.add_argument("--initial-slots", help="JSON path for simple initial slots")
    execute_parser.add_argument(
        "--training-payload",
        help="Optional external training payload JSON",
    )
    execute_parser.add_argument("--training-payload-kind", default="TrainingRunSpec")
    execute_parser.add_argument("--training-payload-schema-id")
    execute_parser.add_argument("--training-payload-schema-version")
    execute_parser.add_argument("--training-payload-ref")
    execute_parser.add_argument(
        "--manifest-metadata-projection",
        help="Governed training-manifest metadata projection envelope JSON path",
    )
    execution_context_group = execute_parser.add_mutually_exclusive_group()
    execution_context_group.add_argument(
        "--execution-context",
        help=(
            "NativeExecutionProducerContext JSON carrying assembly identities, row "
            "provenance, and diagnostics inputs"
        ),
    )
    execution_context_group.add_argument(
        "--execution-context-json",
        help="Inline NativeExecutionProducerContext JSON supplied by orchestration",
    )
    execute_parser.add_argument(
        "--plugin",
        action="append",
        help=(
            "Import a module that registers Feedbax training methods or governed "
            "manifest metadata projections before validation; may be repeated."
        ),
    )
    execute_parser.add_argument("--resume", action="store_true")
    execute_parser.add_argument("--stop-after-barrier")
    execute_parser.add_argument(
        "--update-budget",
        type=int,
        help=(
            "Execute at most this positive delta in completed training batches, "
            "checkpoint the terminal batch, and exit successfully."
        ),
    )
    execute_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the default stderr progress printer.",
    )
    preflight_parser = subparsers.add_parser(
        "preflight-training-run-manifest",
        help="Validate final TrainingRunManifest spec payloads without training.",
    )
    preflight_parser.add_argument("spec", help="TrainingRunSpec JSON path")
    preflight_parser.add_argument("--row-id", help="Row id for failure context")
    preflight_parser.add_argument(
        "--training-payload",
        help="Optional external training payload JSON",
    )
    preflight_parser.add_argument("--training-payload-kind", default="TrainingRunSpec")
    preflight_parser.add_argument("--training-payload-schema-id")
    preflight_parser.add_argument("--training-payload-schema-version")
    preflight_parser.add_argument("--training-payload-ref")
    preflight_parser.add_argument(
        "--manifest-metadata-projection",
        help="Governed training-manifest metadata projection envelope JSON path",
    )
    preflight_parser.add_argument("--task-binding-spec")
    preflight_parser.add_argument(
        "--plugin",
        action="append",
        help=(
            "Import a module that registers Feedbax training methods or governed "
            "manifest metadata projections before validation; may be repeated."
        ),
    )
    envelope_parser = subparsers.add_parser(
        "preflight-experiment-envelope",
        help=(
            "Compile one authored experiment envelope through the registered downstream "
            "compiler claiming its schema."
        ),
    )
    envelope_parser.add_argument("envelope", help="Authored experiment envelope JSON path")
    envelope_parser.add_argument(
        "--out-dir",
        default="generated",
        help="Directory the compiler writes its compile lock and document into.",
    )
    envelope_parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root the envelope's relative references resolve against.",
    )
    envelope_parser.add_argument(
        "--plugin",
        action="append",
        help=(
            "Import a module that registers a downstream experiment envelope compiler "
            "before dispatch; may be repeated."
        ),
    )
    adopt_root = subparsers.add_parser(
        "adopt-legacy-checkpoint",
        help="Dump or adopt legacy Equinox tree_serialise_leaves checkpoint streams.",
    )
    adopt_subparsers = adopt_root.add_subparsers(dest="adopt_command", required=True)
    dump_parser = adopt_subparsers.add_parser(
        "dump-manifest",
        help=(
            "Create a producing-commit worktree and run a downstream builder hook "
            "there to emit a LeafManifest JSON file."
        ),
    )
    dump_parser.add_argument("--commit", help="Producing Git commit")
    dump_parser.add_argument(
        "--spec",
        action="append",
        help="Run spec/config path to pass to the producing-commit builder hook",
    )
    dump_parser.add_argument("--repo", default=".", help="Repository root for git worktree")
    dump_parser.add_argument(
        "--builder",
        required=True,
        help="Old-checkout Python hook as module:function; returns model and optimizer templates",
    )
    dump_parser.add_argument("--output", help="Output manifest path for a single --spec")
    dump_parser.add_argument(
        "--batch",
        action="append",
        help="JSON object with commit, spec, and output fields; may be repeated.",
    )
    dump_parser.add_argument(
        "--skip-uv-sync",
        action="store_true",
        help="Reuse the producing checkout environment instead of running uv sync.",
    )
    adopt_parser = adopt_subparsers.add_parser(
        "adopt",
        help=(
            "Adopt manifest-verified legacy model/optimizer .eqx streams into "
            "current checkpoint custody."
        ),
    )
    adopt_parser.add_argument("--manifest", required=True, help="LeafManifest JSON path")
    adopt_parser.add_argument("--model-stream", required=True, help="Legacy model.eqx path")
    adopt_parser.add_argument("--optimizer-stream", help="Legacy optimizer_state.eqx path")
    adopt_parser.add_argument(
        "--fresh-optimizer",
        action="store_true",
        help="Keep the current optimizer template instead of adopting optimizer_state.eqx.",
    )
    adopt_parser.add_argument(
        "--current-slots",
        required=True,
        help="Pickle containing current checkpoint slot templates keyed by slot name",
    )
    adopt_parser.add_argument("--run-spec", required=True, help="Current TrainingRunSpec JSON")
    adopt_parser.add_argument("--checkpoint-root", required=True, help="Custody root to write")
    adopt_parser.add_argument("--barrier", required=True, help="Checkpoint barrier name")
    adopt_parser.add_argument("--run-id", required=True, help="Run id for checkpoint metadata")
    adopt_parser.add_argument("--phase", required=True, help="Completed phase name")
    adopt_parser.add_argument("--global-step", required=True, type=int)
    adopt_parser.add_argument("--completed-barrier", required=True)
    adopt_parser.add_argument("--model-slot", default="model")
    adopt_parser.add_argument("--optimizer-slot", default="optimizer")
    adopt_parser.add_argument("--path-mapping", help="Optional path mapping registry JSON")
    adopt_parser.add_argument(
        "--plugin",
        action="append",
        help=(
            "Import a module that registers Feedbax training methods before "
            "TrainingRunSpec validation; may be repeated."
        ),
    )
    adopt_parser.add_argument(
        "--resume-slot-transform",
        help=(
            "Optional current-environment module:function that transforms loaded slots "
            "before strict round-trip validation, for downstream optimizer resize hooks."
        ),
    )
    checkpoint_root = subparsers.add_parser(
        "checkpoint",
        help="Checkpoint custody maintenance commands.",
    )
    checkpoint_subparsers = checkpoint_root.add_subparsers(
        dest="checkpoint_command",
        required=True,
    )
    fork_parser = checkpoint_subparsers.add_parser(
        "fork",
        help="Fork one custody checkpoint to one or more target run contracts.",
    )
    fork_parser.add_argument("--source", required=True, help="Source checkpoint root")
    fork_parser.add_argument(
        "--target",
        action="append",
        help="Target as '<TrainingRunSpec JSON>:<checkpoint root>'; may be repeated.",
    )
    fork_parser.add_argument(
        "--targets",
        nargs="+",
        help="One or more '<TrainingRunSpec JSON>:<checkpoint root>' targets.",
    )
    fork_parser.add_argument(
        "--expected-slots",
        help=(
            "Optional pickle containing target checkpoint slot templates. If omitted, "
            "the source slots after transforms are used as strict-load templates."
        ),
    )
    fork_parser.add_argument(
        "--slot-transform",
        action="append",
        help="Per-slot transform as SLOT=module:function; may be repeated.",
    )
    fork_parser.add_argument(
        "--plugin",
        action="append",
        help=(
            "Import a module that registers Feedbax training methods before "
            "TrainingRunSpec validation; may be repeated."
        ),
    )
    fork_parser.add_argument("--tool-version", help="Tool version to record in provenance")
    plan_parser = checkpoint_subparsers.add_parser(
        "fork-plan",
        help="Execute one content-pinned portable checkpoint fork plan.",
    )
    plan_parser.add_argument("--plan", required=True, help="CheckpointForkPlan JSON")
    plan_parser.add_argument(
        "--dependencies", required=True, help="Typed execution-dependency JSON or JSON path"
    )
    plan_parser.add_argument("--bindings", required=True, help="Runtime bindings v1 JSON")
    plan_parser.add_argument(
        "--plugin",
        action="append",
        help="Import a module that registers methods or durable fork transforms.",
    )
    relock_parser = checkpoint_subparsers.add_parser(
        "relock",
        help="Classify or migrate a content-pinned checkpoint fork plan's derived digests.",
    )
    relock_parser.add_argument("--plan", required=True, help="CheckpointForkPlan JSON path")
    relock_parser.add_argument("--bindings", required=True, help="Runtime bindings v1 JSON")
    relock_mode = relock_parser.add_mutually_exclusive_group(required=True)
    relock_mode.add_argument(
        "--check",
        action="store_true",
        help="Report the per-field drift classification and exit nonzero on any drift.",
    )
    relock_mode.add_argument(
        "--write",
        action="store_true",
        help="Migrate supported run-contract algorithm drift and rewrite the plan in place.",
    )
    relock_parser.add_argument(
        "--historical-evidence",
        help=(
            "JSON path mapping target_id to authenticated historical run-contract "
            "projection evidence."
        ),
    )
    relock_parser.add_argument(
        "--requalification",
        action="append",
        help="Required re-qualification duty recorded with the migration; may be repeated.",
    )
    relock_parser.add_argument(
        "--owner-attestation",
        action="store_true",
        help=(
            "Enter the explicit owner-attestation path when authenticated historical "
            "evidence is unavailable; records proof_mode 'owner-attestation'."
        ),
    )
    relock_parser.add_argument(
        "--plugin",
        action="append",
        help="Import a module that registers methods before run-spec validation.",
    )

    harness_parser = subparsers.add_parser(
        "matrix-harness",
        help="Materialize evaluation condition rows with manifests and custody records.",
    )
    harness_parser.add_argument(
        "spec", help="EvaluationRunMatrixSpec or EvaluationRunMatrixDeltaSpec JSON path"
    )
    harness_parser.add_argument("--manifest-root", required=True)
    harness_parser.add_argument("--repo-root")
    harness_parser.add_argument(
        "--plugin",
        action="append",
        help=(
            "Import a module that registers Feedbax training methods and analysis recipes; "
            "may be repeated."
        ),
    )
    harness_parser.add_argument(
        "--escape-hatch-reason",
        help="Required reason when materializing a flat spec outside the row model.",
    )
    harness_parser.add_argument("--parent-manifest-root")
    harness_parser.add_argument("--execution-descriptor")
    harness_parser.add_argument("--artifact-provider", action="append")
    harness_parser.add_argument("--checkpoint-custody", action="append")
    harness_parser.add_argument("--orchestration-bundle")
    harness_parser.add_argument("--orchestration-inputs-root")
    harness_parser.add_argument("--orchestration-row-id")
    harness_parser.add_argument("--lifecycle-result")
    harness_parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Execute the whole matrix through its registered batch recipe; this bypasses "
            "the execution policy's per-row row-count refusal."
        ),
    )
    harness_parser.add_argument(
        "--locked-spec",
        help=(
            "Require the runtime matrix's exact ordered row_id sequence to equal the locked "
            "evaluation matrix sequence; the locked document may be either authoring kind."
        ),
    )
    harness_parser.add_argument(
        "--execution-policy",
        help=(
            "JSON policy requiring positive-integer per_row_max_rows, non-blank "
            "threshold_source, and override_flag set to --allow-large-per-row."
        ),
    )
    harness_parser.add_argument(
        "--allow-large-per-row",
        action="store_true",
        help=(
            "Explicitly bypass an execution policy's oversized per-row refusal; the decision "
            "is recorded in matrix-harness evidence."
        ),
    )

    args = parser.parse_args(argv)
    bootstrap_state = asyncio.run(
        compose_application(modules=tuple(getattr(args, "plugin", None) or ()))
    )
    registries = bootstrap_state.bundle
    if args.command == "matrix-harness":
        from feedbax.analysis.harness import main as harness_main

        harness_argv = [args.spec, "--manifest-root", args.manifest_root]
        if args.repo_root is not None:
            harness_argv.extend(("--repo-root", args.repo_root))
        for module_name in args.plugin or ():
            harness_argv.extend(("--plugin", module_name))
        if args.escape_hatch_reason is not None:
            harness_argv.extend(("--escape-hatch-reason", args.escape_hatch_reason))
        if args.parent_manifest_root is not None:
            harness_argv.extend(("--parent-manifest-root", args.parent_manifest_root))
        if args.execution_descriptor is not None:
            harness_argv.extend(("--execution-descriptor", args.execution_descriptor))
        for binding in args.artifact_provider or ():
            harness_argv.extend(("--artifact-provider", binding))
        for binding in args.checkpoint_custody or ():
            harness_argv.extend(("--checkpoint-custody", binding))
        for option, value in (
            ("--orchestration-bundle", args.orchestration_bundle),
            ("--orchestration-inputs-root", args.orchestration_inputs_root),
            ("--orchestration-row-id", args.orchestration_row_id),
            ("--lifecycle-result", args.lifecycle_result),
        ):
            if value is not None:
                harness_argv.extend((option, value))
        if args.batch:
            harness_argv.append("--batch")
        if args.locked_spec is not None:
            harness_argv.extend(("--locked-spec", args.locked_spec))
        if args.execution_policy is not None:
            harness_argv.extend(("--execution-policy", args.execution_policy))
        if args.allow_large_per_row:
            harness_argv.append("--allow-large-per-row")
        return harness_main(harness_argv, bootstrap_state=bootstrap_state)
    if args.command == "preflight-experiment-envelope":
        return _preflight_experiment_envelope(args, registries)
    if args.command == "execute-training-run-spec":
        run_spec = validate_training_run_spec(_read_json(args.spec))
        resolved_method = resolve_training_run_spec(run_spec, registries.training_methods)
        method_registration = resolved_method.registration
        preparation_registration = registries.execution_preparations.get(run_spec.method_ref.key)
        mapping_levels, _slot_axis_bindings = resolve_execution_mapping(run_spec.worker_execution)
        if method_registration.requires_execution_preparation:
            require_execution_preparation_provider(
                method_ref=run_spec.method_ref.key,
                preparation_registry=registries.execution_preparations,
            )
        if mapping_levels and args.initial_slots:
            raise ValueError(
                "--initial-slots is invalid when mapping_levels are active; mapped runs require "
                "a Feedbax-owned execution-preparation plan"
            )
        if preparation_registration is not None and args.initial_slots:
            raise ValueError(
                "--initial-slots cannot be combined with an execution-preparation provider for "
                f"method_ref {run_spec.method_ref.key!r}"
            )
        initial_slots = _read_json(args.initial_slots) if args.initial_slots else None
        preparation = None
        if preparation_registration is not None:
            prepared = registries.execution_preparations.prepare(
                ExecutionPreparationRequest(
                    run_spec=run_spec,
                    method_payload=resolved_method.payload,
                    method_contract=resolved_method.contract,
                    effective_phase=resolved_method.effective_phase,
                    run_id=args.run_id,
                    resume=args.resume,
                )
            )
            if isinstance(prepared, ExecutionPreparationPlan):
                if mapping_levels:
                    prepared = materialize_execution_preparation(
                        ExecutionPreparationRequest(
                            run_spec=run_spec,
                            method_payload=resolved_method.payload,
                            method_contract=resolved_method.contract,
                            effective_phase=resolved_method.effective_phase,
                            run_id=args.run_id,
                            resume=args.resume,
                        ),
                        prepared,
                        provider_identity=preparation_registration.owner,
                    )
                else:
                    prepared = lower_zero_level_preparation_plan(
                        prepared, resume_template=args.resume
                    )
            preparation = prepared
            initial_slots = None
        training_payload = _read_json(args.training_payload) if args.training_payload else None
        metadata_projection = (
            _read_json(args.manifest_metadata_projection)
            if args.manifest_metadata_projection
            else None
        )
        execution_context = (
            _read_json(args.execution_context)
            if args.execution_context
            else (json.loads(args.execution_context_json) if args.execution_context_json else None)
        )
        started_at = time.perf_counter()
        emitter = RunEventEmitter.from_env(render_batch_lines=False)
        with RunInterruptionController() as interruption:
            try:
                result = execute_training_run_spec(
                    run_spec,
                    run_id=args.run_id,
                    preparation=preparation,
                    initial_slots=initial_slots,
                    manifest_root=args.manifest_root,
                    checkpoint_root=args.checkpoint_root,
                    registry=registries.training_methods,
                    training_spec_payload=training_payload,
                    training_spec_payload_kind=args.training_payload_kind,
                    training_spec_payload_schema_id=args.training_payload_schema_id,
                    training_spec_payload_schema_version=args.training_payload_schema_version,
                    training_spec_payload_ref=args.training_payload_ref,
                    manifest_metadata_projection=metadata_projection,
                    resume=args.resume,
                    stop_after_barrier=args.stop_after_barrier,
                    update_budget=args.update_budget,
                    progress_callback=(
                        None if args.no_progress else _console_progress_printer(started_at)
                    ),
                    run_event_emitter=emitter,
                    cancellation_probe=lambda _coordinate: interruption.poll(),
                    execution_started_at_monotonic=command_started_at,
                    execution_start_semantics="worker_command_entry",
                    execution_context=execution_context,
                )
            finally:
                if emitter is not None:
                    emitter.close()
        json.dump(
            {
                "run_id": result.run_id,
                "status": result.status,
                "start_completed_batches": result.start_completed_batches,
                "end_completed_batches": result.end_completed_batches,
                "payload_binding_status": result.payload_binding_status,
                "manifest_path": str(result.manifest_path),
                "manifest_payload": result.manifest.model_dump(mode="json", exclude_none=True),
            },
            fp=sys.stdout,
            indent=2,
            sort_keys=True,
        )
        print()
        return 1 if result.status == "failed" else 0
    if args.command == "preflight-training-run-manifest":
        training_payload = _read_json(args.training_payload) if args.training_payload else None
        metadata_projection = (
            _read_json(args.manifest_metadata_projection)
            if args.manifest_metadata_projection
            else None
        )
        task_binding_spec = _read_json(args.task_binding_spec) if args.task_binding_spec else None
        run_spec = validate_training_run_spec(_read_json(args.spec))
        payloads = preflight_training_run_manifest_payloads(
            run_spec,
            training_spec_payload=training_payload,
            training_spec_payload_kind=args.training_payload_kind,
            training_spec_payload_schema_id=args.training_payload_schema_id,
            training_spec_payload_schema_version=args.training_payload_schema_version,
            training_spec_payload_ref=args.training_payload_ref,
            task_binding_spec=task_binding_spec,
            row_id=args.row_id,
            spec_path=args.spec,
        )
        projection_custody = prepare_training_manifest_metadata_projection(
            metadata_projection,
            registry=registries.training_methods,
            run_spec=run_spec,
            training_spec_payload=training_payload,
            training_spec_payload_kind=args.training_payload_kind,
            training_spec_payload_schema_id=args.training_payload_schema_id,
            training_spec_payload_schema_version=args.training_payload_schema_version,
        )
        output = payloads.model_dump()
        if projection_custody is not None:
            output["metadata_projection_custody"] = projection_custody.model_dump(
                mode="json",
                exclude_none=True,
            )
        json.dump(output, fp=sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    if args.command == "adopt-legacy-checkpoint":
        if args.adopt_command == "dump-manifest":
            results = dump_leaf_manifests_via_worktrees(
                _dump_manifest_requests(
                    args.spec or (),
                    commit=args.commit or "",
                    output=args.output,
                    batch=args.batch,
                ),
                repo=args.repo,
                builder=args.builder,
                run_uv_sync=not args.skip_uv_sync,
            )
            json.dump(
                {
                    "manifests": [
                        {
                            "commit": result.commit,
                            "spec": str(result.spec_path),
                            "output": str(result.output_path),
                        }
                        for result in results
                    ]
                },
                fp=sys.stdout,
                indent=2,
                sort_keys=True,
            )
            print()
            return 0
        if args.adopt_command == "adopt":
            run_spec = validate_training_run_spec_semantics(
                TrainingRunSpec.model_validate(_read_json(args.run_spec)),
                registries.training_methods,
            )
            phase_program = run_spec.worker_execution.method_contract.phase_program
            model_mapping, optimizer_mapping = _load_path_mapping(args.path_mapping)
            result = adopt_legacy_checkpoint(
                checkpoint_root=args.checkpoint_root,
                run_spec=run_spec,
                phase_program=phase_program,
                barrier_name=args.barrier,
                coordinate=ProgressCoordinate(
                    run_id=args.run_id,
                    phase=args.phase,
                    program_step=args.program_step,
                    completed_barrier=args.completed_barrier,
                ),
                current_slots=_read_pickle(args.current_slots),
                leaf_manifest=load_leaf_manifest(args.manifest),
                model_stream=args.model_stream,
                optimizer_stream=args.optimizer_stream,
                model_slot=args.model_slot,
                optimizer_slot=args.optimizer_slot,
                fresh_optimizer=args.fresh_optimizer,
                model_mapping_rules=model_mapping,
                optimizer_mapping_rules=optimizer_mapping,
                resume_slot_transform=_load_callable(args.resume_slot_transform),
            )
            json.dump(
                {
                    "transaction_id": result.write.manifest.transaction_id,
                    "manifest_path": str(result.write.manifest_path),
                    "latest_pointer_path": str(result.write.latest_pointer_path),
                    "model_assigned_paths": list(result.model_report.assigned_paths),
                    "optimizer_assigned_paths": (
                        list(result.optimizer_report.assigned_paths)
                        if result.optimizer_report is not None
                        else []
                    ),
                    "model_static_paths": [
                        report.__dict__ for report in result.model_report.static_paths
                    ],
                },
                fp=sys.stdout,
                indent=2,
                sort_keys=True,
            )
            print()
            return 0
    if args.command == "checkpoint":
        if args.checkpoint_command == "fork-plan":
            plan = CheckpointForkPlan.model_validate(
                default_spec_registry.migrate("CheckpointForkPlan", _read_json(args.plan)).payload
            )
            dependency_json = (
                args.dependencies
                if args.dependencies.lstrip().startswith("[")
                else Path(args.dependencies).read_text(encoding="utf-8")
            )
            dependencies = TypeAdapter(list[ExecutionDependency]).validate_json(dependency_json)
            validate_checkpoint_fork_execution_dependencies(plan, dependencies)
            bindings = _load_checkpoint_fork_plan_bindings(
                args.bindings,
                registries.training_methods,
            )
            results = fork_checkpoint_plan(plan, bindings)
            json.dump(
                {
                    "targets": {
                        target_id: {
                            "transaction_id": result.manifest.transaction_id,
                        }
                        for target_id, result in sorted(results.items())
                    },
                },
                fp=sys.stdout,
                indent=2,
                sort_keys=True,
            )
            print()
            return 0
        if args.checkpoint_command == "relock":
            bindings = _load_checkpoint_fork_plan_bindings(
                args.bindings,
                registries.training_methods,
            )
            evidence = _load_run_contract_historical_evidence(args.historical_evidence)
            if args.write:
                result = relock_checkpoint_fork_plan_file(
                    args.plan,
                    bindings,
                    requalification_requirements=args.requalification or (),
                    historical_evidence=evidence,
                    owner_attestation=args.owner_attestation,
                )
                json.dump(
                    {
                        "status": "migrated",
                        "source_plan_canonical_sha256": result.source_plan_canonical_sha256,
                        "target_plan_canonical_sha256": result.target_plan_canonical_sha256,
                        "record": result.record.model_dump(mode="json", exclude_none=True),
                    },
                    fp=sys.stdout,
                    indent=2,
                    sort_keys=True,
                )
                print()
                return 0
            plan = CheckpointForkPlan.model_validate(
                default_spec_registry.migrate("CheckpointForkPlan", _read_json(args.plan)).payload
            )
            classification = classify_checkpoint_fork_plan_derived_digests(
                plan, bindings, historical_evidence=evidence
            )
            has_drift = any(not item.is_unchanged for item in classification.classifications)
            json.dump(
                {
                    "status": "drift" if has_drift else "clean",
                    "targets": {
                        item.target_id: {
                            "run_contract_class": item.run_contract_class,
                            "sentinel_input_drift_paths": list(item.sentinel_input_drift_paths),
                            "proof_mode": item.proof_mode,
                            "difference_paths": list(item.difference_paths),
                        }
                        for item in classification.classifications
                    },
                },
                fp=sys.stdout,
                indent=2,
                sort_keys=True,
            )
            print()
            return 1 if has_drift else 0
        if args.checkpoint_command == "fork":
            expected_slots = _read_pickle(args.expected_slots) if args.expected_slots else None
            slot_transforms = _load_slot_transforms(args.slot_transform)
            target_summaries: list[dict[str, Any]] = []
            had_error = False
            for raw_target in _checkpoint_fork_targets(args):
                summary: dict[str, Any] = {"target": raw_target}
                try:
                    spec_path, checkpoint_root = _parse_checkpoint_fork_target(raw_target)
                    run_spec = validate_training_run_spec_semantics(
                        TrainingRunSpec.model_validate(_read_json(str(spec_path))),
                        registries.training_methods,
                    )
                    phase_program = run_spec.worker_execution.method_contract.phase_program
                    result = fork_checkpoint_transaction(
                        args.source,
                        checkpoint_root,
                        target_run_spec=run_spec,
                        target_phase_program=phase_program,
                        expected_slots=expected_slots,
                        slot_transforms=slot_transforms,
                        tool_version=args.tool_version,
                    )
                    summary.update(
                        {
                            "status": "ok",
                            "spec": str(spec_path),
                            "root": str(checkpoint_root),
                            "transaction_id": result.manifest.transaction_id,
                            "manifest_path": str(result.manifest_path),
                            "latest_pointer_path": str(result.latest_pointer_path),
                            "slot_transfer_modes": dict(result.slot_transfer_modes),
                        }
                    )
                except Exception as exc:
                    had_error = True
                    summary.update(
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                target_summaries.append(summary)
            json.dump(
                {"targets": target_summaries},
                fp=sys.stdout,
                indent=2,
                sort_keys=True,
            )
            print()
            return 1 if had_error else 0
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
