"""Command-line entrypoint for ``python -m feedbax``."""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.training.executor import execute_training_run_spec
from feedbax.training.checkpoint_custody import fork_checkpoint_transaction
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


def _load_training_method_plugins(module_names: Sequence[str] | None) -> None:
    from feedbax.plugins import load_training_method_plugins

    load_training_method_plugins(modules=module_names)


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
        raise ValueError(
            "checkpoint fork targets must use '<run-spec-json>:<checkpoint-root>'"
        )
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
        batch = int(coordinate.get("global_step") or 0)
        loss = _progress_loss(metrics)
        loss_text = "nan" if loss is None else f"{loss:.6g}"
        elapsed = time.perf_counter() - started_at
        print(
            f"batch={batch} loss={loss_text} elapsed={elapsed:.2f}s",
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


def main(argv: Sequence[str] | None = None) -> int:
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
        "--plugin",
        action="append",
        help=(
            "Import a module that registers Feedbax training methods before "
            "TrainingRunSpec validation; may be repeated."
        ),
    )
    execute_parser.add_argument("--resume", action="store_true")
    execute_parser.add_argument("--stop-after-barrier")
    execute_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the default stderr progress printer.",
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

    args = parser.parse_args(argv)
    if args.command == "execute-training-run-spec":
        _load_training_method_plugins(args.plugin)
        initial_slots = _read_json(args.initial_slots) if args.initial_slots else None
        training_payload = (
            _read_json(args.training_payload) if args.training_payload else None
        )
        started_at = time.perf_counter()
        result = execute_training_run_spec(
            _read_json(args.spec),
            run_id=args.run_id,
            initial_slots=initial_slots,
            manifest_root=args.manifest_root,
            checkpoint_root=args.checkpoint_root,
            training_spec_payload=training_payload,
            training_spec_payload_kind=args.training_payload_kind,
            training_spec_payload_schema_id=args.training_payload_schema_id,
            training_spec_payload_schema_version=args.training_payload_schema_version,
            training_spec_payload_ref=args.training_payload_ref,
            resume=args.resume,
            stop_after_barrier=args.stop_after_barrier,
            progress_callback=(
                None if args.no_progress else _console_progress_printer(started_at)
            ),
        )
        json.dump(
            {
                "run_id": result.run_id,
                "status": result.status,
                "manifest_path": str(result.manifest_path),
                "manifest_payload": result.manifest.model_dump(mode="json", exclude_none=True),
            },
            fp=sys.stdout,
            indent=2,
            sort_keys=True,
        )
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
            _load_training_method_plugins(args.plugin)
            run_spec = TrainingRunSpec.model_validate(_read_json(args.run_spec))
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
                    global_step=args.global_step,
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
        if args.checkpoint_command == "fork":
            _load_training_method_plugins(args.plugin)
            expected_slots = (
                _read_pickle(args.expected_slots) if args.expected_slots else None
            )
            slot_transforms = _load_slot_transforms(args.slot_transform)
            target_summaries: list[dict[str, Any]] = []
            had_error = False
            for raw_target in _checkpoint_fork_targets(args):
                summary: dict[str, Any] = {"target": raw_target}
                try:
                    spec_path, checkpoint_root = _parse_checkpoint_fork_target(raw_target)
                    run_spec = TrainingRunSpec.model_validate(_read_json(str(spec_path)))
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
