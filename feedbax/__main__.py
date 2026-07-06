"""Command-line entrypoint for ``python -m feedbax``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from feedbax.training.executor import execute_training_run_spec


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    execute_parser.add_argument("--training-payload", help="Optional external training payload JSON")
    execute_parser.add_argument("--training-payload-kind", default="TrainingRunSpec")
    execute_parser.add_argument("--training-payload-schema-id")
    execute_parser.add_argument("--training-payload-schema-version")
    execute_parser.add_argument("--training-payload-ref")
    execute_parser.add_argument("--resume", action="store_true")
    execute_parser.add_argument("--stop-after-barrier")
    execute_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the default stderr progress printer.",
    )

    args = parser.parse_args(argv)
    if args.command == "execute-training-run-spec":
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
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
