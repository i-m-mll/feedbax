"""CLI helpers for Studio provider pipeline materialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from feedbax.integrations.provider import validate_task_spec, validate_training_spec


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _materialize_training(args: argparse.Namespace) -> int:
    graph_spec = _read_json(args.graph)
    training_spec = _read_json(args.training)
    task_spec = _read_json(args.task)
    task_binding_spec = _read_json(args.task_binding) if args.task_binding else None

    training_result = validate_training_spec(training_spec, graph_spec=graph_spec)
    task_result = validate_task_spec(task_spec)
    errors = [
        *[issue.model_dump(mode="json", exclude_none=True) for issue in training_result.errors],
        *[issue.model_dump(mode="json", exclude_none=True) for issue in task_result.errors],
    ]
    warnings = [
        *[issue.model_dump(mode="json", exclude_none=True) for issue in training_result.warnings],
        *[issue.model_dump(mode="json", exclude_none=True) for issue in task_result.warnings],
    ]
    if errors:
        json.dump({"valid": False, "errors": errors, "warnings": warnings}, sys.stderr)
        sys.stderr.write("\n")
        return 1

    summary = {
        "kind": "StudioTrainingValidationArtifact",
        "status": "validated",
        "runner": "studio_validation_only",
        "total_batches": training_spec.get("n_batches"),
        "batch_size": training_spec.get("batch_size"),
        "validation": training_result.model_dump(mode="json", exclude_none=True),
        "task_type": task_spec.get("type"),
        "task_binding_spec": task_binding_spec,
        "warnings": warnings,
        "notes": [
            "This artifact records validation only and contains no training metrics.",
            "Use Studio live training or python -m feedbax execute-training-run-spec "
            "for executor-backed training output.",
        ],
    }
    _write_json(args.output, summary)
    json.dump({"valid": True, "output": args.output, "warnings": warnings}, sys.stdout)
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    training_parser = subparsers.add_parser(
        "materialize-training",
        help="Validate a Studio training snapshot and write an MVP training artifact.",
    )
    training_parser.add_argument("--graph", required=True, help="GraphSpec JSON path")
    training_parser.add_argument("--training", required=True, help="TrainingSpec JSON path")
    training_parser.add_argument("--task", required=True, help="TaskSpec JSON path")
    training_parser.add_argument(
        "--task-binding",
        default=None,
        help="StudioTaskBindingSpec JSON path",
    )
    training_parser.add_argument("--output", required=True, help="Output JSON artifact path")

    args = parser.parse_args(argv)
    if args.command == "materialize-training":
        return _materialize_training(args)
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
