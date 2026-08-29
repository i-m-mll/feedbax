"""Command-line access to the Feedbax provider contract."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from feedbax.contracts.intent_diff import StalenessReport, staleness_report
from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    CompositionNode,
    CompositionParent,
    flatten_composition,
)
from feedbax.persistence.manifest_index import rebuild_manifest_index
from feedbax.integrations.provider import (
    health,
    provider_manifest,
    registry_snapshot,
    validate_spec,
)
from feedbax.execution.backends import write_modal_app
from feedbax.execution.models import ExecutionSpec
from feedbax.execution.planning import (
    load_execution_spec,
    prepare_execution_plan,
    write_execution_plan,
)
from feedbax.execution.local import run_local_execution
from feedbax.plugins.composition import compose_application


def _read_json(path: str) -> dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json", exclude_none=True)
    else:
        payload = value
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Emit provider health JSON")
    subparsers.add_parser("manifest", help="Emit the provider capability manifest")

    registry_parser = subparsers.add_parser("registry", help="Emit a registry snapshot")
    registry_parser.add_argument(
        "kind", choices=["components", "tasks", "losses", "analyses", "protocols"]
    )

    validate_parser = subparsers.add_parser("validate", help="Validate a Feedbax spec JSON file")
    validate_parser.add_argument(
        "kind", choices=["graph", "training", "task", "evaluation", "analysis", "report"]
    )
    validate_parser.add_argument("path", help="Path to JSON spec, or '-' for stdin")
    validate_parser.add_argument(
        "--graph",
        help="Optional GraphSpec JSON path for validating graph-dependent training/loss selectors",
    )

    index_parser = subparsers.add_parser("rebuild-index", help="Rebuild the local manifest index")
    index_parser.add_argument(
        "root", nargs="?", help="Manifest root; defaults to FEEDBAX_RUNS_DIR or cwd"
    )

    plan_parser = subparsers.add_parser("execution-plan", help="Prepare an execution plan")
    plan_parser.add_argument("path", help="ExecutionSpec JSON path")
    plan_parser.add_argument("--output", help="Optional path to write the plan JSON")

    local_parser = subparsers.add_parser(
        "run-local", help="Run an explicitly local ExecutionSpec and emit a manifest"
    )
    local_parser.add_argument("path", help="ExecutionSpec JSON path")
    local_parser.add_argument("--root", help="Manifest root; defaults to FEEDBAX_RUNS_DIR")
    local_parser.add_argument("--timeout", type=float, help="Optional command timeout in seconds")

    modal_parser = subparsers.add_parser("modal-app", help="Render a Modal app from a spec")
    modal_parser.add_argument("path", help="Modal ExecutionSpec JSON path")
    modal_parser.add_argument(
        "--output",
        required=True,
        help="Path to write the generated Modal app",
    )

    staleness_parser = subparsers.add_parser(
        "staleness", help="Report stale contract identities in a composed spec"
    )
    staleness_parser.add_argument("path", help="CompositionNode JSON path, or '-' for stdin")
    staleness_parser.add_argument("--json", action="store_true", help="Emit report JSON")
    staleness_parser.add_argument(
        "--strict", action="store_true", help="Exit nonzero when stale identities are found"
    )

    args = parser.parse_args(argv)
    registries = asyncio.run(compose_application()).bundle

    if args.command == "health":
        _write_json(health())
        return 0
    if args.command == "manifest":
        _write_json(provider_manifest())
        return 0
    if args.command == "registry":
        _write_json(
            registry_snapshot(
                args.kind,
                component_registry=registries.components,
                experiment_registry=registries.experiment_packages,
                analysis_registry=registries.analysis_recipes,
            )
        )
        return 0
    if args.command == "validate":
        graph_spec = _read_json(args.graph) if args.graph else None
        result = validate_spec(
            args.kind,
            _read_json(args.path),
            graph_spec=graph_spec,
            component_registry=registries.components,
            training_method_registry=registries.training_programs,
            analysis_registry=registries.analysis_recipes,
        )
        _write_json(result)
        return 0 if result.valid else 1
    if args.command == "rebuild-index":
        _write_json({"index_path": str(rebuild_manifest_index(args.root))})
        return 0
    if args.command == "execution-plan":
        spec = load_execution_spec(args.path)
        plan = prepare_execution_plan(spec)
        if args.output:
            write_execution_plan(plan, args.output)
        _write_json(plan)
        return 0
    if args.command == "run-local":
        spec = ExecutionSpec.model_validate(_read_json(args.path))
        result = run_local_execution(spec, root=args.root, timeout=args.timeout)
        _write_json(result)
        return 0 if result.return_code == 0 else result.return_code
    if args.command == "modal-app":
        spec = load_execution_spec(args.path)
        output = write_modal_app(spec, args.output)
        _write_json({"path": str(output)})
        return 0
    if args.command == "staleness":
        report = _composition_staleness_report(args.path)
        if args.json:
            _write_json(report)
        else:
            _write_staleness_report(report)
        return 1 if args.strict and report.stale else 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


def _composition_staleness_report(path: str) -> StalenessReport:
    root = Path.cwd() if path == "-" else Path(path).resolve().parent
    node = CompositionNode.model_validate(_read_json(path))

    def resolve_parent(parent: CompositionParent) -> CompositionNode | dict[str, Any]:
        payload = _read_json(str(root / parent.ref))
        if isinstance(parent, AuthoredIntentParent):
            return CompositionNode.model_validate(payload)
        return payload

    return staleness_report(flatten_composition(node, resolve_parent))


def _write_staleness_report(report: StalenessReport) -> None:
    if not report.stale and not report.unresolved:
        sys.stdout.write("No stale or unresolved contract identities.\n")
        return
    if report.stale:
        sys.stdout.write("Stale contract identities:\n")
        for entry in report.stale:
            layer = entry.layer or "<base>"
            sys.stdout.write(
                f"- {entry.schema_id}: {entry.found_version} -> {entry.current_version} "
                f"(layer: {layer})\n"
            )
    if report.unresolved:
        sys.stdout.write("Unresolved contract identities:\n")
        for entry in report.unresolved:
            layer = entry.layer or "<base>"
            sys.stdout.write(f"- {entry.schema_id}: {entry.schema_version} (layer: {layer})\n")


if __name__ == "__main__":
    raise SystemExit(main())
