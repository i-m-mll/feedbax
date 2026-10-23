"""Command-line access to the Feedbax provider contract."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from feedbax.contracts.intent_diff import StalenessReport, staleness_report
from feedbax.contracts.run_composition import (
    ResolvedOutputParent,
    flatten_repo_composition,
    parse_composition_node,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.persistence.manifest_index import rebuild_manifest_index
from feedbax.integrations.provider import (
    health,
    provider_manifest,
    registry_snapshot,
    validate_spec,
)
from feedbax.plugins.composition import compose_application


class ProviderStalenessResolutionError(ValueError):
    """Raised when composed-spec staleness lacks exact parent authority."""


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

    staleness_parser = subparsers.add_parser(
        "staleness", help="Report stale contract identities in a composed spec"
    )
    staleness_parser.add_argument("path", help="CompositionNode JSON path, or '-' for stdin")
    staleness_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Project root for repository-relative authored parents (default: cwd)",
    )
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
    if args.command == "staleness":
        report = _composition_staleness_report(args.path, repo_root=args.repo_root)
        if args.json:
            _write_json(report)
        else:
            _write_staleness_report(report)
        return 1 if args.strict and report.stale else 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


def _composition_staleness_report(
    path: str,
    *,
    repo_root: Path | None = None,
    resolved_output_resolver: Callable[[ResolvedOutputParent], Mapping[str, Any]] | None = None,
) -> StalenessReport:
    root = (repo_root or Path.cwd()).resolve()
    source_ref: str | None = None
    if path != "-":
        source_path = Path(path).resolve()
        try:
            source_ref = source_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProviderStalenessResolutionError(
                f"composition document is outside repository root {root}: {source_path}"
            ) from exc
    try:
        node = parse_composition_node(_read_json(path))
        flattened = flatten_repo_composition(
            node,
            repo_root=root,
            source_ref=source_ref,
            resolved_output_resolver=(
                resolved_output_resolver or _resolve_mandible_artifact_parent
            ),
        )
    except ProviderStalenessResolutionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderStalenessResolutionError(
            f"composition staleness parent resolution failed: {exc}"
        ) from exc
    return staleness_report(flattened)


def _resolve_mandible_artifact_parent(parent: ResolvedOutputParent) -> dict[str, Any]:
    if not re.fullmatch(r"artifact-blob:[0-9a-f]{64}", parent.ref):
        raise ProviderStalenessResolutionError(
            "resolved-output staleness requires an exact artifact-blob:<sha256> "
            f"custody ref; got {parent.ref!r}"
        )
    with tempfile.TemporaryDirectory(prefix="feedbax-staleness-") as temporary:
        output = Path(temporary) / "parent.json"
        command = [
            "mandible",
            "artifact",
            "content",
            parent.ref,
            "--output",
            str(output),
            "--json",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError as exc:
            raise ProviderStalenessResolutionError(
                f"Mandible custody provider is unavailable for {parent.ref!r}"
            ) from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProviderStalenessResolutionError(
                f"Mandible custody provider failed for {parent.ref!r}: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no provider detail"
            raise ProviderStalenessResolutionError(
                f"Mandible custody could not resolve {parent.ref!r}: {detail[:1000]}"
            )
        if not output.is_file():
            raise ProviderStalenessResolutionError(
                f"Mandible custody returned no bytes for {parent.ref!r}"
            )
        try:
            payload = strict_json_loads(output.read_bytes(), ref=parent.ref)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ProviderStalenessResolutionError(
                f"Mandible custody bytes are not strict JSON for {parent.ref!r}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderStalenessResolutionError(
                f"Mandible custody bytes are not a JSON object for {parent.ref!r}"
            )
        return payload


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
