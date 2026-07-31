"""Run and inspect declarative Feedbax figure specs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from feedbax.analysis.figures import execute_figure_spec
from feedbax.config.yaml import get_yaml_loader
from feedbax.plugins.composition import compose_application
from feedbax.plot.constructors import (
    constructor_catalog,
    registered_figure_pieces,
    registered_figure_templates,
)


def _load_spec(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yml", ".yaml"}:
        return get_yaml_loader().load(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a FigureSpec JSON/YAML file")
    run_parser.add_argument("spec", type=Path)
    run_parser.add_argument("--root", type=Path, default=None)

    list_parser = subparsers.add_parser("list", help="List figure registry entries")
    list_parser.add_argument("kind", choices=("constructors", "templates", "pieces"))

    args = parser.parse_args(argv)
    registry = asyncio.run(compose_application()).bundle.figures
    if args.command == "run":
        manifest, path = execute_figure_spec(
            _load_spec(args.spec), root=args.root, registry=registry
        )
        _print_json(
            {
                "manifest_id": manifest.id,
                "manifest_path": str(path),
                "status": manifest.status,
                "artifacts": [
                    artifact.model_dump(mode="json", exclude_none=True)
                    for artifact in manifest.artifacts
                ],
            }
        )
        return 0
    if args.kind == "constructors":
        _print_json(constructor_catalog(registry=registry))
    elif args.kind == "templates":
        _print_json(
            [
                item.model_dump(mode="json", exclude_none=True)
                for item in registered_figure_templates(registry=registry)
            ]
        )
    else:
        _print_json(
            [
                item.model_dump(mode="json", exclude_none=True)
                for item in registered_figure_pieces(registry=registry)
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
