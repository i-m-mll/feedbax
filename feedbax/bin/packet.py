"""Import and export Feedbax manifest packets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from feedbax.contracts.manifest_packet import (
    export_manifest_packet,
    import_manifest_packet,
)


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

    export_parser = subparsers.add_parser("export", help="Export a manifest packet directory")
    export_parser.add_argument("root_ids", nargs="+", help="Manifest ids that seed the closure")
    export_parser.add_argument("--root", help="Manifest root; defaults to FEEDBAX_RUNS_DIR")
    export_parser.add_argument("--dest", required=True, help="Destination packet directory")
    export_parser.add_argument(
        "--direction",
        choices=["ancestors", "descendants", "both"],
        default="descendants",
        help="Lineage closure direction",
    )
    export_parser.add_argument(
        "--include-artifacts",
        choices=["auto", "include", "external"],
        default="auto",
        help="Artifact inclusion policy",
    )
    export_parser.add_argument(
        "--max-included-artifact-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="Auto mode total artifact byte limit",
    )
    export_parser.add_argument(
        "--producer",
        help="Optional producer JSON object to store in packet.json",
    )

    import_parser = subparsers.add_parser("import", help="Import a manifest packet directory")
    import_parser.add_argument("packet", help="Packet directory containing packet.json")
    import_parser.add_argument("--root", help="Manifest root; defaults to FEEDBAX_RUNS_DIR")

    args = parser.parse_args(argv)
    if args.command == "export":
        producer = json.loads(args.producer) if args.producer else {}
        if not isinstance(producer, dict):
            raise SystemExit("--producer must be a JSON object")
        result = export_manifest_packet(
            args.root_ids,
            root=Path(args.root) if args.root else None,
            dest=args.dest,
            direction=args.direction,
            include_artifacts=args.include_artifacts,
            max_included_artifact_bytes=args.max_included_artifact_bytes,
            producer=producer,
        )
        _write_json(result)
        return 0

    if args.command == "import":
        result = import_manifest_packet(
            args.packet,
            root=Path(args.root) if args.root else None,
        )
        _write_json(result)
        return 0

    raise SystemExit(f"Unknown packet command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
