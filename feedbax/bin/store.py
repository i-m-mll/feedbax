"""Inspect the declared project Store without changing it."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def configure_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="store_command", required=True)
    gc = commands.add_parser("gc", help="Report retained and potentially collectable Store bytes.")
    gc.add_argument("--dry-run", action="store_true", required=True)
    gc.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Project declaration file or directory to search upward (default: current directory).",
    )


def run(args: argparse.Namespace) -> int:
    from feedbax.contracts.project_io import locate_text_project, read_text_project_snapshot
    from feedbax.host import HostClient
    from feedbax.project_retention import inspect_project_retention

    try:
        root, declaration = locate_text_project(args.project_dir)
        text = read_text_project_snapshot(root, declaration=declaration)
        report = inspect_project_retention(text, HostClient())
    except (ValueError, OSError) as error:
        print(f"feedbax store gc: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"dry_run": True, **asdict(report)}, indent=2))
    return 2 if report.objects.gaps else 0
