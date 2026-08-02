"""Argument parsing for the project-shaped ``feedbax`` commands.

These commands act on a repository rather than on a spec, so they live together
and share one convention: the project root is an explicit positional argument
that defaults to the current directory, never something discovered by walking
upwards or reading an environment variable.

Exit codes follow the framework contract — ``0`` succeeded, ``2`` a stable
refusal with an actionable diagnostic on stderr, ``1`` an infrastructure
failure — except that ``instructions check`` reports a distinct nonzero code per
unhealthy state, because a caller needs to know *which* thing is stale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_REFUSED = 2


def build_init_parser() -> argparse.ArgumentParser:
    """Build the parser for ``feedbax init``."""
    parser = argparse.ArgumentParser(
        prog="feedbax init",
        description=(
            "Create or validate this project's Feedbax skeleton: declaration, "
            "authoring budgets, science-surface policy, minimal package, and agent "
            "instructions. Deterministic, non-interactive, and transactional."
        ),
    )
    parser.add_argument(
        "path", nargs="?", default=".", help="Project root to initialize (default: cwd)."
    )
    parser.add_argument("--project", help="Project name (default: the directory's name).")
    parser.add_argument(
        "--package", help="Importable package name (default: the project name with - as _)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report every outcome and write nothing.",
    )
    return parser


def run_init(argv: Sequence[str]) -> int:
    """Run ``feedbax init`` and print its exact per-entry report."""
    from feedbax.governance.project_init import ProjectInitError, initialize

    args = build_init_parser().parse_args(list(argv))
    try:
        report = initialize(
            Path(args.path),
            project=args.project,
            package=args.package,
            dry_run=args.dry_run,
        )
    except ProjectInitError as exc:
        print(f"feedbax init: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except OSError as exc:
        print(f"feedbax init failed on infrastructure: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    stream = sys.stderr if report.conflicts else sys.stdout
    print(report.describe(), file=stream)
    return report.exit_code


#: Every project command this module implements, by name.
PROJECT_COMMAND_RUNNERS = {
    "init": run_init,
}


def run_project_command(name: str, argv: Sequence[str]) -> int:
    """Dispatch one project command by name."""
    runner = PROJECT_COMMAND_RUNNERS.get(name)
    if runner is None:
        raise KeyError(f"no feedbax project command named {name!r}")
    return runner(argv)


__all__ = [
    "EXIT_INFRASTRUCTURE",
    "EXIT_OK",
    "EXIT_REFUSED",
    "PROJECT_COMMAND_RUNNERS",
    "build_init_parser",
    "run_init",
    "run_project_command",
]
