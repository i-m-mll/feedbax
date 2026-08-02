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


def build_instructions_parser() -> argparse.ArgumentParser:
    """Build the parser for the ``feedbax instructions`` family."""
    parser = argparse.ArgumentParser(
        prog="feedbax instructions",
        description=(
            "Install and check the Feedbax-maintained block of agent instructions in "
            "a repository's instruction files."
        ),
    )
    subparsers = parser.add_subparsers(dest="instructions_command", required=True)

    install = subparsers.add_parser(
        "install",
        help="Install or update the managed block, leaving every other byte alone.",
    )
    install.add_argument(
        "path", nargs="?", default=".", help="Repository root (default: cwd)."
    )
    install.add_argument(
        "--target",
        help="Install into this exact file instead of reconciling the agent files.",
    )
    install.add_argument(
        "--mode",
        choices=("managed-block", "standalone"),
        default="managed-block",
        help=(
            "managed-block injects a delimited block; standalone writes a whole "
            "generated fragment for repositories that already compile instructions."
        ),
    )
    install.add_argument(
        "--dry-run", action="store_true", help="Report every outcome and write nothing."
    )

    check = subparsers.add_parser(
        "check", help="Report whether the installed managed block is current."
    )
    check.add_argument("path", nargs="?", default=".", help="Repository root (default: cwd).")
    check.add_argument("--target", help="Check this exact file instead of the agent files.")
    return parser


def run_instructions(argv: Sequence[str]) -> int:
    """Run one ``feedbax instructions`` subcommand.

    ``install`` refuses (exit 2) rather than writing anything it cannot write
    safely. ``check`` is read-only and returns one distinct code per unhealthy
    state, so a caller can branch on *which* thing is wrong.
    """
    from feedbax.governance.agent_instructions import (
        AgentInstructionsError,
        check as check_instructions,
        install as install_instructions,
    )

    args = build_instructions_parser().parse_args(list(argv))
    if args.instructions_command == "install":
        try:
            report = install_instructions(
                Path(args.path),
                target=args.target,
                mode=args.mode,
                dry_run=args.dry_run,
            )
        except AgentInstructionsError as exc:
            print(f"feedbax instructions install: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        except OSError as exc:
            print(f"feedbax instructions install failed: {exc}", file=sys.stderr)
            return EXIT_INFRASTRUCTURE
        print(report.describe())
        return EXIT_OK
    try:
        verdict = check_instructions(Path(args.path), target=args.target)
    except OSError as exc:
        print(f"feedbax instructions check failed: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    stream = sys.stdout if verdict.exit_code == EXIT_OK else sys.stderr
    print(verdict.describe(), file=stream)
    return verdict.exit_code


#: Every project command this module implements, by name.
PROJECT_COMMAND_RUNNERS = {
    "init": run_init,
    "instructions": run_instructions,
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
    "build_instructions_parser",
    "run_init",
    "run_instructions",
    "run_project_command",
]
