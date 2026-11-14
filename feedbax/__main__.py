"""Command-line entry point for project and analysis operations."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib import import_module

from feedbax.bin import project_commands


_STANDALONE_COMMANDS = {
    "matrix-harness": ("feedbax.analysis.harness", "Execute an evaluation matrix."),
    "check-project-science-surface": (
        "feedbax.governance.science_surface",
        "Verify production Python against the policy on a protected baseline ref.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m feedbax")
    subparsers = parser.add_subparsers(dest="command", required=True)
    project_commands.register_engine_commands(subparsers)
    for name, (_, summary) in _STANDALONE_COMMANDS.items():
        subparsers.add_parser(name, help=summary)
    return parser


def engine_command_names() -> tuple[str, ...]:
    managed = tuple(name for name, command in project_commands.PROJECT_COMMANDS.items() if not command.delegated)
    return (*managed, *_STANDALONE_COMMANDS)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in _STANDALONE_COMMANDS:
        module, _ = _STANDALONE_COMMANDS[args[0]]
        return import_module(module).main(args[1:])
    if args and args[0] in engine_command_names():
        return project_commands.run_project_command(args[0], args[1:])
    build_parser().parse_args(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
