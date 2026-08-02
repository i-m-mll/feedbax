"""The one ``feedbax`` console entry point.

Feedbax used to be a handful of sibling executables — ``feedbax-run``,
``feedbax-analysis``, ``feedbax-figure``, and friends — plus a ``python -m
feedbax`` module entry point that nothing installed a name for. A person or an
agent arriving at a project had no single thing to type, and no single ``--help``
that told them what the framework can do.

This module is that single thing. It owns no behavior of its own: every
subcommand either delegates to the existing console main that already implements
it, or is routed into the engine parser in :mod:`feedbax.__main__`. The engine
inventory is not copied here — :func:`engine_commands` reads it off that
parser — so the unified surface cannot drift into a second, subtly different
command list.

Dispatch happens before any delegate module is imported, so running one
subcommand never pays for the imports of the other five.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from functools import cache
from typing import Sequence

#: Exit code for a usage error: an absent, unknown, or unroutable command.
EXIT_USAGE = 2


@dataclass(frozen=True)
class DelegatedCommand:
    """One subcommand this entry point routes to an existing console main."""

    name: str
    module: str
    attribute: str
    summary: str

    def entrypoint(self):
        """Import the delegate lazily and return its callable main."""
        return getattr(importlib.import_module(self.module), self.attribute)


#: The console mains this entry point absorbs, in help order. Each one keeps its
#: own argument parser and its own documented exit codes; nothing is
#: reinterpreted on the way through.
DELEGATED_COMMANDS: tuple[DelegatedCommand, ...] = (
    DelegatedCommand(
        "run",
        "feedbax.bin.run",
        "main",
        "Run an experiment package's training or analysis entry point.",
    ),
    DelegatedCommand(
        "analysis",
        "feedbax.bin.analysis",
        "main",
        "Execute analysis, evaluation, report, and bundle specs.",
    ),
    DelegatedCommand(
        "figure",
        "feedbax.bin.figure",
        "main",
        "Execute and resolve figure specs.",
    ),
    DelegatedCommand(
        "train",
        "feedbax.bin.train",
        "main",
        "Retired; use `feedbax execute-training-run-spec`.",
    ),
    DelegatedCommand(
        "provider",
        "feedbax.bin.provider",
        "main",
        "Emit provider health, capability manifests, and registry snapshots.",
    ),
    DelegatedCommand(
        "orchestrate",
        "feedbax.bin.orchestrate",
        "main",
        "Drive run-set orchestration: preflight, launch, status, watch.",
    ),
)

DELEGATED_BY_NAME: dict[str, DelegatedCommand] = {
    command.name: command for command in DELEGATED_COMMANDS
}

#: Commands this entry point implements itself, because they are about the
#: project rather than about running science in one.
PROJECT_COMMANDS: dict[str, str] = {
    "init": "Create or validate this project's Feedbax skeleton.",
}

_HELP_FLAGS = frozenset({"-h", "--help", "help"})


@cache
def engine_commands() -> tuple[str, ...]:
    """Return every ``python -m feedbax`` subcommand name, from that parser itself.

    Reading the inventory off the engine parser is what keeps the two entry
    points honest: a command added there is reachable through ``feedbax``
    immediately, and one removed there stops being advertised here.
    """
    from feedbax.__main__ import engine_command_names

    return engine_command_names()


def _delegated_help() -> str:
    width = max(len(command.name) for command in DELEGATED_COMMANDS)
    return "\n".join(
        f"  {command.name.ljust(width)}  {command.summary}" for command in DELEGATED_COMMANDS
    )


def _project_help() -> str:
    width = max(len(name) for name in PROJECT_COMMANDS)
    return "\n".join(
        f"  {name.ljust(width)}  {summary}" for name, summary in PROJECT_COMMANDS.items()
    )


def usage() -> str:
    """Render the top-level help text."""
    return "\n".join(
        (
            "usage: feedbax <command> [args...]",
            "",
            "Feedbax: build, compile, run, and account for computational experiments.",
            "",
            "project commands:",
            _project_help(),
            "",
            "commands:",
            _delegated_help(),
            "",
            "engine commands (also available as `python -m feedbax <command>`):",
            *(f"  {name}" for name in engine_commands()),
            "",
            "Run `feedbax <command> --help` for a command's own options.",
        )
    )


def _normalize_exit(result: object) -> int:
    """Normalize a delegate's return value into a process exit code."""
    if result is None:
        return 0
    if isinstance(result, bool):
        return int(result)
    if isinstance(result, int):
        return result
    raise TypeError(f"feedbax subcommand returned a non-exit-code value: {result!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Route one ``feedbax`` invocation to the command that implements it."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(usage(), file=sys.stderr)
        return EXIT_USAGE
    name, rest = args[0], args[1:]
    if name in _HELP_FLAGS:
        print(usage())
        return 0
    if name in PROJECT_COMMANDS:
        from feedbax.governance.cli import run_project_command

        return run_project_command(name, rest)
    delegated = DELEGATED_BY_NAME.get(name)
    if delegated is not None:
        return _normalize_exit(delegated.entrypoint()(rest))
    if name in engine_commands():
        from feedbax.__main__ import main as engine_main

        return _normalize_exit(engine_main(args))
    print(f"feedbax: unknown command {name!r}", file=sys.stderr)
    print(usage(), file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
