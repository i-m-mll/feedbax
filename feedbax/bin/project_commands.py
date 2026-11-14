"""Dispatch and output for the managed project commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from feedbax.contracts.project_resolution import ProjectDiagnostic

EXIT_CODES = (
    "Exit codes:\n"
    "  0  Success.\n"
    "  1  Review drift or a selected run did not succeed.\n"
    "  2  Invalid project or command could not be carried out."
)


@dataclass(frozen=True)
class ProjectCommand:
    module: str
    summary: str
    configure: str = "configure_parser"
    execute: str = "run"
    delegated: bool = False


PROJECT_COMMANDS = {
    "check": ProjectCommand("check", "Verify a committed result closure or the declared Store."),
    "plan": ProjectCommand("project_plan", "Compile an authored text project for review."),
    "status": ProjectCommand(
        "project_plan", "Show saved project run and element states.",
        configure="configure_status_parser", execute="run_status",
    ),
    "run": ProjectCommand(
        "run", "Submit saved project rows for local admission and print their dispositions.",
        execute="run_saved_project", delegated=True,
    ),
    "describe": ProjectCommand("describe", "Show authored schemas and registry declarations."),
}


class CommandOutput:
    def __init__(self, command: str, *, json: bool) -> None:
        self.command = command
        self.json = json

    def write(self, line: str) -> None:
        if not self.json:
            print(line)

    def response(self, result: BaseModel, *, exclude_unset: bool = True) -> None:
        print(result.model_dump_json(by_alias=True, exclude_unset=exclude_unset))

    def error(self, code: str, detail: str, *, path: str, address: str, text: str | None = None) -> None:
        if self.json:
            from feedbax.contracts.jobs import StudioProjectWindowMapProblem

            problem = StudioProjectWindowMapProblem(
                code=code, path=path, address=address, detail=detail,
                variant=None, step=None, row=None, document=None,
            )
            print(problem.model_dump_json(exclude_unset=True), file=sys.stderr)
        else:
            print(text or f"feedbax {self.command}: {detail}", file=sys.stderr)

    def failure(
        self, error: Exception, *, path: str | Path, detail: str | None = None,
    ) -> None:
        from feedbax.contracts.project_io import ProjectReadError

        source = error if isinstance(error, ProjectReadError) else error.__cause__
        if isinstance(source, ProjectReadError):
            path = source.source
            code = "invalid_project_input"
        elif isinstance(error, OSError):
            path = error.filename or path
            code = "io_error"
        else:
            code = getattr(error, "code", "command_failed")
        address = getattr(error, "plugin_id", None) or str(path)
        self.error(code, detail or str(error), path=str(path), address=address)

    def diagnostic(self, problem: ProjectDiagnostic, *, declaration: str) -> None:
        from feedbax.contracts.jobs import StudioProjectWindowMapProblem

        projected = StudioProjectWindowMapProblem.from_diagnostic(problem, declaration=declaration)
        if self.json:
            print(projected.model_dump_json(exclude_unset=True), file=sys.stderr)
        else:
            layer = f" ({problem.layer})" if problem.layer else ""
            stored_path = f":{projected.authored_path}" if projected.authored_path else ""
            document_path = f":{problem.document_path}" if problem.document_path else ""
            document = f" [documents/{problem.document}.json{document_path}]" if problem.document else ""
            address = f" model={problem.model_address.to_text()}" if problem.model_address else ""
            print(
                f"{projected.address}{stored_path}{layer}{document} {problem.path}{address} "
                f"[{problem.code}]: {problem.detail}",
                file=sys.stderr,
            )


class UsageError(ValueError):
    pass


class ProjectArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def wants_json(argv: Sequence[str]) -> bool:
    for argument in argv:
        if argument == "--":
            return False
        if argument == "--json":
            return True
    return False


def _configure(parser: argparse.ArgumentParser, command: ProjectCommand):
    module = import_module(f"feedbax.bin.{command.module}")
    parser.add_argument("--json", action="store_true", help="Write the shared response contract as JSON.")
    getattr(module, command.configure)(parser)
    return getattr(module, command.execute)


def register_engine_commands(subparsers) -> None:
    for name, command in PROJECT_COMMANDS.items():
        if not command.delegated:
            parser = subparsers.add_parser(
                name, prog=f"feedbax {name}", help=command.summary,
                allow_abbrev=False, epilog=EXIT_CODES,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            _configure(parser, command)


def run_project_command(name: str, argv: Sequence[str]) -> int:
    command = PROJECT_COMMANDS[name]
    parser = ProjectArgumentParser(
        prog=f"feedbax {name}", description=command.summary,
        exit_on_error=False, allow_abbrev=False, epilog=EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    execute = _configure(parser, command)
    parse = parser.parse_known_intermixed_args if command.delegated else parser.parse_known_args
    try:
        args, extras = parse(argv)
    except (argparse.ArgumentError, UsageError) as error:
        address = getattr(error, "argument_name", None) or name
        return _usage_error(parser, argv, str(error), address)
    if extras:
        return _usage_error(parser, argv, f"unrecognized arguments: {' '.join(extras)}", extras[0])
    return execute(args, CommandOutput(name, json=args.json))


def _usage_error(
    parser: ProjectArgumentParser, argv: Sequence[str], detail: str, address: str,
) -> int:
    output = CommandOutput(parser.prog.removeprefix("feedbax "), json=wants_json(argv))
    if not output.json:
        parser.print_usage(sys.stderr)
    output.error(
        "invalid_arguments", detail, path=address, address=address,
        text=f"{parser.prog}: error: {detail}",
    )
    return 2
