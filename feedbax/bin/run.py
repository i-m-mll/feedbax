#!/usr/bin/env python
"""Admit saved project runs or dispatch an explicit legacy subcommand."""

import argparse
import asyncio
import importlib
import os
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence
from uuid import uuid4

from ._setup import setup
from .project_commands import CommandOutput, run_project_command
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.plugins.composition import compose_application

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


def _call_module_main(mod: ModuleType, argv: Sequence[str], bootstrap_state: BootstrapState) -> int:
    """
    Try to call mod.main() if present; otherwise run it as __main__
    with sys.argv patched to the sub-argv.
    """
    # If module exposes a `main()`, prefer it (cleaner; avoids top-level side effects).
    main_fn: Optional[Callable] = getattr(mod, "main", None)
    if callable(main_fn):
        # Patch sys.argv so their argparse sees only the subcommand args.
        old_argv = sys.argv
        try:
            sys.argv = [sys.argv[0]] + list(argv)
            if mod.__name__ == "feedbax.bin.analysis":
                _ = main_fn(
                    list(argv),
                    bootstrap_state=bootstrap_state,
                    application_setup=False,
                )
            else:
                _ = main_fn()
            return int(0)
        finally:
            sys.argv = old_argv

    # Fallback: execute the module as if run via `python -m <mod>`
    old_argv = sys.argv
    try:
        sys.argv = [mod.__name__] + list(argv)
        # run_module returns the globals dict; no exit code, assume 0 if no exception.
        runpy.run_module(mod.__name__, run_name="__main__")
        return 0
    finally:
        sys.argv = old_argv


def _dispatch(cmd: str, sub_argv: Sequence[str], bootstrap_state: BootstrapState) -> int:
    # Try candidates in order until one imports successfully.
    mod = importlib.import_module("." + cmd, package="feedbax.bin")
    if mod is not None:
        return _call_module_main(mod, sub_argv, bootstrap_state)
    raise SystemExit(f"Could not locate a module for subcommand '{cmd}'. ")


def parse_args(argv: Sequence[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        prog="run",
        description="Unified entrypoint for training and analysis.",
        add_help=False,  # we'll let subcommands handle their own -h/--help
    )
    parser.add_argument("command", choices=("train", "analysis"))
    # Everything after the subcommand is passed through verbatim
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)
    # If user wrote `run train --help`, they probably want train's help.
    if ns.rest and ns.rest[0] == "--":
        ns.rest = ns.rest[1:]
    return ns.command, ns.rest


def _run_legacy(argv: Sequence[str]) -> int:
    cmd, sub_argv = parse_args(argv)
    bootstrap_state = asyncio.run(compose_application())
    if cmd not in ("analysis", "train"):
        raise SystemExit(f"Unknown command '{cmd}'; expected 'train' or 'analysis'.")
    # Switch to slightly different naming conventions for subpackages (versus CLI commands)
    if cmd == "train":
        domain = "training"
    else:
        domain = "analysis"
    # Set up logging, warnings, and globals config for the chosen domain
    setup(
        argv=sub_argv,
        domain=domain,
        queue_mode="flush",
        registry=bootstrap_state.bundle.experiment_packages,
    )
    # Import and run the chosen script
    return _dispatch(cmd, sub_argv, bootstrap_state)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "project_dir", nargs="?", type=Path, default=Path("."),
        help="Project declaration file or directory. Use . before selectors for the current project.",
    )
    parser.add_argument("runs", nargs="*", metavar="step[@row]")
    parser.add_argument(
        "--label", default=os.environ.get("FEEDBAX_ACTOR_LABEL"),
        help="Label the submission (default: FEEDBAX_ACTOR_LABEL).",
    )


def run_saved_project(args: argparse.Namespace, output: CommandOutput) -> int:
    from feedbax.bin.project_plan import write_diagnostics
    from feedbax.contracts.jobs import StudioSubmissionActor
    from feedbax.contracts.project_io import ProjectReadError, locate_text_project, project_declaration_name
    from feedbax.host import HostClient, HostError
    from feedbax.jobs import JobConflict, JobService, evaluation_candidate_configured
    from feedbax.plugins.bootstrap import BootstrapError
    from feedbax.project_folder import ProjectFolder, ProjectFolderInvalid, ProjectFolderLocation

    try:
        root, declaration = locate_text_project(args.project_dir)
        registries = asyncio.run(compose_application()).bundle
        folders = ProjectFolder(
            root.parent, operations=registries.operations, components=registries.components,
            functions=registries.expression_functions, python_tasks=registries.python_tasks,
            local_folder=ProjectFolderLocation.selected(root),
        )
        reviewed = folders.read_plan(root.name, revision=None, declaration=declaration)
        if reviewed.plan.diagnostics:
            write_diagnostics(output, reviewed.plan, None, declaration=project_declaration_name(
                reviewed.configuration.project,
            ))
            return 2
        enabled = evaluation_candidate_configured()
        keys = [run.key for run in reviewed.plan.runs]
        selected = dict.fromkeys(keys) if not args.runs else {}
        for selector in args.runs:
            step, separator, row = selector.partition("@")
            matches = [
                key for key in keys
                if key.step == step and (not separator or key.row == row)
            ]
            if not matches:
                output.error(
                    "unknown_run", f"No runnable saved rows match {selector!r}.",
                    path=selector, address=selector,
                )
                return 2
            selected.update(dict.fromkeys(matches))
        jobs = JobService(HostClient(), registries, enable_evaluation_candidate=enabled)
        submission = jobs.admit_plan(
            folders, reviewed, run_keys=tuple(selected),
            token=uuid4().hex, actor=StudioSubmissionActor(client="cli", label=args.label),
        )
    except (BootstrapError, HostError, JobConflict, OSError, ProjectFolderInvalid, ProjectReadError) as error:
        output.failure(error, path=args.project_dir)
        return 2
    if output.json:
        output.response(submission)
        for item in submission.items:
            if item.disposition == "rejected":
                output.error(
                    "admission_rejected", item.reason or "Admission rejected.",
                    path="/runs", address=item.run_key,
                )
    else:
        output.write(f"Submission {submission.id}")
        for item in submission.items:
            detail = item.job_id or item.reason or ""
            output.write(f"{item.run_key} {item.disposition} {detail}".rstrip())
    return int(any(item.disposition == "rejected" for item in submission.items))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"train", "analysis"}:
        return _run_legacy(args)
    return run_project_command("run", args)


if __name__ == "__main__":
    raise SystemExit(main())
