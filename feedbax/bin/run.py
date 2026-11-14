"""Admit saved project runs."""

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from feedbax.plugins.composition import compose_application

from .project_commands import CommandOutput, run_project_command


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
    from feedbax.jobs import JobConflict, JobService
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
        jobs = JobService(HostClient(), registries)
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
    return run_project_command("run", args)


if __name__ == "__main__":
    raise SystemExit(main())
