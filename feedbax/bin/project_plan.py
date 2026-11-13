"""Read-only planning and status for an authored text project."""

import argparse
from collections.abc import Mapping
import os
from pathlib import Path

from feedbax.bin.project_commands import CommandOutput

from feedbax._atomic import write_bytes_atomic
from feedbax.contracts.project_io import (
    PROJECT_DECLARATION_FILENAME,
    PROJECT_FILE_FAMILIES,
    STUDIO_PATCH_JOURNAL_FILENAME,
    STUDIO_PRESENTATION_FILENAME,
    STUDIO_WRITE_LOCK_FILENAME,
    ProjectReadError,
    TextProject,
    TextProjectExternalInputs,
    find_text_project,
    read_text_project_snapshot,
)
from feedbax.contracts.project_plan import (
    ElementChange,
    ElementKey,
    PlanRequests,
    ProjectPlan,
    RunKey,
    compile_project,
    diff_plan_requests,
)
from feedbax.contracts.project_plan_io import (
    dump_plan_requests,
    load_plan_requests,
    run_key,
    select_plan_requests,
)
from feedbax.contracts.project_resolution import BaseReaders, ProjectDiagnostic


def configure_status_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory to search upward for feedbax.project.json (default: current directory).",
    )
    parser.add_argument(
        "--variant", action="append", help="Show this variant. Repeat to select more."
    )


def configure_parser(parser: argparse.ArgumentParser) -> None:
    configure_status_parser(parser)
    review = parser.add_mutually_exclusive_group()
    review.add_argument("--out", type=Path, metavar="FILE", help="Write a JSON review Plan.")
    review.add_argument(
        "--check",
        type=Path,
        metavar="FILE",
        help="Compare a saved Plan. Request drift exits 1.",
    )


def _address(key: RunKey | ElementKey) -> str:
    run = run_key(key)
    address = f"{run.variant}/{run.step}/{run.row}"
    return f"{address}/{key.name}" if isinstance(key, ElementKey) else address


def _output_path(path: Path, text: TextProject, external_paths: set[Path]) -> Path:
    root = text.root
    destination = path.parent.resolve() / path.name
    resolved = destination.resolve()
    owned = {
        root / name
        for name in (
            *text.file_revisions,
            PROJECT_DECLARATION_FILENAME,
            STUDIO_PATCH_JOURNAL_FILENAME,
            STUDIO_PRESENTATION_FILENAME,
            STUDIO_WRITE_LOCK_FILENAME,
        )
    }
    owned.update(external_paths)
    owned.update(path.resolve() for path in tuple(owned))
    families = {root / name for name in PROJECT_FILE_FAMILIES}
    families.update(path.resolve() for path in tuple(families))
    store = text.store_path
    for candidate in (Path(os.path.abspath(path)), destination, resolved):
        if candidate.is_relative_to(store):
            raise ProjectReadError(path, "the review file cannot be written in the declared Store")
        if any(candidate.is_relative_to(file) for file in owned):
            raise ProjectReadError(path, "the review file overlaps an authored project file")
        for family in families:
            if candidate.is_relative_to(family):
                parts = candidate.relative_to(family).parts
                if not parts or Path(parts[0]).suffix == ".json":
                    raise ProjectReadError(
                        path, "the review file overlaps the project's authored files"
                    )
    return destination


def _write_review(
    output: CommandOutput,
    changes: Mapping[RunKey | ElementKey, ElementChange],
    current: PlanRequests,
    before: PlanRequests,
) -> None:
    for key, change in changes.items():
        request = current.entries.get(key) or before.entries[key]
        status = f"changed({change.path})" if change.state == "changed" else change.state
        origin = f" step=steps/{run_key(key).step}.json" if change.path else ""
        upstream = f" upstream={_address(change.upstream)}" if change.upstream else ""
        output.write(f"{_address(key)} {status} {request.request_id}{origin}{upstream}")


def _read_baseline(path: Path) -> tuple[PlanRequests, tuple[str, ...] | None]:
    try:
        return load_plan_requests(path.read_bytes())
    except (ValueError, RecursionError) as error:
        raise ProjectReadError(path, str(error)) from error


def _read_plan(project_dir: Path) -> tuple[TextProject, ProjectPlan, set[Path]]:
    from feedbax.analysis.builtin_operations import register_builtin_operations
    from feedbax.analysis.operations import OperationRegistry

    root = find_text_project(project_dir)
    text = read_text_project_snapshot(root)

    operations = OperationRegistry()
    register_builtin_operations(operations)
    operations.seal()
    external = TextProjectExternalInputs(text)
    readers = BaseReaders(external=external.read, figure_external=external.read)
    return text, compile_project(text.project, operations=operations, readers=readers), external.paths


def _variants(selected: list[str] | None) -> tuple[str, ...] | None:
    return tuple(sorted(set(selected))) if selected else None


def _validate_variants(
    root: Path,
    variants: tuple[str, ...] | None,
    known_variants: set[str],
) -> None:
    unknown = set(variants or ()) - known_variants
    if unknown:
        candidates = ", ".join(sorted(known_variants)) or "(none)"
        raise ProjectReadError(
            root,
            f"unknown variant {', '.join(sorted(unknown))}. Available variants: {candidates}",
        )


def write_diagnostics(
    output: CommandOutput,
    plan: ProjectPlan,
    variants: tuple[str, ...] | None,
) -> list[ProjectDiagnostic]:
    problems = [
        problem
        for problem in plan.diagnostics
        if variants is None or problem.variant is None or problem.variant in variants
    ]
    for problem in problems:
        output.diagnostic(problem)
    return problems


def _plan(args: argparse.Namespace, output: CommandOutput) -> int:
    text, plan, external_paths = _read_plan(args.project_dir)
    current = PlanRequests.from_plan(plan)
    known_variants = set(plan.variants)
    variants = _variants(args.variant)
    before = None
    if args.check is not None:
        before, saved_variants = _read_baseline(args.check)
        if (
            saved_variants is not None
            and variants is not None
            and set(variants) - set(saved_variants)
        ):
            raise ProjectReadError(args.check, "the saved Plan excludes a requested variant")
        variants = variants if variants is not None else saved_variants
        known_variants.update(saved_variants or ())
        known_variants.update(run_key(key).variant for key in before.entries)
    _validate_variants(text.root, variants, known_variants)
    scoped = select_plan_requests(current, variants)
    problems = write_diagnostics(output, plan, variants)
    changes: Mapping[RunKey | ElementKey, ElementChange] = {}
    if before is not None:
        before = select_plan_requests(before, variants)
        changes = diff_plan_requests(before, scoped)
    elif not output.json:
        for key, request in scoped.entries.items():
            output.write(f"{_address(key)} {request.request_id}")

    if args.out is not None and not problems:
        destination = _output_path(args.out, text, external_paths)
        write_bytes_atomic(destination, dump_plan_requests(current, variants))

    if output.json:
        from feedbax.contracts.jobs import StudioPlanReviewEntry
        from feedbax.web.services.project_folder import SettledProjectPlan
        from feedbax.web.services.project_plan import ProjectPlanRead, plan_request_address

        view = ProjectPlanRead.without_history(SettledProjectPlan.from_text(text.root.name, text, plan))
        result = view.response(include_window_map=True, variants=variants)
        if before is not None:
            result.review = [
                StudioPlanReviewEntry(
                    **plan_request_address(key),
                    request_id=(scoped.entries.get(key) or before.entries[key]).request_id,
                    state=change.state, path=change.path,
                    upstream=plan_request_address(change.upstream) if change.upstream else None,
                )
                for key, change in changes.items()
            ]
        output.response(result)
    elif before is not None:
        _write_review(output, changes, scoped, before)
    return 2 if problems else int(any(change.state != "unchanged" for change in changes.values()))


def run(args: argparse.Namespace, output: CommandOutput) -> int:
    try:
        return _plan(args, output)
    except (ProjectReadError, OSError) as error:
        output.failure(error, path=args.project_dir)
        return 2


def _status_cell(text: str) -> str:
    return text.replace("\r", r"\r").replace("\n", r"\n").replace("\t", r"\t")


def run_status(args: argparse.Namespace, output: CommandOutput) -> int:
    from feedbax.host import HostClient, HostError
    from feedbax.jobs import PlanJobReader, evaluation_candidate_configured
    from feedbax.web.services.project_folder import ProjectFolderInvalid, SettledProjectPlan
    from feedbax.web.services.project_plan import read_settled_project_plan

    try:
        text, plan, _ = _read_plan(args.project_dir)
        variants = _variants(args.variant)
        _validate_variants(text.root, variants, set(plan.variants))
        problems = write_diagnostics(output, plan, variants)
        settled = SettledProjectPlan.from_text(text.root.name, text, plan)
        view = read_settled_project_plan(
            settled,
            PlanJobReader(
                HostClient(),
                enable_evaluation_candidate=evaluation_candidate_configured(),
            ),
        )
    except HostError as error:
        output.failure(
            error, path=args.project_dir, detail=f"saved result history is unavailable: {error}",
        )
        return 2
    except (ProjectReadError, ProjectFolderInvalid, OSError) as error:
        output.failure(error, path=args.project_dir)
        return 2

    if output.json:
        output.response(view.response(include_window_map=True, variants=variants))
        return 2 if problems else 0

    output.write("ADDRESS\tROW\tSTATE\tREASON")
    for row in view.status_rows():
        if variants is None or run_key(row.key).variant in variants:
            output.write(
                "\t".join(
                    (
                        _address(row.key),
                        _status_cell(row.label),
                        row.status.value,
                        _status_cell(row.reason or ""),
                    )
                )
            )
    return 2 if problems else 0
