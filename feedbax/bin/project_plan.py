"""Read-only planning for an authored text project."""

import argparse
import os
from pathlib import Path
import sys

from feedbax._atomic import write_bytes_atomic
from feedbax.contracts.project_io import (
    PROJECT_DECLARATION_FILENAME,
    PROJECT_FILE_FAMILIES,
    STUDIO_PATCH_JOURNAL_FILENAME,
    STUDIO_PRESENTATION_FILENAME,
    STUDIO_WRITE_LOCK_FILENAME,
    ProjectReadError,
    TextProject,
    find_text_project,
    read_text_project_snapshot,
)
from feedbax.contracts.project_plan import (
    ElementKey,
    PlanRequests,
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


def configure_parser(parser: argparse.ArgumentParser) -> None:
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


def _output_path(path: Path, text: TextProject) -> Path:
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


def _check(current: PlanRequests, before: PlanRequests) -> int:
    changes = diff_plan_requests(before, current)
    for key, change in changes.items():
        request = current.entries.get(key) or before.entries[key]
        status = f"changed({change.path})" if change.state == "changed" else change.state
        origin = f" step=steps/{run_key(key).step}.json" if change.path else ""
        upstream = f" upstream={_address(change.upstream)}" if change.upstream else ""
        print(f"{_address(key)} {status} {request.request_id}{origin}{upstream}")
    return int(any(change.state != "unchanged" for change in changes.values()))


def _read_baseline(path: Path) -> tuple[PlanRequests, tuple[str, ...] | None]:
    try:
        return load_plan_requests(path.read_bytes())
    except (ValueError, RecursionError) as error:
        raise ProjectReadError(path, str(error)) from error


def _plan(args: argparse.Namespace) -> int:
    from feedbax.analysis.builtin_operations import register_builtin_operations
    from feedbax.analysis.operations import OperationRegistry

    root = find_text_project(args.project_dir)
    text = read_text_project_snapshot(root)

    operations = OperationRegistry()
    register_builtin_operations(operations)
    operations.seal()
    plan = compile_project(text.project, operations=operations)
    current = PlanRequests.from_plan(plan)
    known_variants = set(plan.variants)
    variants = tuple(sorted(set(args.variant))) if args.variant else None
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
    unknown = set(variants or ()) - known_variants
    if unknown:
        candidates = ", ".join(sorted(known_variants)) or "(none)"
        raise ProjectReadError(
            root,
            f"unknown variant {', '.join(sorted(unknown))}. Available variants: {candidates}",
        )

    scoped = select_plan_requests(current, variants)
    problems = [
        problem
        for problem in plan.diagnostics
        if variants is None or problem.variant is None or problem.variant in variants
    ]
    for problem in problems:
        source = f"steps/{problem.step}.json" if problem.step else PROJECT_DECLARATION_FILENAME
        layer = f" ({problem.layer})" if problem.layer else ""
        document = f" [documents/{problem.document}.json]" if problem.document else ""
        print(
            f"{source}{layer}{document} {problem.path} [{problem.code}]: {problem.detail}",
            file=sys.stderr,
        )
    if before is not None:
        status = _check(scoped, select_plan_requests(before, variants))
    else:
        status = 0
        for key, request in scoped.entries.items():
            print(f"{_address(key)} {request.request_id}")
        if args.out is not None and not problems:
            destination = _output_path(args.out, text)
            write_bytes_atomic(destination, dump_plan_requests(current, variants))
    return 2 if problems else status


def run(args: argparse.Namespace) -> int:
    try:
        return _plan(args)
    except (ProjectReadError, OSError) as error:
        print(f"feedbax plan: {error}", file=sys.stderr)
        return 2
