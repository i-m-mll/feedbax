"""Verify a saved project's Store and advise on authored member families."""

import argparse
from pathlib import Path

from feedbax.bin.project_commands import CommandOutput


def _absent(path: Path) -> bool:
    """Distinguish an absent Store path from an existing path with broken contents."""
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    return False


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "project_dir", nargs="?", type=Path, default=Path("."),
        help="Project declaration file or directory to search upward (default: current directory).",
    )
    parser.add_argument("--attempt", help="Check this committed result and its bound inputs.")
    parser.epilog = (
        "Exit codes:\n"
        "  0  Success. The requested scope verified.\n"
        "  1  Damaged objects.\n"
        "  2  Invalid project, selection, or incomplete check scope."
    )


def run(args: argparse.Namespace, output: CommandOutput) -> int:
    from feedbax.bin.family_advice import unrolled_family_advice
    from feedbax.contracts.project_io import locate_text_project, read_text_project_snapshot
    from feedbax.contracts.store_check import StoreCheckResult
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.store_check import check_store

    try:
        root, declaration = locate_text_project(args.project_dir)
        project = read_text_project_snapshot(root, declaration=declaration)
        store = ImmutableArtifactBlobProvider(project.store_path)
        if args.attempt is None and _absent(project.root / project.configuration.store):
            report = StoreCheckResult(
                store=str(store.root), attempt=None, receipt=None, verified_objects=0, verified_bytes=0, findings=(),
            )
        else:
            report = check_store(store, attempt=args.attempt)
    except (OSError, ValueError) as error:
        output.failure(error, path=args.project_dir)
        return 2

    if args.attempt is None and (advice := unrolled_family_advice(project.project)):
        report = report.model_copy(update={"advice": advice})
    if output.json:
        output.response(report)
    else:
        scope = report.attempt or report.store
        output.write(f"{scope}: verified {report.verified_objects} objects, {report.verified_bytes} bytes")
        for item in report.advice:
            output.write(f"Advice {item.address}:{item.path} {item.detail}")
    for finding in report.findings:
        output.error(
            finding.code, finding.detail, path=finding.path, address=finding.address,
            text=f"{'Damaged' if finding.code == 'damaged_object' else 'Incomplete'} "
                 f"{finding.address} ({finding.path}): {finding.detail}",
        )
    if any(finding.code == "damaged_object" for finding in report.findings):
        return 1
    return 2 if report.findings else 0
