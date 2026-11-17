"""Verify a saved project's immutable Store without contacting the Host."""

import argparse
from pathlib import Path

from feedbax.bin.project_commands import CommandOutput


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
    from feedbax.contracts.project_io import locate_text_project, read_text_project_snapshot
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.store_check import check_store

    try:
        root, declaration = locate_text_project(args.project_dir)
        project = read_text_project_snapshot(root, declaration=declaration)
        store = ImmutableArtifactBlobProvider(project.store_path)
        report = check_store(store, attempt=args.attempt)
    except (OSError, ValueError) as error:
        output.failure(error, path=args.project_dir)
        return 2

    if output.json:
        output.response(report)
    else:
        scope = report.attempt or report.store
        output.write(f"{scope}: verified {report.verified_objects} objects, {report.verified_bytes} bytes")
    for finding in report.findings:
        output.error(
            finding.code, finding.detail, path=finding.path, address=finding.address,
            text=f"{'Damaged' if finding.code == 'damaged_object' else 'Incomplete'} "
                 f"{finding.address} ({finding.path}): {finding.detail}",
        )
    if any(finding.code == "damaged_object" for finding in report.findings):
        return 1
    return 2 if report.findings else 0
