"""The choke point: tracked compiled bytes must be what the engine produces now.

An authored experiment enters a repository as an envelope and leaves as compiled
output. That is only true if the tracked output is *provably* what the envelope
compiles to today — otherwise a hand edit to a generated document survives, and
the envelope stops being the source of truth.

The check is mechanical and has no opinion about science: recompile every
envelope, emit its outputs into the canonical on-disk form, and compare against
the bytes on disk. Three findings are possible per output — it matches, it
differs, or it is absent — plus one per unclaimed file: a compiled document that
no envelope in the repository produces.

The report is structured rather than printed. The engine decides *what drifted*;
the project's thin entry point decides how to say it and what exit code to use,
because the remediation command is the project's own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from feedbax.contracts.authored_canonical import emit_text
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    PendingProductCustodyError,
)
from feedbax.envelope.compile import EnvelopeKernel


class ChokeFinding(StrEnum):
    """What the choke point concluded about one path or one envelope."""

    IDENTICAL = "identical"
    DIFFERS = "differs"
    MISSING = "missing"
    ORPHANED = "orphaned"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass(frozen=True)
class ChokeEntry:
    """One structured finding, naming what was judged and why."""

    finding: ChokeFinding
    path: str
    envelope_ref: str | None = None
    detail: str | None = None

    @property
    def drifted(self) -> bool:
        """Whether this entry means the tracked tree is not what compiles now."""
        return self.finding is not ChokeFinding.IDENTICAL

    def describe(self) -> str:
        """Return a stable one-line description of this finding."""
        source = f" from {self.envelope_ref}" if self.envelope_ref is not None else ""
        reason = f": {self.detail}" if self.detail is not None else ""
        return f"{self.path}{source} is {self.finding.value}{reason}"


@dataclass(frozen=True)
class ChokeReport:
    """Every finding from one choke-point pass, in deterministic order."""

    entries: tuple[ChokeEntry, ...]

    @property
    def ok(self) -> bool:
        """Whether the tracked tree is exactly what the envelopes compile to."""
        return not any(entry.drifted for entry in self.entries)

    @property
    def drift(self) -> tuple[ChokeEntry, ...]:
        """Return only the entries that mean something must be regenerated."""
        return tuple(entry for entry in self.entries if entry.drifted)

    def by_finding(self, finding: ChokeFinding) -> tuple[ChokeEntry, ...]:
        """Return every entry with one finding."""
        return tuple(entry for entry in self.entries if entry.finding is finding)

    def describe(self) -> str:
        """Return every drifted entry, one per line."""
        return "\n".join(entry.describe() for entry in self.drift)


def compare_tracked_outputs(
    kernel: EnvelopeKernel,
    repo_root: Path,
    *,
    out_dir: Path | None = None,
    check_orphans: bool = True,
) -> ChokeReport:
    """Recompile every envelope and compare its outputs to the tracked bytes.

    Args:
        kernel: The engine bound to the project whose envelopes are checked.
        repo_root: Root the envelope and output directories resolve against.
        out_dir: Where compiled output is tracked. Defaults to the layout's
            output directory under ``repo_root``.
        check_orphans: Whether to report a tracked compiled document that no
            envelope produces. A project that files other documents alongside its
            compiled output turns this off.

    Returns:
        A structured report. An envelope that no longer compiles is a finding
        rather than a raised exception, so one broken envelope does not hide the
        state of every other one.
    """
    output_root = out_dir if out_dir is not None else repo_root / kernel.layout.output_directory
    entries: list[ChokeEntry] = []
    claimed: set[Path] = set()

    for envelope_path in kernel.envelopes(repo_root):
        envelope_ref = envelope_path.relative_to(repo_root).as_posix()
        try:
            outcome = kernel.compile_envelope_file(envelope_path, repo_root=repo_root)
        except ExperimentEnvelopeRejection as rejection:
            entries.append(
                ChokeEntry(
                    ChokeFinding.REJECTED,
                    envelope_ref,
                    envelope_ref=envelope_ref,
                    detail=f"no longer compiles: {rejection}",
                )
            )
            continue
        except PendingProductCustodyError as pending:
            entries.append(
                ChokeEntry(
                    ChokeFinding.PENDING,
                    envelope_ref,
                    envelope_ref=envelope_ref,
                    detail=str(pending),
                )
            )
            continue
        expected: Mapping[str, Any] = {
            str(kernel.output_paths(outcome, output_root)["compile_lock"]): outcome.compile_lock,
            str(kernel.output_paths(outcome, output_root)["document"]): outcome.document,
        }
        for raw_path, document in expected.items():
            path = Path(raw_path)
            claimed.add(path)
            entries.append(_compare_one(path, document, repo_root, envelope_ref))

    if check_orphans and output_root.is_dir():
        for path in sorted(output_root.glob("*.json")):
            if path in claimed:
                continue
            entries.append(
                ChokeEntry(
                    ChokeFinding.ORPHANED,
                    _relative(path, repo_root),
                    detail="no envelope in this repository produces it",
                )
            )
    return ChokeReport(tuple(entries))


def _compare_one(
    path: Path, document: Any, repo_root: Path, envelope_ref: str
) -> ChokeEntry:
    relative = _relative(path, repo_root)
    wanted = emit_text(document)
    if not path.is_file():
        return ChokeEntry(
            ChokeFinding.MISSING,
            relative,
            envelope_ref=envelope_ref,
            detail="the envelope compiles to it but it is not tracked",
        )
    if path.read_text(encoding="utf-8") != wanted:
        return ChokeEntry(
            ChokeFinding.DIFFERS,
            relative,
            envelope_ref=envelope_ref,
            detail="a generated document is never hand-edited; regenerate it",
        )
    return ChokeEntry(ChokeFinding.IDENTICAL, relative, envelope_ref=envelope_ref)


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def format_choke_findings(report: ChokeReport, *, prefix: str, remediation: str) -> Sequence[str]:
    """Render a report's drift as the lines a project's entry point prints."""
    return tuple(
        f"{prefix}: {entry.describe()}. Regenerate with `{remediation}`"
        for entry in report.drift
    )


__all__ = [
    "ChokeEntry",
    "ChokeFinding",
    "ChokeReport",
    "compare_tracked_outputs",
    "format_choke_findings",
]
