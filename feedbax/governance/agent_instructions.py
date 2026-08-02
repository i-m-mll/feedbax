"""One centrally versioned block of agent instructions, installed into a repo.

Every project that uses Feedbax needs the same orientation text, and every copy
of it goes stale the moment the framework moves. Copying a manual into each
repository guarantees drift; generating it does not. So Feedbax owns one
template, ships it in the wheel, and installs it into a repository's agent
instruction file as a delimited *managed block*.

The block is delimited by markers that carry their own identity::

    <!-- feedbax:agent-instructions:start
    schema=feedbax.agent_instructions.v1
    template=1
    generated-by=feedbax-0.2.0
    sha256=<hash of the body between the markers>
    -->
    ...generated body...
    <!-- feedbax:agent-instructions:end -->

Those four header fields are what make the block safe to rewrite. The schema
says which reader may touch it, the template version says how old it is, the
hash says whether a human edited it, and ``generated-by`` records the Feedbax
that wrote it without participating in freshness — a new release that does not
change the text leaves the file byte-identical.

Everything here fails closed and never guesses:

* bytes outside the markers are never read, reordered, or rewritten;
* a duplicated, overlapping, or malformed marker pair refuses without writing;
* an unknown schema, or a template newer than this Feedbax understands, refuses
  rather than downgrading somebody else's work;
* nothing is merged, discarded, or converted when two agent files disagree.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Sequence

#: The schema identity every managed block declares.
AGENT_INSTRUCTIONS_SCHEMA_ID = "feedbax.agent_instructions"
AGENT_INSTRUCTIONS_SCHEMA_VERSION = "feedbax.agent_instructions.v1"

#: The template revision this Feedbax knows how to write and validate. A block
#: carrying a higher number was written by a newer Feedbax and is never rewritten.
AGENT_INSTRUCTIONS_TEMPLATE_VERSION = 1

#: The template resource shipped in the wheel.
AGENT_INSTRUCTIONS_TEMPLATE_PACKAGE = "feedbax.governance.templates"
AGENT_INSTRUCTIONS_TEMPLATE_NAME = "agent_instructions.v1.md"

BLOCK_START_MARKER = "<!-- feedbax:agent-instructions:start"
BLOCK_HEADER_END = "-->"
BLOCK_END_MARKER = "<!-- feedbax:agent-instructions:end -->"

#: The exact ordered header keys a well-formed block states.
BLOCK_HEADER_KEYS: tuple[str, ...] = ("schema", "template", "generated-by", "sha256")

#: Where ``--mode standalone`` writes its generated fragment.
STANDALONE_FRAGMENT_PATH = ".agent-instructions/20-feedbax.generated.md"

#: The conventional agent instruction file names, in reconciliation order.
AGENT_FILE_NAMES: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")

_BLOCK_PATTERN = re.compile(
    rf"^{re.escape(BLOCK_START_MARKER)}\n(?P<header>.*?)^{re.escape(BLOCK_HEADER_END)}\n"
    rf"(?P<body>.*?)^{re.escape(BLOCK_END_MARKER)}\n?",
    re.DOTALL | re.MULTILINE,
)


class AgentInstructionsError(Exception):
    """Raised when a managed block cannot be read or written safely."""


class BlockStatus(Enum):
    """What one agent instruction file's managed block currently is."""

    FRESH = "fresh"
    STALE = "stale"
    EDITED = "edited"
    FUTURE = "future"
    MISSING = "missing"
    MALFORMED = "malformed"


#: One distinct nonzero exit code per unhealthy state, so a caller can act on
#: *which* thing is wrong without parsing prose. ``FRESH`` is the only success.
BLOCK_STATUS_EXIT_CODES: dict[BlockStatus, int] = {
    BlockStatus.FRESH: 0,
    BlockStatus.STALE: 3,
    BlockStatus.EDITED: 4,
    BlockStatus.FUTURE: 5,
    BlockStatus.MISSING: 6,
    BlockStatus.MALFORMED: 7,
}

#: Exit code for an infrastructure failure: unreadable path, broken link.
EXIT_INFRASTRUCTURE = 1


def feedbax_version() -> str:
    """Return the running Feedbax version, or a stable marker when unknown."""
    try:
        from feedbax import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - only when metadata is absent
        return "unknown"


def template_body() -> str:
    """Return the canonical instruction body, read from the shipped template."""
    resource = resources.files(AGENT_INSTRUCTIONS_TEMPLATE_PACKAGE).joinpath(
        AGENT_INSTRUCTIONS_TEMPLATE_NAME
    )
    body = resource.read_text(encoding="utf-8").replace("\r\n", "\n")
    return body if body.endswith("\n") else body + "\n"


def body_sha256(body: str) -> str:
    """Return the canonical hash of one instruction body."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def render_block(body: str | None = None, *, generated_by: str | None = None) -> str:
    """Render one complete managed block, markers and header included."""
    text = template_body() if body is None else body
    header = "\n".join(
        (
            f"schema={AGENT_INSTRUCTIONS_SCHEMA_VERSION}",
            f"template={AGENT_INSTRUCTIONS_TEMPLATE_VERSION}",
            f"generated-by=feedbax-{generated_by or feedbax_version()}",
            f"sha256={body_sha256(text)}",
        )
    )
    return f"{BLOCK_START_MARKER}\n{header}\n{BLOCK_HEADER_END}\n{text}{BLOCK_END_MARKER}\n"


@dataclass(frozen=True)
class ParsedBlock:
    """One managed block found in a file, with its declared identity."""

    schema: str
    template_version: int
    generated_by: str
    declared_sha256: str
    body: str
    start: int
    end: int

    @property
    def body_matches_declaration(self) -> bool:
        """Whether the body still hashes to what its own header claims."""
        return body_sha256(self.body) == self.declared_sha256


def _parse_header(raw: str, source: str) -> dict[str, str]:
    lines = [line for line in raw.split("\n") if line != ""]
    keys = tuple(line.partition("=")[0] for line in lines)
    if keys != BLOCK_HEADER_KEYS:
        raise AgentInstructionsError(
            f"{source}: managed block header must state exactly "
            f"{list(BLOCK_HEADER_KEYS)} in order, found {list(keys)}"
        )
    header: dict[str, str] = {}
    for line in lines:
        key, sep, value = line.partition("=")
        if not sep or not value.strip():
            raise AgentInstructionsError(f"{source}: managed block header line is malformed: {line!r}")
        header[key] = value.strip()
    return header


def parse_block(text: str, *, source: str = "<text>") -> ParsedBlock | None:
    """Return the one managed block in *text*, or ``None`` when there is none.

    Raises:
        AgentInstructionsError: If markers are duplicated, unpaired, out of
            order, or carry a header this Feedbax does not recognize. Nothing is
            written on any of those paths; the caller stops.
    """
    starts = text.count(BLOCK_START_MARKER)
    ends = text.count(BLOCK_END_MARKER)
    if starts == 0 and ends == 0:
        return None
    if starts > 1 or ends > 1:
        raise AgentInstructionsError(
            f"{source}: found {starts} start and {ends} end markers; exactly one "
            "managed block may exist and this file is not safe to rewrite"
        )
    if starts != ends:
        raise AgentInstructionsError(
            f"{source}: managed block markers are unpaired "
            f"({starts} start, {ends} end)"
        )
    if text.index(BLOCK_END_MARKER) < text.index(BLOCK_START_MARKER):
        raise AgentInstructionsError(
            f"{source}: managed block end marker precedes its start marker"
        )
    match = _BLOCK_PATTERN.search(text)
    if match is None:
        raise AgentInstructionsError(
            f"{source}: managed block markers are present but the block is malformed"
        )
    header = _parse_header(match.group("header"), source)
    if header["schema"] != AGENT_INSTRUCTIONS_SCHEMA_VERSION:
        raise AgentInstructionsError(
            f"{source}: managed block declares schema {header['schema']!r}, which this "
            f"Feedbax does not know; expected {AGENT_INSTRUCTIONS_SCHEMA_VERSION!r}"
        )
    try:
        template_version = int(header["template"])
    except ValueError as exc:
        raise AgentInstructionsError(
            f"{source}: managed block template version is not an integer: "
            f"{header['template']!r}"
        ) from exc
    return ParsedBlock(
        schema=header["schema"],
        template_version=template_version,
        generated_by=header["generated-by"],
        declared_sha256=header["sha256"],
        body=match.group("body"),
        start=match.start(),
        end=match.end(),
    )


def classify(block: ParsedBlock | None) -> BlockStatus:
    """Classify one parsed block against the template this Feedbax ships."""
    if block is None:
        return BlockStatus.MISSING
    if block.template_version > AGENT_INSTRUCTIONS_TEMPLATE_VERSION:
        return BlockStatus.FUTURE
    if not block.body_matches_declaration:
        return BlockStatus.EDITED
    if block.template_version < AGENT_INSTRUCTIONS_TEMPLATE_VERSION:
        return BlockStatus.STALE
    return BlockStatus.FRESH if block.body == template_body() else BlockStatus.EDITED


def apply_block(text: str, *, source: str = "<text>") -> str:
    """Return *text* with the current managed block installed or updated.

    A first installation prepends exactly one block. An update replaces only the
    bytes between the recognized markers, leaving every other byte alone. A block
    newer than this Feedbax is never rewritten.
    """
    block = parse_block(text, source=source)
    if block is None:
        rendered = render_block()
        return rendered if not text else f"{rendered}\n{text}"
    status = classify(block)
    if status is BlockStatus.FUTURE:
        raise AgentInstructionsError(
            f"{source}: the installed managed block is template version "
            f"{block.template_version}, newer than this Feedbax "
            f"({AGENT_INSTRUCTIONS_TEMPLATE_VERSION}); upgrade Feedbax rather than "
            "downgrading the block"
        )
    if status is BlockStatus.FRESH:
        return text
    return text[: block.start] + render_block() + text[block.end :]


def write_atomic(path: Path, text: str) -> None:
    """Write *text* to *path* atomically, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.feedbax-tmp"
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only on a failed replace
            temporary.unlink()


# --- which files a repository's instructions actually live in ----------------


@dataclass(frozen=True)
class AgentFilePlan:
    """The real files to install into, and the links to create beside them."""

    targets: tuple[Path, ...]
    links: tuple[tuple[Path, str], ...] = ()

    def describe(self) -> str:
        return ", ".join(str(target) for target in self.targets)


def _resolved_target(path: Path) -> Path:
    """Return the real file one agent path stands for, or fail on a broken link."""
    if path.is_symlink() and not path.exists():
        raise AgentInstructionsError(
            f"{path} is a broken symlink; repair or remove it before installing"
        )
    return path.resolve()


def plan_agent_files(root: Path) -> AgentFilePlan:
    """Decide which real files hold this repository's instructions.

    The rules are deliberately timid. A new repository gets one real
    ``AGENTS.md`` and a relative ``CLAUDE.md`` link beside it. An existing
    symlink is followed to its real target rather than repointed. When only one
    of the two exists, the missing one becomes a link to the one that does. When
    both exist as regular files, both are written and neither is merged,
    discarded, or converted — divergent human text is never resolved by a tool.
    """
    agents, claude = (root / name for name in AGENT_FILE_NAMES)
    agents_here = agents.exists() or agents.is_symlink()
    claude_here = claude.exists() or claude.is_symlink()
    if not agents_here and not claude_here:
        return AgentFilePlan(targets=(agents,), links=((claude, AGENT_FILE_NAMES[0]),))
    if agents_here and not claude_here:
        return AgentFilePlan(
            targets=(_resolved_target(agents),), links=((claude, AGENT_FILE_NAMES[0]),)
        )
    if claude_here and not agents_here:
        return AgentFilePlan(
            targets=(_resolved_target(claude),), links=((agents, AGENT_FILE_NAMES[1]),)
        )
    targets: list[Path] = []
    for path in (agents, claude):
        resolved = _resolved_target(path)
        if resolved not in targets:
            targets.append(resolved)
    return AgentFilePlan(targets=tuple(targets))


# --- install -----------------------------------------------------------------


@dataclass(frozen=True)
class FileOutcome:
    """What installing into one path did, or would do."""

    path: Path
    action: str
    status: BlockStatus | None = None

    def describe(self) -> str:
        detail = f" ({self.status.value})" if self.status is not None else ""
        return f"{self.action:<9} {self.path}{detail}"


@dataclass(frozen=True)
class InstallReport:
    """Everything one ``instructions install`` did, or would have done."""

    outcomes: tuple[FileOutcome, ...]
    dry_run: bool

    @property
    def changed(self) -> bool:
        return any(outcome.action in ("created", "updated", "linked") for outcome in self.outcomes)

    def describe(self) -> str:
        prefix = "would " if self.dry_run else ""
        lines = [f"{prefix}{outcome.describe()}" for outcome in self.outcomes]
        return "\n".join(lines)


def _install_into(path: Path, *, dry_run: bool) -> FileOutcome:
    existed = path.exists()
    original = path.read_text(encoding="utf-8") if existed else ""
    updated = apply_block(original, source=str(path))
    if existed and updated == original:
        return FileOutcome(path, "unchanged", classify(parse_block(original, source=str(path))))
    if not dry_run:
        write_atomic(path, updated)
    return FileOutcome(path, "created" if not existed else "updated")


def _link(path: Path, target_name: str, *, dry_run: bool) -> FileOutcome:
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target_name)
    return FileOutcome(path, "linked")


def install(
    root: Path | str,
    *,
    target: Path | str | None = None,
    mode: str = "managed-block",
    dry_run: bool = False,
) -> InstallReport:
    """Install or update the managed block in one repository.

    Args:
        root: The repository root to install into.
        target: An explicit file to install into, instead of reconciling the
            conventional agent files.
        mode: ``managed-block`` injects a delimited block into an instruction
            file; ``standalone`` writes a whole generated fragment for
            repositories that already compile instructions from fragments.
        dry_run: Compute every outcome and write nothing.

    Raises:
        AgentInstructionsError: On any unsafe or unrecognized existing state.
            Nothing is written when this is raised.
    """
    root_path = Path(root)
    if mode not in ("managed-block", "standalone"):
        raise AgentInstructionsError(
            f"unknown instructions mode {mode!r}; expected 'managed-block' or 'standalone'"
        )
    if mode == "standalone":
        fragment = Path(target) if target is not None else root_path / STANDALONE_FRAGMENT_PATH
        rendered = render_block()
        if fragment.exists() and fragment.read_text(encoding="utf-8") == rendered:
            return InstallReport((FileOutcome(fragment, "unchanged", BlockStatus.FRESH),), dry_run)
        action = "updated" if fragment.exists() else "created"
        if not dry_run:
            write_atomic(fragment, rendered)
        return InstallReport((FileOutcome(fragment, action),), dry_run)
    if target is not None:
        return InstallReport((_install_into(Path(target), dry_run=dry_run),), dry_run)
    plan = plan_agent_files(root_path)
    outcomes = [_install_into(path, dry_run=dry_run) for path in plan.targets]
    outcomes.extend(_link(path, name, dry_run=dry_run) for path, name in plan.links)
    return InstallReport(tuple(outcomes), dry_run)


# --- check -------------------------------------------------------------------


@dataclass(frozen=True)
class CheckOutcome:
    """One checked file's managed-block state and why."""

    path: Path
    status: BlockStatus
    detail: str

    def describe(self) -> str:
        return f"{self.status.value:<9} {self.path}: {self.detail}"


@dataclass(frozen=True)
class CheckReport:
    """The freshness verdict for one repository's instruction files."""

    outcomes: tuple[CheckOutcome, ...]

    @property
    def status(self) -> BlockStatus:
        """The worst state found, since a caller must act on the worst one."""
        order = (
            BlockStatus.MALFORMED,
            BlockStatus.FUTURE,
            BlockStatus.MISSING,
            BlockStatus.EDITED,
            BlockStatus.STALE,
            BlockStatus.FRESH,
        )
        found = {outcome.status for outcome in self.outcomes}
        for candidate in order:
            if candidate in found:
                return candidate
        return BlockStatus.MISSING

    @property
    def exit_code(self) -> int:
        return BLOCK_STATUS_EXIT_CODES[self.status]

    def describe(self) -> str:
        return "\n".join(outcome.describe() for outcome in self.outcomes)


_STATUS_DETAILS: dict[BlockStatus, str] = {
    BlockStatus.FRESH: "the managed block matches the shipped template",
    BlockStatus.STALE: (
        "the managed block is an older template revision; run `feedbax instructions install`"
    ),
    BlockStatus.EDITED: (
        "the managed block was edited locally or is corrupt; run "
        "`feedbax instructions install` to restore it"
    ),
    BlockStatus.FUTURE: (
        "the managed block is newer than this Feedbax; upgrade Feedbax rather than "
        "downgrading the block"
    ),
    BlockStatus.MISSING: (
        "no managed block is installed; run `feedbax instructions install`"
    ),
    BlockStatus.MALFORMED: "the managed block markers are not safe to rewrite",
}


def _check_path(path: Path) -> CheckOutcome:
    if not path.exists():
        return CheckOutcome(path, BlockStatus.MISSING, f"{path} does not exist")
    try:
        block = parse_block(path.read_text(encoding="utf-8"), source=str(path))
    except AgentInstructionsError as exc:
        return CheckOutcome(path, BlockStatus.MALFORMED, str(exc))
    status = classify(block)
    return CheckOutcome(path, status, _STATUS_DETAILS[status])


def check(root: Path | str, *, target: Path | str | None = None) -> CheckReport:
    """Report whether a repository's managed block is current.

    Freshness is decided by the template hash, not by the Feedbax version that
    wrote the block: a release that does not change the instructions leaves an
    installed block fresh.
    """
    root_path = Path(root)
    if target is not None:
        return CheckReport((_check_path(Path(target)),))
    try:
        plan = plan_agent_files(root_path)
    except AgentInstructionsError as exc:
        return CheckReport((CheckOutcome(root_path, BlockStatus.MALFORMED, str(exc)),))
    paths: Sequence[Path] = plan.targets or (root_path / AGENT_FILE_NAMES[0],)
    return CheckReport(tuple(_check_path(path) for path in paths))


__all__ = [
    "AGENT_FILE_NAMES",
    "AGENT_INSTRUCTIONS_SCHEMA_ID",
    "AGENT_INSTRUCTIONS_SCHEMA_VERSION",
    "AGENT_INSTRUCTIONS_TEMPLATE_NAME",
    "AGENT_INSTRUCTIONS_TEMPLATE_PACKAGE",
    "AGENT_INSTRUCTIONS_TEMPLATE_VERSION",
    "BLOCK_END_MARKER",
    "BLOCK_HEADER_KEYS",
    "BLOCK_START_MARKER",
    "BLOCK_STATUS_EXIT_CODES",
    "EXIT_INFRASTRUCTURE",
    "STANDALONE_FRAGMENT_PATH",
    "AgentFilePlan",
    "AgentInstructionsError",
    "BlockStatus",
    "CheckOutcome",
    "CheckReport",
    "FileOutcome",
    "InstallReport",
    "ParsedBlock",
    "apply_block",
    "body_sha256",
    "check",
    "classify",
    "feedbax_version",
    "install",
    "parse_block",
    "plan_agent_files",
    "render_block",
    "template_body",
    "write_atomic",
]
