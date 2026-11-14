"""Create a managed project declaration and repository scaffolding transactionally."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import (
    PROJECT_DECLARATION_SUFFIX,
    ProjectConfiguration,
    ProjectReadError,
    dump_text_project,
    load_text_project,
)
from feedbax.governance.agent_instructions import (
    AGENT_FILE_NAMES,
    AgentInstructionsError,
    BlockStatus,
    InstallReport,
    classify,
    install as install_agent_instructions,
    parse_block,
    write_atomic,
)
from feedbax.governance.science_surface import POLICY_SCHEMA_VERSION

DEFAULT_STORE = "results"
DEFAULT_POLICY_REF = "ci/feedbax-science-surface.toml"

#: Exit codes. Initialization follows the same contract as every other Feedbax
#: command: 0 succeeded, 2 a stable refusal with an actionable diagnostic.
EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_CONFLICT = 2


class ProjectInitError(Exception):
    """Raised when a project skeleton cannot be created safely."""


@dataclass(frozen=True)
class EntryOutcome:
    """What one entry is, or would be, after this run."""

    path: str
    action: str
    detail: str = ""

    @property
    def is_conflict(self) -> bool:
        return self.action == "conflict"

    def describe(self) -> str:
        suffix = f": {self.detail}" if self.detail else ""
        return f"{self.action:<9} {self.path}{suffix}"


@dataclass(frozen=True)
class InitReport:
    """Everything one ``feedbax init`` did, or would have done."""

    root: Path
    project: str
    package: str
    outcomes: tuple[EntryOutcome, ...]
    dry_run: bool
    policy_ref: str = DEFAULT_POLICY_REF

    @property
    def conflicts(self) -> tuple[EntryOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.is_conflict)

    @property
    def exit_code(self) -> int:
        return EXIT_CONFLICT if self.conflicts else EXIT_OK

    def describe(self) -> str:
        prefix = "would " if self.dry_run else ""
        lines = [f"feedbax init {self.root} (project={self.project}, package={self.package})"]
        lines.extend(f"  {prefix}{outcome.describe()}" for outcome in self.outcomes)
        if self.conflicts:
            lines.append(
                "refused: the entries above marked `conflict` already exist in a state "
                "this command will not overwrite. Nothing was written."
            )
        else:
            lines.append(
                f"note: {self.policy_ref} is not authoritative until it is committed and "
                "ratified on your protected baseline ref. Until then "
                "`feedbax check-project-science-surface` fails closed, which is the gate "
                "working correctly."
            )
        return "\n".join(lines)


# --- default file contents ---------------------------------------------------


def default_science_policy(package: str | None) -> str:
    """Return the deny-by-default science-surface policy ``init`` writes.

    It authorizes exactly what this command creates and nothing else. It never
    inventories existing source: silently ratifying whatever a repository
    already contains would turn the gate into a rubber stamp.
    """
    lines = [
        "# Deny-by-default science surface for this project, written by `feedbax init`.",
        "#",
        "# Every `*.py` file under `source_roots` must be listed below, with every",
        "# top-level symbol it defines. This file is authoritative only as committed on",
        "# your protected baseline ref: a branch cannot authorize itself by editing it.",
        "",
        f"schema_version = {POLICY_SCHEMA_VERSION}",
        "",
        'source_roots = ["src"]',
        "",
        "# Path patterns ratified as forbidden forever, in pathlib glob syntax.",
        "banned_paths = []",
        "",
    ]
    if package is not None:
        lines.extend(
            (
                "[[allowed_file]]",
                f'path = "src/{package}/__init__.py"',
                "symbols = []",
                'reason = "package marker created by feedbax init"',
                "",
            )
        )
    return "\n".join(lines)


def default_pyproject(project: str, package: str, *, feedbax_requirement: str) -> str:
    """Return the minimal installable project ``init`` writes for a new repo."""
    return f"""[project]
name = "{project}"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["{feedbax_requirement}"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{package}"]
"""


def default_package_marker(package: str) -> str:
    """Return the empty package marker ``init`` writes for a new repo."""
    return f'"""Modular science for {package}."""\n'


# --- naming ------------------------------------------------------------------


def derive_project_name(root: Path) -> str:
    """Derive a project name from a directory, deterministically."""
    name = root.resolve().name.strip()
    if not name:
        raise ProjectInitError(f"cannot derive a project name from {root}; pass --project")
    return name


def derive_package_name(project: str) -> str:
    """Derive an importable package name from a project name."""
    package = project.strip().replace("-", "_").replace(" ", "_")
    if not package.isidentifier():
        raise ProjectInitError(
            f"cannot derive an importable package name from {project!r}; pass --package"
        )
    return package


# --- planning ----------------------------------------------------------------


@dataclass
class _PlannedWrite:
    """One file this run would write and the bytes to restore on failure."""

    path: Path
    content: str | bytes
    previous_bytes: bytes | None = None


@dataclass
class _Plan:
    outcomes: list[EntryOutcome] = field(default_factory=list)
    writes: list[_PlannedWrite] = field(default_factory=list)

    def record(self, outcome: EntryOutcome) -> None:
        self.outcomes.append(outcome)

    def create(self, relative: str, path: Path, text: str | bytes) -> None:
        self.writes.append(_PlannedWrite(path, text))
        self.record(EntryOutcome(relative, "created"))

    def update(self, relative: str, path: Path, content: bytes, previous_bytes: bytes) -> None:
        self.writes.append(_PlannedWrite(path, content, previous_bytes))
        self.record(EntryOutcome(relative, "updated", "added the Store ignore rule"))


def _plan_durable(
    plan: _Plan,
    *,
    relative: str,
    path: Path,
    text: str,
    validate: Callable[[Path], str | None],
) -> None:
    """Plan one durable file: create it, or validate and keep what is there."""
    if not path.exists():
        plan.create(relative, path, text)
        return
    problem = validate(path)
    if problem is None:
        plan.record(EntryOutcome(relative, "validated", "already present; not rewritten"))
    else:
        plan.record(EntryOutcome(relative, "conflict", problem))


def _store_ignore_rule(store_directory: str) -> str:
    """Return one literal root-anchored Git ignore rule for Store contents."""
    escaped = "".join(f"\\{char}" if char in "\\*?[] " else char for char in store_directory)
    return f"/{escaped}/"


def _plan_store_ignore(plan: _Plan, root: Path, store_directory: str) -> None:
    """Add the Store ignore rule while preserving existing rules."""
    relative = ".gitignore"
    path = root / relative
    rule = _store_ignore_rule(store_directory)
    if not path.exists():
        plan.create(relative, path, f"{rule}\n")
        return
    if path.is_symlink():
        plan.record(
            EntryOutcome(
                relative,
                "conflict",
                "is a symlink; replace it with a project-owned file before initialization",
            )
        )
        return
    try:
        previous_bytes = path.read_bytes()
    except OSError as exc:
        plan.record(EntryOutcome(relative, "conflict", f"is not readable: {exc}"))
        return
    rule_bytes = rule.encode("utf-8")
    if rule_bytes in previous_bytes.splitlines():
        plan.record(EntryOutcome(relative, "validated", "Store ignore rule already present"))
        return
    separator = b"" if not previous_bytes or previous_bytes.endswith(b"\n") else b"\n"
    plan.update(
        relative,
        path,
        previous_bytes + separator + rule_bytes + b"\n",
        previous_bytes,
    )


def _validate_policy(path: Path) -> str | None:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"is not readable TOML: {exc}"
    version = document.get("schema_version")
    if version != POLICY_SCHEMA_VERSION:
        return (
            f"declares schema_version {version!r}, but this Feedbax supports "
            f"{POLICY_SCHEMA_VERSION}"
        )
    return None


def _plan_agent_files(plan: _Plan, root: Path) -> None:
    """Plan the agent instruction files without writing, using a dry install."""
    try:
        report: InstallReport = install_agent_instructions(root, dry_run=True)
    except AgentInstructionsError as exc:
        for name in AGENT_FILE_NAMES:
            plan.record(EntryOutcome(name, "conflict", str(exc)))
        return
    for outcome in report.outcomes:
        try:
            relative = str(outcome.path.relative_to(root))
        except ValueError:
            relative = str(outcome.path)
        plan.record(EntryOutcome(relative, outcome.action))


def plan_init(
    root: Path | str,
    *,
    project: str | None = None,
    package: str | None = None,
    feedbax_requirement: str | None = None,
) -> tuple[_Plan, str, str]:
    """Compute every outcome for one ``init`` run without writing anything."""
    root_path = Path(root)
    requirement = feedbax_requirement or _default_feedbax_requirement()
    plan = _Plan()

    declarations = tuple(root_path.glob(f"*{PROJECT_DECLARATION_SUFFIX}"))
    if declarations:
        try:
            existing = load_text_project(root_path)
            if project is not None and project != existing.configuration.project:
                raise ProjectInitError(f"the existing project is named {existing.configuration.project!r}")
        except (ProjectReadError, ProjectInitError) as exc:
            plan.record(EntryOutcome(declarations[0].name, "conflict", str(exc)))
            return plan, project or derive_project_name(root_path), package or ""
        project_name = existing.configuration.project
        package_name = package or derive_package_name(project_name)
        plan.record(EntryOutcome(existing.declaration, "validated", "already present"))
        store = existing.configuration.store
    else:
        project_name = project or derive_project_name(root_path)
        package_name = package or derive_package_name(project_name)
        try:
            configuration = ProjectConfiguration(project=project_name, store=DEFAULT_STORE)
        except ValidationError as exc:
            raise ProjectInitError(str(exc)) from exc
        for path, content in dump_text_project(Project(), configuration).items():
            plan.create(path, root_path / path, content)
        store = DEFAULT_STORE

    resolved_root = root_path.resolve()
    store_path = (resolved_root / store).resolve()
    if store_path.is_relative_to(resolved_root) and store_path != resolved_root:
        _plan_store_ignore(plan, root_path, store_path.relative_to(resolved_root).as_posix())

    is_new_project = not (root_path / "pyproject.toml").exists()
    marker_relative = f"src/{package_name}/__init__.py"
    marker_path = root_path / marker_relative
    creates_marker = is_new_project and not marker_path.exists()

    _plan_durable(
        plan,
        relative=DEFAULT_POLICY_REF,
        path=root_path / DEFAULT_POLICY_REF,
        text=default_science_policy(package_name if creates_marker else None),
        validate=_validate_policy,
    )

    if is_new_project:
        plan.create(
            "pyproject.toml",
            root_path / "pyproject.toml",
            default_pyproject(project_name, package_name, feedbax_requirement=requirement),
        )
    else:
        plan.record(
            EntryOutcome(
                "pyproject.toml",
                "skipped",
                "this repository already declares a Python project",
            )
        )

    if not is_new_project:
        plan.record(
            EntryOutcome(
                marker_relative, "skipped", "this repository already declares a Python project"
            )
        )
    elif marker_path.exists():
        plan.record(EntryOutcome(marker_relative, "validated", "already present; not rewritten"))
    else:
        plan.create(marker_relative, marker_path, default_package_marker(package_name))

    _plan_agent_files(plan, root_path)
    return plan, project_name, package_name


def _default_feedbax_requirement() -> str:
    from feedbax.governance.agent_instructions import feedbax_version

    version = feedbax_version()
    return "feedbax" if version == "unknown" else f"feedbax>={version}"


# --- applying ----------------------------------------------------------------


def initialize(
    root: Path | str,
    *,
    project: str | None = None,
    package: str | None = None,
    dry_run: bool = False,
    feedbax_requirement: str | None = None,
) -> InitReport:
    """Create or validate one project skeleton, transactionally.

    Every outcome is computed first. If any entry conflicts, nothing at all is
    written. If a write fails part-way, every file this run created is removed
    again, so a failed ``init`` leaves the repository as it found it.
    """
    root_path = Path(root)
    plan, project_name, package_name = plan_init(
        root_path, project=project, package=package, feedbax_requirement=feedbax_requirement
    )
    report = InitReport(
        root=root_path,
        project=project_name,
        package=package_name,
        outcomes=tuple(plan.outcomes),
        dry_run=dry_run,
    )
    if dry_run or report.conflicts:
        return report
    absent_before = [
        root_path / name
        for name in AGENT_FILE_NAMES
        if not (root_path / name).exists() and not (root_path / name).is_symlink()
    ]
    written: list[_PlannedWrite] = []
    try:
        for planned in plan.writes:
            write_atomic(planned.path, planned.content)
            written.append(planned)
        install_agent_instructions(root_path)
    except Exception:
        for planned in reversed(written):
            if planned.previous_bytes is None:
                planned.path.unlink(missing_ok=True)
            else:
                write_atomic(planned.path, planned.previous_bytes)
        for path in absent_before:
            path.unlink(missing_ok=True)
        raise
    return report


def agent_block_status(root: Path | str) -> BlockStatus:
    """Return the managed block's state in one repository's primary agent file."""
    path = Path(root) / AGENT_FILE_NAMES[0]
    if not path.exists():
        return BlockStatus.MISSING
    return classify(parse_block(path.read_text(encoding="utf-8"), source=str(path)))


def created_entry_paths(report: InitReport) -> tuple[str, ...]:
    """Return the entries this run created, in report order."""
    return tuple(outcome.path for outcome in report.outcomes if outcome.action == "created")


#: The files a fresh project initializes, in report order.
INIT_ENTRIES: tuple[str, ...] = (
    "<project>.feedbax",
    ".gitignore",
    DEFAULT_POLICY_REF,
    "pyproject.toml",
    "src/<package>/__init__.py",
    AGENT_FILE_NAMES[0],
)


def entry_paths(project: str, package: str) -> tuple[str, ...]:
    """Return :data:`INIT_ENTRIES` with the package placeholder resolved."""
    return tuple(entry.replace("<project>", project).replace("<package>", package) for entry in INIT_ENTRIES)


__all__ = [
    "DEFAULT_POLICY_REF",
    "DEFAULT_STORE",
    "EXIT_CONFLICT",
    "EXIT_INFRASTRUCTURE",
    "EXIT_OK",
    "INIT_ENTRIES",
    "EntryOutcome",
    "InitReport",
    "ProjectInitError",
    "agent_block_status",
    "created_entry_paths",
    "default_package_marker",
    "default_pyproject",
    "default_science_policy",
    "derive_package_name",
    "derive_project_name",
    "entry_paths",
    "initialize",
    "plan_init",
]
