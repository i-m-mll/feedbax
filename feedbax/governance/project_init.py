"""``feedbax init``: the deterministic, transactional cold start for a project.

A project that uses Feedbax needs eight things and no more: a declaration of its
layout, the budgets that bound how large an authored envelope may be, a
root-anchored ignore rule for reproducible compiler output, a deny-by-default
authorization for its production Python, a minimal installable package to put
science in, and one agent instruction file reachable under both conventional
names. This module creates exactly those, and nothing else.

What it deliberately does *not* create is the point. No example envelope, no
fake base, no placeholder science, no project-local compiler or wrapper CLI, no
``generated/`` directory, no Git commit, no dependency resolution, and no
network access. An empty scaffold that means nothing is worse than no scaffold:
it invites someone to fill in the shape instead of authoring the science.

Two properties make it safe to run twice. It is **transactional** — every
outcome is computed before anything is written, any conflict refuses the whole
run, and a failure part-way through removes what this run created. And it is
**non-destructive** — a durable file that already exists is validated and left
exactly as it is. The single thing that upgrades itself in place is the
generated agent-instructions block, which is Feedbax's to maintain.

Initialization never makes the science-surface gate pass. That gate reads its
policy from a protected baseline ref, so a freshly written policy file is not
authoritative until it is committed and ratified there. ``init`` says so out
loud rather than producing green output that means nothing.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from feedbax.contracts.project_experiment import (
    PROJECT_DECLARATION_FILENAME,
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID,
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
    ProjectExperimentDeclarationError,
    load_project_declaration,
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

#: Conventional locations. Any of them may be moved by editing the declaration
#: afterwards; ``init`` itself is not configurable through a second file.
DEFAULT_ENVELOPE_DIRECTORY = "specs/experiment"
DEFAULT_OUTPUT_DIRECTORY = "generated"
DEFAULT_BUDGET_REF = f"{DEFAULT_ENVELOPE_DIRECTORY}/authoring_budget.json"
DEFAULT_POLICY_REF = "ci/feedbax-science-surface.toml"

AUTHORING_BUDGET_SCHEMA_ID = "feedbax.spec.authoring_budget"
AUTHORING_BUDGET_SCHEMA_VERSION = f"{AUTHORING_BUDGET_SCHEMA_ID}.v1"

#: The owner-ratified framework-wide default caps, one section per dialect layer.
#: A training envelope states a native composition delta and is the largest
#: authored document by construction; a report states ordered bindings and prose,
#: so its scalars may be longer while its structure stays flatter.
DEFAULT_AUTHORING_BUDGET_LAYERS: dict[str, dict[str, int]] = {
    "training": {
        "max_lines": 128,
        "max_bytes": 8192,
        "max_scalar_bytes": 512,
        "max_ast_nodes": 160,
        "max_depth": 16,
        "max_items": 32,
        "max_object_keys": 32,
        "max_assertions": 4,
    },
    "evaluation": {
        "max_lines": 80,
        "max_bytes": 4096,
        "max_scalar_bytes": 512,
        "max_ast_nodes": 120,
        "max_depth": 12,
        "max_items": 32,
        "max_object_keys": 32,
        "max_assertions": 4,
    },
    "analysis": {
        "max_lines": 80,
        "max_bytes": 4096,
        "max_scalar_bytes": 512,
        "max_ast_nodes": 120,
        "max_depth": 12,
        "max_items": 32,
        "max_object_keys": 32,
        "max_assertions": 4,
    },
    "figure": {
        "max_lines": 80,
        "max_bytes": 4096,
        "max_scalar_bytes": 512,
        "max_ast_nodes": 120,
        "max_depth": 12,
        "max_items": 32,
        "max_object_keys": 32,
        "max_assertions": 4,
    },
    "comparison": {
        "max_lines": 80,
        "max_bytes": 4096,
        "max_scalar_bytes": 512,
        "max_ast_nodes": 120,
        "max_depth": 12,
        "max_items": 32,
        "max_object_keys": 32,
        "max_assertions": 4,
    },
    "report": {
        "max_lines": 96,
        "max_bytes": 8192,
        "max_scalar_bytes": 4096,
        "max_ast_nodes": 120,
        "max_depth": 10,
        "max_items": 32,
        "max_object_keys": 32,
        "max_assertions": 2,
    },
}

DEFAULT_BUDGET_RATIONALE = (
    "Framework defaults from `feedbax init`. Every cap is a ceiling on one "
    "authored envelope, not a target: an envelope that needs more usually wants "
    "a different base rather than a larger budget."
)

#: Exit codes. Initialization follows the same contract as every other Feedbax
#: command: 0 succeeded, 2 a stable refusal with an actionable diagnostic.
EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_CONFLICT = 2


class ProjectInitError(Exception):
    """Raised when a project skeleton cannot be created safely."""


@dataclass(frozen=True)
class EntryOutcome:
    """What one of the eight entries is, or would be, after this run."""

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


def default_declaration_document(project: str) -> dict[str, Any]:
    """Return the declaration ``init`` writes for a project of this name."""
    return {
        "schema_id": PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID,
        "schema_version": PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
        "project": project,
        "envelope_directory": DEFAULT_ENVELOPE_DIRECTORY,
        "output_directory": DEFAULT_OUTPUT_DIRECTORY,
        "authoring_budget": DEFAULT_BUDGET_REF,
    }


def default_budget_document(project: str) -> dict[str, Any]:
    """Return the authoring-budget document ``init`` writes."""
    return {
        "schema_id": AUTHORING_BUDGET_SCHEMA_ID,
        "schema_version": AUTHORING_BUDGET_SCHEMA_VERSION,
        "budget_id": f"{project}.envelope_budgets.v1",
        "rationale": DEFAULT_BUDGET_RATIONALE,
        "layers": DEFAULT_AUTHORING_BUDGET_LAYERS,
    }


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

    def create(self, relative: str, path: Path, text: str) -> None:
        self.writes.append(_PlannedWrite(path, text))
        self.record(EntryOutcome(relative, "created"))

    def update(self, relative: str, path: Path, content: bytes, previous_bytes: bytes) -> None:
        self.writes.append(_PlannedWrite(path, content, previous_bytes))
        self.record(EntryOutcome(relative, "updated", "added the compiler-output rule"))


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


def _validate_declaration(root: Path) -> Callable[[Path], str | None]:
    def validate(_path: Path) -> str | None:
        try:
            load_project_declaration(root)
        except ProjectExperimentDeclarationError as exc:
            return str(exc)
        return None

    return validate


def _output_ignore_rule(output_directory: str) -> str:
    """Return one literal root-anchored Git ignore rule for compiler output."""
    escaped = "".join(f"\\{char}" if char in "\\*?[] " else char for char in output_directory)
    return f"/{escaped}/"


def _plan_output_ignore(plan: _Plan, root: Path, output_directory: str) -> None:
    """Install the declared compiler-output rule without rewriting other rules."""
    relative = ".gitignore"
    path = root / relative
    rule = _output_ignore_rule(output_directory)
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
        plan.record(EntryOutcome(relative, "validated", "compiler-output rule already present"))
        return
    separator = b"" if not previous_bytes or previous_bytes.endswith(b"\n") else b"\n"
    plan.update(
        relative,
        path,
        previous_bytes + separator + rule_bytes + b"\n",
        previous_bytes,
    )


def _validate_budget(path: Path) -> str | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"is not readable JSON: {exc}"
    if not isinstance(document, dict):
        return "is not one JSON object"
    identity = (document.get("schema_id"), document.get("schema_version"))
    if identity != (AUTHORING_BUDGET_SCHEMA_ID, AUTHORING_BUDGET_SCHEMA_VERSION):
        return (
            f"declares schema identity {identity}, expected "
            f"{(AUTHORING_BUDGET_SCHEMA_ID, AUTHORING_BUDGET_SCHEMA_VERSION)}"
        )
    if not isinstance(document.get("layers"), dict) or not document["layers"]:
        return "states no layer sections"
    return None


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
    project_name = project or derive_project_name(root_path)
    package_name = package or derive_package_name(project_name)
    requirement = feedbax_requirement or _default_feedbax_requirement()
    plan = _Plan()

    declaration_path = root_path / PROJECT_DECLARATION_FILENAME
    _plan_durable(
        plan,
        relative=PROJECT_DECLARATION_FILENAME,
        path=declaration_path,
        text=_json_text(default_declaration_document(project_name)),
        validate=_validate_declaration(root_path),
    )
    output_directory = DEFAULT_OUTPUT_DIRECTORY
    if declaration_path.exists():
        try:
            output_directory = load_project_declaration(root_path).output_directory
        except ProjectExperimentDeclarationError:
            pass
    _plan_output_ignore(plan, root_path, output_directory)
    _plan_durable(
        plan,
        relative=DEFAULT_BUDGET_REF,
        path=root_path / DEFAULT_BUDGET_REF,
        text=_json_text(default_budget_document(project_name)),
        validate=_validate_budget,
    )

    # A repository that already declares a Python project owns its own packaging
    # and its own source layout. Initialization states that it is staying out of
    # both rather than quietly writing a competing one.
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


def _json_text(document: Any) -> str:
    return json.dumps(document, indent=2) + "\n"


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


#: The eight entries a fresh ``feedbax init`` is responsible for, in report
#: order. Tests assert against this rather than against prose.
INIT_ENTRIES: tuple[str, ...] = (
    PROJECT_DECLARATION_FILENAME,
    ".gitignore",
    DEFAULT_BUDGET_REF,
    DEFAULT_POLICY_REF,
    "pyproject.toml",
    "src/<package>/__init__.py",
    AGENT_FILE_NAMES[0],
    AGENT_FILE_NAMES[1],
)


def entry_paths(package: str) -> tuple[str, ...]:
    """Return :data:`INIT_ENTRIES` with the package placeholder resolved."""
    return tuple(entry.replace("<package>", package) for entry in INIT_ENTRIES)


__all__ = [
    "AUTHORING_BUDGET_SCHEMA_ID",
    "AUTHORING_BUDGET_SCHEMA_VERSION",
    "DEFAULT_AUTHORING_BUDGET_LAYERS",
    "DEFAULT_BUDGET_RATIONALE",
    "DEFAULT_BUDGET_REF",
    "DEFAULT_ENVELOPE_DIRECTORY",
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_POLICY_REF",
    "EXIT_CONFLICT",
    "EXIT_INFRASTRUCTURE",
    "EXIT_OK",
    "INIT_ENTRIES",
    "EntryOutcome",
    "InitReport",
    "ProjectInitError",
    "agent_block_status",
    "created_entry_paths",
    "default_budget_document",
    "default_declaration_document",
    "default_package_marker",
    "default_pyproject",
    "default_science_policy",
    "derive_package_name",
    "derive_project_name",
    "entry_paths",
    "initialize",
    "plan_init",
]
