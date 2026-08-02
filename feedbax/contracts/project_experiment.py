"""What a project declares about its authored experiments: data, and only data.

A project tells Feedbax three things — where its envelopes live, where its
compiled outputs go, and where its authoring budgets are. That is the whole
declaration. It binds no callable, names no layer, claims no envelope family,
and states no compiler contract, because under the single
``feedbax.experiment_envelope.v1`` dialect there is nothing left for a project
to decide about *how* compilation works:

* the dialect is fixed, so there is no family to claim;
* the five layers are Feedbax's own artifact families, so there is no layer
  binding, no authored model, and no lowerer;
* the compiler contract is global
  (:data:`~feedbax.contracts.experiment_envelope_dialect.EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION`),
  so there is no per-project contract id;
* applicability is authored in the envelope or decided by a versioned Feedbax
  structural rule, so there is no applicability callback.

What remains is repository layout and one budget resource, which are facts about
where a project keeps its files. A declaration is therefore inert: nothing in it
can change what a compiled document *means*, only which directory it is read
from and written to. That is the property that makes two projects' corpora
comparable.

Because the declaration is data and nothing else, it is *read* as data. One
project root holds one :data:`PROJECT_DECLARATION_FILENAME` file, and
:func:`load_project_declaration` parses it against an explicit schema identity.
There is no entry point, no registry, and no discovery: a caller states the root
it means, and an unknown schema id or version fails closed rather than being
inferred. Entry points remain what they always were — the way a project
registers genuine scientific implementations (components, recipes, figure
constructors), which is exactly the vocabulary this declaration deliberately
says nothing about.
"""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID = "feedbax.project_experiment"
PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION = "feedbax.project_experiment.v1"

#: Every declaration schema version this Feedbax admits. One entry means one
#: shape; a second version arrives with a migration rule or an explicit refusal,
#: never with a silent reinterpretation of the version it replaces.
PROJECT_EXPERIMENT_DECLARATION_SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
)

#: The one filename a project root uses to declare its experiment layout.
PROJECT_DECLARATION_FILENAME = "feedbax.project.json"

#: The exact key set an admitted declaration document states. An absent key is
#: incomplete and an unknown key is a misunderstanding; both fail closed rather
#: than being defaulted or ignored.
PROJECT_DECLARATION_DOCUMENT_KEYS: frozenset[str] = frozenset(
    {
        "schema_id",
        "schema_version",
        "project",
        "envelope_directory",
        "output_directory",
        "authoring_budget",
    }
)


class ProjectExperimentDeclarationError(ValueError):
    """Raised when a project experiment declaration cannot be admitted."""


@runtime_checkable
class AuthoringBudgetRoot(Protocol):
    """The resource-root surface Feedbax needs from a budget reference."""

    def joinpath(self, *names: str) -> Any:
        """Return a child of this resource root."""
        ...

    def is_dir(self) -> bool:
        """Return whether this resource root is a directory."""
        ...


@dataclass(frozen=True)
class AuthoringBudgetResource:
    """Where a project's authoring budgets live, as an already-resolved reference.

    The project resolves its own resource root and hands the result over. This
    module never reads, parses, or imports anything through it; the budget's
    content is the budget lane's business.
    """

    resource_id: str
    root: AuthoringBudgetRoot
    document_name: str = "authoring_budget.json"

    def __post_init__(self) -> None:
        if not self.document_name.strip() or "/" in self.document_name:
            raise ProjectExperimentDeclarationError(
                f"authoring budget document name must be a bare filename: "
                f"{self.document_name!r}"
            )
        if not self.resource_id.strip():
            raise ProjectExperimentDeclarationError(
                "authoring budget resource must declare a nonempty resource id"
            )
        if not isinstance(self.root, AuthoringBudgetRoot):
            raise ProjectExperimentDeclarationError(
                f"authoring budget resource {self.resource_id!r} must supply a resource root"
            )


def _validate_directory(value: str, name: str) -> str:
    """Refuse anything but a normalized, relative, non-escaping repo directory."""
    if not value.strip():
        raise ProjectExperimentDeclarationError(f"{name} must be nonempty")
    if PurePosixPath(value).is_absolute() or value.startswith("/"):
        raise ProjectExperimentDeclarationError(f"{name} must be repo-relative: {value!r}")
    normalized = posixpath.normpath(value)
    if normalized != value or normalized.startswith(".."):
        raise ProjectExperimentDeclarationError(
            f"{name} must be a normalized repo-relative directory: {value!r}"
        )
    return normalized


def path_is_within(ref: str, directory: str) -> bool:
    """Return whether repo-relative *ref* lies anywhere under *directory*.

    Both sides are declared repo-relative POSIX paths, and a declared directory
    may have any number of segments — ``_validate_directory`` accepts
    ``specs/experiment`` exactly as it accepts ``studies``. So the comparison is
    a full segment-prefix match: matching only the first segment would say that
    ``specs/base/x.json`` lives in ``specs/experiment``, and would say that
    nothing at all lives in a nested directory.

    The reference is normalized first, so ``a/x``, ``./a/x``, and ``b/../a/x``
    are one rule rather than three holes.
    """
    normalized = PurePosixPath(posixpath.normpath(ref.replace("\\", "/")))
    expected = PurePosixPath(directory).parts
    return normalized.parts[: len(expected)] == expected


@dataclass(frozen=True)
class ProjectExperimentDeclaration:
    """Everything one project declares about its authored experiments.

    Attributes:
        project: The project's name, used for diagnostics and for registry
            identity. It is never read by the compiler to decide anything.
        declaration_source: Where this declaration was read from, so every
            diagnostic names the file a reader must go and fix.
        envelope_directory: Repo-relative directory holding authored envelopes.
            An envelope alias resolves inside it and nowhere else.
        output_directory: Repo-relative directory holding compiled documents and
            their locks. A base may never point into it.
        authoring_budget: The project's already-resolved budget resource.
    """

    project: str
    declaration_source: str
    envelope_directory: str
    output_directory: str
    authoring_budget: AuthoringBudgetResource
    schema_id: str = PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID
    schema_version: str = PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("project", "declaration_source"):
            if not str(getattr(self, name)).strip():
                raise ProjectExperimentDeclarationError(
                    f"project experiment declaration {name} must be nonempty"
                )
        if self.schema_id != PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID:
            raise ProjectExperimentDeclarationError(
                f"unsupported project experiment declaration schema_id: {self.schema_id!r}"
            )
        if self.schema_version not in PROJECT_EXPERIMENT_DECLARATION_SUPPORTED_SCHEMA_VERSIONS:
            raise ProjectExperimentDeclarationError(
                "unsupported project experiment declaration schema_version: "
                f"{self.schema_version!r}; supported="
                f"{list(PROJECT_EXPERIMENT_DECLARATION_SUPPORTED_SCHEMA_VERSIONS)}; "
                "migration_intentionally_absent"
            )
        object.__setattr__(
            self, "envelope_directory", _validate_directory(self.envelope_directory, "envelope_directory")
        )
        object.__setattr__(
            self, "output_directory", _validate_directory(self.output_directory, "output_directory")
        )
        if self.envelope_directory == self.output_directory:
            raise ProjectExperimentDeclarationError(
                f"project {self.project!r} files authored envelopes and compiled output in "
                f"the same directory {self.envelope_directory!r}; a compiled document is "
                "never an authored parent, so the two cannot share a home"
            )
        if not isinstance(self.authoring_budget, AuthoringBudgetResource):
            raise ProjectExperimentDeclarationError(
                f"project {self.project!r} must declare an AuthoringBudgetResource"
            )

    def owns_envelope_ref(self, envelope_ref: str) -> bool:
        """Return whether *envelope_ref* lies in this project's envelope directory."""
        return path_is_within(envelope_ref, self.envelope_directory)


def project_declaration_path(root: Path | str) -> Path:
    """Return where *root* states its declaration; the name is never searched for."""
    return Path(root) / PROJECT_DECLARATION_FILENAME


def _required_string(document: Any, key: str, source: str) -> str:
    value = document[key]
    if not isinstance(value, str) or not value.strip():
        raise ProjectExperimentDeclarationError(
            f"{source}#{key} must be a nonempty string, found {value!r}"
        )
    return value


def _budget_resource(budget_ref: str, budget_root: AuthoringBudgetRoot, source: str) -> (
    AuthoringBudgetResource
):
    """Resolve the declared repo-relative budget path against a resource root."""
    if budget_ref != budget_ref.strip() or budget_ref.endswith("/"):
        raise ProjectExperimentDeclarationError(
            f"{source}#authoring_budget must be a repo-relative file path: {budget_ref!r}"
        )
    directory, _, document_name = budget_ref.rpartition("/")
    if not document_name:
        raise ProjectExperimentDeclarationError(
            f"{source}#authoring_budget must name a budget document: {budget_ref!r}"
        )
    root = budget_root
    if directory:
        parts = _validate_directory(directory, f"{source}#authoring_budget").split("/")
        root = budget_root.joinpath(*parts)
    return AuthoringBudgetResource(
        resource_id=budget_ref, root=root, document_name=document_name
    )


def parse_project_declaration(
    raw: bytes,
    *,
    budget_root: AuthoringBudgetRoot,
    source: str,
) -> ProjectExperimentDeclaration:
    """Parse one declaration document's bytes against its stated schema identity.

    Args:
        raw: The declaration document's exact bytes.
        budget_root: The resource root the declared budget path resolves under,
            which is the project root for a repository on disk.
        source: Where these bytes came from, recorded as the declaration source
            and used to qualify every diagnostic.

    Raises:
        ProjectExperimentDeclarationError: If the bytes are not one JSON object
            stating exactly the admitted keys at a supported schema identity.
    """
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProjectExperimentDeclarationError(
            f"{source} is not a readable JSON project declaration: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ProjectExperimentDeclarationError(
            f"{source} must be one JSON object, found {type(document).__name__}"
        )
    found_id = document.get("schema_id")
    if found_id != PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID:
        raise ProjectExperimentDeclarationError(
            f"{source}#schema_id must be "
            f"{PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID!r}, found {found_id!r}"
        )
    found_version = document.get("schema_version")
    if found_version not in PROJECT_EXPERIMENT_DECLARATION_SUPPORTED_SCHEMA_VERSIONS:
        raise ProjectExperimentDeclarationError(
            f"{source}#schema_version {found_version!r} is unsupported; supported="
            f"{list(PROJECT_EXPERIMENT_DECLARATION_SUPPORTED_SCHEMA_VERSIONS)}; "
            "migration_intentionally_absent"
        )
    present = set(document)
    missing = sorted(PROJECT_DECLARATION_DOCUMENT_KEYS - present)
    if missing:
        raise ProjectExperimentDeclarationError(f"{source} omits required keys: {missing}")
    unknown = sorted(present - PROJECT_DECLARATION_DOCUMENT_KEYS)
    if unknown:
        raise ProjectExperimentDeclarationError(
            f"{source} states keys this declaration schema does not define: {unknown}"
        )
    return ProjectExperimentDeclaration(
        project=_required_string(document, "project", source),
        declaration_source=source,
        envelope_directory=_required_string(document, "envelope_directory", source),
        output_directory=_required_string(document, "output_directory", source),
        authoring_budget=_budget_resource(
            _required_string(document, "authoring_budget", source), budget_root, source
        ),
        schema_id=str(found_id),
        schema_version=str(found_version),
    )


def load_project_declaration(root: Path | str) -> ProjectExperimentDeclaration:
    """Load the declaration one stated project root authors, or fail closed.

    The root is always explicit. Nothing walks upwards, scans siblings, consults
    an environment variable, or infers a project from a package name: a caller
    that cannot say which root it means does not yet know what it is compiling.
    """
    path = project_declaration_path(root)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProjectExperimentDeclarationError(
            f"no project declaration at {path}: {exc}; a Feedbax project states its "
            f"experiment layout in {PROJECT_DECLARATION_FILENAME} at its root"
        ) from exc
    return parse_project_declaration(raw, budget_root=path.parent, source=str(path))


__all__ = [
    "PROJECT_DECLARATION_DOCUMENT_KEYS",
    "PROJECT_DECLARATION_FILENAME",
    "PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID",
    "PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION",
    "PROJECT_EXPERIMENT_DECLARATION_SUPPORTED_SCHEMA_VERSIONS",
    "AuthoringBudgetResource",
    "AuthoringBudgetRoot",
    "ProjectExperimentDeclaration",
    "ProjectExperimentDeclarationError",
    "load_project_declaration",
    "parse_project_declaration",
    "project_declaration_path",
]
