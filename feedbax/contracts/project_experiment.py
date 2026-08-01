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

The declaration is still registered through the ordinary ``feedbax.plugins``
entry point inside the transactional bootstrap, with its registry injected like
every other family. A project that also registers real scientific
implementations — components, recipes, figure constructors — registers those the
same way it always did; they are what validate the vocabulary this declaration
deliberately says nothing about.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from feedbax.registry_errors import RegistryCollisionError

PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID = "feedbax.project_experiment"
PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION = "feedbax.project_experiment.v1"


class ProjectExperimentDeclarationError(ValueError):
    """Raised when a project experiment declaration cannot be admitted."""


class ProjectExperimentCollisionError(
    ProjectExperimentDeclarationError, RegistryCollisionError
):
    """Raised when two declarations claim one project name or one directory."""


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


@dataclass(frozen=True)
class ProjectExperimentDeclaration:
    """Everything one project declares about its authored experiments.

    Attributes:
        project: The project's name, used for diagnostics and for registry
            identity. It is never read by the compiler to decide anything.
        declaration_source: Where this declaration was authored, so a collision
            names both claimants.
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
        if self.schema_version != PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION:
            raise ProjectExperimentDeclarationError(
                "unsupported project experiment declaration schema_version: "
                f"{self.schema_version!r}"
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
        normalized = posixpath.normpath(envelope_ref.replace("\\", "/"))
        return PurePosixPath(normalized).parts[:1] == (self.envelope_directory,)


class ProjectExperimentRegistry:
    """Injected registry resolving one declaration per project and per directory."""

    def __init__(self) -> None:
        self._sealed = False
        self._declarations: dict[str, ProjectExperimentDeclaration] = {}
        self._by_directory: dict[str, str] = {}

    def register(self, declaration: ProjectExperimentDeclaration) -> None:
        """Register one project declaration; any duplicate claim fails closed."""
        if self._sealed:
            raise RuntimeError("project experiment registry is sealed")
        if not isinstance(declaration, ProjectExperimentDeclaration):
            raise TypeError("declaration must be a ProjectExperimentDeclaration")
        existing = self._declarations.get(declaration.project)
        if existing is not None:
            raise ProjectExperimentCollisionError(
                f"project {declaration.project!r} is already declared by "
                f"{existing.declaration_source!r}"
            )
        owner = self._by_directory.get(declaration.envelope_directory)
        if owner is not None:
            raise ProjectExperimentCollisionError(
                f"envelope directory {declaration.envelope_directory!r} is already claimed "
                f"by project {owner!r}"
            )
        self._declarations[declaration.project] = declaration
        self._by_directory[declaration.envelope_directory] = declaration.project

    def get(self, project: str) -> ProjectExperimentDeclaration:
        """Return the declaration for *project* or fail closed."""
        declaration = self._declarations.get(project)
        if declaration is None:
            known = ", ".join(sorted(self._declarations)) or "none"
            raise ProjectExperimentDeclarationError(
                f"no registered project experiment declaration for {project!r}; "
                f"registered projects: {known}"
            )
        return declaration

    def for_envelope_ref(self, envelope_ref: str) -> ProjectExperimentDeclaration:
        """Return the one project whose envelope directory holds *envelope_ref*.

        Resolution is by declared directory, which is data. It is never by
        project name, package name, or anything read out of the envelope, so an
        envelope cannot select which project's budgets it is judged under.
        """
        matches = [
            declaration
            for declaration in self._declarations.values()
            if declaration.owns_envelope_ref(envelope_ref)
        ]
        if len(matches) == 1:
            return matches[0]
        directories = ", ".join(sorted(self._by_directory)) or "none"
        if not matches:
            raise ProjectExperimentDeclarationError(
                f"{envelope_ref!r} is in no declared envelope directory; "
                f"declared directories: {directories}"
            )
        raise ProjectExperimentCollisionError(
            f"{envelope_ref!r} is claimed by projects "
            f"{sorted(item.project for item in matches)}"
        )

    def available_keys(self) -> tuple[str, ...]:
        """Return every declared project name, in stable order."""
        return tuple(sorted(self._declarations))

    def seal(self) -> None:
        """Prevent further registration after bootstrap completes."""
        self._sealed = True


__all__ = [
    "PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID",
    "PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION",
    "AuthoringBudgetResource",
    "AuthoringBudgetRoot",
    "ProjectExperimentCollisionError",
    "ProjectExperimentDeclaration",
    "ProjectExperimentDeclarationError",
    "ProjectExperimentRegistry",
]
