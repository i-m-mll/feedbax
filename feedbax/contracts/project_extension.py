"""Typed project extension declaration for the authored-experiment engine.

A downstream project tells Feedbax what it extends by exposing exactly one
:class:`ProjectExtensionDeclaration` through the single ``feedbax.plugins``
entry point, registered inside the ordinary transactional bootstrap with its
registry injected like every other family. There is no decorator, no
import-time discovery, no module-path or package-scanning convention, and no
second component registry: the declaration is inert typed data plus the
callables it binds.

What the declaration carries, and nothing more:

* the project name and the diagnostic source that authored the declaration;
* the envelope families the project accepts and the ones it retired;
* one binding per layer label — authored model, lowerer, output family;
* versioned applicability-rule ids mapped to their callbacks;
* the project's authoring-budget resource reference;
* the logical compiler-contract id and version.

What it deliberately does not carry: serialization, migration algorithms,
custody code, executor logic, and module paths for dynamic import. A binding
names a Python object the project already resolved; Feedbax never imports on a
declaration's behalf.

Label resolution is the one behavior this module adds, and it is fail-closed:

* the project package failed to load, a declaration is malformed, or two
  declarations collide — that is an infrastructure failure. It is raised as an
  ordinary ``BootstrapError`` while the bootstrap is still running, so it never
  reaches the authoring command and ``python -m feedbax`` exits 1;
* the project loaded but an authored label is absent from its declaration —
  that is an authored rejection under the stable
  ``unresolved-extension-label`` category, and
  ``preflight-experiment-envelope`` returns 2 with the full field set on
  stderr.

Both decisions are reached before the compiler runs and therefore before any
output file exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Protocol, runtime_checkable

from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
    envelope_schema_of,
)
from feedbax.registry_errors import RegistryCollisionError

PROJECT_EXTENSION_DECLARATION_SCHEMA_ID = "feedbax.plugin.project_extension_declaration"
PROJECT_EXTENSION_DECLARATION_SCHEMA_VERSION = "feedbax.plugin.project_extension_declaration.v1"


class ProjectExtensionPlugPoint(StrEnum):
    """Closed set of plug points an authored envelope may name a label at."""

    LAYER = "layer"
    APPLICABILITY_RULE = "applicability_rule"


class EnvelopeFamilyStatus(StrEnum):
    """Whether a declared envelope family is still authored or was retired."""

    ACCEPTED = "accepted"
    RETIRED = "retired"


class ProjectExtensionDeclarationError(ValueError):
    """Raised when a project extension declaration cannot be admitted."""


class ProjectExtensionCollisionError(ProjectExtensionDeclarationError, RegistryCollisionError):
    """Raised when two declarations claim one project name or envelope family."""


@runtime_checkable
class AuthoringBudgetRoot(Protocol):
    """The resource-root surface Feedbax needs from a budget reference."""

    def joinpath(self, *names: str) -> Any:
        """Return a child of this resource root."""

    def is_dir(self) -> bool:
        """Return whether this resource root is a directory."""


@dataclass(frozen=True)
class EnvelopeFamilyDeclaration:
    """One authored envelope family the project owns, accepted or retired.

    ``schema_version`` is the exact string an envelope declares in its
    ``schema`` field; ``schema_id`` is its unversioned identity, following the
    ordinary Feedbax convention that a version string extends its id.
    """

    schema_id: str
    schema_version: str
    status: EnvelopeFamilyStatus = EnvelopeFamilyStatus.ACCEPTED
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        if not self.schema_id.strip() or not self.schema_version.strip():
            raise ProjectExtensionDeclarationError(
                "envelope family must declare a nonempty schema id and schema version"
            )
        if not self.schema_version.startswith(f"{self.schema_id}."):
            raise ProjectExtensionDeclarationError(
                f"envelope family schema version {self.schema_version!r} does not extend "
                f"schema id {self.schema_id!r}"
            )
        status = EnvelopeFamilyStatus(self.status)
        if status is EnvelopeFamilyStatus.RETIRED and not (self.superseded_by or "").strip():
            raise ProjectExtensionDeclarationError(
                f"retired envelope family {self.schema_version!r} must name what replaced it"
            )
        if status is EnvelopeFamilyStatus.ACCEPTED and self.superseded_by is not None:
            raise ProjectExtensionDeclarationError(
                f"accepted envelope family {self.schema_version!r} cannot name a replacement"
            )


@dataclass(frozen=True)
class LayerBinding:
    """One labeled layer: what it is authored as, lowered by, and compiles into."""

    label: str
    authored_model: type
    lowerer: Callable[..., Any]
    output_family: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ProjectExtensionDeclarationError("layer binding must declare a nonempty label")
        if not isinstance(self.authored_model, type):
            raise ProjectExtensionDeclarationError(
                f"layer {self.label!r} must bind an authored model type"
            )
        if not callable(self.lowerer):
            raise ProjectExtensionDeclarationError(
                f"layer {self.label!r} must bind a callable lowerer"
            )
        if not self.output_family.strip():
            raise ProjectExtensionDeclarationError(
                f"layer {self.label!r} must declare a nonempty output family"
            )


@dataclass(frozen=True)
class ApplicabilityRuleBinding:
    """One versioned applicability rule id bound to the callback that decides it.

    The rule id carries its own version as its final dotted segment, exactly as
    Feedbax schema versions do, so an authored envelope naming a rule names the
    version it was written against.
    """

    rule_id: str
    version: str
    applies: Callable[..., Any]

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.version.strip():
            raise ProjectExtensionDeclarationError(
                "applicability rule must declare a nonempty id and version"
            )
        if not self.rule_id.endswith(f".{self.version}"):
            raise ProjectExtensionDeclarationError(
                f"applicability rule id {self.rule_id!r} does not end with its declared "
                f"version {self.version!r}"
            )
        if not callable(self.applies):
            raise ProjectExtensionDeclarationError(
                f"applicability rule {self.rule_id!r} must bind a callable"
            )


@dataclass(frozen=True)
class AuthoringBudgetResource:
    """Where a project's authoring budgets live, as an already-resolved reference.

    The project resolves its own resource root and hands the result over. This
    module never reads, parses, or imports anything through it; the budget's
    content is the budget lane's business.
    """

    resource_id: str
    root: AuthoringBudgetRoot

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ProjectExtensionDeclarationError(
                "authoring budget resource must declare a nonempty resource id"
            )
        if not isinstance(self.root, AuthoringBudgetRoot):
            raise ProjectExtensionDeclarationError(
                f"authoring budget resource {self.resource_id!r} must supply a resource root"
            )


@dataclass(frozen=True)
class AuthoredLabelSite:
    """The authored field a plug point's label is read from.

    ``field`` is a dotted path of mapping keys into the authored envelope. The
    site is a data address, not dialect semantics: Feedbax reads the string
    that is there and resolves it, and leaves whether the field was required to
    the project's own compiler.
    """

    plug_point: ProjectExtensionPlugPoint
    field: str

    def __post_init__(self) -> None:
        if not self.field.strip() or any(not part for part in self.field.split(".")):
            raise ProjectExtensionDeclarationError(
                f"authored label site field {self.field!r} is not a dotted authored path"
            )
        ProjectExtensionPlugPoint(self.plug_point)


@dataclass(frozen=True)
class ProjectExtensionDeclaration:
    """Everything one project declares about the engine it extends."""

    project: str
    declaration_source: str
    compiler_contract_id: str
    compiler_contract_version: str
    authoring_budget: AuthoringBudgetResource
    envelope_families: tuple[EnvelopeFamilyDeclaration, ...] = ()
    layers: tuple[LayerBinding, ...] = ()
    applicability_rules: tuple[ApplicabilityRuleBinding, ...] = ()
    label_sites: tuple[AuthoredLabelSite, ...] = ()
    schema_id: str = PROJECT_EXTENSION_DECLARATION_SCHEMA_ID
    schema_version: str = PROJECT_EXTENSION_DECLARATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("project", "declaration_source", "compiler_contract_id"):
            if not str(getattr(self, name)).strip():
                raise ProjectExtensionDeclarationError(
                    f"project extension declaration {name} must be nonempty"
                )
        if self.schema_id != PROJECT_EXTENSION_DECLARATION_SCHEMA_ID:
            raise ProjectExtensionDeclarationError(
                f"unsupported project extension declaration schema_id: {self.schema_id!r}"
            )
        if self.schema_version != PROJECT_EXTENSION_DECLARATION_SCHEMA_VERSION:
            raise ProjectExtensionDeclarationError(
                f"unsupported project extension declaration schema_version: {self.schema_version!r}"
            )
        if not self.compiler_contract_version.startswith(f"{self.compiler_contract_id}."):
            raise ProjectExtensionDeclarationError(
                f"compiler contract version {self.compiler_contract_version!r} does not "
                f"extend contract id {self.compiler_contract_id!r}"
            )
        if not isinstance(self.authoring_budget, AuthoringBudgetResource):
            raise ProjectExtensionDeclarationError(
                f"project {self.project!r} must declare an AuthoringBudgetResource"
            )
        self._validate_unique(
            "envelope family", (item.schema_version for item in self.envelope_families)
        )
        self._validate_unique("layer label", (item.label for item in self.layers))
        self._validate_unique(
            "applicability rule id", (item.rule_id for item in self.applicability_rules)
        )
        self._validate_unique(
            "authored label site",
            (f"{item.plug_point.value}:{item.field}" for item in self.label_sites),
        )
        for site in self.label_sites:
            if not self.labels(site.plug_point):
                raise ProjectExtensionDeclarationError(
                    f"project {self.project!r} reads a {site.plug_point.value!r} label from "
                    f"{site.field!r} but declares no {site.plug_point.value} binding"
                )

    def _validate_unique(self, kind: str, values: Any) -> None:
        seen: set[str] = set()
        for value in values:
            if value in seen:
                raise ProjectExtensionDeclarationError(
                    f"project {self.project!r} declares duplicate {kind} {value!r}"
                )
            seen.add(value)

    @property
    def accepted_families(self) -> tuple[EnvelopeFamilyDeclaration, ...]:
        """Return every still-authored envelope family, in declaration order."""
        return tuple(
            item
            for item in self.envelope_families
            if EnvelopeFamilyStatus(item.status) is EnvelopeFamilyStatus.ACCEPTED
        )

    @property
    def retired_families(self) -> tuple[EnvelopeFamilyDeclaration, ...]:
        """Return every retired envelope family, in declaration order."""
        return tuple(
            item
            for item in self.envelope_families
            if EnvelopeFamilyStatus(item.status) is EnvelopeFamilyStatus.RETIRED
        )

    def accepts_envelope_schema(self, envelope_schema: str) -> bool:
        """Return whether this project still authors *envelope_schema*."""
        return any(item.schema_version == envelope_schema for item in self.accepted_families)

    def binding(self, plug_point: ProjectExtensionPlugPoint, label: str) -> object | None:
        """Return the binding *label* names at *plug_point*, or ``None``."""
        point = ProjectExtensionPlugPoint(plug_point)
        if point is ProjectExtensionPlugPoint.LAYER:
            return next((item for item in self.layers if item.label == label), None)
        return next((item for item in self.applicability_rules if item.rule_id == label), None)

    def labels(self, plug_point: ProjectExtensionPlugPoint) -> tuple[str, ...]:
        """Return every label declared at *plug_point*, in stable sorted order."""
        point = ProjectExtensionPlugPoint(plug_point)
        if point is ProjectExtensionPlugPoint.LAYER:
            return tuple(sorted(item.label for item in self.layers))
        return tuple(sorted(item.rule_id for item in self.applicability_rules))


class ProjectExtensionRegistry:
    """Injected registry resolving one declaration per project and per family."""

    def __init__(self) -> None:
        self._sealed = False
        self._declarations: dict[str, ProjectExtensionDeclaration] = {}
        self._by_family: dict[str, str] = {}

    def register(self, declaration: ProjectExtensionDeclaration) -> None:
        """Register one project declaration; any duplicate claim fails closed."""
        if self._sealed:
            raise RuntimeError("project extension registry is sealed")
        if not isinstance(declaration, ProjectExtensionDeclaration):
            raise TypeError("declaration must be a ProjectExtensionDeclaration")
        existing = self._declarations.get(declaration.project)
        if existing is not None:
            raise ProjectExtensionCollisionError(
                f"project {declaration.project!r} is already declared by "
                f"{existing.declaration_source!r}"
            )
        for family in declaration.accepted_families:
            owner = self._by_family.get(family.schema_version)
            if owner is not None:
                raise ProjectExtensionCollisionError(
                    f"envelope family {family.schema_version!r} is already accepted by "
                    f"project {owner!r}"
                )
        self._declarations[declaration.project] = declaration
        for family in declaration.accepted_families:
            self._by_family[family.schema_version] = declaration.project

    def get(self, project: str) -> ProjectExtensionDeclaration:
        """Return the declaration for *project* or fail closed."""
        declaration = self._declarations.get(project)
        if declaration is None:
            known = ", ".join(sorted(self._declarations)) or "none"
            raise ProjectExtensionDeclarationError(
                f"no registered project extension declaration for {project!r}; "
                f"registered projects: {known}"
            )
        return declaration

    def for_envelope_schema(self, envelope_schema: str) -> ProjectExtensionDeclaration | None:
        """Return the project accepting *envelope_schema*, or ``None`` if none does."""
        project = self._by_family.get(envelope_schema)
        return None if project is None else self._declarations[project]

    def available_keys(self) -> tuple[str, ...]:
        """Return every declared project name, in stable order."""
        return tuple(sorted(self._declarations))

    def seal(self) -> None:
        """Prevent further registration after bootstrap completes."""
        self._sealed = True


class UnresolvedExtensionLabelRejection(ExperimentEnvelopeRejection):
    """An authored envelope named a label the project's declaration does not bind."""

    def __init__(
        self,
        *,
        project: str,
        family: str,
        field: str,
        label: str,
        plug_point: ProjectExtensionPlugPoint,
        available_labels: tuple[str, ...],
        declaration_source: str,
        repair: str,
    ) -> None:
        point = ProjectExtensionPlugPoint(plug_point)
        super().__init__(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_EXTENSION_LABEL,
            f"project {project!r} declares no {point.value} named {label!r}",
            field=field,
            correct_home=repair,
        )
        self.project = project
        self.family = family
        self.label = label
        self.plug_point = point
        self.available_labels = tuple(sorted(available_labels))
        self.declaration_source = declaration_source
        self.repair = repair

    def render(self) -> str:
        """Return the one-line diagnostic written to stderr."""
        available = ", ".join(self.available_labels) or "none"
        parts = [
            f"category={self.category.value}",
            f"project={self.project}",
            f"family={self.family}",
            f"plug_point={self.plug_point.value}",
            f"field={self.field}",
            f"label={self.label}",
            f"available_labels={available}",
            f"declaration={self.declaration_source}",
            str(self),
        ]
        return "envelope rejected: " + "; ".join(parts) + f"; repair: {self.repair}"


@dataclass(frozen=True)
class ResolvedExtensionLabel:
    """One authored label, the site it came from, and the binding it resolved to."""

    plug_point: ProjectExtensionPlugPoint
    field: str
    label: str
    binding: object


def _authored_value(envelope: Mapping[str, Any], field: str) -> Any:
    """Return the value at a dotted mapping path, or ``None`` if it is not there."""
    value: Any = envelope
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def resolve_authored_extension_labels(
    envelope: Mapping[str, Any],
    registry: ProjectExtensionRegistry,
) -> tuple[ResolvedExtensionLabel, ...]:
    """Resolve every authored extension label before the compiler is invoked.

    A project that accepts no envelope family matching this document has
    nothing to resolve and nothing to say about it. When a project does accept
    it, every label the declaration says it reads must bind, or the envelope is
    rejected here — before compilation, and before any output file is written.
    """
    schema = envelope_schema_of(envelope)
    declaration = registry.for_envelope_schema(schema)
    if declaration is None:
        return ()
    resolved: list[ResolvedExtensionLabel] = []
    for site in declaration.label_sites:
        label = _authored_value(envelope, site.field)
        if not isinstance(label, str) or not label:
            continue
        binding = declaration.binding(site.plug_point, label)
        if binding is None:
            available = declaration.labels(site.plug_point)
            raise UnresolvedExtensionLabelRejection(
                project=declaration.project,
                family=schema,
                field=site.field,
                label=label,
                plug_point=site.plug_point,
                available_labels=available,
                declaration_source=declaration.declaration_source,
                repair=(
                    f"author one of the {site.plug_point.value} labels project "
                    f"{declaration.project!r} declares "
                    f"({', '.join(available) or 'none'}) at {site.field!r}, or add the "
                    f"binding to the declaration at {declaration.declaration_source}"
                ),
            )
        resolved.append(
            ResolvedExtensionLabel(
                plug_point=ProjectExtensionPlugPoint(site.plug_point),
                field=site.field,
                label=label,
                binding=binding,
            )
        )
    return tuple(resolved)


__all__ = [
    "PROJECT_EXTENSION_DECLARATION_SCHEMA_ID",
    "PROJECT_EXTENSION_DECLARATION_SCHEMA_VERSION",
    "ApplicabilityRuleBinding",
    "AuthoredLabelSite",
    "AuthoringBudgetResource",
    "AuthoringBudgetRoot",
    "EnvelopeFamilyDeclaration",
    "EnvelopeFamilyStatus",
    "LayerBinding",
    "ProjectExtensionCollisionError",
    "ProjectExtensionDeclaration",
    "ProjectExtensionDeclarationError",
    "ProjectExtensionPlugPoint",
    "ProjectExtensionRegistry",
    "ResolvedExtensionLabel",
    "UnresolvedExtensionLabelRejection",
    "resolve_authored_extension_labels",
]
