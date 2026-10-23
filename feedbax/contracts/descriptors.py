"""Semantic variable and component descriptor contracts."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.canonical_json import canonical_json_v1_bytes
from feedbax.contracts.value_schema import ValueSchema


VARIABLE_DESCRIPTOR_SCHEMA_ID = "feedbax.spec.descriptor.variable"
VARIABLE_DESCRIPTOR_SCHEMA_VERSION = "feedbax.spec.descriptor.variable.v1"
COMPONENT_DESCRIPTOR_SCHEMA_ID = "feedbax.spec.descriptor.component"
COMPONENT_DESCRIPTOR_SCHEMA_VERSION = "feedbax.spec.descriptor.component.v1"
DESCRIPTOR_BASIS_SCHEMA_ID = "feedbax.spec.descriptor.basis"
DESCRIPTOR_BASIS_SCHEMA_VERSION = "feedbax.spec.descriptor.basis.v1"
SELECTOR_ROLE_IDENTITY_SCHEMA_ID = "feedbax.spec.selector.role_identity"
SELECTOR_ROLE_IDENTITY_SCHEMA_VERSION = "feedbax.spec.selector.role_identity.v1"
COMPONENT_SELECTOR_SYNTAX_SCHEMA_ID = "feedbax.spec.selector.component_syntax"
COMPONENT_SELECTOR_SYNTAX_SCHEMA_VERSION = "feedbax.spec.selector.component_syntax.v1"
SELECTOR_FALLBACK_POLICY_SCHEMA_ID = "feedbax.spec.selector.fallback_policy"
SELECTOR_FALLBACK_POLICY_SCHEMA_VERSION = "feedbax.spec.selector.fallback_policy.v1"

DescriptorSourceKind = Literal["model", "graph", "training", "run", "analysis", "external"]
DescriptorTiming = Literal[
    "static",
    "per_trial",
    "per_timestep",
    "pre_step",
    "post_step",
    "post_run",
    "unknown",
]
DescriptorSign = Literal["native", "positive", "negative", "signed_magnitude", "unknown"]
DescriptorFallbackPolicy = Literal["forbid", "allow_declared"]


class DescriptorValidationError(ValueError):
    """Raised when descriptor records are semantically inconsistent."""


class DescriptorResolutionError(ValueError):
    """Raised when a descriptor selection cannot be resolved fail-closed."""


class DescriptorModel(BaseModel):
    """Base model for descriptor contract records."""

    model_config = ConfigDict(extra="forbid")


class NamespacedSelectorIdentity(DescriptorModel):
    """Shared shape for governed selector-vocabulary identities."""

    namespace: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(default="v1", min_length=1)
    compatible_with: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def identity(self) -> str:
        """Return the stable namespaced identity."""
        return f"{self.namespace}:{self.name}:{self.version}"

    @model_validator(mode="after")
    def _validate_segments(self) -> "NamespacedSelectorIdentity":
        for field_name, value in (
            ("namespace", self.namespace),
            ("name", self.name),
            ("version", self.version),
        ):
            if value.strip() != value or not value.strip():
                raise ValueError(f"{field_name} must be non-empty and trimmed")
        return self


class SelectorRoleIdentity(NamespacedSelectorIdentity):
    """First-class identity for a selector role such as model input or target."""

    schema_id: str = SELECTOR_ROLE_IDENTITY_SCHEMA_ID
    schema_version: str = SELECTOR_ROLE_IDENTITY_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "SelectorRoleIdentity":
        _require_schema(
            self.schema_id,
            self.schema_version,
            SELECTOR_ROLE_IDENTITY_SCHEMA_ID,
            SELECTOR_ROLE_IDENTITY_SCHEMA_VERSION,
            "SelectorRoleIdentity",
        )
        return self


class ComponentSelectorSyntax(NamespacedSelectorIdentity):
    """First-class identity for component selector syntax compatibility."""

    schema_id: str = COMPONENT_SELECTOR_SYNTAX_SCHEMA_ID
    schema_version: str = COMPONENT_SELECTOR_SYNTAX_SCHEMA_VERSION
    grammar: str = Field(
        default="descriptor://{basis_id}/{descriptor_id}",
        min_length=1,
    )

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "ComponentSelectorSyntax":
        _require_schema(
            self.schema_id,
            self.schema_version,
            COMPONENT_SELECTOR_SYNTAX_SCHEMA_ID,
            COMPONENT_SELECTOR_SYNTAX_SCHEMA_VERSION,
            "ComponentSelectorSyntax",
        )
        return self


class SelectorFallbackPolicyIdentity(NamespacedSelectorIdentity):
    """First-class identity for selector fallback behavior."""

    schema_id: str = SELECTOR_FALLBACK_POLICY_SCHEMA_ID
    schema_version: str = SELECTOR_FALLBACK_POLICY_SCHEMA_VERSION
    policy: DescriptorFallbackPolicy = "forbid"

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "SelectorFallbackPolicyIdentity":
        _require_schema(
            self.schema_id,
            self.schema_version,
            SELECTOR_FALLBACK_POLICY_SCHEMA_ID,
            SELECTOR_FALLBACK_POLICY_SCHEMA_VERSION,
            "SelectorFallbackPolicyIdentity",
        )
        return self


class ComponentSlice(DescriptorModel):
    """Half-open vector slice covered by one component descriptor."""

    start: int = Field(ge=0)
    stop: int = Field(gt=0)
    step: int = Field(default=1, gt=0)

    @property
    def width(self) -> int:
        """Return the number of vector positions addressed by this slice."""
        return len(range(self.start, self.stop, self.step))

    @model_validator(mode="after")
    def _validate_slice(self) -> "ComponentSlice":
        if self.stop <= self.start:
            raise ValueError("component slice stop must be greater than start")
        if self.width <= 0:
            raise ValueError("component slice must cover at least one position")
        return self


class VariableDescriptor(DescriptorModel):
    """Descriptor for a selectable model, run, or spec value."""

    schema_id: str = VARIABLE_DESCRIPTOR_SCHEMA_ID
    schema_version: str = VARIABLE_DESCRIPTOR_SCHEMA_VERSION
    descriptor_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str | None = None
    value_schema: ValueSchema
    source_kind: DescriptorSourceKind = "model"
    source_path: str = Field(min_length=1)
    timing: DescriptorTiming = "unknown"
    role: SelectorRoleIdentity | None = None
    selector_syntax: ComponentSelectorSyntax | None = None
    fallback_policy: SelectorFallbackPolicyIdentity | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    extensions: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_descriptor(self) -> "VariableDescriptor":
        _require_schema(
            self.schema_id,
            self.schema_version,
            VARIABLE_DESCRIPTOR_SCHEMA_ID,
            VARIABLE_DESCRIPTOR_SCHEMA_VERSION,
            "VariableDescriptor",
        )
        if not self.scope:
            raise ValueError("VariableDescriptor scope must not be empty")
        return self


class ComponentDescriptor(DescriptorModel):
    """Descriptor for a scalar or slice-level component of a variable."""

    schema_id: str = COMPONENT_DESCRIPTOR_SCHEMA_ID
    schema_version: str = COMPONENT_DESCRIPTOR_SCHEMA_VERSION
    descriptor_id: str = Field(min_length=1)
    variable_descriptor_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str | None = None
    slice: ComponentSlice
    value_schema: ValueSchema
    source_kind: DescriptorSourceKind = "model"
    source_path: str = Field(min_length=1)
    timing: DescriptorTiming = "unknown"
    sign: DescriptorSign = "native"
    transform: str = "identity"
    tags: list[str] = Field(default_factory=list)
    extensions: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_descriptor(self) -> "ComponentDescriptor":
        _require_schema(
            self.schema_id,
            self.schema_version,
            COMPONENT_DESCRIPTOR_SCHEMA_ID,
            COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
            "ComponentDescriptor",
        )
        if not self.transform.strip():
            raise ValueError("ComponentDescriptor transform must not be empty")
        return self


class DescriptorBasisComponent(DescriptorModel):
    """Basis identity projection for one ordered component descriptor."""

    component_descriptor_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    slice: ComponentSlice
    value_schema: ValueSchema
    sign: DescriptorSign = "native"
    transform: str = "identity"
    timing: DescriptorTiming = "unknown"
    source_path: str = Field(min_length=1)


class DescriptorBasisIdentity(DescriptorModel):
    """Whole-basis identity pinned by consumers and data products."""

    schema_id: str = DESCRIPTOR_BASIS_SCHEMA_ID
    schema_version: str = DESCRIPTOR_BASIS_SCHEMA_VERSION
    basis_id: str = Field(min_length=1)
    basis_schema_version: str = DESCRIPTOR_BASIS_SCHEMA_VERSION
    variable_descriptor_id: str = Field(min_length=1)
    value_schema: ValueSchema
    components: list[DescriptorBasisComponent] = Field(default_factory=list)
    timing: DescriptorTiming = "unknown"
    source_kind: DescriptorSourceKind = "model"
    source_path: str = Field(min_length=1)
    scope: dict[str, Any] = Field(default_factory=dict)
    descriptor_basis_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_basis(self) -> "DescriptorBasisIdentity":
        _require_schema(
            self.schema_id,
            self.schema_version,
            DESCRIPTOR_BASIS_SCHEMA_ID,
            DESCRIPTOR_BASIS_SCHEMA_VERSION,
            "DescriptorBasisIdentity",
        )
        if self.basis_schema_version != DESCRIPTOR_BASIS_SCHEMA_VERSION:
            raise ValueError(
                "unsupported DescriptorBasisIdentity basis_schema_version: "
                f"{self.basis_schema_version!r}, expected "
                f"{DESCRIPTOR_BASIS_SCHEMA_VERSION!r}"
            )
        if not self.scope:
            raise ValueError("DescriptorBasisIdentity scope must not be empty")
        _validate_unique_component_ids(self.components)
        _validate_basis_width(self.value_schema, self.components)
        computed = descriptor_basis_hash(self)
        if self.descriptor_basis_hash is not None and self.descriptor_basis_hash != computed:
            raise ValueError(
                "DescriptorBasisIdentity descriptor_basis_hash does not match basis identity: "
                f"descriptor_basis_hash={self.descriptor_basis_hash!r}, computed={computed!r}"
            )
        object.__setattr__(self, "descriptor_basis_hash", computed)
        return self


class ResolvedDescriptorView(DescriptorModel):
    """Resolved view for a descriptor selected under one verified basis."""

    descriptor_id: str
    descriptor_kind: Literal["variable", "component"]
    basis_id: str
    descriptor_basis_hash: str
    variable_descriptor: VariableDescriptor
    component_descriptor: ComponentDescriptor | None = None
    slice: ComponentSlice | None = None
    value_schema: ValueSchema
    scope: dict[str, Any]


def descriptor_basis_from_descriptors(
    *,
    basis_id: str,
    variable: VariableDescriptor,
    components: list[ComponentDescriptor],
    timing: DescriptorTiming | None = None,
    source_kind: DescriptorSourceKind | None = None,
    source_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DescriptorBasisIdentity:
    """Build a basis identity record from one variable and ordered components."""
    for component in components:
        if component.variable_descriptor_id != variable.descriptor_id:
            raise DescriptorValidationError(
                "component descriptor belongs to a different variable: "
                f"component={component.descriptor_id!r}, "
                f"component.variable_descriptor_id={component.variable_descriptor_id!r}, "
                f"variable={variable.descriptor_id!r}"
            )
    return DescriptorBasisIdentity(
        basis_id=basis_id,
        variable_descriptor_id=variable.descriptor_id,
        value_schema=variable.value_schema,
        components=[
            DescriptorBasisComponent(
                component_descriptor_id=component.descriptor_id,
                component_id=component.component_id,
                slice=component.slice,
                value_schema=component.value_schema,
                sign=component.sign,
                transform=component.transform,
                timing=component.timing,
                source_path=component.source_path,
            )
            for component in components
        ],
        timing=timing or variable.timing,
        source_kind=source_kind or variable.source_kind,
        source_path=source_path or variable.source_path,
        scope=dict(variable.scope),
        metadata=dict(metadata or {}),
    )


def descriptor_basis_identity_envelope(basis: DescriptorBasisIdentity) -> dict[str, Any]:
    """Return the canonical basis identity envelope used for hashing."""
    payload = basis.model_dump(mode="json", exclude_none=True)
    payload.pop("descriptor_basis_hash", None)
    payload.pop("metadata", None)
    return payload


def descriptor_basis_hash(basis: DescriptorBasisIdentity) -> str:
    """Hash a descriptor basis identity using ``canonical_json_v1``."""
    data = canonical_json_v1_bytes(descriptor_basis_identity_envelope(basis))
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def resolve_descriptor_view(
    *,
    descriptor_id: str,
    variable: VariableDescriptor,
    components: list[ComponentDescriptor],
    basis: DescriptorBasisIdentity,
    descriptor_basis_hash: str,
) -> ResolvedDescriptorView:
    """Resolve a descriptor selection under a verified basis identity.

    The caller supplies descriptor identity plus the pinned basis hash. Raw
    component indices are intentionally not accepted as selection inputs.
    """
    computed = descriptor_basis_hash_fn(basis)
    if descriptor_basis_hash != computed or basis.descriptor_basis_hash != computed:
        raise DescriptorResolutionError(
            "descriptor basis mismatch: "
            f"requested={descriptor_basis_hash!r}, basis={basis.descriptor_basis_hash!r}, "
            f"computed={computed!r}"
        )
    if basis.variable_descriptor_id != variable.descriptor_id:
        raise DescriptorResolutionError(
            "descriptor basis variable mismatch: "
            f"basis.variable_descriptor_id={basis.variable_descriptor_id!r}, "
            f"variable.descriptor_id={variable.descriptor_id!r}"
        )

    if descriptor_id == variable.descriptor_id:
        return ResolvedDescriptorView(
            descriptor_id=descriptor_id,
            descriptor_kind="variable",
            basis_id=basis.basis_id,
            descriptor_basis_hash=computed,
            variable_descriptor=variable,
            value_schema=variable.value_schema,
            scope=dict(basis.scope),
        )

    component_by_descriptor = {component.descriptor_id: component for component in components}
    component = component_by_descriptor.get(descriptor_id)
    if component is None:
        raise DescriptorResolutionError(f"unknown descriptor_id for basis: {descriptor_id!r}")

    basis_component = next(
        (
            item
            for item in basis.components
            if item.component_descriptor_id == component.descriptor_id
        ),
        None,
    )
    if basis_component is None:
        raise DescriptorResolutionError(
            "component descriptor is not present in the requested basis: "
            f"{component.descriptor_id!r}"
        )
    _validate_component_matches_basis(component, basis_component)
    return ResolvedDescriptorView(
        descriptor_id=descriptor_id,
        descriptor_kind="component",
        basis_id=basis.basis_id,
        descriptor_basis_hash=computed,
        variable_descriptor=variable,
        component_descriptor=component,
        slice=component.slice,
        value_schema=component.value_schema,
        scope=dict(basis.scope),
    )


def descriptor_basis_hash_fn(basis: DescriptorBasisIdentity) -> str:
    """Return the computed basis hash without trusting the stored field."""
    return descriptor_basis_hash(basis)


def _validate_component_matches_basis(
    component: ComponentDescriptor,
    basis_component: DescriptorBasisComponent,
) -> None:
    fields = (
        "component_id",
        "slice",
        "value_schema",
        "sign",
        "transform",
        "timing",
        "source_path",
    )
    for field_name in fields:
        component_value = getattr(component, field_name)
        basis_value = getattr(basis_component, field_name)
        if isinstance(component_value, BaseModel):
            component_value = component_value.model_dump(mode="json", exclude_none=True)
        if isinstance(basis_value, BaseModel):
            basis_value = basis_value.model_dump(mode="json", exclude_none=True)
        if component_value != basis_value:
            raise DescriptorResolutionError(
                "component descriptor does not match basis identity: "
                f"field={field_name!r}, descriptor={component_value!r}, basis={basis_value!r}"
            )


def _require_schema(
    schema_id: str,
    schema_version: str,
    expected_schema_id: str,
    expected_schema_version: str,
    family: str,
) -> None:
    if schema_id != expected_schema_id:
        raise ValueError(
            f"unsupported {family} schema_id: {schema_id!r}, expected {expected_schema_id!r}"
        )
    if schema_version != expected_schema_version:
        raise ValueError(
            "unsupported "
            f"{family} schema_version: {schema_version!r}, "
            f"expected {expected_schema_version!r}"
        )


def _validate_unique_component_ids(components: list[DescriptorBasisComponent]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for component in components:
        if component.component_id in seen:
            duplicates.append(component.component_id)
        seen.add(component.component_id)
    if duplicates:
        raise ValueError(f"duplicate component IDs in descriptor basis: {sorted(duplicates)!r}")


def _validate_basis_width(
    value_schema: ValueSchema,
    components: list[DescriptorBasisComponent],
) -> None:
    width = _declared_vector_width(value_schema)
    if width is None:
        return
    component_width = sum(component.slice.width for component in components)
    if component_width != width:
        raise ValueError(
            "descriptor basis component width mismatch: "
            f"value_schema_width={width}, component_width={component_width}"
        )


def _declared_vector_width(value_schema: ValueSchema) -> int | None:
    shape = value_schema.shape
    if not shape:
        return None
    width = shape[-1]
    return width if isinstance(width, int) else None
