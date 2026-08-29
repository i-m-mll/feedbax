"""Public scientific declarations and resolved runtime protocols."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from feedbax.training.environment import ObjectiveProtocol, TrialSourceProtocol

from .core import Declaration, Facet


PayloadT = TypeVar("PayloadT", bound=BaseModel)


@dataclass(frozen=True)
class ResolvedTrialSource:
    declaration: Declaration
    source: TrialSourceProtocol

    def __post_init__(self) -> None:
        if self.declaration.kind != "trial_source":
            raise TypeError("resolved trial source requires kind='trial_source'")
        if not isinstance(self.source, TrialSourceProtocol):
            raise TypeError("trial-source runtime facet does not satisfy TrialSourceProtocol")


@dataclass(frozen=True)
class ResolvedObjective:
    declaration: Declaration
    objective: ObjectiveProtocol

    def __post_init__(self) -> None:
        if self.declaration.kind != "objective":
            raise TypeError("resolved objective requires kind='objective'")
        if not isinstance(self.objective, ObjectiveProtocol):
            raise TypeError("objective runtime facet does not satisfy ObjectiveProtocol")


@runtime_checkable
class OperationProtocol(Protocol):
    """Resolved typed operation independent of provider realization."""

    def execute(self, **inputs: object) -> object: ...


@runtime_checkable
class BackendProtocol(Protocol):
    """Resolved capability backend independent of scientific semantics."""

    def realize(self, capability: str, request: object) -> object: ...


@dataclass(frozen=True)
class ResolvedOperation:
    declaration: Declaration
    operation: OperationProtocol

    def __post_init__(self) -> None:
        if self.declaration.kind != "operation":
            raise TypeError("resolved operation requires kind='operation'")
        if not isinstance(self.operation, OperationProtocol):
            raise TypeError("operation runtime facet does not satisfy OperationProtocol")


@dataclass(frozen=True)
class ResolvedBackend:
    declaration: Declaration
    backend: BackendProtocol

    def __post_init__(self) -> None:
        if self.declaration.kind != "backend":
            raise TypeError("resolved backend requires kind='backend'")
        if not isinstance(self.backend, BackendProtocol):
            raise TypeError("backend runtime facet does not satisfy BackendProtocol")


@dataclass(frozen=True)
class RuntimeFacet:
    implementation: object


@dataclass(frozen=True)
class CompilerFacet:
    lower: Callable[..., object]


@dataclass(frozen=True)
class AuthoringFacet:
    defaults: Callable[[], object]
    operations: Mapping[str, Callable[..., object]]


@dataclass(frozen=True)
class StudioFacet:
    label: str
    palette_group: str
    editor_schema: type[BaseModel] | None = None


@dataclass(frozen=True)
class SerializationFacet:
    encode: Callable[[object], Mapping[str, Any]]
    rejected_versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationFacet:
    input_types: Mapping[str, str]
    output_types: Mapping[str, str]
    execute: Callable[..., object]


@dataclass(frozen=True)
class BackendFacet:
    realize: Callable[..., object]


def scientific_declaration(
    *,
    kind: str,
    type_id: str,
    schema_id: str,
    schema_version: str,
    capabilities: Sequence[str],
    runtime_protocol: type[Any],
    owner: str,
) -> Declaration:
    if kind not in {"trial_source", "objective", "training_program", "operation", "backend"}:
        raise ValueError(f"unsupported scientific declaration kind {kind!r}")
    return Declaration(
        kind=kind,
        type_id=type_id,
        schema_id=schema_id,
        schema_version=schema_version,
        capabilities=frozenset(capabilities),
        runtime_protocol=runtime_protocol,
        owner=owner,
    )


def facet(declaration: Declaration, layer: str, value: object) -> Facet:
    return Facet(
        kind=declaration.kind,
        type_id=declaration.type_id,
        layer=layer,
        schema_version=declaration.schema_version,
        value=value,
    )
