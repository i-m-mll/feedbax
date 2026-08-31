"""Component declarations composed from layer-local facets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.component import (
    ComponentIdentity,
    ComponentMigrationInfo,
    DynamicPortPolicy,
    PortTypeSpec,
)
from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
from feedbax.contracts.graph import (
    CanvasPositionSpec,
    GraphSpec,
    GraphUIState,
    NodeUIState,
    ParamSchema,
)
from feedbax.contracts.representation import RepresentationSpec

if TYPE_CHECKING:
    from feedbax.contracts.graph import ParamValue
    from feedbax.runtime.graph import Component


ComponentBuilder = Callable[[Mapping[str, Any]], "Component"]
OutputPrototypeFn = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


class MissingPrototypeInput(ValueError):
    """An output prototype needs an input prototype not yet known."""


@dataclass(frozen=True)
class ComponentDeclaration:
    """Identity and capabilities owned by one component declaration."""

    type_id: str
    schema_id: str
    schema_version: str
    capabilities: frozenset[str]
    owner: str

    def __post_init__(self) -> None:
        identities = (self.type_id, self.schema_id, self.schema_version, self.owner)
        if any(not value.strip() for value in identities):
            raise ValueError("component declaration identities and owner must be non-empty")


@dataclass(frozen=True)
class ComponentCompilerFacet:
    param_schema: tuple[ParamSchema, ...]
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    port_types: PortTypeSpec | None = None
    dynamic_port_policy: DynamicPortPolicy | None = None
    domain: str = CAUSAL_DOMAIN_ID
    interior_domain: str | None = None
    is_composite: bool = False
    output_prototype_fn: OutputPrototypeFn | None = None


@dataclass(frozen=True)
class ComponentRuntimeFacet:
    builder: ComponentBuilder


@dataclass(frozen=True)
class ComponentAuthoringFacet:
    template_graph: GraphSpec | AcausalGraphSpec | None = None
    template_id: str | None = None
    template_kind: str | None = None


@dataclass(frozen=True)
class ComponentStudioFacet:
    category: str
    description: str
    icon: str = "box"
    template_ui_state: GraphUIState | None = None
    representation: RepresentationSpec | None = None


@dataclass(frozen=True)
class ComponentSerializationFacet:
    param_schema_version: str
    supported_param_schema_versions: tuple[str, ...]
    migrations: tuple[ComponentMigrationInfo, ...] = ()


@dataclass(frozen=True)
class ComponentTrainingFacet:
    trainable_by_default: bool


@dataclass(frozen=True)
class DeclaredComponent:
    """Derived component projection used after application composition."""

    declaration: ComponentDeclaration
    compiler: ComponentCompilerFacet
    serialization: ComponentSerializationFacet
    runtime: ComponentRuntimeFacet | None = None
    authoring: ComponentAuthoringFacet | None = None
    studio: ComponentStudioFacet | None = None
    training: ComponentTrainingFacet | None = None
    provenance: str | None = None
    identity: ComponentIdentity | None = None

    @property
    def name(self) -> str:
        return self.declaration.type_id

    @property
    def owner(self) -> str:
        return self.declaration.owner

    @property
    def category(self) -> str:
        return self.studio.category if self.studio is not None else ""

    @property
    def description(self) -> str:
        return self.studio.description if self.studio is not None else ""

    @property
    def icon(self) -> str:
        return self.studio.icon if self.studio is not None else "box"

    @property
    def param_schema(self) -> list[ParamSchema]:
        return list(self.compiler.param_schema)

    @property
    def input_ports(self) -> list[str]:
        return list(self.compiler.input_ports)

    @property
    def output_ports(self) -> list[str]:
        return list(self.compiler.output_ports)

    @property
    def port_types(self) -> PortTypeSpec | None:
        return self.compiler.port_types

    @property
    def dynamic_port_policy(self) -> DynamicPortPolicy | None:
        return self.compiler.dynamic_port_policy

    @property
    def domain(self) -> str:
        return self.compiler.domain

    @property
    def interior_domain(self) -> str | None:
        return self.compiler.interior_domain

    @property
    def is_composite(self) -> bool:
        return self.compiler.is_composite

    @property
    def output_prototype_fn(self) -> OutputPrototypeFn | None:
        return self.compiler.output_prototype_fn

    @property
    def builder(self) -> ComponentBuilder | None:
        return self.runtime.builder if self.runtime is not None else None

    @property
    def template_graph(self) -> GraphSpec | AcausalGraphSpec | None:
        return self.authoring.template_graph if self.authoring is not None else None

    @property
    def template_id(self) -> str | None:
        return self.authoring.template_id if self.authoring is not None else None

    @property
    def template_kind(self) -> str | None:
        return self.authoring.template_kind if self.authoring is not None else None

    @property
    def template_ui_state(self) -> GraphUIState | None:
        return self.studio.template_ui_state if self.studio is not None else None

    @property
    def representation(self) -> RepresentationSpec | None:
        return self.studio.representation if self.studio is not None else None

    @property
    def param_schema_version(self) -> str:
        return self.serialization.param_schema_version

    @property
    def supported_param_schema_versions(self) -> list[str]:
        return list(self.serialization.supported_param_schema_versions)

    @property
    def migrations(self) -> list[ComponentMigrationInfo]:
        return list(self.serialization.migrations)

    @property
    def trainable_by_default(self) -> bool:
        return self.training.trainable_by_default if self.training is not None else False

    @property
    def default_params(self) -> dict[str, ParamValue]:
        return {schema.name: schema.default for schema in self.compiler.param_schema}


def declare_component(
    *,
    name: str,
    param_schema: Sequence[ParamSchema],
    input_ports: Sequence[str],
    output_ports: Sequence[str],
    category: str = "",
    description: str = "",
    icon: str = "box",
    port_types: PortTypeSpec | None = None,
    dynamic_port_policy: DynamicPortPolicy | None = None,
    domain: str = CAUSAL_DOMAIN_ID,
    interior_domain: str | None = None,
    is_composite: bool = False,
    template_graph: GraphSpec | AcausalGraphSpec | None = None,
    template_ui_state: GraphUIState | None = None,
    template_id: str | None = None,
    template_kind: str | None = None,
    builder: ComponentBuilder | None = None,
    output_prototype_fn: OutputPrototypeFn | None = None,
    provenance: str | None = None,
    identity: ComponentIdentity | None = None,
    owner: str | None = None,
    param_schema_version: str = "1",
    supported_param_schema_versions: Sequence[str] = (),
    migrations: Sequence[ComponentMigrationInfo] = (),
    trainable_by_default: bool = False,
    representation: RepresentationSpec | None = None,
) -> DeclaredComponent:
    """Declare only the component facets supplied by its owning layers."""
    if is_composite and interior_domain is None:
        interior_domain = CAUSAL_DOMAIN_ID
    supported = tuple(dict.fromkeys((*supported_param_schema_versions, param_schema_version)))
    capabilities = {"compile", "serialize"}
    if builder is not None:
        capabilities.add("runtime")
    if template_graph is not None:
        capabilities.add("authoring")
    if category or description or representation is not None or template_ui_state is not None:
        capabilities.add("studio")
    if trainable_by_default:
        capabilities.add("training")
    declaration = ComponentDeclaration(
        type_id=name,
        schema_id=f"{owner or provenance or 'local'}.component.{name}",
        schema_version=param_schema_version,
        capabilities=frozenset(capabilities),
        owner=owner or provenance or "local",
    )
    return DeclaredComponent(
        declaration=declaration,
        compiler=ComponentCompilerFacet(
            param_schema=tuple(param_schema),
            input_ports=tuple(input_ports),
            output_ports=tuple(output_ports),
            port_types=port_types,
            dynamic_port_policy=dynamic_port_policy,
            domain=domain,
            interior_domain=interior_domain,
            is_composite=is_composite,
            output_prototype_fn=output_prototype_fn,
        ),
        serialization=ComponentSerializationFacet(
            param_schema_version=param_schema_version,
            supported_param_schema_versions=supported,
            migrations=tuple(migrations),
        ),
        runtime=ComponentRuntimeFacet(builder) if builder is not None else None,
        authoring=(
            ComponentAuthoringFacet(template_graph, template_id, template_kind)
            if template_graph is not None or template_id is not None or template_kind is not None
            else None
        ),
        studio=(
            ComponentStudioFacet(category, description, icon, template_ui_state, representation)
            if "studio" in capabilities
            else None
        ),
        training=(ComponentTrainingFacet(True) if trainable_by_default else None),
        provenance=provenance,
        identity=identity,
    )


def _template_ui_state(
    positions: Mapping[str, tuple[float, float]],
    *,
    subgraph_states: Mapping[str, GraphUIState] | None = None,
) -> GraphUIState:
    return GraphUIState(
        node_states={
            node_id: NodeUIState(position=CanvasPositionSpec(x=x, y=y))
            for node_id, (x, y) in positions.items()
        },
        subgraph_states=dict(subgraph_states or {}),
    )
