"""Component declarations composed from layer-local facets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ValidationError

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.component import (
    ComponentIdentity,
    ComponentMigrationInfo,
    DynamicPortPolicy,
    DynamicPortPolicyError,
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
from feedbax.contracts.portable import (
    MissingPrototypeInput as MissingPrototypeInput,
    InputPrototypeParameter,
)
from feedbax.contracts.representation import RepresentationSpec

if TYPE_CHECKING:
    from feedbax.contracts.graph import ParamValue
    from feedbax.runtime.graph import Component
    from feedbax.contracts.portable import ComponentOwner


ComponentBuilder = Callable[[Mapping[str, Any]], "Component"]
OutputPrototypeFn = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
ComponentParamModel = type[BaseModel]


class ComponentParamValidationError(DynamicPortPolicyError):
    """A component parameter payload failed its declared closed contract."""

    def __init__(
        self,
        *,
        component_type: str,
        revision: int,
        failures: tuple["ComponentParamFailure", ...],
    ) -> None:
        self.component_type = component_type
        self.revision = revision
        self.failures = failures
        details = "; ".join(
            f"{'.'.join(str(part) for part in failure.path) or '<root>'}: "
            f"{failure.message} ({failure.code})"
            for failure in failures
        )
        canonical_hint = (
            " Only canonical Channel noise parameters are accepted."
            if component_type == "feedbax.Channel"
            else ""
        )
        super().__init__(
            f"Invalid parameters for component {component_type!r} revision "
            f"{revision!r}: {details}.{canonical_hint}"
        )


@dataclass(frozen=True)
class ComponentParamFailure:
    path: tuple[str | int, ...]
    code: str
    message: str


@dataclass(frozen=True)
class ComponentParamValidation:
    value: BaseModel | None = None
    failures: tuple[ComponentParamFailure, ...] = ()

    @property
    def ok(self) -> bool:
        return self.value is not None


def _param_schema_from_model(model: ComponentParamModel) -> tuple[ParamSchema, ...]:
    schemas: list[ParamSchema] = []
    for name, model_field in model.model_fields.items():
        extra = (model_field.json_schema_extra or {}).get("feedbax", {})
        annotation = model_field.annotation
        origin = get_origin(annotation)
        args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
        base = args[0] if origin in (Union, UnionType) and len(args) == 1 else annotation
        if extra.get("type"):
            field_type = extra["type"]
        elif get_origin(base) is Literal:
            field_type = "enum"
        elif base is bool:
            field_type = "bool"
        elif base is int:
            field_type = "int"
        elif base is float:
            field_type = "float"
        elif base is str:
            field_type = "str"
        elif get_origin(base) in (list, tuple):
            field_type = "array"
        elif get_origin(base) in (dict, Mapping):
            field_type = "object"
        else:
            field_type = "object"
        default = None if model_field.is_required() else model_field.default
        schemas.append(
            ParamSchema(
                name=name,
                type=field_type,
                default=default,
                required=bool(extra.get("required", model_field.is_required())),
                description=model_field.description or "",
                min=next(
                    (item.ge for item in model_field.metadata if hasattr(item, "ge")),
                    None,
                ),
                max=next(
                    (item.le for item in model_field.metadata if hasattr(item, "le")),
                    None,
                ),
                step=extra.get("step"),
                options=extra.get("options")
                or (list(get_args(base)) if get_origin(base) is Literal else None),
                option_descriptions=extra.get("option_descriptions"),
                nested_schema=extra.get("nested_schema"),
            )
        )
    return tuple(schemas)


@dataclass(frozen=True)
class ComponentParamContract:
    """Closed authority for one component's build and serialization parameters."""

    component_type: str
    model: ComponentParamModel
    revision: int
    runtime_type: type["ComponentOwner"]
    build_context_fields: frozenset[str] = frozenset()

    @property
    def param_schema(self) -> tuple[ParamSchema, ...]:
        return _param_schema_from_model(self.model)

    @property
    def default_params(self) -> dict[str, Any]:
        values = {
            name: model_field.get_default(call_default_factory=True)
            for name, model_field in self.model.model_fields.items()
            if not model_field.is_required()
        }
        return self.model.model_construct(**values).model_dump(
            mode="json",
            exclude_unset=True,
        )

    @property
    def classification(self) -> str:
        return "generic"

    def validate(self, params: Mapping[str, Any]) -> ComponentParamValidation:
        try:
            from feedbax.contracts.portable import validate_authoring

            value = validate_authoring(self.model, params)
            return ComponentParamValidation(value=value)
        except ValidationError as exc:
            return ComponentParamValidation(
                failures=tuple(
                    ComponentParamFailure(
                        tuple(item["loc"]),
                        str(item["type"]),
                        str(item["msg"]),
                    )
                    for item in exc.errors()
                )
            )

    def require(self, params: Mapping[str, Any]) -> BaseModel:
        result = self.validate(params)
        if result.value is None:
            raise ComponentParamValidationError(
                component_type=self.component_type,
                revision=self.revision,
                failures=result.failures,
            )
        return result.value

    def require_effective(self, params: Mapping[str, Any]) -> BaseModel:
        from feedbax.contracts.portable import validate_effective

        return validate_effective(self.model, params)

    def build(self, params: Mapping[str, Any]) -> "Component":
        context = {
            name: value for name, value in params.items() if name in self.build_context_fields
        }
        authored = {name: value for name, value in params.items() if name not in context}
        validated = self.require(authored)
        return self.runtime_type.from_config(validated, **context)

    def build_resolved(
        self, params: Mapping[str, Any], *, config: BaseModel | None = None
    ) -> "Component":
        """Construct canonical compiler parameters without reapplying their schema."""
        values = dict(params)
        context = {
            name: values.pop(name) for name in tuple(values) if name in self.build_context_fields
        }
        return self.runtime_type.from_config(
            config if config is not None else self.require_effective(values), **context
        )

    def extract(self, component: "ComponentOwner") -> BaseModel:
        return component.configuration()

    def dump(self, component: "Component") -> dict[str, Any]:
        return self.extract(component).model_dump(mode="json", warnings="error")


@dataclass(frozen=True)
class ComponentDeclaration:
    """Identity and capabilities owned by one component declaration."""

    type_id: str
    revision: int
    owner: str

    def __post_init__(self) -> None:
        identities = (self.type_id, self.owner)
        if any(not value.strip() for value in identities):
            raise ValueError("component declaration identities and owner must be non-empty")


@dataclass(frozen=True)
class ComponentCompilerFacet:
    parameter_contract: ComponentParamContract
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    port_types: PortTypeSpec | None = None
    dynamic_port_policy: DynamicPortPolicy | None = None
    domain: str = CAUSAL_DOMAIN_ID
    interior_domain: str | None = None
    is_composite: bool = False
    output_prototype_fn: OutputPrototypeFn | None = None
    prototype_depends_on_connected_inputs: bool = False
    input_prototype_parameter: InputPrototypeParameter | None = None
    # Bind runtime-only constructor parameters to resolved output ports.
    build_output_prototypes: Mapping[str, str] = field(default_factory=dict)

    @property
    def param_schema(self) -> tuple[ParamSchema, ...]:
        return self.parameter_contract.param_schema


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
    source: Path | None = field(default=None, compare=False)

    def __post_init__(self):
        import inspect

        frame = inspect.currentframe().f_back.f_back
        filename = frame.f_code.co_filename
        if self.source is None and not filename.startswith("<"):
            object.__setattr__(self, "source", Path(filename).resolve())


@dataclass(frozen=True)
class ComponentSerializationFacet:
    revision: int
    migrations: tuple[ComponentMigrationInfo, ...] = ()


@dataclass(frozen=True)
class ComponentTrainingFacet:
    trainable_by_default: bool


@dataclass(frozen=True)
class DeclaredComponent:
    """Derived component projection used after application composition."""

    declaration: ComponentDeclaration
    params: ComponentParamContract
    compiler: ComponentCompilerFacet
    serialization: ComponentSerializationFacet
    runtime: ComponentRuntimeFacet | None = None
    authoring: ComponentAuthoringFacet | None = None
    studio: ComponentStudioFacet | None = None
    training: ComponentTrainingFacet | None = None
    provenance: str | None = None
    identity: ComponentIdentity | None = None
    source_paths: tuple[Path | None, ...] = ()
    owner_class: type | None = None
    alias_of: str | None = None

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
        return list(self.params.param_schema)

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
        return self.params.build

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
    def revision(self) -> int:
        return self.serialization.revision

    @property
    def supported_revisions(self) -> list[int]:
        return list(range(1, self.revision + 1))

    @property
    def migrations(self) -> list[ComponentMigrationInfo]:
        return list(self.serialization.migrations)

    @property
    def trainable_by_default(self) -> bool:
        return self.training.trainable_by_default if self.training is not None else False

    @property
    def default_params(self) -> dict[str, ParamValue]:
        return self.params.default_params


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
