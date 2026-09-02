"""Component declarations composed from layer-local facets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import UnionType
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

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
ComponentParamModel = type[BaseModel]
ComponentParamExtractor = Callable[["Component", BaseModel], BaseModel]


@dataclass(frozen=True)
class ComponentConstructorPlan:
    """Declarative projection from validated fields to constructor arguments."""

    renames: Mapping[str, str] = field(default_factory=dict)
    omit: frozenset[str] = frozenset()
    injections: Mapping[str, Any] = field(default_factory=dict)
    groups: Mapping[str, tuple[type[Any], tuple[str, ...]]] = field(default_factory=dict)
    context_adapters: Mapping[str, tuple[str, Literal["array_prototype"]]] = field(
        default_factory=dict
    )


def _unsupported_declared_builder(component_type: str) -> ComponentBuilder:
    message = (
        f"Component type {component_type!r} has no runtime builder. "
        "It is a display-only abstraction used in composite subgraph templates."
    )

    def _builder(params: Mapping[str, Any]) -> "Component":
        del params
        raise NotImplementedError(message)

    _builder._feedbax_unsupported_builder = True  # type: ignore[attr-defined]
    _builder._feedbax_unsupported_builder_message = message  # type: ignore[attr-defined]
    return _builder


class MissingPrototypeInput(ValueError):
    """An output prototype needs an input prototype not yet known."""


class ComponentParamValidationError(ValueError):
    """A component parameter payload failed its declared closed contract."""

    def __init__(
        self,
        *,
        component_type: str,
        schema_version: str,
        failures: tuple["ComponentParamFailure", ...],
    ) -> None:
        self.component_type = component_type
        self.schema_version = schema_version
        self.failures = failures
        details = "; ".join(
            f"{'.'.join(str(part) for part in failure.path) or '<root>'}: "
            f"{failure.message} ({failure.code})"
            for failure in failures
        )
        canonical_hint = (
            " Only canonical Channel noise parameters are accepted."
            if component_type == "Channel"
            else ""
        )
        super().__init__(
            f"Invalid parameters for component {component_type!r} schema "
            f"{schema_version!r}: {details}.{canonical_hint}"
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


class ParameterField(ParamSchema):
    """One authoritative parameter field declaration.

    Feedbax built-ins declare these fields once. The declaration is compiled to
    both the strict runtime model and the public Studio ``ParamSchema`` view.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


def _field_annotation(parameter: ParamSchema) -> Any:
    annotation: Any
    if parameter.type == "bool":
        annotation = bool
    elif parameter.type == "int":
        annotation = int
    elif parameter.type == "float":
        annotation = float
    elif parameter.type == "str":
        annotation = str
    elif parameter.type == "enum" and parameter.options:
        annotation = Literal.__getitem__(tuple(parameter.options))
    elif parameter.type == "bounds2d":
        annotation = list[list[float]]
    elif parameter.type == "array":
        annotation = list[Any] if isinstance(parameter.default, list) else Any
    elif parameter.type == "object":
        # ``object`` is the GraphSpec spelling for structured JSON.  Some
        # declarations accept more than one JSON shape (for example sparse
        # object or dense array), so their nested schema remains the authority.
        annotation = Any
    else:
        annotation = Any
    if parameter.default is None and get_origin(annotation) not in (Union, UnionType):
        annotation = annotation | None
    return annotation


def _model_from_param_schema(
    component_type: str,
    param_schema: Sequence[ParamSchema],
) -> ComponentParamModel:
    fields: dict[str, tuple[Any, Any]] = {}
    for parameter in param_schema:
        default = parameter.default
        field_info = Field(
            default=default,
            ge=parameter.min,
            le=parameter.max,
            description=parameter.description or None,
            json_schema_extra={
                "feedbax": {
                    "type": parameter.type,
                    "required": parameter.required,
                    "step": parameter.step,
                    "options": parameter.options,
                    "option_descriptions": parameter.option_descriptions,
                    "nested_schema": [
                        item.model_dump(mode="json") for item in parameter.nested_schema
                    ]
                    if parameter.nested_schema
                    else None,
                }
            },
        )
        fields[parameter.name] = (_field_annotation(parameter), field_info)
    return create_model(
        f"{component_type.replace('.', '_')}Params",
        __config__=ConfigDict(extra="forbid", strict=True),
        **fields,
    )


def _strict_param_model(component_type: str, model: ComponentParamModel) -> ComponentParamModel:
    """Compile a caller model into the registry's closed validation boundary."""

    config = dict(model.model_config)
    if config.get("extra") == "forbid" and config.get("strict") is True:
        return model
    config.update(extra="forbid", strict=True)
    return type(
        f"{component_type.replace('.', '_')}StrictParams",
        (model,),
        {"model_config": ConfigDict(**config)},
    )


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


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Component parameter value is not canonical JSON: {type(value).__name__}")


def _read_attribute_path(component: "Component", path: str) -> Any:
    value: Any = component
    for part in path.split("."):
        if not hasattr(value, part):
            raise AttributeError(
                f"component {type(component).__name__!r} has no declared parameter path {path!r}"
            )
        value = getattr(value, part)
    return value


@dataclass(frozen=True)
class ComponentParamContract:
    """Closed authority for one component's build and serialization parameters."""

    component_type: str
    model: ComponentParamModel
    schema_version: str
    required_fields: frozenset[str] = frozenset()
    build_context_fields: frozenset[str] = frozenset()
    builder: ComponentBuilder | None = None
    runtime_type: type["Component"] | None = None
    constructor: ComponentConstructorPlan = field(default_factory=ComponentConstructorPlan)
    attribute_paths: Mapping[str, str] = field(default_factory=dict)
    extractor: ComponentParamExtractor | None = None
    override_reason: str | None = None

    @property
    def param_schema(self) -> tuple[ParamSchema, ...]:
        return _param_schema_from_model(self.model)

    @property
    def default_params(self) -> dict[str, Any]:
        values = {
            name: model_field.default
            for name, model_field in self.model.model_fields.items()
            if not model_field.is_required()
        }
        return self.model.model_construct(**values).model_dump(
            mode="json",
            exclude_unset=True,
        )

    @property
    def classification(self) -> str:
        if self.runtime_type is None and self.builder is None:
            return "unsupported"
        return "custom" if self.builder is not None or self.extractor is not None else "generic"

    def validate(self, params: Mapping[str, Any]) -> ComponentParamValidation:
        missing = self.required_fields - set(params)
        if missing:
            return ComponentParamValidation(
                failures=tuple(
                    ComponentParamFailure((name,), "missing", "Field required")
                    for name in sorted(missing)
                )
            )
        try:
            return ComponentParamValidation(value=self.model.model_validate(dict(params)))
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
                schema_version=self.schema_version,
                failures=result.failures,
            )
        return result.value

    def build(self, params: Mapping[str, Any]) -> "Component":
        context = {
            name: value for name, value in params.items() if name in self.build_context_fields
        }
        authored = {name: value for name, value in params.items() if name not in context}
        validated = self.require(authored)
        values = validated.model_dump(mode="python")
        values.update(context)
        if self.builder is not None:
            return self.builder(values)
        if self.runtime_type is None:
            raise NotImplementedError(
                f"Component type {self.component_type!r} is registered for metadata "
                "but has no executable builder"
            )
        constructor_values = {
            self.constructor.renames.get(name, name): value
            for name, value in values.items()
            if name not in self.constructor.omit and name not in self.constructor.context_adapters
        }
        for source, (target, adapter) in self.constructor.context_adapters.items():
            if adapter == "array_prototype":
                from feedbax.contracts.graphs.prototypes import array_proto_from_shape

                constructor_values[target] = array_proto_from_shape(values.get(source))
        for target, (group_type, names) in self.constructor.groups.items():
            constructor_values[target] = group_type(
                **{
                    self.constructor.renames.get(name, name): constructor_values.pop(
                        self.constructor.renames.get(name, name)
                    )
                    for name in names
                }
            )
        constructor_values.update(self.constructor.injections)
        return self.runtime_type(**constructor_values)

    def extract(self, component: "Component") -> BaseModel:
        empty = self.model.model_construct()
        if self.extractor is not None:
            return self.extractor(component, empty)
        raw = {}
        for name, model_field in self.model.model_fields.items():
            path = self.attribute_paths.get(name, name)
            value = (
                model_field.default if path == "$default" else _read_attribute_path(component, path)
            )
            raw[name] = _json_value(value)
        return self.require(raw)

    def dump(self, component: "Component") -> dict[str, Any]:
        return self.extract(component).model_dump(mode="json")


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
    parameter_contract: ComponentParamContract
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    port_types: PortTypeSpec | None = None
    dynamic_port_policy: DynamicPortPolicy | None = None
    domain: str = CAUSAL_DOMAIN_ID
    interior_domain: str | None = None
    is_composite: bool = False
    output_prototype_fn: OutputPrototypeFn | None = None

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
    params: ComponentParamContract
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
        if self.params.classification == "unsupported":
            return _unsupported_declared_builder(self.name)
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
        return self.params.default_params


def declare_component(
    *,
    name: str,
    param_schema: Sequence[ParamSchema] | None = None,
    parameter_fields: Sequence[ParameterField] | None = None,
    param_model: ComponentParamModel | None = None,
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
    runtime_type: type["Component"] | None = None,
    constructor: ComponentConstructorPlan | None = None,
    attribute_paths: Mapping[str, str] | None = None,
    build_context_fields: Sequence[str] = (),
    extractor: ComponentParamExtractor | None = None,
    override_reason: str | None = None,
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
    schema_sources = sum(
        source is not None for source in (param_schema, parameter_fields, param_model)
    )
    if schema_sources > 1:
        raise ValueError(f"Component {name!r} must declare exactly one parameter schema source")
    source_schema = tuple(param_schema or parameter_fields or ())
    model = (
        _strict_param_model(name, param_model)
        if param_model is not None
        else _model_from_param_schema(name, source_schema)
    )
    public_schema = _param_schema_from_model(model)
    required_fields = frozenset(schema.name for schema in public_schema if schema.required)
    if extractor is not None and override_reason is None:
        raise ValueError(f"Component {name!r} extractor override requires a reason")
    if override_reason is not None and builder is None and extractor is None:
        raise ValueError(f"Component {name!r} override reason has no override")
    if runtime_type is None and builder is not None:
        from feedbax.runtime.graph import Component

        inferred = get_type_hints(builder).get("return")
        if isinstance(inferred, type) and issubclass(inferred, Component):
            runtime_type = inferred
    param_contract = ComponentParamContract(
        component_type=name,
        model=model,
        schema_version=param_schema_version,
        required_fields=required_fields,
        build_context_fields=frozenset(build_context_fields),
        builder=builder,
        runtime_type=runtime_type,
        constructor=constructor or ComponentConstructorPlan(),
        attribute_paths=dict(attribute_paths or {}),
        extractor=extractor,
        override_reason=override_reason,
    )
    if is_composite and interior_domain is None:
        interior_domain = CAUSAL_DOMAIN_ID
    supported = tuple(dict.fromkeys((*supported_param_schema_versions, param_schema_version)))
    capabilities = {"compile", "serialize"}
    if builder is not None:
        capabilities.add("runtime")
    if runtime_type is not None:
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
        params=param_contract,
        compiler=ComponentCompilerFacet(
            parameter_contract=param_contract,
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
        runtime=(
            ComponentRuntimeFacet(param_contract.build)
            if builder is not None or runtime_type is not None
            else None
        ),
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
