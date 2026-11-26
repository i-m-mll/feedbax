"""The authored Model v2 payload and its ordered projection to runtime Models."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
import json
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator
from pydantic_core import to_jsonable_python

from feedbax.contracts.acausal import AcausalPhysicalDomain
from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_storage import StoredExpression, StoredTime, store_expression
from feedbax.contracts.expression_text import print_stored_expression
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import TIME, CallValue, Expression, LiteralValue, UnresolvedValue
from feedbax.contracts.model_address import FieldElement, ModelAddress, ModelElementId, SignalSource, SignalTarget
from feedbax.contracts.model_document import (
    ComponentPortCount, ComponentRevision, CompositeComponent, DelayAttachment, DelayInitialTask,
    DelayInitialValue, DelayInitialWire, GraphInterior,
    HoldAttachment, InputDefault, InterventionAttachment, LeafComponent, ModelComponent,
    ModelDelay, ModelDocument, ModelName, ModelPhysicalConnection, ModelPortAttachments, ModelPorts,
    ModelSignalSource, ModelStepSeconds, ModelSubsteps, ModelTime, ModelWire, ModelWorld, PhysicalInterior,
    PhysicalPort, PhysicalPortType, ProbeAttachment, SignalInput, SignalOutput, SignalType,
    WireAttachment,
)
from feedbax.contracts.stored_members import StoredMemberKey, StoredMembers, keyed_members, project_member
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.time_units import TimeValue, literal_time
from feedbax.contracts.window import StoredWindow, WindowOffset, lower_window_offset, stored_window_offset

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry


STORED_MODEL_DOCUMENT_SCHEMA_ID = "feedbax.spec.model.v2"
_STORED_EXPRESSION = TypeAdapter(StoredExpression)
_RUNTIME_EXPRESSION = TypeAdapter(Expression)
_STORED_TIME = TypeAdapter(StoredTime)
_RUNTIME_TIME = TypeAdapter(TimeValue | UnresolvedValue)
_RUNTIME_OFFSET = TypeAdapter(WindowOffset)


def model_keyed_collection(path: tuple[str, ...]) -> bool:
    from feedbax.contracts.project_storage import stored_keyed_collection

    return stored_keyed_collection(STORED_MODEL_DOCUMENT_SCHEMA_ID, path)


def authored_model_payload(model: ModelDocument, components: ComponentRegistry | None = None) -> dict[str, JsonValue]:
    """Project a typed Model into a complete keyed payload for authored comparisons."""
    return StoredModelDocument.from_lowered(model, components).model_dump(mode="json")


def model_member_order(
    value: JsonValue, path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], list[str]]]:
    """List the member orders that canonical JSON's key sorting would erase."""
    if isinstance(value, dict):
        if model_keyed_collection(path):
            yield path, list(value)
        for key, child in value.items():
            yield from model_member_order(child, (*path, key))


def _held_expression(expression: Expression) -> UnresolvedValue:
    text = print_stored_expression(expression)
    return UnresolvedValue(text=text if isinstance(text, str) else json.dumps(text))


def _lower_period(time: TimeValue | UnresolvedValue) -> ModelStepSeconds:
    if isinstance(time, UnresolvedValue):
        return time
    if time.unit == "s" and time.value > 0:
        return time.value
    return _held_expression(LiteralValue(value=time.value, unit=time.unit))


def _store_period(seconds: ModelStepSeconds) -> TimeValue | UnresolvedValue:
    return seconds if isinstance(seconds, UnresolvedValue) else TimeValue(value=seconds, unit="s")


def _lower_delay(time: TimeValue | UnresolvedValue) -> ModelDelay:
    if isinstance(time, UnresolvedValue):
        return time
    if time.value > 0:
        return ModelTime(value=time.value, unit=time.unit)
    return _held_expression(LiteralValue(value=time.value, unit=time.unit))


def _store_delay(time: ModelDelay) -> TimeValue | UnresolvedValue:
    return time if isinstance(time, UnresolvedValue) else TimeValue(value=time.value, unit=time.unit)


def _number(expression: Expression) -> int | float | None:
    if isinstance(expression, LiteralValue) and not expression.unit and type(expression.value) in (int, float):
        return expression.value
    return None


def _lower_substeps(expression: Expression) -> ModelSubsteps:
    count = _number(expression)
    return count if type(count) is int and count >= 1 else _held_expression(expression)


def _store_substeps(count: ModelSubsteps) -> Expression:
    return count if isinstance(count, UnresolvedValue) else LiteralValue(value=count)


def _lower_initial(expression: Expression) -> float | tuple[float, ...] | UnresolvedValue:
    if (scalar := _number(expression)) is not None:
        return float(scalar)
    if isinstance(expression, CallValue) and expression.function == "vector" and not expression.kwargs:
        members = tuple(_number(item) for item in expression.args)
        if None not in members:
            return tuple(float(item) for item in members)
    return _held_expression(expression)


def _store_initial(value: float | tuple[float, ...] | UnresolvedValue) -> Expression:
    if isinstance(value, UnresolvedValue):
        return value
    if isinstance(value, tuple):
        return CallValue(function="vector", args=tuple(LiteralValue(value=item) for item in value))
    return LiteralValue(value=value)


def _lower_signal_source(source: StoredModelSignalSource) -> ModelSignalSource:
    if isinstance(source, StoredDelayInitialValue):
        return project_member(source, DelayInitialValue, {"value": _lower_initial})
    return source


def _store_signal_source(source: ModelSignalSource) -> StoredModelSignalSource:
    if isinstance(source, DelayInitialValue):
        return project_member(source, StoredDelayInitialValue, {"value": _store_initial})
    return source


def lower_model_numeric_field(component_type: str, name: str, kind: Literal["int", "float"], value: JsonValue) -> JsonValue:
    """Restore a numeric leaf's runtime value from its authored expression."""
    if value is None:
        return None
    expression = _STORED_EXPRESSION.validate_python(value)
    if (scalar := _number(expression)) is not None and (kind == "float" or type(scalar) is int):
        return scalar
    if component_type == "feedbax.VanillaRNN" and name == "tau" and isinstance(expression, LiteralValue):
        if type(expression.value) in (int, float) and (unit := UNITS.get(expression.unit)) and unit.dimensions == TIME:
            time = literal_time(expression)
            return LiteralValue(value=time.value, unit=time.unit).model_dump(mode="json")
    return _held_expression(expression).model_dump(mode="json")


def stored_model_numeric_field(value: JsonValue) -> JsonValue:
    """Project a declared numeric runtime field without interpreting other leaf values."""
    if value is None:
        return None
    expression = _RUNTIME_EXPRESSION.validate_python(value) if isinstance(value, dict) else LiteralValue(value=value)
    return _STORED_EXPRESSION.dump_python(store_expression(expression), mode="json")


@cache
def _numeric_params(model: type[BaseModel]) -> dict[str, Literal["int", "float"]]:
    from feedbax.component_registry.declarations import parameter_schema_from_model

    return {schema.name: schema.type for schema in parameter_schema_from_model(model) if schema.type in ("int", "float")}


def _numeric_fields(
    component: StoredLeafComponent | LeafComponent, components: ComponentRegistry | None,
) -> dict[str, Literal["int", "float"]]:
    if components is None:
        return {}
    try:
        declaration = components.declaration_record(component.type).value
    except KeyError:
        return {}
    return _numeric_params(declaration.params.model) if component.revision in (None, declaration.revision) else {}


def _leaf_fields(
    component: StoredLeafComponent | LeafComponent, components: ComponentRegistry | None, *, storing: bool,
) -> dict[str, JsonValue]:
    if not component.fields:
        return component.fields
    fields = component.fields.copy()
    for name, kind in _numeric_fields(component, components).items():
        if name in fields:
            value = fields[name]
            fields[name] = stored_model_numeric_field(value) if storing else lower_model_numeric_field(
                component.type, name, kind, value,
            )
    return fields


def lower_model_field_value(
    model: ModelDocument, address: ModelAddress, value: JsonValue, components: ComponentRegistry | None = None,
) -> JsonValue:
    """Lower a persisted addressed write. Numeric sweep coordinates retain the destination's unit."""
    element = address.element
    if not isinstance(element, FieldElement) or value is None or type(value) in (int, float):
        return value
    path = element.path
    match element.parent.kind, path:
        case "graph", ("step_seconds",):
            return to_jsonable_python(_lower_period(_STORED_TIME.validate_python(value)))
        case "component", ("substeps",):
            return to_jsonable_python(_lower_substeps(_STORED_EXPRESSION.validate_python(value)))
        case "attachment", ("delay",):
            return to_jsonable_python(_lower_delay(_STORED_TIME.validate_python(value)))
        case "attachment", ("window", "spans", _, _, "offset"):
            return to_jsonable_python(lower_window_offset(_STORED_EXPRESSION.validate_python(value)))
        case "attachment", (("initial_source",) | ("operands", _)):
            return to_jsonable_python(_lower_signal_source(_STORED_SIGNAL_SOURCE.validate_python(value)))
        case "attachment", (("initial_source", "value") | ("operands", _, "value")):
            return to_jsonable_python(_lower_initial(_STORED_EXPRESSION.validate_python(value)))
        case "component", ("fields", str(name)) if components is not None:
            branch = model
            for key in address.composites:
                child = branch.interior.components.get(key)
                if not isinstance(child, CompositeComponent):
                    return value
                branch = child
            component = branch.interior.components.get(element.parent.id)
            if isinstance(component, LeafComponent) and (kind := _numeric_fields(component, components).get(name)):
                return lower_model_numeric_field(component.type, name, kind, value)
    return value


def stored_model_field_value(address: ModelAddress, value: JsonValue) -> JsonValue:
    """Project a typed edit on a known numeric, time or expression field to its authored slot."""
    element = address.element
    if not isinstance(element, FieldElement) or value is None:
        return value
    match element.parent.kind, element.path:
        case "graph", ("step_seconds",):
            if isinstance(value, dict):
                time = _RUNTIME_TIME.validate_python(value)
            else:
                time = _store_period(value)
            return _STORED_TIME.dump_python(time, mode="json")
        case "attachment", ("delay",):
            return _STORED_TIME.dump_python(_RUNTIME_TIME.validate_python(value), mode="json")
        case "attachment", ("window", "spans", _, _, "offset"):
            offset = stored_window_offset(_RUNTIME_OFFSET.validate_python(value))
            return _STORED_EXPRESSION.dump_python(offset, mode="json")
        case "attachment", (("initial_source",) | ("operands", _)):
            source = _store_signal_source(_RUNTIME_SIGNAL_SOURCE.validate_python(value))
            return _STORED_SIGNAL_SOURCE.dump_python(source, mode="json")
        case "attachment", (("initial_source", "value") | ("operands", _, "value")):
            initial = DelayInitialValue(kind="value", value=value)
            return _STORED_EXPRESSION.dump_python(_store_initial(initial.value), mode="json")
        case _:
            return stored_model_numeric_field(value)


class StoredSignalInput(StrictModel):
    label: ModelName
    type: SignalType
    default: InputDefault | None = None


class StoredSignalOutput(StrictModel):
    label: ModelName
    type: SignalType


class StoredPhysicalPort(StrictModel):
    label: ModelName
    type: PhysicalPortType


class StoredModelPorts(StrictModel):
    inputs: StoredMembers[StoredSignalInput] = Field(default_factory=dict)
    outputs: StoredMembers[StoredSignalOutput] = Field(default_factory=dict)
    physical: StoredMembers[StoredPhysicalPort] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_ports(self) -> StoredModelPorts:
        validate_cross_field_refinements("StoredModelPorts", self)
        return self


class StoredDelayInitialValue(StrictModel):
    kind: Literal["value"]
    value: StoredExpression


type StoredModelSignalSource = Annotated[
    StoredDelayInitialValue | DelayInitialTask | DelayInitialWire, Field(discriminator="kind")
]

_STORED_SIGNAL_SOURCE = TypeAdapter(StoredModelSignalSource)
_RUNTIME_SIGNAL_SOURCE = TypeAdapter(ModelSignalSource)


class StoredDelayAttachment(StrictModel):
    kind: Literal["delay"]
    delay: StoredTime
    initial_source: StoredModelSignalSource = Field(default_factory=lambda: StoredDelayInitialValue(
        kind="value", value=LiteralValue(value=0),
    ))


class StoredHoldAttachment(StrictModel):
    kind: Literal["hold"]


class StoredInterventionAttachment(StrictModel):
    kind: Literal["intervention"]
    operation: str
    operands: dict[str, StoredModelSignalSource] = Field(default_factory=dict)
    window: StoredWindow | None = None


class StoredProbeAttachment(StrictModel):
    kind: Literal["probe"]
    label: ModelName


StoredWireAttachment = Annotated[
    StoredDelayAttachment | StoredHoldAttachment | StoredInterventionAttachment | StoredProbeAttachment,
    Field(discriminator="kind"),
]
StoredSourceAttachment = Annotated[StoredInterventionAttachment | StoredProbeAttachment, Field(discriminator="kind")]


class StoredModelPortAttachments(StrictModel):
    sources: dict[StoredMemberKey, StoredMembers[StoredSourceAttachment]] = Field(default_factory=dict)
    sinks: dict[StoredMemberKey, StoredMembers[StoredProbeAttachment]] = Field(default_factory=dict)


class StoredModelWire(StrictModel):
    source: SignalSource
    target: SignalTarget
    attachments: StoredMembers[StoredWireAttachment] = Field(default_factory=dict)


class StoredLeafComponent(StrictModel):
    kind: Literal["leaf"]
    label: ModelName
    type: ModelName
    revision: ComponentRevision | None = None
    fields: dict[str, JsonValue] = Field(default_factory=dict)
    port_counts: dict[ModelElementId, ComponentPortCount] = Field(default_factory=dict)
    attachments: StoredModelPortAttachments = Field(default_factory=StoredModelPortAttachments)


class StoredCompositeComponent(StrictModel):
    kind: Literal["composite"]
    label: ModelName
    ports: StoredModelPorts
    interior: StoredGraphInterior | StoredPhysicalInterior
    substeps: StoredExpression = Field(default_factory=lambda: LiteralValue(value=1))
    attachments: StoredModelPortAttachments = Field(default_factory=StoredModelPortAttachments)

    @model_validator(mode="after")
    def _physical_interface(self) -> StoredCompositeComponent:
        validate_cross_field_refinements("StoredCompositeComponent", self)
        return self


StoredModelComponent = Annotated[StoredLeafComponent | StoredCompositeComponent, Field(discriminator="kind")]


class _StoredModelInterior(StrictModel):
    components: StoredMembers[StoredModelComponent] = Field(default_factory=dict)
    wires: StoredMembers[StoredModelWire] = Field(default_factory=dict)
    attachments: StoredModelPortAttachments = Field(default_factory=StoredModelPortAttachments)


class StoredGraphInterior(_StoredModelInterior):
    physical_domain: None = None

    @model_validator(mode="after")
    def _unique_elements(self) -> StoredGraphInterior:
        validate_cross_field_refinements("StoredGraphInterior", self)
        return self


class StoredPhysicalInterior(_StoredModelInterior):
    physical_domain: AcausalPhysicalDomain
    connections: StoredMembers[ModelPhysicalConnection] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_elements(self) -> StoredPhysicalInterior:
        validate_cross_field_refinements("StoredPhysicalInterior", self)
        return self


class StoredModelDocument(StrictModel):
    model_config = ConfigDict(allow_inf_nan=False)

    step_seconds: StoredTime
    ports: StoredModelPorts
    interior: StoredGraphInterior
    world: ModelWorld | None = None

    @model_validator(mode="after")
    def _physical_interface(self) -> StoredModelDocument:
        validate_cross_field_refinements("StoredModelDocument", self)
        return self

    def lower(self, components: ComponentRegistry | None = None) -> ModelDocument:
        """Restore the runtime Model with the same port addresses and attachment sequence."""
        interior = _lower_interior(self.interior, components)
        assert isinstance(interior, GraphInterior)
        return project_member(
            self, ModelDocument, {"step_seconds": _lower_period}, ports=_lower_ports(self.ports), interior=interior,
        )

    @classmethod
    def from_lowered(cls, model: ModelDocument, components: ComponentRegistry | None = None) -> StoredModelDocument:
        """Project a producer's typed Model into the authored form."""
        interior = _store_interior(model.interior, components)
        assert isinstance(interior, StoredGraphInterior)
        return project_member(
            model, cls, {"step_seconds": _store_period}, ports=_store_ports(model.ports), interior=interior,
        )


def _lower_ports(ports: StoredModelPorts) -> ModelPorts:
    return ModelPorts(
        inputs=tuple(project_member(port, SignalInput, id=key, name=port.label) for key, port in ports.inputs.items()),
        outputs=tuple(project_member(port, SignalOutput, id=key, name=port.label) for key, port in ports.outputs.items()),
        physical=tuple(project_member(port, PhysicalPort, id=key, name=port.label) for key, port in ports.physical.items()),
    )


def _store_ports(ports: ModelPorts) -> StoredModelPorts:
    return StoredModelPorts(
        inputs=keyed_members((port.id, project_member(port, StoredSignalInput, label=port.name)) for port in ports.inputs),
        outputs=keyed_members((port.id, project_member(port, StoredSignalOutput, label=port.name)) for port in ports.outputs),
        physical=keyed_members((port.id, project_member(port, StoredPhysicalPort, label=port.name)) for port in ports.physical),
    )


def lower_model_attachment(key: str, item: StoredWireAttachment) -> WireAttachment:
    match item:
        case StoredDelayAttachment():
            return project_member(item, DelayAttachment, {
                "delay": _lower_delay, "initial_source": _lower_signal_source,
            }, id=key)
        case StoredHoldAttachment():
            return HoldAttachment(kind="hold", id=key)
        case StoredInterventionAttachment():
            return project_member(item, InterventionAttachment, {
                "operands": lambda sources: {name: _lower_signal_source(source) for name, source in sources.items()},
                "window": lambda window: window.lower() if window is not None else None,
            }, id=key)
        case StoredProbeAttachment():
            return ProbeAttachment(kind="probe", id=key, name=item.label)


def stored_model_attachment(item: WireAttachment) -> StoredWireAttachment:
    match item:
        case DelayAttachment():
            return project_member(item, StoredDelayAttachment, {
                "delay": _store_delay, "initial_source": _store_signal_source,
            })
        case HoldAttachment():
            return StoredHoldAttachment(kind="hold")
        case InterventionAttachment():
            return project_member(item, StoredInterventionAttachment, {
                "operands": lambda sources: {name: _store_signal_source(source) for name, source in sources.items()},
                "window": lambda window: StoredWindow.from_lowered(window) if window is not None else None,
            })
        case ProbeAttachment():
            return StoredProbeAttachment(kind="probe", label=item.name)


def _lower_port_attachments(ports: StoredModelPortAttachments) -> ModelPortAttachments:
    return project_member(
        ports, ModelPortAttachments,
        sources={port: tuple(lower_model_attachment(key, item) for key, item in chain.items()) for port, chain in ports.sources.items()},
        sinks={port: tuple(lower_model_attachment(key, item) for key, item in chain.items()) for port, chain in ports.sinks.items()},
    )


def _store_port_attachments(ports: ModelPortAttachments) -> StoredModelPortAttachments:
    return project_member(
        ports, StoredModelPortAttachments,
        sources={port: keyed_members((item.id, stored_model_attachment(item)) for item in chain) for port, chain in ports.sources.items()},
        sinks={port: keyed_members((item.id, stored_model_attachment(item)) for item in chain) for port, chain in ports.sinks.items()},
    )


def lower_model_component(
    component: StoredModelComponent, components: ComponentRegistry | None = None,
) -> ModelComponent:
    attachments = _lower_port_attachments(component.attachments)
    if isinstance(component, StoredLeafComponent):
        return project_member(component, LeafComponent, {
            "fields": lambda _: _leaf_fields(component, components, storing=False),
        }, name=component.label, attachments=attachments)
    return project_member(
        component, CompositeComponent, {"substeps": _lower_substeps},
        name=component.label, ports=_lower_ports(component.ports),
        interior=_lower_interior(component.interior, components), attachments=attachments,
    )


def stored_model_component(
    component: ModelComponent, components: ComponentRegistry | None = None,
) -> StoredModelComponent:
    attachments = _store_port_attachments(component.attachments)
    if isinstance(component, LeafComponent):
        return project_member(component, StoredLeafComponent, {
            "fields": lambda _: _leaf_fields(component, components, storing=True),
        }, label=component.name, attachments=attachments)
    return project_member(
        component, StoredCompositeComponent, {"substeps": _store_substeps},
        label=component.name, ports=_store_ports(component.ports),
        interior=_store_interior(component.interior, components), attachments=attachments,
    )


def lower_model_wire(wire: StoredModelWire) -> ModelWire:
    return project_member(
        wire, ModelWire,
        attachments=tuple(lower_model_attachment(key, item) for key, item in wire.attachments.items()),
    )


def _lower_interior(
    interior: StoredGraphInterior | StoredPhysicalInterior, components: ComponentRegistry | None,
) -> GraphInterior | PhysicalInterior:
    target = GraphInterior if isinstance(interior, StoredGraphInterior) else PhysicalInterior
    return project_member(
        interior, target,
        components={key: lower_model_component(component, components) for key, component in interior.components.items()},
        wires={key: lower_model_wire(wire) for key, wire in interior.wires.items()},
        attachments=_lower_port_attachments(interior.attachments),
    )


def _store_interior(
    interior: GraphInterior | PhysicalInterior, components: ComponentRegistry | None,
) -> StoredGraphInterior | StoredPhysicalInterior:
    target = StoredGraphInterior if isinstance(interior, GraphInterior) else StoredPhysicalInterior
    return project_member(
        interior, target,
        components={key: stored_model_component(component, components) for key, component in interior.components.items()},
        wires={
            key: project_member(wire, StoredModelWire, attachments=keyed_members(
                (item.id, stored_model_attachment(item)) for item in wire.attachments
            ))
            for key, wire in interior.wires.items()
        },
        attachments=_store_port_attachments(interior.attachments),
    )


StoredModelDocument.model_rebuild()
