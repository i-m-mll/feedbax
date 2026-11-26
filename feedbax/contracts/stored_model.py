"""The authored Model v2 payload and its ordered projection to runtime Models."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from feedbax.contracts.acausal import AcausalPhysicalDomain
from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelElementId, SignalSource, SignalTarget
from feedbax.contracts.model_document import (
    ComponentPortCount, ComponentRevision, CompositeComponent, DelayAttachment, GraphInterior,
    HoldAttachment, InputDefault, InterventionAttachment, LeafComponent, ModelComponent,
    ModelDelay, ModelDocument, ModelName, ModelPhysicalConnection, ModelPortAttachments, ModelPorts,
    ModelSignalSource, ModelStepSeconds, ModelSubsteps, ModelWire, ModelWorld, PhysicalInterior,
    PhysicalPort, PhysicalPortType, ProbeAttachment, SignalInput, SignalOutput, SignalType,
    WireAttachment, default_delay_initial_source,
)
from feedbax.contracts.stored_members import StoredMemberKey, StoredMembers, keyed_members, project_member
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.window import Window


STORED_MODEL_DOCUMENT_SCHEMA_ID = "feedbax.spec.model.v2"


def model_keyed_collection(path: tuple[str, ...]) -> bool:
    from feedbax.contracts.project_storage import stored_keyed_collection

    return stored_keyed_collection(STORED_MODEL_DOCUMENT_SCHEMA_ID, path)


def authored_model_payload(model: ModelDocument) -> dict[str, JsonValue]:
    """Project a typed Model into a complete keyed payload for authored comparisons."""
    return StoredModelDocument.from_lowered(model).model_dump(mode="json")


def model_member_order(
    value: JsonValue, path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], list[str]]]:
    """List the member orders that canonical JSON's key sorting would erase."""
    if isinstance(value, dict):
        if model_keyed_collection(path):
            yield path, list(value)
        for key, child in value.items():
            yield from model_member_order(child, (*path, key))


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


class StoredDelayAttachment(StrictModel):
    kind: Literal["delay"]
    delay: ModelDelay
    initial_source: ModelSignalSource = Field(default_factory=default_delay_initial_source)


class StoredHoldAttachment(StrictModel):
    kind: Literal["hold"]


class StoredInterventionAttachment(StrictModel):
    kind: Literal["intervention"]
    operation: str
    operands: dict[str, ModelSignalSource] = Field(default_factory=dict)
    window: Window | None = None


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
    substeps: ModelSubsteps = 1
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

    step_seconds: ModelStepSeconds
    ports: StoredModelPorts
    interior: StoredGraphInterior
    world: ModelWorld | None = None

    @model_validator(mode="after")
    def _physical_interface(self) -> StoredModelDocument:
        validate_cross_field_refinements("StoredModelDocument", self)
        return self

    def lower(self) -> ModelDocument:
        """Restore the runtime Model with the same port addresses and attachment sequence."""
        interior = _lower_interior(self.interior)
        assert isinstance(interior, GraphInterior)
        return project_member(self, ModelDocument, ports=_lower_ports(self.ports), interior=interior)

    @classmethod
    def from_lowered(cls, model: ModelDocument) -> StoredModelDocument:
        """Project a producer's typed Model into the authored form."""
        interior = _store_interior(model.interior)
        assert isinstance(interior, StoredGraphInterior)
        return project_member(model, cls, ports=_store_ports(model.ports), interior=interior)


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


def _lower_attachment(key: str, item: StoredWireAttachment) -> WireAttachment:
    match item:
        case StoredDelayAttachment():
            return project_member(item, DelayAttachment, id=key)
        case StoredHoldAttachment():
            return HoldAttachment(kind="hold", id=key)
        case StoredInterventionAttachment():
            return project_member(item, InterventionAttachment, id=key)
        case StoredProbeAttachment():
            return ProbeAttachment(kind="probe", id=key, name=item.label)


def _store_attachment(item: WireAttachment) -> StoredWireAttachment:
    match item:
        case DelayAttachment():
            return project_member(item, StoredDelayAttachment)
        case HoldAttachment():
            return StoredHoldAttachment(kind="hold")
        case InterventionAttachment():
            return project_member(item, StoredInterventionAttachment)
        case ProbeAttachment():
            return StoredProbeAttachment(kind="probe", label=item.name)


def _lower_port_attachments(ports: StoredModelPortAttachments) -> ModelPortAttachments:
    return project_member(
        ports, ModelPortAttachments,
        sources={port: tuple(_lower_attachment(key, item) for key, item in chain.items()) for port, chain in ports.sources.items()},
        sinks={port: tuple(_lower_attachment(key, item) for key, item in chain.items()) for port, chain in ports.sinks.items()},
    )


def _store_port_attachments(ports: ModelPortAttachments) -> StoredModelPortAttachments:
    return project_member(
        ports, StoredModelPortAttachments,
        sources={port: keyed_members((item.id, _store_attachment(item)) for item in chain) for port, chain in ports.sources.items()},
        sinks={port: keyed_members((item.id, _store_attachment(item)) for item in chain) for port, chain in ports.sinks.items()},
    )


def _lower_component(component: StoredModelComponent) -> ModelComponent:
    attachments = _lower_port_attachments(component.attachments)
    if isinstance(component, StoredLeafComponent):
        return project_member(component, LeafComponent, name=component.label, attachments=attachments)
    return project_member(
        component, CompositeComponent, name=component.label, ports=_lower_ports(component.ports),
        interior=_lower_interior(component.interior), attachments=attachments,
    )


def _store_component(component: ModelComponent) -> StoredModelComponent:
    attachments = _store_port_attachments(component.attachments)
    if isinstance(component, LeafComponent):
        return project_member(component, StoredLeafComponent, label=component.name, attachments=attachments)
    return project_member(
        component, StoredCompositeComponent, label=component.name, ports=_store_ports(component.ports),
        interior=_store_interior(component.interior), attachments=attachments,
    )


def _lower_interior(interior: StoredGraphInterior | StoredPhysicalInterior) -> GraphInterior | PhysicalInterior:
    target = GraphInterior if isinstance(interior, StoredGraphInterior) else PhysicalInterior
    return project_member(
        interior, target,
        components={key: _lower_component(component) for key, component in interior.components.items()},
        wires={
            key: project_member(wire, ModelWire, attachments=tuple(
                _lower_attachment(identifier, item) for identifier, item in wire.attachments.items()
            ))
            for key, wire in interior.wires.items()
        },
        attachments=_lower_port_attachments(interior.attachments),
    )


def _store_interior(interior: GraphInterior | PhysicalInterior) -> StoredGraphInterior | StoredPhysicalInterior:
    target = StoredGraphInterior if isinstance(interior, GraphInterior) else StoredPhysicalInterior
    return project_member(
        interior, target,
        components={key: _store_component(component) for key, component in interior.components.items()},
        wires={
            key: project_member(wire, StoredModelWire, attachments=keyed_members(
                (item.id, _store_attachment(item)) for item in wire.attachments
            ))
            for key, wire in interior.wires.items()
        },
        attachments=_store_port_attachments(interior.attachments),
    )


StoredModelDocument.model_rebuild()
