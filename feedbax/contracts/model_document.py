"""The stored model tree and its scientific projection."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, JsonValue, StringConstraints, model_validator

from feedbax.contracts.acausal import AcausalPhysicalDomain, EquationsDomain
from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import SignalSpace, UnresolvedValue
from feedbax.contracts.model_address import ModelElementId, PhysicalEnd, SignalSource, SignalTarget
from feedbax.contracts.request_identity_guard import IdentityAuthoredAlternative
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.time_units import TimeValue
from feedbax.contracts.window import Window

MODEL_DOCUMENT_SCHEMA_ID = "feedbax.spec.model"
MODEL_DOCUMENT_SCHEMA_VERSION = "feedbax.spec.model.v1"
ModelName = Annotated[str, StringConstraints(min_length=1)]
PositivePeriod = Annotated[float, Field(gt=0, allow_inf_nan=False)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


class SignalType(StrictModel):
    base: str | None = None
    shape: tuple[int | str | None, ...] | None = None
    unit: str | None = None
    space: SignalSpace | None = Field(default=None, exclude_if=lambda value: value is None)


class ModelTime(TimeValue):
    """A positive time, such as a Delay or a component's physical time constant."""

    value: PositivePeriod


ModelDelay = Annotated[ModelTime | UnresolvedValue, IdentityAuthoredAlternative(UnresolvedValue)]
ModelStepSeconds = Annotated[PositivePeriod | UnresolvedValue, IdentityAuthoredAlternative(UnresolvedValue)]
ModelSubsteps = Annotated[Annotated[int, Field(ge=1)] | UnresolvedValue, IdentityAuthoredAlternative(UnresolvedValue)]
ComponentRevision = Annotated[int, Field(strict=True, ge=1)]
ComponentPortCount = Annotated[int, Field(ge=0)]


class DelayInitialValue(StrictModel):
    kind: Literal["value"]
    value: Annotated[FiniteNumber | tuple[FiniteNumber, ...] | UnresolvedValue, IdentityAuthoredAlternative(UnresolvedValue)]


class DelayInitialTask(StrictModel):
    kind: Literal["task"]
    variable: ModelName


class DelayInitialWire(StrictModel):
    kind: Literal["wire"]
    source: SignalSource | None = Field(default=None, exclude_if=lambda value: value is None)


type ModelSignalSource = Annotated[DelayInitialValue | DelayInitialTask | DelayInitialWire, Field(discriminator="kind")]


def default_delay_initial_source() -> DelayInitialValue:
    return DelayInitialValue(kind="value", value=0.0)


class DelayAttachment(StrictModel):
    kind: Literal["delay"]
    id: ModelElementId
    delay: ModelDelay
    initial_source: ModelSignalSource = Field(default_factory=default_delay_initial_source)


class HoldAttachment(StrictModel):
    kind: Literal["hold"]
    id: ModelElementId


class InterventionAttachment(StrictModel):
    kind: Literal["intervention"]
    id: ModelElementId
    operation: str
    operands: dict[str, ModelSignalSource] = Field(default_factory=dict)
    window: Window | None = None


class ProbeAttachment(StrictModel):
    kind: Literal["probe"]
    id: ModelElementId
    name: ModelName


WireAttachment = Annotated[
    DelayAttachment | HoldAttachment | InterventionAttachment | ProbeAttachment, Field(discriminator="kind")
]
SourceAttachment = Annotated[
    InterventionAttachment | ProbeAttachment, Field(discriminator="kind")
]


class InputDefault(StrictModel):
    value: JsonValue


class SignalInput(StrictModel):
    id: ModelElementId
    name: ModelName
    type: SignalType
    default: InputDefault | None = None


class SignalOutput(StrictModel):
    id: ModelElementId
    name: ModelName
    type: SignalType


PhysicalPortType = Literal[EquationsDomain, "frame"]


class PhysicalPort(StrictModel):
    id: ModelElementId
    name: ModelName
    type: PhysicalPortType


class ModelPorts(StrictModel):
    inputs: tuple[SignalInput, ...] = ()
    outputs: tuple[SignalOutput, ...] = ()
    physical: tuple[PhysicalPort, ...] = ()

    @model_validator(mode="after")
    def _unique_ports(self) -> ModelPorts:
        validate_cross_field_refinements("ModelPorts", self)
        return self


class ModelPortAttachments(StrictModel):
    sources: dict[ModelElementId, tuple[SourceAttachment, ...]] = Field(default_factory=dict)
    sinks: dict[ModelElementId, tuple[ProbeAttachment, ...]] = Field(default_factory=dict)


class ModelWire(StrictModel):
    source: SignalSource
    target: SignalTarget
    attachments: tuple[WireAttachment, ...] = ()


class ModelPhysicalConnection(StrictModel):
    a: PhysicalEnd
    b: PhysicalEnd


class LeafComponent(StrictModel):
    kind: Literal["leaf"]
    name: ModelName
    type: ModelName
    revision: ComponentRevision | None = None
    fields: dict[str, JsonValue] = Field(default_factory=dict)
    port_counts: dict[ModelElementId, ComponentPortCount] = Field(default_factory=dict)
    attachments: ModelPortAttachments = Field(default_factory=ModelPortAttachments)


class CompositeComponent(StrictModel):
    kind: Literal["composite"]
    name: ModelName
    ports: ModelPorts
    interior: GraphInterior | PhysicalInterior
    substeps: ModelSubsteps = 1
    attachments: ModelPortAttachments = Field(default_factory=ModelPortAttachments)

    @model_validator(mode="after")
    def _physical_interface(self) -> CompositeComponent:
        validate_cross_field_refinements("CompositeComponent", self)
        return self


ModelComponent = Annotated[LeafComponent | CompositeComponent, Field(discriminator="kind")]


class ModelInterior(StrictModel):
    components: dict[ModelElementId, ModelComponent] = Field(default_factory=dict)
    wires: dict[ModelElementId, ModelWire] = Field(default_factory=dict)
    attachments: ModelPortAttachments = Field(default_factory=ModelPortAttachments)


class GraphInterior(ModelInterior):
    physical_domain: None = None

    @model_validator(mode="after")
    def _unique_elements(self) -> GraphInterior:
        validate_cross_field_refinements("GraphInterior", self)
        return self

    @property
    def kind(self) -> Literal["graph"]:
        return "graph"


class PhysicalInterior(ModelInterior):
    physical_domain: AcausalPhysicalDomain
    connections: dict[ModelElementId, ModelPhysicalConnection] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_elements(self) -> PhysicalInterior:
        validate_cross_field_refinements("PhysicalInterior", self)
        return self

    @property
    def kind(self) -> Literal["equations", "mechanics"]:
        if self.physical_domain in {"translational", "rotational"}:
            return "equations"
        return "mechanics"


class ModelWorld(StrictModel):
    gravity: tuple[FiniteNumber, FiniteNumber, FiniteNumber]


class ModelDocument(StrictModel):
    model_config = ConfigDict(allow_inf_nan=False)

    schema_version: Literal[MODEL_DOCUMENT_SCHEMA_VERSION] = MODEL_DOCUMENT_SCHEMA_VERSION
    step_seconds: ModelStepSeconds
    ports: ModelPorts
    interior: GraphInterior
    world: ModelWorld | None = None

    @model_validator(mode="after")
    def _physical_interface(self) -> ModelDocument:
        validate_cross_field_refinements("ModelDocument", self)
        return self

    def with_keyed_root_ports(self) -> ModelDocument:
        """Use stable port keys for the external names of a managed graph."""
        updated = {}
        for direction in ("inputs", "outputs", "physical"):
            ports = getattr(self.ports, direction)
            if any(port.name != port.id for port in ports):
                updated[direction] = tuple(port.model_copy(update={"name": port.id}) for port in ports)
        if not updated:
            return self
        return self.model_copy(update={"ports": self.ports.model_copy(update=updated)})

    def scientific_projection(self, *, keyed_root_ports: bool = False) -> dict[str, JsonValue]:
        value = self.model_dump(mode="json")
        omit_model_display_names(value, keyed_root_ports=keyed_root_ports)
        return value


def _display_names(value: dict[str, Any], *, keyed_root_ports: bool) -> Iterator[tuple[dict[str, Any], str]]:
    if keyed_root_ports:
        for direction in ("inputs", "outputs", "physical"):
            for port in value["ports"].get(direction, ()):
                yield port, port["id"]
    for identifier, component in value["interior"].get("components", {}).items():
        yield component, identifier
        if component["kind"] == "composite":
            yield from _display_names(component, keyed_root_ports=True)


def omit_model_display_names(value: dict[str, Any], *, keyed_root_ports: bool = False) -> None:
    """Remove presentation labels from a serialized Model."""
    for element, _ in _display_names(value, keyed_root_ports=keyed_root_ports):
        element.pop("name", None)


def restore_model_display_names(value: dict[str, Any], *, keyed_root_ports: bool = False) -> None:
    """Supply the names the runtime Model requires from the stored keys."""
    for element, identifier in _display_names(value, keyed_root_ports=keyed_root_ports):
        element["name"] = identifier


ModelDocument.model_rebuild()
