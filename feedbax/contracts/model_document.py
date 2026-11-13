"""The stored model tree and its scientific projection."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, JsonValue, StringConstraints, model_validator

from feedbax.contracts.acausal import AcausalPhysicalDomain
from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelElementId, PhysicalEnd, SignalSource, SignalTarget
from feedbax.contracts.studio_refinements import validate_cross_field_refinements

MODEL_DOCUMENT_SCHEMA_ID = "feedbax.spec.model"
MODEL_DOCUMENT_SCHEMA_VERSION = "feedbax.spec.model.v1"
ModelName = Annotated[str, StringConstraints(min_length=1)]
PositivePeriod = Annotated[float, Field(gt=0, allow_inf_nan=False)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


class SignalType(StrictModel):
    base: str | None = None
    shape: tuple[int | str | None, ...] | None = None
    unit: str | None = None


class ModelTime(StrictModel):
    value: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    unit: Literal["s", "st"]


class DelayAttachment(StrictModel):
    kind: Literal["delay"]
    id: ModelElementId
    delay: ModelTime
    initial: JsonValue = 0


class InterventionAttachment(StrictModel):
    kind: Literal["intervention"]
    id: ModelElementId
    operation: str
    fields: dict[str, JsonValue] = Field(default_factory=dict)
    window: JsonValue = None


class ProbeAttachment(StrictModel):
    kind: Literal["probe"]
    id: ModelElementId
    name: ModelName


WireAttachment = Annotated[
    DelayAttachment | InterventionAttachment | ProbeAttachment, Field(discriminator="kind")
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


class PhysicalPort(StrictModel):
    id: ModelElementId
    name: ModelName
    type: Literal["translational", "rotational", "frame"]


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
    revision: Annotated[int, Field(strict=True, ge=1)] | None = None
    fields: dict[str, JsonValue] = Field(default_factory=dict)
    port_counts: dict[ModelElementId, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    attachments: ModelPortAttachments = Field(default_factory=ModelPortAttachments)


class CompositeComponent(StrictModel):
    kind: Literal["composite"]
    name: ModelName
    ports: ModelPorts
    interior: GraphInterior | PhysicalInterior
    substeps: Annotated[int, Field(ge=1)] = 1
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
    step_seconds: PositivePeriod
    ports: ModelPorts
    interior: GraphInterior
    world: ModelWorld | None = None

    @model_validator(mode="after")
    def _physical_interface(self) -> ModelDocument:
        validate_cross_field_refinements("ModelDocument", self)
        return self

    def scientific_projection(self) -> dict[str, JsonValue]:
        value = self.model_dump(mode="json")
        _project_interior(value["interior"])
        return value


def _project_interior(interior: dict[str, Any]) -> None:
    for component in interior["components"].values():
        del component["name"]
        if component["kind"] == "composite":
            for ports in component["ports"].values():
                for port in ports:
                    del port["name"]
            _project_interior(component["interior"])


ModelDocument.model_rebuild()
