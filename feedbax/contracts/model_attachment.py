"""Authored Model attachment differences, shared by Project and Studio."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import (
    ComponentElement, ModelAddress, ModelElementId, SignalSource, SignalTarget, WireElement,
)
from feedbax.contracts.stored_model import StoredLeafComponent, StoredWireAttachment
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_refinements import validate_cross_field_refinements


class ExistingModelAttachment(StrictModel):
    address: ModelAddress

    @model_validator(mode="after")
    def _attachment(self) -> "ExistingModelAttachment":
        validate_cross_field_refinements("ExistingModelAttachment", self)
        return self


class AddModelAttachment(ExistingModelAttachment):
    op: Literal["add"]
    attachment: StoredWireAttachment
    after: ModelElementId | None

    @model_validator(mode="after")
    def _wire_hold(self) -> "AddModelAttachment":
        validate_cross_field_refinements("AddModelAttachment", self)
        return self


class ChangeModelAttachment(ExistingModelAttachment):
    op: Literal["change"]
    attachment: StoredWireAttachment
    after: ModelElementId | None = None


class RemoveModelAttachment(ExistingModelAttachment):
    op: Literal["remove"]


class SetField(StrictModel):
    op: Literal["set_field"]
    address: ModelAddress
    value: JsonValue


class AddModelComponent(StrictModel):
    op: Literal["add_component"]
    address: ModelAddress
    component: StoredLeafComponent

    @model_validator(mode="after")
    def _component_address(self) -> "AddModelComponent":
        if not isinstance(self.address.element, ComponentElement):
            raise ValueError("A component addition requires a component address")
        return self


class ModelWireDifference(StrictModel):
    op: Literal["add_wire", "rewire"]
    address: ModelAddress
    source: SignalSource
    target: SignalTarget

    @model_validator(mode="after")
    def _wire_address(self) -> "ModelWireDifference":
        if not isinstance(self.address.element, WireElement):
            raise ValueError("A wire difference requires a wire address")
        return self


AttachmentDifference = AddModelAttachment | ChangeModelAttachment | RemoveModelAttachment
ModelAttachmentDifference = Annotated[AttachmentDifference, Field(discriminator="op")]
EditableModelDifference = Annotated[AttachmentDifference | SetField, Field(discriminator="op")]
ModelDifference = Annotated[
    AttachmentDifference | SetField | AddModelComponent | ModelWireDifference,
    Field(discriminator="op"),
]

MODEL_ATTACHMENTS = "model_attachments"
MODEL_ATTACHMENT_BATCHES = TypeAdapter(dict[ProjectName, list[ModelAttachmentDifference]])
MODEL_DIFFERENCE_BATCHES = TypeAdapter(dict[ProjectName, list[ModelDifference]])


def is_trial_binding(name: str) -> bool:
    """Name reserved for a generated Set or intervention input."""
    token = name.removeprefix("set-")
    return token != name and len(token) == 28 and all(char in "0123456789abcdef" for char in token)


def is_generated_attachment_node(name: str, type_id: str) -> bool:
    """Recognize stateless occurrences owned by attachment lowering."""
    if is_trial_binding(name):
        return type_id in {"feedbax.Input", "feedbax.Switch"}
    for prefix, types in (
        ("_model_attachment_", {
            "feedbax.Scale", "feedbax.Saturation", "feedbax.ConstantInput",
            "feedbax.Sum", "feedbax.Multiply", "feedbax.Clamp",
        }),
        ("_model_intervention_constant_", {"feedbax.Constant"}),
        ("_model_hold_", {"feedbax.Input"}),
        ("_model_delay_", {"feedbax.Input"}),
        ("_model_output_", {"feedbax.Input"}),
        ("_model_input_", {"feedbax.Input"}),
    ):
        if name.startswith(prefix):
            token = name.removeprefix(prefix)
            return type_id in types and len(token) == 64 and all(char in "0123456789abcdef" for char in token)
    return False
