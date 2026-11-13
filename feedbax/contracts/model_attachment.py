"""Authored Model attachment differences, shared by Project and Studio."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import AttachmentElement, ModelAddress, ModelElementId
from feedbax.contracts.model_document import WireAttachment
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_refinements import validate_cross_field_refinements


class AddModelAttachment(StrictModel):
    op: Literal["add"]
    owner: ModelAddress
    attachment: WireAttachment
    after: ModelElementId | None

    @model_validator(mode="after")
    def _owner(self) -> "AddModelAttachment":
        validate_cross_field_refinements("AddModelAttachment", self)
        return self

    @property
    def address(self) -> ModelAddress:
        return self.owner.model_copy(update={"element": AttachmentElement(
            kind="attachment", parent=self.owner.element, id=self.attachment.id,
        )})


class ExistingModelAttachment(StrictModel):
    address: ModelAddress

    @model_validator(mode="after")
    def _attachment(self) -> "ExistingModelAttachment":
        validate_cross_field_refinements("ExistingModelAttachment", self)
        return self


class ChangeModelAttachment(ExistingModelAttachment):
    op: Literal["change"]
    attachment: WireAttachment
    after: ModelElementId | None = None

    @model_validator(mode="after")
    def _id(self) -> "ChangeModelAttachment":
        validate_cross_field_refinements("ChangeModelAttachment", self)
        return self


class RemoveModelAttachment(ExistingModelAttachment):
    op: Literal["remove"]


class SetField(StrictModel):
    op: Literal["set_field"]
    address: ModelAddress
    value: JsonValue


ModelAttachmentDifference = Annotated[
    AddModelAttachment | ChangeModelAttachment | RemoveModelAttachment, Field(discriminator="op")
]
ModelDifference = Annotated[
    AddModelAttachment | ChangeModelAttachment | RemoveModelAttachment | SetField,
    Field(discriminator="op"),
]

MODEL_ATTACHMENTS = "model_attachments"
MODEL_ATTACHMENT_BATCHES = TypeAdapter(dict[ProjectName, list[ModelAttachmentDifference]])
MODEL_DIFFERENCE_BATCHES = TypeAdapter(dict[ProjectName, list[ModelDifference]])


def is_generated_attachment_node(name: str, type_id: str) -> bool:
    """Recognize stateless occurrences owned by attachment lowering."""
    for prefix, types in (
        ("_model_attachment_", {"feedbax.Scale", "feedbax.Saturation"}),
        ("_model_delay_", {"feedbax.Input"}),
        ("_model_output_", {"feedbax.Input"}),
    ):
        if name.startswith(prefix):
            token = name.removeprefix(prefix)
            return type_id in types and len(token) == 64 and all(char in "0123456789abcdef" for char in token)
    return False
