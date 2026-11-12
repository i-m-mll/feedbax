"""Location-free kernel references, shared by requests and receipts."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from feedbax.checkpoint_coordinates import (
    MAX_CHECKPOINT_COORDINATE, MAX_CHECKPOINT_MEMBERS, cadence_coordinates, finite_coordinates,
)

RequestKind = Literal["train", "evaluate", "analyze", "render", "report"]
RequestId = Annotated[
    str, StringConstraints(pattern=rf"^({'|'.join(get_args(RequestKind))}):[0-9a-f]{{64}}$")
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(min_length=1)]


class ReferenceValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BatchMember(ReferenceValue):
    kind: Literal["batch"] = "batch"
    batch: int = Field(ge=0, strict=True)


BatchCoordinate = Annotated[int, Field(gt=0, le=MAX_CHECKPOINT_COORDINATE, strict=True)]


class CheckpointCadence(ReferenceValue):
    every_k: BatchCoordinate
    through: BatchCoordinate

    def coordinates(self) -> tuple[int, ...]:
        return cadence_coordinates(self.every_k, self.through)

    @model_validator(mode="after")
    def _bounded(self):
        self.coordinates()
        return self


class CheckpointBatches(ReferenceValue):
    batches: tuple[BatchCoordinate, ...] = Field(min_length=1, max_length=MAX_CHECKPOINT_MEMBERS)

    @field_validator("batches")
    @classmethod
    def _ordered_distinct(cls, value):
        return finite_coordinates(value)


class BatchSetMember(CheckpointBatches):
    """Fixed batches at the standard Train after-batch barrier, in authored order."""

    kind: Literal["batch_set"] = "batch_set"
    cadence: CheckpointCadence | None = None

    @model_validator(mode="after")
    def _compiled_coordinates(self):
        if self.cadence is not None and self.batches != self.cadence.coordinates():
            raise ValueError("checkpoint batches differ from the bounded cadence")
        return self


class SlotMember(ReferenceValue):
    kind: Literal["slot"] = "slot"
    slot: Name


class NamedMember(ReferenceValue):
    kind: Literal["member"] = "member"
    member: Name


class BarrierMember(ReferenceValue):
    kind: Literal["barrier"] = "barrier"
    name: Name


class LastMember(ReferenceValue):
    kind: Literal["last"] = "last"


class BestMember(ReferenceValue):
    kind: Literal["best_by"] = "best_by"
    metric: Name


ProductSelector = Annotated[
    BatchMember | BatchSetMember | SlotMember | NamedMember | BarrierMember | LastMember | BestMember,
    Field(discriminator="kind"),
]


class ProductRef(ReferenceValue):
    """An unresolved product of the named request."""

    identity_family: ClassVar[str] = "product"

    request_id: RequestId
    role: Name
    select: ProductSelector | None = None


class ContentRef(ReferenceValue):
    """Resolved immutable content, or a digest-bound input."""

    identity_family: ClassVar[str] = "content"

    sha256: Sha256
    size: int = Field(ge=0, strict=True)
    media_type: Name | None = None
    tree: bool = Field(default=False, strict=True)


class InputMember(ReferenceValue):
    key: Name
    reference: ProductRef | ContentRef


RequestInputs = dict[Name, tuple[InputMember, ...]]
