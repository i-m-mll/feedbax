"""Location-free kernel references, shared by requests and receipts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, field_validator, model_validator

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


class UpdateCheckpoint(ReferenceValue):
    at: BatchCoordinate


class UpdateCadence(ReferenceValue):
    every: BatchCoordinate
    from_: BatchCoordinate | None = Field(default=None, alias="from")
    through: BatchCoordinate | None = None

    def coordinates(self, horizon: int) -> tuple[int, ...]:
        start = self.from_ or 1
        end = min(self.through or horizon, horizon)
        first = ((start + self.every - 1) // self.every) * self.every
        count = (end - first) // self.every + 1
        if not 1 <= count <= MAX_CHECKPOINT_MEMBERS:
            raise ValueError("update cadence must select one to 64 completed updates")
        return tuple(range(first, end + 1, self.every))


type UpdateRule = (
    UpdateCadence | Annotated[tuple[UpdateCheckpoint, ...], Field(min_length=1, max_length=MAX_CHECKPOINT_MEMBERS)]
)


class UpdateSeriesMember(ReferenceValue):
    """Requested updates of one Train row, with explicit gaps at execution."""

    kind: Literal["update_series"] = "update_series"
    rule: UpdateRule
    updates: tuple[BatchCoordinate, ...] = Field(min_length=1, max_length=MAX_CHECKPOINT_MEMBERS)

    @model_validator(mode="after")
    def _compiled_coordinates(self):
        updates = finite_coordinates(self.updates)
        if isinstance(self.rule, UpdateCadence):
            expected = self.rule.coordinates(updates[-1])
        else:
            expected = tuple(point.at for point in self.rule)
        if updates != expected:
            raise ValueError("requested updates differ from the update-series rule")
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
    BatchMember | BatchSetMember | UpdateSeriesMember | SlotMember | NamedMember | BarrierMember | LastMember | BestMember,
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

EffectiveBindingId = Annotated[str, StringConstraints(pattern=r"^effective:[0-9a-f]{64}$")]


class EffectiveInputMember(ReferenceValue):
    """The exact input content and semantic descriptor presented to a computation."""

    key: Name
    content: ContentRef
    descriptor: JsonValue = None


EffectiveInputs = Mapping[Name, tuple[EffectiveInputMember, ...]]
