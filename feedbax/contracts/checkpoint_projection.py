"""Deterministic continuation inputs and exact checkpoint occurrence provenance."""

from typing import Literal

from pydantic import Field, JsonValue

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.request_refs import Name, ReferenceValue, Sha256
from feedbax.contracts.studio_training import StudioTrainContinuation, StudioTrainSlot
from feedbax.contracts.training import LrScheduleSpec, OptimizerSpec

STANDARD_TRAIN_STATE_MEDIA_TYPE = "application/vnd.feedbax.standard-train-state+safetensors"


class CheckpointOccurrenceRef(ReferenceValue):
    run_id: Name
    transaction_id: Name
    manifest_sha256: Sha256


class CheckpointClock(ReferenceValue):
    completed_batches: int = Field(ge=0)
    segment_start_batch: int = Field(ge=0)
    phase: Name
    barrier: Name
    program_step: int = Field(ge=0)
    inner_step: int | None
    outer_step: int | None
    schedule_origin_step: int | None
    phase_boundary: tuple[Name | None] | None


class ContinuationSlot(ReferenceValue):
    content: ContentObjectRef
    array_descriptor_sha256: Sha256
    semantic_abi_sha256: Sha256


class CarriedOptimizer(ReferenceValue):
    kind: Name
    parameters: dict[str, JsonValue]
    grad_clip: float | None

    @classmethod
    def from_spec(cls, optimizer: OptimizerSpec, grad_clip: float | None) -> "CarriedOptimizer":
        return cls(
            kind=optimizer.type,
            parameters={key: value for key, value in optimizer.params.items() if key != "learning_rate"},
            grad_clip=grad_clip,
        )


class ContinuationEncoding(ReferenceValue):
    schema_version: Literal["feedbax.train.checkpoint-projection.v1"] = (
        "feedbax.train.checkpoint-projection.v1"
    )
    clock: CheckpointClock
    learning_rate: LrScheduleSpec | float
    carried_optimizer: CarriedOptimizer | None = Field(default=None, exclude_if=lambda value: value is None)
    model: SemanticModelEncoding
    model_semantic_abi_sha256: Sha256
    batch_counter: ContinuationSlot
    optimizer: ContinuationSlot | None = Field(default=None, exclude_if=lambda value: value is None)
    prng: ContinuationSlot | None = Field(default=None, exclude_if=lambda value: value is None)

    @property
    def array_slots(self) -> dict[StudioTrainSlot, ContinuationSlot]:
        slots: dict[StudioTrainSlot, ContinuationSlot] = {"batch_counter": self.batch_counter}
        if self.optimizer is not None:
            slots["optimizer"] = self.optimizer
        if self.prng is not None:
            slots["prng"] = self.prng
        return slots


class ContinuationOccurrence(ReferenceValue):
    provenance: CheckpointOccurrenceRef
    clock: CheckpointClock
    learning_rate: LrScheduleSpec | float
    carried_optimizer: CarriedOptimizer
    model_semantic_abi_sha256: Sha256
    batch_counter: ContinuationSlot
    optimizer: ContinuationSlot
    prng: ContinuationSlot

    def project(
        self, model: SemanticModelEncoding, policy: StudioTrainContinuation
    ) -> ContinuationEncoding:
        slots = policy.source_slots
        return ContinuationEncoding(
            clock=self.clock,
            learning_rate=self.learning_rate,
            carried_optimizer=self.carried_optimizer if "optimizer" in slots else None,
            model=model,
            model_semantic_abi_sha256=self.model_semantic_abi_sha256,
            batch_counter=self.batch_counter,
            optimizer=self.optimizer if "optimizer" in slots else None,
            prng=self.prng if "prng" in slots else None,
        )
