"""Deterministic continuation inputs and exact checkpoint occurrence provenance."""

from typing import Literal

from pydantic import Field

from feedbax.contracts.method_checkpoint import MethodCheckpointClock, MethodCheckpointExtra
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.request_refs import Name, ReferenceValue, Sha256
from feedbax.contracts.request_specs import TrainContinuation


class CheckpointOccurrenceRef(ReferenceValue):
    run_id: Name
    transaction_id: Name
    manifest_sha256: Sha256


class CarriedMethodState(ReferenceValue):
    method_key: str
    extra: MethodCheckpointExtra


class ContinuationEncoding(ReferenceValue):
    schema_version: Literal["feedbax.train.checkpoint-projection.v2"] = (
        "feedbax.train.checkpoint-projection.v2"
    )
    completed_updates: int
    schedule_origin: int | None = Field(default=None, exclude_if=lambda value: value is None)
    seed: int | None = Field(default=None, exclude_if=lambda value: value is None)
    model: SemanticModelEncoding
    state: CarriedMethodState | None = Field(default=None, exclude_if=lambda value: value is None)


class ContinuationOccurrence(ReferenceValue):
    provenance: CheckpointOccurrenceRef
    clock: MethodCheckpointClock
    state: CarriedMethodState

    def project(self, model: SemanticModelEncoding, policy: TrainContinuation) -> ContinuationEncoding:
        return ContinuationEncoding(
            completed_updates=self.clock.completed_updates,
            schedule_origin=self.clock.schedule_origin if policy.schedule == "continue" else None,
            seed=self.clock.seed if policy.rng == "carry" else None,
            model=model,
            state=self.state if policy.state == "preserve" else None,
        )
