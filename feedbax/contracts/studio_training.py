"""Governed authored contract for Studio worker training launches."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.checkpoints import CheckpointContinuationRequest, CheckpointLatestPointer
from feedbax.contracts.training import MethodRefSpec, OptimizerSpec, TrainingRunSpec

STUDIO_GRAPH_METHOD = MethodRefSpec(package="feedbax", name="studio_graph", version="v1")
STUDIO_GRAPH_METHOD_PAYLOAD_ID = "feedbax.spec.training_method.studio_graph"
STUDIO_GRAPH_METHOD_PAYLOAD_VERSION = f"{STUDIO_GRAPH_METHOD_PAYLOAD_ID}.v1"
STUDIO_INDEPENDENT_KEY_SCHEDULE = "independent_v1"

StudioTrainSlot = Literal["model", "optimizer", "prng", "batch_counter"]


class StudioTrainContinuation(StrictModel):
    """Explicit custody and clock policy for a stateful Studio successor."""

    mode: Literal["continue_from"]
    progress: CheckpointContinuationRequest
    optimizer: Literal["preserve", "reset"]
    schedule: Literal["continue", "restart"]
    rng: Literal["carry", "reset"]
    cursor: Literal["carry"]

    @property
    def source_slots(self) -> frozenset[StudioTrainSlot]:
        slots: set[StudioTrainSlot] = {"model", "batch_counter"}
        if self.optimizer == "preserve":
            slots.add("optimizer")
        if self.rng == "carry":
            slots.add("prng")
        return frozenset(slots)


class StudioGraphMethodPayload(StrictModel):
    """Numerical settings for one checkpointable Studio graph training run."""

    optimizer: OptimizerSpec
    target_total: int = Field(gt=0)
    seed: int = Field(default=0, ge=0, lt=2**32, strict=True)
    key_schedule: Literal["independent_v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    checkpoint_initialization: Literal["initialize_from"] | StudioTrainContinuation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    def require_supported_launch(self, *, resume: bool) -> None:
        # A resume takes its model and carried update key from checkpoint custody.
        # An unstarted legacy seed-zero request would silently change numerics.
        if not resume and self.key_schedule is None and self.seed == 0:
            raise ValueError("Create a new Studio request to start training with seed 0")

    @property
    def continuation(self) -> StudioTrainContinuation | None:
        policy = self.checkpoint_initialization
        return policy if isinstance(policy, StudioTrainContinuation) else None

    @model_validator(mode="after")
    def _validate_schedule(self) -> StudioGraphMethodPayload:
        schedule = self.optimizer.lr_schedule
        policy = self.continuation
        if policy is None:
            if schedule is not None and (schedule.kind != "constant" or schedule.origin.kind != "run_start"):
                raise ValueError("/optimizer/lr_schedule: independent Train requires a run-start constant rate")
            return self
        origin = "segment_start" if policy.schedule == "restart" else "run_start"
        if (schedule is None and origin != "run_start") or (
            schedule is not None and schedule.origin.kind != origin
        ):
            raise ValueError(f"/optimizer/lr_schedule/origin: expected {origin} for {policy.schedule}")
        if schedule is not None and schedule.kind != "constant" and policy.optimizer != "preserve":
            raise ValueError("/checkpoint_initialization/optimizer: nonconstant schedule requires preserved count")
        if policy.progress.target_total != self.target_total:
            raise ValueError("/checkpoint_initialization/progress: continuation total differs from target_total")
        return self


class PublishedTrainingCheckpoint(StrictModel):
    checkpoint_root: Path
    completed_updates: int = Field(ge=0)
    pointer: CheckpointLatestPointer


class StudioTrainingJobInput(StrictModel):
    spec: TrainingRunSpec
    project_id: str
    request_id: str
    manifest_root: Path
    bootstrap_modules: tuple[str, ...] = ()
    resume: PublishedTrainingCheckpoint | None = None
