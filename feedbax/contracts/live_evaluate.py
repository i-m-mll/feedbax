"""Durable delivery records for checkpoint-bound live Evaluate."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.base import StrictModel
from feedbax.contracts.code_validity import ComputationUnavailable
from feedbax.contracts.computation_validity import ComputationTrace, DriverIdentity, EvaluatePrefix
from feedbax.contracts.dataset import DatasetProductRef
from feedbax.contracts.jobs import JobError, StudioEvaluationCheckpointGap
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.request_refs import Sha256
from feedbax.contracts.rollout import ResolvedRolloutMember
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint

LIVE_EVALUATE_POINT = "evaluation_point"
EVALUATE_DRIVER = DriverIdentity(name="evaluate.rollout", revision=3479529567574847)
EVALUATE_PREFIX_STEPS = 5


def rollout_slot(coordinate: int | None, replica: str) -> str:
    return pointer(("initial" if coordinate is None else str(coordinate), replica))


def require_evaluation_coverage(
    computation: ComputationTrace | ComputationUnavailable | None, slots: tuple[str, ...],
) -> None:
    if not slots and computation is None:
        return
    if computation is not None and computation.driver.name == EVALUATE_DRIVER.name and (
        computation.completed_steps == len(slots)
    ):
        if isinstance(computation, ComputationUnavailable):
            if computation.slots == slots:
                return
        elif (
            tuple(start.slot for start in computation.starts) == slots
            and tuple(program.slot for program in computation.programs) == slots
            and isinstance(computation.prefix, EvaluatePrefix)
            and len(computation.prefix.outputs) == min(EVALUATE_PREFIX_STEPS, len(slots))
        ):
            return
    raise ValueError("Evaluate computation does not cover its exact checkpoint and replica slots")


class ScoredEvaluationPoint(StrictModel):
    status: Literal["scored"] = "scored"
    states: DatasetProductRef
    member: ResolvedRolloutMember
    objective: Sha256
    total: float = Field(allow_inf_nan=False)
    evidence: ContentObjectRef


class FailedEvaluationPoint(StrictModel):
    status: Literal["failed"] = "failed"
    error: JobError


class LiveEvaluationPoint(StrictModel):
    evaluation_job_id: str
    checkpoint: PublishedTrainingCheckpoint
    result: Annotated[ScoredEvaluationPoint | FailedEvaluationPoint, Field(discriminator="status")]

    @property
    def root_id(self) -> str:
        return f"evaluate-point:{self.evaluation_job_id}:{self.checkpoint.pointer.transaction_id}"



class BoundLiveEvaluationPoint(LiveEvaluationPoint):
    publication_cursor: int = Field(gt=0)


class SelectedLiveEvaluationPoint(StrictModel):
    point: BoundLiveEvaluationPoint
    producer_job_id: str
    producer_request_id: str
    binding_sha256: Sha256


class LiveEvaluationObservation(StrictModel):
    """The Host-selected evidence for one active Evaluate attempt."""

    lineage: list[str] = Field(min_length=1)
    gaps: tuple[StudioEvaluationCheckpointGap, ...]
    points: tuple[SelectedLiveEvaluationPoint, ...]
