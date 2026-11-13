"""Durable delivery records for checkpoint-bound live Evaluate."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import DatasetProductRef
from feedbax.contracts.jobs import JobError
from feedbax.contracts.request_refs import Sha256
from feedbax.contracts.rollout import ResolvedRolloutMember
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint

LIVE_EVALUATE_POINT = "evaluation_point"


class ScoredEvaluationPoint(StrictModel):
    status: Literal["scored"] = "scored"
    states: DatasetProductRef
    member: ResolvedRolloutMember
    objective: Sha256
    total: float = Field(allow_inf_nan=False)


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
