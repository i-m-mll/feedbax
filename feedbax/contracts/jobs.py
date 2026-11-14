"""Studio's typed job service and event-stream contract."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field, FiniteFloat, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import StepKind
from feedbax.contracts.project_inputs import StepReference, SubsetTerms
from feedbax.contracts.project_inputs import CheckpointRule
from feedbax.contracts.project_sources import SourceKind
from feedbax.contracts.validation_placement import ValidationPlacement


if TYPE_CHECKING:
    from feedbax.contracts.project_resolution import ProjectDiagnostic


Token = Annotated[str, Field(min_length=1, max_length=128)]
_DigestPattern = r"^[a-f0-9]{64}$"
StopMode = Literal["checkpoint", "force"]
JobAction = Literal["checkpoint", "force", "resume"]


class JobStatus(StrEnum):
    NOT_RUN = "not_run"
    CANNOT_RUN = "cannot_run"
    STALE = "stale"
    QUEUED = "queued"
    AWAITING_AUTHORIZATION = "awaiting_authorization"
    BLOCKED = "blocked"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COLLECTING = "collecting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    CORRUPT = "corrupt"


class JobError(StrictModel):
    code: str
    message: str
    phase: str | None = None
    path: str | None = None


class JobProgress(StrictModel):
    completed_updates: int = Field(ge=0)
    total_updates: int = Field(ge=0)


class JobReceipt(StrictModel):
    manifest_id: str
    manifest_sha256: str = Field(pattern=_DigestPattern)


class StudioTrainCheckpointSource(StrictModel):
    job_id: str
    request_id: str
    completed_updates: int = Field(ge=0)


class StudioSavedTrainAttempt(StrictModel):
    job_id: str
    status: JobStatus
    reason: str | None
    progress: JobProgress | None
    successful_receipt: JobReceipt | None
    continues: str | None
    checkpoint_source: StudioTrainCheckpointSource | None
    stopped_checkpoint: int | None = Field(default=None, gt=0, exclude_if=lambda value: value is None)


class StudioSavedTrainKey(StrictModel):
    variant: str
    step: str
    row: str


class StudioEvaluationCheckpointGap(StrictModel):
    update: int = Field(gt=0)
    reason: Literal["checkpoint_unavailable"]


class StudioEvaluationCheckpointSeries(StrictModel):
    subject: StudioSavedTrainKey
    requested: list[int]


class StudioEvaluationPendingCheckpoint(StrictModel):
    producer_job_id: str
    producer_request_id: str
    state: Literal["queued", "assigned", "running", "stopping", "resumable"]
    next_update: int | None = Field(default=None, gt=0, exclude_if=lambda value: value is None)
    recovery_update: int | None = Field(default=None, gt=0, exclude_if=lambda value: value is None)


class StudioEvaluationPointProgress(StudioTrainCheckpointSource):
    publication_cursor: int = Field(gt=0)
    transaction_root_sha256: str = Field(pattern=_DigestPattern)
    status: Literal["scored", "failed"]


class StudioEvaluationCheckpointAttempt(StrictModel):
    job_id: str
    placement: ValidationPlacement
    pending: StudioEvaluationPendingCheckpoint | None = Field(default=None, exclude_if=lambda value: value is None)
    points: list[StudioEvaluationPointProgress] = Field(default_factory=list, exclude_if=lambda value: not value)


class StudioBoundBestSubject(StrictModel):
    key: str
    variant: str
    row: str
    consumer_job_id: str
    producer_job_id: str
    producer_request_id: str
    completed_updates: int
    selection_sha256: str = Field(pattern=_DigestPattern)


class StudioSavedEvaluationResult(StrictModel):
    job_id: str
    receipt: JobReceipt
    checkpoint_source: StudioTrainCheckpointSource | None
    size_bytes: int = Field(ge=0)
    checkpoint_sources: list[StudioTrainCheckpointSource] | None = Field(default=None, exclude_if=lambda value: value is None)
    checkpoint_gaps: list[StudioEvaluationCheckpointGap] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    placement: Literal["ordinary", "live"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    bound_subject: StudioBoundBestSubject | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioSavedEvaluationRow(StrictModel):
    key: StudioSavedTrainKey
    request_id: str
    status: JobStatus
    reason: str | None
    can_run: bool
    trials: list[str]
    result: StudioSavedEvaluationResult | None
    subject_rank_key: str | None = Field(default=None, exclude_if=lambda value: value is None)
    checkpoint_series: StudioEvaluationCheckpointSeries | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    checkpoint_attempt: StudioEvaluationCheckpointAttempt | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    checkpoint_rule: CheckpointRule | None = Field(default=None, exclude_if=lambda value: value is None)
    bound_subject: StudioBoundBestSubject | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioSavedEvaluationRead(StrictModel):
    revision: str = Field(pattern=_DigestPattern)
    key: StudioSavedTrainKey
    job_id: str


class StudioSavedEvaluationReplayRequest(StudioSavedEvaluationRead):
    trial: str
    checkpoint: int | None = Field(default=None, ge=0)


class StudioEvaluationCheckpoint(StudioTrainCheckpointSource):
    transaction: JobReceipt
    transaction_root_sha256: str = Field(pattern=_DigestPattern)
    binding_sha256: str = Field(pattern=_DigestPattern)
    publication_cursor: int = Field(gt=0)


class StudioEvaluationObjectiveTerm(StrictModel):
    path: str = Field(pattern=r"^/(?:[^~/]|~[01])*$")
    name: str
    value: FiniteFloat
    coverage: int = Field(strict=True, ge=0)


class StudioEvaluationObjectivePoint(StrictModel):
    checkpoint: StudioEvaluationCheckpoint
    total: FiniteFloat
    terms: list[StudioEvaluationObjectiveTerm]


class StudioEvaluationObjectiveSeries(StrictModel):
    job_id: str
    request_id: str
    receipt: JobReceipt
    lineage: list[str]
    condition: str | None = Field(pattern=_DigestPattern)
    trials: list[str]
    seed: int
    objective: str = Field(pattern=_DigestPattern)
    requested: list[int]
    points: list[StudioEvaluationObjectivePoint]
    gaps: list[StudioEvaluationCheckpointGap]


class StudioSavedTrainResult(StrictModel):
    request_id: str
    job_id: str
    receipt: JobReceipt
    progress: JobProgress | None


class StudioSavedTrainChange(StrictModel):
    path: str
    before: JsonValue
    after: JsonValue


class StudioSavedTrainRow(StrictModel):
    key: StudioSavedTrainKey
    label: str
    request_id: str
    attempts: list[StudioSavedTrainAttempt]
    status: JobStatus
    reason: str | None
    result: StudioSavedTrainResult | None
    previous: StudioSavedTrainResult | None
    change: StudioSavedTrainChange | None
    previous_attempts: list[StudioSavedTrainAttempt] = Field(default_factory=list, exclude_if=lambda value: not value)


class StudioProjectWindowMapCoordinate(StrictModel):
    value_id: str
    label: str | None
    value: JsonValue


class StudioProjectWindowMapRow(StrictModel):
    key: StudioSavedTrainKey
    label: str
    coordinates: dict[str, StudioProjectWindowMapCoordinate]
    member_key: str | None = Field(default=None, exclude_if=lambda value: value is None)
    status: JobStatus
    reason: str | None = Field(default=None, exclude_if=lambda value: value is None)
    evaluation_conditions: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioBestRankGroup(StrictModel):
    coordinates: dict[str, str]
    ranks: list[str]


class StudioProjectWindowMapSubset(StrictModel):
    terms: SubsetTerms
    count: int
    total: int
    document: str | None
    members: list[StudioSavedTrainKey] | None = Field(default=None, exclude_if=lambda value: value is None)
    ranks: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    rank_groups: list[StudioBestRankGroup] = Field(default_factory=list, exclude_if=lambda value: not value)


class StudioProjectWindowMapInput(StrictModel):
    id: str
    source_step_id: str
    subset: StudioProjectWindowMapSubset | None
    checkpoint: CheckpointRule | None = Field(default=None, exclude_if=lambda value: value is None)
    checkpoint_axis: str | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioProjectWindowMapStep(StrictModel):
    variant: str
    step_id: str
    kind: StepKind
    group: str | None
    inputs: list[StudioProjectWindowMapInput]
    document_ids: dict[str, str]
    document_uses: list[str] = Field(default_factory=list)
    status_counts: dict[str, int]


class StudioProjectWindowMapEdge(StrictModel):
    from_key: StudioSavedTrainKey
    to_key: StudioSavedTrainKey


class StudioProjectWindowMapProblem(StrictModel):
    code: str
    path: str
    detail: str
    variant: str | None
    step: str | None
    row: str | None
    document: str | None
    element: str | None = Field(default=None, exclude_if=lambda value: value is None)
    document_path: str | None = Field(default=None, exclude_if=lambda value: value is None)
    model_address: ModelAddress | None = Field(default=None, exclude_if=lambda value: value is None)

    address: str | None = Field(default=None, exclude_if=lambda value: value is None)
    layer: str | None = Field(default=None, exclude_if=lambda value: value is None)
    authored_path: str | None = Field(default=None, exclude_if=lambda value: value is None)

    @classmethod
    def from_diagnostic(cls, diagnostic: ProjectDiagnostic, *, declaration: str) -> StudioProjectWindowMapProblem:
        source = f"steps/{diagnostic.step}.json" if diagnostic.step else declaration
        problem = cls.model_validate(diagnostic, from_attributes=True)
        problem.address = diagnostic.authored_file or source
        problem.layer = diagnostic.layer
        return problem


class StudioPlanRequestAddress(StrictModel):
    key: StudioSavedTrainKey
    element: str | None = None


class StudioPlanRequest(StudioPlanRequestAddress):
    request_id: str


class StudioPlanReviewSource(StrictModel):
    kind: SourceKind
    name: str
    path: str


class StudioPlanReviewEntry(StudioPlanRequest):
    state: Literal["unchanged", "changed", "added", "removed"]
    path: str | None = None
    upstream: StudioPlanRequestAddress | None = None
    before_source: StudioPlanReviewSource | None = Field(default=None, exclude_if=lambda value: value is None)
    after_source: StudioPlanReviewSource | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioProjectWindowMapFigure(StrictModel):
    key: StudioSavedTrainKey
    name: str
    data: list[str]
    caption: str | None


class StudioProjectWindowMapElement(StrictModel):
    key: StudioSavedTrainKey
    name: str
    kind: Literal["analyze", "render"]
    request_id: str
    status: Literal["not_run"]
    figure: StepReference | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioProjectWindowMap(StrictModel):
    variants: list[str]
    steps: list[StudioProjectWindowMapStep]
    rows: list[StudioProjectWindowMapRow]
    edges: list[StudioProjectWindowMapEdge]
    problems: list[StudioProjectWindowMapProblem]
    requests: list[StudioPlanRequest] = Field(default_factory=list)
    evaluations: list[StudioSavedEvaluationRow] = Field(default_factory=list)
    analysis_elements: list[StudioProjectWindowMapElement] = Field(default_factory=list, exclude_if=lambda value: not value)
    analysis_figures: list[StudioProjectWindowMapFigure] = Field(default_factory=list, exclude_if=lambda value: not value)


class StudioSavedTrainRuns(StrictModel):
    revision: str
    rows: list[StudioSavedTrainRow]
    window_map: StudioProjectWindowMap | None = None
    review: list[StudioPlanReviewEntry] | None = None


class StudioSavedTrainRunRequest(StrictModel):
    revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_keys: list[StudioSavedTrainKey] = Field(min_length=1)
    token: Token


class JobOutcome(StrictModel):
    outcome: Literal["succeeded", "stopped", "failed", "interrupted", "blocked"]
    error: JobError | None = None
    receipt: JobReceipt | None = None
    signal: int | None = None
    exit: int | None = None


class StudioSubmissionActor(StrictModel):
    model_config = ConfigDict(frozen=True)

    client: Literal["studio", "cli"]
    label: str | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioActivityJob(StrictModel):
    job_id: str
    project_id: str
    request_id: str
    kind: Literal["train", "evaluate"]
    run_key: str
    stage_id: str
    scenario_id: str
    label: str
    coordinates: dict[str, JsonValue]
    status: JobStatus
    phase: str | None = None
    reason: str | None = None
    can_resume: bool = False
    allowed_actions: list[JobAction]
    continues: str | None = None
    progress: JobProgress | None = None
    eta_seconds: float | None = None
    outcome: JobOutcome | None = None
    earlier_definition: bool = False
    seen: bool = False
    actor: StudioSubmissionActor | None = Field(default=None, exclude_if=lambda value: value is None)
    placement: Literal["live", "ordinary"] | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioLocalSession(StrictModel):
    id: Literal["local"] = "local"
    state: Literal["open"] = "open"


class StudioJobSnapshot(StrictModel):
    jobs: list[StudioActivityJob]
    cursor: int = Field(ge=0)
    oldest_cursor: int = Field(ge=0)
    sessions: list[StudioLocalSession]


class StudioSubmissionItem(StrictModel):
    run_key: str
    job_id: str | None = None
    disposition: Literal["admitted", "attached", "reused", "rejected"]
    reason: str | None = None


class StudioJobSubmission(StrictModel):
    id: str
    token: str
    project_id: str
    items: list[StudioSubmissionItem]
    actor: StudioSubmissionActor | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioJobStopRequest(StrictModel):
    job_ids: list[str] = Field(min_length=1)
    mode: StopMode


class StudioJobStopItem(StrictModel):
    job_id: str
    job: StudioActivityJob | None = None
    error: JobError | None = None


class StudioJobResumeRequest(StrictModel):
    token: Token


class StudioJobEvent(StrictModel):
    cursor: int = Field(ge=1)
    job_id: str
    project_id: str
    request_id: str
    kind: str
    data: dict[str, JsonValue]
    activity: StudioActivityJob | None = None


class StudioJobEventHistory(StrictModel):
    cursor: int = Field(ge=0)
    oldest_cursor: int = Field(ge=0)
    events: list[StudioJobEvent]


class StudioEventWatermark(StrictModel):
    cursor: int = Field(ge=0)


class StudioEventReset(StudioEventWatermark):
    oldest_cursor: int = Field(ge=0)
