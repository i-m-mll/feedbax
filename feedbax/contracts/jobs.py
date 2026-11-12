"""Studio's typed job service and event-stream contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.graph import GraphSpec, StudioWorkspaceSpec
from feedbax.contracts.project import StepKind
from feedbax.contracts.project_inputs import SubsetTerms


Token = Annotated[str, Field(min_length=1, max_length=128)]
StopMode = Literal["checkpoint", "force"]
JobAction = Literal["checkpoint", "force", "resume"]


class StudioRunSelection(StrictModel):
    stage_id: str = Field(min_length=1)
    run_keys: list[str] | None = Field(default=None, min_length=1)


class StudioRunPreviewRequest(StrictModel):
    project_id: str = Field(min_length=1, max_length=128)
    workspace: StudioWorkspaceSpec
    graph: GraphSpec
    selections: list[StudioRunSelection] = Field(min_length=1)


class StudioJobSubmissionRequest(StudioRunPreviewRequest):
    token: Token
    policy: Literal["out_of_date", "all"] = "out_of_date"


class StudioPreviewRow(StrictModel):
    run_key: str
    request_id: str
    stage_id: str
    scenario_id: str
    label: str
    coordinates: dict[str, JsonValue]


class StudioPreviewIssue(StrictModel):
    stage_id: str
    path: str
    message: str


class StudioRunPreview(StrictModel):
    rows: list[StudioPreviewRow]
    issues: list[StudioPreviewIssue]


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
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


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


class StudioSavedTrainKey(StrictModel):
    variant: str
    step: str
    row: str


class StudioSavedEvaluationResult(StrictModel):
    job_id: str
    receipt: JobReceipt
    checkpoint_source: StudioTrainCheckpointSource | None
    size_bytes: int = Field(ge=0)
    checkpoint_sources: list[StudioTrainCheckpointSource] | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioSavedEvaluationRow(StrictModel):
    key: StudioSavedTrainKey
    request_id: str
    status: JobStatus
    reason: str | None
    can_run: bool
    trials: list[str]
    result: StudioSavedEvaluationResult | None


class StudioSavedEvaluationReplayRequest(StrictModel):
    revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    key: StudioSavedTrainKey
    job_id: str
    trial: str
    checkpoint: int | None = Field(default=None, ge=0)


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


class StudioProjectWindowMapCoordinate(StrictModel):
    value_id: str
    label: str | None
    value: JsonValue


class StudioProjectWindowMapRow(StrictModel):
    key: StudioSavedTrainKey
    label: str
    coordinates: dict[str, StudioProjectWindowMapCoordinate]


class StudioProjectWindowMapSubset(StrictModel):
    terms: SubsetTerms
    count: int
    total: int
    document: str | None


class StudioProjectWindowMapInput(StrictModel):
    id: str
    source_step_id: str
    subset: StudioProjectWindowMapSubset | None


class StudioProjectWindowMapStep(StrictModel):
    variant: str
    step_id: str
    kind: StepKind
    group: str | None
    inputs: list[StudioProjectWindowMapInput]
    document_ids: dict[str, str]
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


class StudioProjectWindowMap(StrictModel):
    variants: list[str]
    steps: list[StudioProjectWindowMapStep]
    rows: list[StudioProjectWindowMapRow]
    edges: list[StudioProjectWindowMapEdge]
    problems: list[StudioProjectWindowMapProblem]
    evaluations: list[StudioSavedEvaluationRow] = Field(default_factory=list)


class StudioSavedTrainRuns(StrictModel):
    revision: str
    rows: list[StudioSavedTrainRow]
    window_map: StudioProjectWindowMap | None = None


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


class StudioActivityJob(StrictModel):
    job_id: str
    project_id: str
    request_id: str
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
