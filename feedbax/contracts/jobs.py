"""Studio's typed job service and event-stream contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.graph import GraphSpec, StudioWorkspaceSpec


Token = Annotated[str, Field(min_length=1, max_length=128)]
StopMode = Literal["checkpoint", "force"]
JobAction = Literal["checkpoint", "force", "resume"]


class StudioRunSelection(StrictModel):
    stage_id: str = Field(min_length=1)
    run_keys: list[str] | None = Field(default=None, min_length=1)


class StudioJobSubmissionRequest(StrictModel):
    token: Token
    project_id: str = Field(min_length=1, max_length=128)
    workspace: StudioWorkspaceSpec
    graph: GraphSpec
    selections: list[StudioRunSelection] = Field(min_length=1)
    policy: Literal["out_of_date", "all"] = "out_of_date"


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
