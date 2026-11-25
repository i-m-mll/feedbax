"""Studio's typed job service and event-stream contract."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, FiniteFloat, JsonValue

from feedbax.contracts.artifact_custody import StoreTransferProgress
from feedbax.contracts.base import StrictModel
from feedbax.contracts.code_validity import AttemptProvenance
from feedbax.contracts.execution_target import AttemptTarget, LOCAL_TARGET, LOCAL_TARGET_ID, OpeningObservation, OpeningStep, TargetId
from feedbax.contracts.expression_values import TextSpan
from feedbax.contracts.loaded_model import ImportedModelProvenance
from feedbax.contracts.receipt_validity import ReceiptValidity
from feedbax.contracts.project import StepKind
from feedbax.contracts.project_inputs import SelectedAxisValueProblem, StepReference, SubsetTerms
from feedbax.contracts.project_inputs import CheckpointRule
from feedbax.contracts.project_sources import SourceKind
from feedbax.contracts.project_sweep import SweepException
from feedbax.contracts.studio_place import StudioPlace, definition_place, path_parts
from feedbax.contracts.trial_clock import TrialMeanSelection
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
    WAITING = "waiting"
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
    request_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    revision: str | None = Field(default=None, pattern=_DigestPattern, exclude_if=lambda value: value is None)
    trials: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)
    receipt: JobReceipt
    checkpoint_source: StudioTrainCheckpointSource | None
    loaded_model: ImportedModelProvenance | None = Field(default=None, exclude_if=lambda value: value is None)
    size_bytes: int = Field(ge=0)
    validity: ReceiptValidity | None = Field(default=None, exclude_if=lambda value: value is None)
    provenance: AttemptProvenance | None = Field(default=None, exclude_if=lambda value: value is None)
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
    previous: StudioSavedEvaluationResult | None = Field(default=None, exclude_if=lambda value: value is None)
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
    trial: str | None = None
    """When absent, select every retained trial at the checkpoint."""
    aggregate: bool = False
    checkpoint: int | None = Field(default=None, ge=0)
    mean: TrialMeanSelection | None = None


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
    receipt: JobReceipt | None
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
    revision: str | None = Field(default=None, pattern=_DigestPattern, exclude_if=lambda value: value is None)
    receipt: JobReceipt
    progress: JobProgress | None
    validity: ReceiptValidity | None = Field(default=None, exclude_if=lambda value: value is None)
    provenance: AttemptProvenance | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioDefinitionChange(StrictModel):
    """The first authored field that differs from the definition of a retained result."""

    path: str
    place: StudioPlace


class StudioSavedTrainChange(StudioDefinitionChange):
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
    """A row's value on one axis: its stored key, label and value, and the step's axis value it is."""

    value_id: str
    label: str | None
    value: JsonValue
    axis_value: str
    """The ID of the value of the step's axis that this coordinate is."""


class StudioProjectWindowMapInputAxis(StrictModel):
    """A row's member of one input axis, with the axes of the row's coordinates the member carries."""

    value_id: str
    label: str | None
    carries: list[str]
    axis_value: str
    """The ID of the value of the step's axis that this member is."""


class StudioSavedLightResult(StrictModel):
    job_id: str
    request_id: str
    revision: str | None = Field(default=None, pattern=_DigestPattern, exclude_if=lambda value: value is None)
    receipt: JobReceipt
    products: dict[str, str]
    validity: ReceiptValidity | None = Field(default=None, exclude_if=lambda value: value is None)
    provenance: AttemptProvenance | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioSavedLightHistory(StrictModel):
    result: StudioSavedLightResult | None = None
    previous: StudioSavedLightResult | None = None
    changed: StudioDefinitionChange | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioProjectWindowMapRow(StrictModel):
    key: StudioSavedTrainKey
    label: str
    coordinates: dict[str, StudioProjectWindowMapCoordinate]
    input_axes: dict[str, StudioProjectWindowMapInputAxis] = Field(
        default_factory=dict, exclude_if=lambda value: not value,
    )
    member_key: str | None = Field(default=None, exclude_if=lambda value: value is None)
    status: JobStatus
    reason: str | None = Field(default=None, exclude_if=lambda value: value is None)
    opening_step: OpeningStep | None = Field(default=None, exclude_if=lambda value: value is None)
    collection_progress: StoreTransferProgress | None = Field(default=None, exclude_if=lambda value: value is None)
    evaluation_conditions: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)
    origin: int | None = Field(default=None, exclude_if=lambda value: value is None)
    """A Train row's completed-update count at its start (0 when fresh); its readers select updates after it."""
    target_total: int | None = Field(default=None, exclude_if=lambda value: value is None)
    """A Train row's completed-update count at its end, which bounds the checkpoints its readers select."""
    light: StudioSavedLightHistory | None = Field(default=None, exclude_if=lambda value: value is None)
    resolved: bool = Field(default=True, exclude_if=lambda value: value)
    """Whether the row has a definition; a row whose own patches or documents are refused has none to show."""


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
    can_sweep_checkpoint: bool = Field(default=False, exclude_if=lambda value: not value)


class StudioProjectWindowMapException(StrictModel):
    id: str
    count: int
    rule: SweepException


class StudioProjectWindowMapAxisValue(StrictModel):
    """One value of a step's axis, gathering its runs' coordinates that are the same value: a number by number, a
    named value by name, an input's member by its key. Its ID is the coordinates' stored key when that key names this
    value alone in the step; otherwise, as when two producers store the same value under different keys, it is ``=``
    and the value, which no stored key can be."""

    id: str
    label: str
    rule: str
    """The value as a run rule writes it."""


class StudioProjectWindowMapAxis(StrictModel):
    """An axis a step's runs carry, with its values in order: numbers by number, other values as the runs give them."""

    id: str
    values: list[StudioProjectWindowMapAxisValue]


class StudioProjectWindowMapStep(StrictModel):
    variant: str
    step_id: str
    kind: StepKind
    group: str | None
    inputs: list[StudioProjectWindowMapInput]
    document_ids: dict[str, str]
    document_uses: list[str] = Field(default_factory=list)
    exceptions: list[StudioProjectWindowMapException] = Field(default_factory=list, exclude_if=lambda value: not value)
    status_counts: dict[str, int]
    axes: list[StudioProjectWindowMapAxis] = Field(default_factory=list, exclude_if=lambda value: not value)
    """Each axis the step's runs carry, in the order the runs first carry them."""
    rule_metrics: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    """The metrics the step's finished Train runs report, which a lowest or highest term ranks by."""


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
    place: StudioPlace | None
    document_path: str | None = Field(default=None, exclude_if=lambda value: value is None)

    address: str | None = Field(default=None, exclude_if=lambda value: value is None)
    layer: str | None = Field(default=None, exclude_if=lambda value: value is None)
    authored_path: str | None = Field(default=None, exclude_if=lambda value: value is None)
    blocked_by: StudioSavedTrainKey | None = Field(default=None, exclude_if=lambda value: value is None)
    """The producer row a ``blocked_input`` row waits on."""
    span: TextSpan | None = Field(default=None, exclude_if=lambda value: value is None)
    selection_value: SelectedAxisValueProblem | None = Field(default=None, exclude_if=lambda value: value is None)

    @classmethod
    def from_diagnostic(cls, diagnostic: ProjectDiagnostic, *, declaration: str) -> StudioProjectWindowMapProblem:
        source = f"steps/{diagnostic.step}.json" if diagnostic.step else declaration
        return cls(
            code=diagnostic.code, path=diagnostic.path, detail=diagnostic.detail,
            variant=diagnostic.variant, step=diagnostic.step, row=diagnostic.row,
            place=definition_place(
                path_parts(diagnostic.path), lambda _slot: diagnostic.document,
                element=diagnostic.element, element_name=diagnostic.element_name, member=diagnostic.model_address,
                variation=lambda _name: diagnostic.variation_target,
            ) if diagnostic.step else None,
            document_path=diagnostic.document_path, address=diagnostic.authored_file or source,
            layer=diagnostic.layer, authored_path=diagnostic.authored_path,
            span=diagnostic.span, selection_value=diagnostic.selection_value,
            blocked_by=StudioSavedTrainKey.model_validate(diagnostic.blocked_by, from_attributes=True)
            if diagnostic.blocked_by is not None else None,
        )


class StudioPlanRequestAddress(StrictModel):
    key: StudioSavedTrainKey
    element: str | None = None
    member: str | None = Field(default=None, exclude_if=lambda value: value is None)
    """The mapped request of ``element``, when the element maps over labels."""


class StudioPlanRequest(StudioPlanRequestAddress):
    request_id: str


class StudioPlanReviewSource(StrictModel):
    kind: SourceKind
    name: str
    path: str
    step: str | None = Field(default=None, exclude_if=lambda value: value is None)
    axis: str | None = Field(default=None, exclude_if=lambda value: value is None)
    value: str | None = Field(default=None, exclude_if=lambda value: value is None)
    variant: str | None = Field(default=None, exclude_if=lambda value: value is None)
    row: str | None = Field(default=None, exclude_if=lambda value: value is None)


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


class StudioProjectWindowMapRequest(StudioSavedLightHistory):
    """One request of an Analyze element."""

    labels: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
    """The request's member id, when its element maps over labels: the names of the values of the labels that vary
    across the element's requests, by label, in label order."""
    request_id: str
    status: JobStatus
    reason: str | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioProjectWindowMapElement(StrictModel):
    """An operation or figure of an Analyze row, with its requests: one, or one per combination of the labels it
    maps over. Its status is their summary."""

    key: StudioSavedTrainKey
    name: str
    kind: Literal["analyze", "render"]
    status: JobStatus
    reason: str | None = Field(default=None, exclude_if=lambda value: value is None)
    figure: StepReference | None = Field(default=None, exclude_if=lambda value: value is None)
    requests: list[StudioProjectWindowMapRequest] = Field(min_length=1)


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
    regroup_refused: list[str] = Field(default_factory=list)
    """Analyze cards whose Split and Merge are disabled: their document is shared or based, or they are swept."""


class StudioSavedTrainRuns(StrictModel):
    revision: str
    rows: list[StudioSavedTrainRow]
    window_map: StudioProjectWindowMap | None = None
    review: list[StudioPlanReviewEntry] | None = None


class StudioSavedMachineQuoteRequest(StrictModel):
    revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_keys: list[StudioSavedTrainKey] = Field(min_length=1)
    target: TargetId = LOCAL_TARGET_ID


class StudioSavedTrainRunRequest(StudioSavedMachineQuoteRequest):
    token: Token
    reuse_completed: bool = False


class StudioMachineQuoteRow(StrictModel):
    key: StudioSavedTrainKey
    label: str
    axes: dict[str, JsonValue]


class StudioMachineQuote(StrictModel):
    id: UUID
    machine: str
    resources: dict[str, JsonValue]
    hourly_microusd: int = Field(ge=0)
    rows: list[StudioMachineQuoteRow] = Field(min_length=1)
    ceiling_microusd: int = Field(ge=0)
    expires_at_ms: int = Field(gt=0)
    idle_timeout_seconds: int | None = Field(gt=0)
    idle_with_results: Literal["on"]
    status: Literal["awaiting_authorization"]


class StudioMachineApproval(StrictModel):
    quote: UUID


class StudioMachineApprovalReceipt(StrictModel):
    session_id: UUID


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
    kind: Literal["train", "evaluate", "analyze", "render", "report"]
    run_key: str
    stage_id: str
    scenario_id: str
    label: str
    coordinates: dict[str, JsonValue]
    status: JobStatus
    submitted_at_ms: int = Field(ge=0)
    """When this job was submitted, in Unix epoch milliseconds."""
    ended_at_ms: int | None = Field(ge=0)
    """When this job reached a terminal state, in Unix epoch milliseconds."""
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
    target: AttemptTarget = LOCAL_TARGET
    log_tail: list[str] = Field(default_factory=list, exclude_if=lambda lines: not lines)
    collection_progress: StoreTransferProgress | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioLocalSession(StrictModel):
    id: Literal["local"] = "local"
    state: Literal["open"] = "open"


class StudioRemoteSession(StrictModel):
    id: UUID
    name: str
    profile: str = Field(min_length=1)
    state: Literal["queued", "opening", "open", "closing", "unreachable", "off"]
    hourly_microusd: int = Field(ge=0)
    estimated_cost_microusd: int = Field(ge=0)
    closes_at_ms: int = Field(gt=0)
    ceiling_microusd: int = Field(ge=0)
    idle_since_ms: int | None = Field(ge=0)
    job_ids: list[str]
    can_close: bool
    region: str | None = Field(default=None, exclude_if=lambda value: value is None)
    opening_observations: tuple[OpeningObservation, ...] = Field(default=(), exclude_if=lambda value: not value)
    opening_failure: str | None = Field(default=None, exclude_if=lambda value: value is None)
    billed_cost_microusd: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    provider_observed_at_ms: int | None = Field(default=None, gt=0, exclude_if=lambda value: value is None)
    power_state: Literal["starting", "on", "off"] | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioVenueStream(StrictModel):
    session_id: UUID
    stream_id: str = Field(min_length=1)


class StudioVenueCursor(StudioVenueStream):
    cursor: int = Field(ge=0)


class StudioUnreadableJob(StrictModel):
    """A run whose stored record Studio could not read into an Activity row."""

    project_id: str
    job_id: str
    label: str | None
    reason: str


class StudioJobSnapshot(StrictModel):
    jobs: list[StudioActivityJob]
    unreadable: list[StudioUnreadableJob]
    cursor: int = Field(ge=0)
    oldest_cursor: int = Field(ge=0)
    sessions: list[StudioLocalSession | StudioRemoteSession]
    remote_version: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    machine_catalog_version: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    venue_cursors: list[StudioVenueCursor] = Field(default_factory=list, exclude_if=lambda value: not value)


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
    target: AttemptTarget = LOCAL_TARGET


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
    unreadable: StudioUnreadableJob | None = Field(default=None, exclude_if=lambda value: value is None)
    venue: StudioVenueStream | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioJobEventHistory(StrictModel):
    cursor: int = Field(ge=0)
    oldest_cursor: int = Field(ge=0)
    events: list[StudioJobEvent]


class StudioEventWatermark(StrictModel):
    cursor: int = Field(ge=0)


class StudioEventReset(StudioEventWatermark):
    oldest_cursor: int = Field(ge=0)
