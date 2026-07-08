"""Studio API transport contracts.

These Pydantic models are the Python-owned source of truth for Studio JSON
transport shapes. The frontend derives runtime validators from these models.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from feedbax.contracts.component import ComponentDefinition
from feedbax.contracts.domain import (
    DomainCompileReport,
    DomainDiagnostic,
    DomainRegistryPayload,
)
from feedbax.contracts.graph import (
    AnalysisPageSpec,
    GraphMetadata,
    GraphSpec,
    GraphUIState,
    StudioWorkspaceSpec,
)
from feedbax.contracts.manifest import ParentRef
from feedbax.contracts.selection import SelectionPreview, SelectionSpec
from feedbax.contracts.workspace_replay import WorkspaceReplaySampleAxis, WorkspaceReplayTrack


STUDIO_API_TRANSPORT_SCHEMA_ID = "feedbax.spec.studio.api_transport"
STUDIO_API_TRANSPORT_SCHEMA_VERSION = "feedbax.spec.studio.api_transport.v2"
TRAINING_TRAJECTORY_SCHEMA_ID = "feedbax.event.studio.training_trajectory"
TRAINING_TRAJECTORY_SCHEMA_VERSION = "feedbax.event.studio.training_trajectory.v1"


class StudioApiModel(BaseModel):
    """Base model for Studio API transport contracts."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[STUDIO_API_TRANSPORT_SCHEMA_ID] = STUDIO_API_TRANSPORT_SCHEMA_ID
    schema_version: Literal[STUDIO_API_TRANSPORT_SCHEMA_VERSION] = (
        STUDIO_API_TRANSPORT_SCHEMA_VERSION
    )


class SuccessPayload(StudioApiModel):
    """Boolean success payload shared by mutation endpoints."""

    success: bool


class SuccessResponse(StudioApiModel):
    """Standard API envelope for success-only mutations."""

    data: SuccessPayload


class GraphListItem(StudioApiModel):
    """A saved graph listing row."""

    id: str
    metadata: GraphMetadata


class GraphListPayload(StudioApiModel):
    """Payload for ``GET /api/graphs``."""

    graphs: list[GraphListItem]


class GraphListResponse(StudioApiModel):
    """Standard API envelope for saved graph listings."""

    data: GraphListPayload


class GraphCreatePayload(StudioApiModel):
    """Payload returned after creating a graph."""

    id: str
    metadata: GraphMetadata


class GraphCreateResponse(StudioApiModel):
    """Standard API envelope for graph creation."""

    data: GraphCreatePayload


class GraphUpdatePayload(StudioApiModel):
    """Payload returned after updating a graph."""

    success: bool
    metadata: GraphMetadata


class GraphUpdateResponse(StudioApiModel):
    """Standard API envelope for graph updates."""

    data: GraphUpdatePayload


class GraphDetailPayload(StudioApiModel):
    """Payload for ``GET /api/graphs/{graph_id}``."""

    graph: GraphSpec
    ui_state: Optional[GraphUIState] = None
    demo_training_data: Optional[Any] = None
    metadata: Optional[GraphMetadata] = None
    analysis_pages: Optional[list[AnalysisPageSpec]] = None
    active_analysis_page_id: Optional[str] = None
    workspace: Optional[StudioWorkspaceSpec] = None
    compile_reports: Optional[dict[str, DomainCompileReport]] = None


class GraphDetailResponse(StudioApiModel):
    """Standard API envelope for graph detail retrieval."""

    data: GraphDetailPayload


class GraphValidationResponse(StudioApiModel):
    """Standard API envelope for graph validation."""

    data: list[DomainDiagnostic]


class GraphExportPayload(StudioApiModel):
    """Payload for graph export endpoints."""

    content: str
    filename: str


class GraphExportResponse(StudioApiModel):
    """Standard API envelope for graph exports."""

    data: GraphExportPayload


class ComponentListPayload(StudioApiModel):
    """Payload for ``GET /api/components``."""

    components: list[ComponentDefinition]


class ComponentListResponse(StudioApiModel):
    """Standard API envelope for component library listings."""

    data: ComponentListPayload


class ComponentDetailResponse(StudioApiModel):
    """Standard API envelope for component detail retrieval."""

    data: ComponentDefinition


class ComponentRefreshPayload(StudioApiModel):
    """Payload for component library refresh results."""

    added: list[str]
    removed: list[str]


class ComponentRefreshResponse(StudioApiModel):
    """Standard API envelope for component library refresh results."""

    data: ComponentRefreshPayload


class DomainListResponse(StudioApiModel):
    """Standard API envelope for domain registry listings."""

    data: DomainRegistryPayload


class TrainingStartPayload(StudioApiModel):
    """Payload returned after starting a training job."""

    job_id: str


class TrainingStartResponse(StudioApiModel):
    """Standard API envelope for ``POST /api/training``."""

    data: TrainingStartPayload


class TrainingStatusPayload(StudioApiModel):
    """Payload for training status polling."""

    status: dict[str, Any]


class TrainingStatusResponse(StudioApiModel):
    """Standard API envelope for training status polling."""

    data: TrainingStatusPayload


class WorkerConnectResponse(StudioApiModel):
    """Payload for connecting Studio to a worker."""

    ok: bool
    url: str


class WorkerConnectEnvelope(StudioApiModel):
    """Standard API envelope for worker connection responses."""

    data: WorkerConnectResponse


class WorkerStatusResponse(StudioApiModel):
    """Payload describing the current worker connection."""

    mode: Literal["local", "remote"]
    url: Optional[str] = None
    connected: bool


class WorkerStatusEnvelope(StudioApiModel):
    """Standard API envelope for worker status responses."""

    data: WorkerStatusResponse


class AnalysisClassInfo(StudioApiModel):
    """Describes a single analysis class available in a package."""

    name: str
    description: str
    category: str
    inputPorts: list[str]
    outputPorts: list[str]
    defaultParams: dict[str, Any]
    icon: str


class AnalysisPackageInfo(StudioApiModel):
    """A group of related analysis classes."""

    name: str
    description: str
    analyses: list[AnalysisClassInfo]


class AnalysisPackagesPayload(StudioApiModel):
    """Payload for analysis package discovery."""

    packages: list[AnalysisPackageInfo]


class AnalysisPackagesResponse(StudioApiModel):
    """Standard API envelope for analysis package discovery."""

    data: AnalysisPackagesPayload


class BundleMissingRoleRecord(StudioApiModel):
    """Required role dependency unavailable in an analysis bundle dry-run."""

    stage: str
    role: str
    required: bool = True
    bind_as: Optional[str] = None
    reason: str


class BundleStageDryRunOutputRecord(StudioApiModel):
    """Predicted output-role status for one analysis bundle dry-run stage."""

    role: str
    required: bool = True
    status: Literal["would_run", "would_skip", "missing", "not_applicable"]
    reason: Optional[str] = None


class BundleStageDryRunRecord(StudioApiModel):
    """Side-effect-free stage plan for analysis bundle preflight."""

    name: str
    kind: Literal["evaluation", "analysis", "materialization", "report"]
    status: Literal["would_run", "would_skip", "missing", "not_applicable"]
    depends_on: list[str] = Field(default_factory=list)
    inputs: list[ParentRef] = Field(default_factory=list)
    outputs: list[BundleStageDryRunOutputRecord] = Field(default_factory=list)
    missing_roles: list[BundleMissingRoleRecord] = Field(default_factory=list)
    reason: Optional[str] = None


class AnalysisBundleDryRunResult(StudioApiModel):
    """Side-effect-free analysis bundle preflight over a matched selection."""

    bundle_name: str
    match_preview: SelectionPreview
    matched_run_ids: list[str] = Field(default_factory=list)
    stages: list[BundleStageDryRunRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisBundleDryRunRequest(StudioApiModel):
    """Body for side-effect-free analysis bundle preflight."""

    bundle: dict[str, Any]
    selection_spec: Optional[SelectionSpec] = None
    root: Optional[str] = None
    preview_limit: int = Field(default=50, ge=0, le=500)


class AnalysisBundleDryRunPayload(StudioApiModel):
    """Payload for analysis bundle preflight results."""

    dry_run: AnalysisBundleDryRunResult


class AnalysisBundleDryRunResponse(StudioApiModel):
    """Standard API envelope for analysis bundle dry-run results."""

    data: AnalysisBundleDryRunPayload


class GenerateAnalysisRequest(StudioApiModel):
    """Body for ``POST /api/analyses/jobs``."""

    node_id: str
    force_rerun: bool = False
    eval_run_id: Optional[str] = None


class GenerateAnalysisPayload(StudioApiModel):
    """Payload returned immediately after scheduling an analysis job."""

    request_id: str
    status: str
    manifest_id: Optional[str] = None


class GenerateAnalysisResponse(StudioApiModel):
    """Standard API envelope for demand-driven analysis job creation."""

    data: GenerateAnalysisPayload


class AnalysisJobStatusPayload(StudioApiModel):
    """Payload returned by demand-driven analysis job polling."""

    request_id: str
    status: str
    figure_hashes: Optional[list[str]] = None
    manifest_id: Optional[str] = None
    manifest_path: Optional[str] = None
    artifact_ids: Optional[list[str]] = None
    artifact_paths: Optional[list[str]] = None
    error: Optional[str] = None


class AnalysisJobStatusResponse(StudioApiModel):
    """Standard API envelope for demand-driven analysis job polling."""

    data: AnalysisJobStatusPayload


class TrainingProgressEvent(StudioApiModel):
    """Training progress event sent over the Studio WebSocket."""

    type: Literal["training_progress"]
    job_id: str
    seq: int
    emitted_at_ms: int
    worker_seq: Optional[int] = None
    batch: int
    total_batches: int
    loss: float
    loss_terms: dict[str, float] = Field(default_factory=dict)
    grad_norm: float = 0.0
    step_time_ms: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    status: str = "running"
    execution: Optional[str] = None


class TrainingLogEvent(StudioApiModel):
    """Training log event sent over the Studio WebSocket."""

    type: Literal["training_log"]
    job_id: str
    seq: int
    emitted_at_ms: int
    worker_seq: Optional[int] = None
    batch: int
    level: Literal["info", "warning", "error"] = "info"
    message: str
    manifest_path: Optional[str] = None
    manifest_id: Optional[str] = None
    execution: Optional[str] = None


class TrainingTrajectoryPayload(StudioApiModel):
    """Trajectory snapshot carried by a training WebSocket event."""

    schema_id: Literal[TRAINING_TRAJECTORY_SCHEMA_ID] = TRAINING_TRAJECTORY_SCHEMA_ID
    schema_version: Literal[TRAINING_TRAJECTORY_SCHEMA_VERSION] = TRAINING_TRAJECTORY_SCHEMA_VERSION
    source_kind: Literal["live_snapshot"] = "live_snapshot"
    fidelity: Literal["lower_fidelity_live_snapshot"] = "lower_fidelity_live_snapshot"
    n_steps: Optional[int] = None
    time: Optional[WorkspaceReplaySampleAxis] = None
    tracks: dict[str, WorkspaceReplayTrack] = Field(default_factory=dict)
    observables: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    effector: Optional[list[Any]] = None
    target: Optional[Any] = None
    t: Optional[list[Any]] = None


class TrainingTrajectoryEvent(StudioApiModel):
    """Training trajectory event sent over the Studio WebSocket."""

    type: Literal["training_trajectory"]
    job_id: str
    seq: int
    emitted_at_ms: int
    worker_seq: Optional[int] = None
    batch: int
    trajectory: TrainingTrajectoryPayload
    execution: Optional[str] = None


class TrainingCompleteEvent(StudioApiModel):
    """Training completion event sent over the Studio WebSocket."""

    type: Literal["training_complete"]
    job_id: str
    seq: int
    emitted_at_ms: int
    worker_seq: Optional[int] = None
    batch: int
    loss: Optional[float] = None
    manifest_path: Optional[str] = None
    manifest_id: Optional[str] = None
    execution: Optional[str] = None


class TrainingErrorEvent(StudioApiModel):
    """Training error event sent over the Studio WebSocket."""

    type: Literal["training_error"]
    job_id: str
    seq: int
    emitted_at_ms: int
    worker_seq: Optional[int] = None
    batch: Optional[int] = None
    error: str
    diagnostics: list[DomainDiagnostic] = Field(default_factory=list)


class TrainingResyncEvent(StudioApiModel):
    """Training stream reconnect/resync marker sent over the Studio WebSocket."""

    type: Literal["training_resync"]
    job_id: str
    seq: int
    emitted_at_ms: int
    worker_seq: Optional[int] = None
    expected_worker_seq: Optional[int] = None
    observed_worker_seq: Optional[int] = None
    missed_events: int = 0
    reason: Literal["resumed", "gap"]
    message: str
