"""Studio API transport contracts.

These Pydantic models are the Python-owned source of truth for Studio JSON
transport shapes. The frontend derives runtime validators from these models.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from feedbax.contracts.component import ComponentDefinition
from feedbax.contracts.graph import (
    AnalysisPageSpec,
    GraphMetadata,
    GraphSpec,
    GraphUIState,
    StudioWorkspaceSpec,
    ValidationResult,
)


class SuccessPayload(BaseModel):
    """Boolean success payload shared by mutation endpoints."""

    success: bool


class SuccessResponse(BaseModel):
    """Standard API envelope for success-only mutations."""

    data: SuccessPayload


class GraphListItem(BaseModel):
    """A saved graph listing row."""

    id: str
    metadata: GraphMetadata


class GraphListPayload(BaseModel):
    """Payload for ``GET /api/graphs``."""

    graphs: list[GraphListItem]


class GraphListResponse(BaseModel):
    """Standard API envelope for saved graph listings."""

    data: GraphListPayload


class GraphCreatePayload(BaseModel):
    """Payload returned after creating a graph."""

    id: str
    metadata: GraphMetadata


class GraphCreateResponse(BaseModel):
    """Standard API envelope for graph creation."""

    data: GraphCreatePayload


class GraphDetailPayload(BaseModel):
    """Payload for ``GET /api/graphs/{graph_id}``."""

    graph: GraphSpec
    ui_state: Optional[GraphUIState] = None
    demo_training_data: Optional[Any] = None
    metadata: Optional[GraphMetadata] = None
    analysis_pages: Optional[list[AnalysisPageSpec]] = None
    active_analysis_page_id: Optional[str] = None
    workspace: Optional[StudioWorkspaceSpec] = None


class GraphDetailResponse(BaseModel):
    """Standard API envelope for graph detail retrieval."""

    data: GraphDetailPayload


class GraphValidationResponse(BaseModel):
    """Standard API envelope for graph validation."""

    data: ValidationResult


class GraphExportPayload(BaseModel):
    """Payload for graph export endpoints."""

    content: str
    filename: str


class GraphExportResponse(BaseModel):
    """Standard API envelope for graph exports."""

    data: GraphExportPayload


class ComponentListPayload(BaseModel):
    """Payload for ``GET /api/components``."""

    components: list[ComponentDefinition]


class ComponentListResponse(BaseModel):
    """Standard API envelope for component library listings."""

    data: ComponentListPayload


class ComponentDetailResponse(BaseModel):
    """Standard API envelope for component detail retrieval."""

    data: ComponentDefinition


class ComponentRefreshPayload(BaseModel):
    """Payload for component library refresh results."""

    added: list[str]
    removed: list[str]


class ComponentRefreshResponse(BaseModel):
    """Standard API envelope for component library refresh results."""

    data: ComponentRefreshPayload


class TrainingStartPayload(BaseModel):
    """Payload returned after starting a training job."""

    job_id: str


class TrainingStartResponse(BaseModel):
    """Standard API envelope for ``POST /api/training``."""

    data: TrainingStartPayload


class TrainingStatusPayload(BaseModel):
    """Payload for training status polling."""

    status: dict[str, Any]


class TrainingStatusResponse(BaseModel):
    """Standard API envelope for training status polling."""

    data: TrainingStatusPayload


class WorkerConnectResponse(BaseModel):
    """Payload for connecting Studio to a worker."""

    ok: bool
    url: str


class WorkerConnectEnvelope(BaseModel):
    """Standard API envelope for worker connection responses."""

    data: WorkerConnectResponse


class WorkerStatusResponse(BaseModel):
    """Payload describing the current worker connection."""

    mode: Literal["local", "remote"]
    url: Optional[str] = None
    connected: bool


class WorkerStatusEnvelope(BaseModel):
    """Standard API envelope for worker status responses."""

    data: WorkerStatusResponse


class AnalysisClassInfo(BaseModel):
    """Describes a single analysis class available in a package."""

    name: str
    description: str
    category: str
    inputPorts: list[str]
    outputPorts: list[str]
    defaultParams: dict[str, Any]
    icon: str


class AnalysisPackageInfo(BaseModel):
    """A group of related analysis classes."""

    name: str
    description: str
    analyses: list[AnalysisClassInfo]


class AnalysisPackagesPayload(BaseModel):
    """Payload for analysis package discovery."""

    packages: list[AnalysisPackageInfo]


class AnalysisPackagesResponse(BaseModel):
    """Standard API envelope for analysis package discovery."""

    data: AnalysisPackagesPayload


class GenerateAnalysisRequest(BaseModel):
    """Body for ``POST /api/analyses/jobs``."""

    node_id: str
    force_rerun: bool = False
    eval_run_id: Optional[str] = None


class GenerateAnalysisPayload(BaseModel):
    """Payload returned immediately after scheduling an analysis job."""

    request_id: str
    status: str


class GenerateAnalysisResponse(BaseModel):
    """Standard API envelope for demand-driven analysis job creation."""

    data: GenerateAnalysisPayload


class AnalysisJobStatusPayload(BaseModel):
    """Payload returned by demand-driven analysis job polling."""

    request_id: str
    status: str
    figure_hashes: Optional[list[str]] = None
    error: Optional[str] = None


class AnalysisJobStatusResponse(BaseModel):
    """Standard API envelope for demand-driven analysis job polling."""

    data: AnalysisJobStatusPayload


class TrainingProgressEvent(BaseModel):
    """Training progress event sent over the Studio WebSocket."""

    type: Literal["training_progress"]
    job_id: str
    batch: int
    total_batches: int
    loss: float
    loss_terms: dict[str, float] = Field(default_factory=dict)
    grad_norm: float = 0.0
    step_time_ms: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    status: str = "running"
    execution: Optional[str] = None


class TrainingLogEvent(BaseModel):
    """Training log event sent over the Studio WebSocket."""

    type: Literal["training_log"]
    job_id: str
    batch: int
    level: Literal["info", "warning", "error"] = "info"
    message: str
    manifest_path: Optional[str] = None
    manifest_id: Optional[str] = None
    execution: Optional[str] = None


class TrainingTrajectoryPayload(BaseModel):
    """Trajectory snapshot carried by a training WebSocket event."""

    effector: list[Any] = Field(default_factory=list)
    target: Optional[Any] = None
    t: list[Any] = Field(default_factory=list)
    observables: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)


class TrainingTrajectoryEvent(BaseModel):
    """Training trajectory event sent over the Studio WebSocket."""

    type: Literal["training_trajectory"]
    job_id: str
    batch: int
    trajectory: TrainingTrajectoryPayload
    execution: Optional[str] = None


class TrainingCompleteEvent(BaseModel):
    """Training completion event sent over the Studio WebSocket."""

    type: Literal["training_complete"]
    job_id: str
    batch: int
    loss: Optional[float] = None
    manifest_path: Optional[str] = None
    manifest_id: Optional[str] = None
    execution: Optional[str] = None


class TrainingErrorEvent(BaseModel):
    """Training error event sent over the Studio WebSocket."""

    type: Literal["training_error"]
    job_id: str
    batch: Optional[int] = None
    error: str
