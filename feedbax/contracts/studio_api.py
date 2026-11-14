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
    DomainRegistryPayload,
)
from feedbax.contracts.base import ParentRef, StrictModel
from feedbax.contracts.manifest import EvaluationStatesConsumptionPolicy
from feedbax.contracts.selection import SelectionPreview, SelectionSpec
from feedbax.contracts.workspace_replay import WorkspaceReplaySampleAxis, WorkspaceReplayTrack


STUDIO_API_TRANSPORT_SCHEMA_ID = "feedbax.spec.studio.api_transport"
STUDIO_API_TRANSPORT_SCHEMA_VERSION = "feedbax.spec.studio.api_transport.v3"
TRAINING_TRAJECTORY_SCHEMA_ID = "feedbax.event.studio.training_trajectory"
TRAINING_TRAJECTORY_SCHEMA_VERSION = "feedbax.event.studio.training_trajectory.v1"


class StudioRenderFailure(StrictModel):
    region: str = Field(min_length=1, max_length=80)
    error_type: str = Field(min_length=1, max_length=80)
    detail: str = Field(max_length=2048)
    component_stack: str = Field(max_length=4096)


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


class PenzaiBuilderInfo(StudioApiModel):
    """Registry-visible Penzai builder metadata."""

    name: str
    description: str
    default_params: dict[str, Any] = Field(default_factory=dict)
    input_shape: Optional[list[str]] = None
    output_shape: Optional[list[str]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PenzaiBuilderListPayload(StudioApiModel):
    """Payload for ``GET /api/penzai/builders``."""

    builders: list[PenzaiBuilderInfo]


class PenzaiBuilderListResponse(StudioApiModel):
    """Standard API envelope for Penzai builder listings."""

    data: PenzaiBuilderListPayload


class PenzaiNodeRequest(StudioApiModel):
    """Penzai adapter node payload for authoring compile and inspector routes."""

    node_path: list[str] = Field(default_factory=list)
    builder_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    input_port: str = "input"
    output_port: str = "output"


class PenzaiInspectorPayload(StudioApiModel):
    """Payload for rendered Penzai inspector content."""

    html: str
    report: DomainCompileReport


class PenzaiInspectorResponse(StudioApiModel):
    """Standard API envelope for Penzai inspector HTML."""

    data: PenzaiInspectorPayload


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
    kind: Literal["evaluation", "analysis", "materialization", "figure", "report"]
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
    job_kind: Literal["analysis", "figure"] = "analysis"
    force_rerun: bool = False
    eval_run_id: Optional[str] = None
    evaluation_states_policy: EvaluationStatesConsumptionPolicy = "recompute"
    figure_spec: Optional[dict[str, Any]] = None


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


class TrainingTrajectoryPayload(StudioApiModel):
    """Trajectory snapshot carried by a live Jobs frame."""

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
