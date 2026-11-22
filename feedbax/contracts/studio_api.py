"""Studio API transport contracts.

These Pydantic models are the Python-owned source of truth for Studio JSON
transport shapes. The frontend derives runtime validators from these models.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from feedbax.contracts.component import ComponentDefinition
from feedbax.contracts.domain import (
    DomainCompileReport,
    DomainRegistryPayload,
)
from feedbax.contracts.base import StrictModel
from feedbax.contracts.workspace_replay import WorkspaceReplaySampleAxis, WorkspaceReplayTrack


STUDIO_API_TRANSPORT_SCHEMA_ID = "feedbax.spec.studio.api_transport"
STUDIO_API_TRANSPORT_SCHEMA_VERSION = "feedbax.spec.studio.api_transport.v3"
TRAINING_TRAJECTORY_SCHEMA_ID = "feedbax.event.studio.training_trajectory"
TRAINING_TRAJECTORY_SCHEMA_VERSION = "feedbax.event.studio.training_trajectory.v1"


class StudioErrorReport(StrictModel):
    """An error Studio showed, logged whole under the ref its element carries."""

    ref: str = Field(pattern=r"^[0-9a-f]{8}$")
    kind: Literal["network", "http", "request", "response", "render", "error"]
    """``request``: Studio's check refused what it was about to send, so nothing was sent; ``response``: the service's
    answer failed its contract; ``render``: a region failed to render."""
    message: str = Field(max_length=4096)
    path: str | None = Field(default=None, max_length=2048)
    method: str | None = Field(default=None, max_length=16)
    status: int | None = None
    contract: str | None = Field(default=None, max_length=120)
    issues: list[str] = Field(default_factory=list, max_length=200)
    sent: JsonValue = None
    """The body Studio sent, or would have sent; its ``revision`` is the project revision it was made against."""
    detail: JsonValue = None
    """The service's own account of a refusal."""
    region: str | None = Field(default=None, max_length=80)
    stack: str = Field(default="", max_length=16384)
    component_stack: str = Field(default="", max_length=8192)


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
