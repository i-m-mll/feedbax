"""Pydantic models for model inspection and visualization."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from feedbax.web.models.graph import GraphSpec


class CycleAnnotationModel(BaseModel):
    """A back edge that was cut to break a cycle in the graph."""

    source: str = Field(description="Source node.port identifier")
    target: str = Field(description="Target node.port identifier")


class TreescopeRequest(BaseModel):
    """Request for Treescope visualization."""

    max_depth: int = Field(default=10, ge=1, le=50, description="Maximum tree depth")
    project_cycles: bool = Field(default=True, description="Include cycle annotations")
    roundtrip_mode: bool = Field(
        default=False, description="Render in reconstructable format"
    )


class InlineTreescopeRequest(BaseModel):
    """Request for Treescope visualization from an in-memory graph spec."""

    graph: GraphSpec = Field(description="The graph spec to render")
    max_depth: int = Field(default=10, ge=1, le=50, description="Maximum tree depth")
    project_cycles: bool = Field(default=True, description="Include cycle annotations")


class TreescopeResponse(BaseModel):
    """Response containing Treescope visualization."""

    html: str = Field(description="Self-contained HTML for iframe embedding")
    has_cycles: bool = Field(default=False, description="Whether the graph has cycles")
    cycle_count: int = Field(default=0, description="Number of cycles detected")
    cycles: List[CycleAnnotationModel] = Field(
        default_factory=list, description="List of cycle annotations"
    )
    execution_order: Optional[List[str]] = Field(
        default=None, description="Topological execution order of nodes"
    )


class InspectionStatusResponse(BaseModel):
    """Response for inspection status endpoint."""

    treescope_available: bool = Field(description="Whether treescope is installed")
    treescope_configured: bool = Field(description="Whether treescope is configured")
    treescope_version: Optional[str] = Field(
        default=None, description="Treescope version if available"
    )
