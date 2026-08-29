from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict
from typing import Optional

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.domain import DomainCompileReport
from feedbax.contracts.graph import (
    GraphSpec,
    StudioWorkspaceSpec,
    WorkspaceDocument,
)
from feedbax.contracts.studio_api import (
    GraphCreateResponse,
    GraphDetailResponse,
    GraphExportResponse,
    GraphListResponse,
    GraphUpdateResponse,
    GraphValidationResponse,
    SuccessPayload,
    SuccessResponse,
    PenzaiNodeRequest,
)
from feedbax.web.services.graph_service import GraphSaveConflictError, GraphService

router = APIRouter()
service = GraphService()


def _component_registry(request: Request):
    bootstrap_state = getattr(request.app.state, "bootstrap_state", None)
    return bootstrap_state.bundle.components if bootstrap_state is not None else None


class GraphCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: GraphSpec
    workspace_document: Optional[WorkspaceDocument] = None
    workspace: Optional[StudioWorkspaceSpec] = None


class GraphUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: Optional[GraphSpec] = None
    workspace_document: Optional[WorkspaceDocument] = None
    workspace: Optional[StudioWorkspaceSpec] = None
    expected_save_revision: Optional[int] = None


class GraphNodeCompileRequest(BaseModel):
    node_path: list[str]
    interior: AcausalGraphSpec


def _parse_if_match_revision(if_match: Optional[str]) -> Optional[int]:
    if if_match is None:
        return None
    candidate = if_match.strip()
    if candidate.startswith("W/"):
        candidate = candidate[2:].strip()
    candidate = candidate.strip('"')
    try:
        revision = int(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid If-Match save revision") from exc
    if revision < 0:
        raise HTTPException(status_code=400, detail="Invalid If-Match save revision")
    return revision


def _conflict_detail(exc: GraphSaveConflictError) -> dict[str, object]:
    return {
        "message": "Project changed in another tab or session. Review the server copy before saving again.",
        "graph_id": exc.graph_id,
        "expected_save_revision": exc.expected_revision,
        "current_save_revision": exc.current_revision,
    }


@router.get("", response_model=GraphListResponse)
async def list_graphs(request: Request, response: Response) -> GraphListResponse:
    response.headers["Cache-Control"] = "no-store"
    return GraphListResponse(
        data={
            "graphs": service.list_graphs(
                component_registry=_component_registry(request),
            )
        }
    )


@router.post("", response_model=GraphCreateResponse)
async def create_graph(payload: GraphCreateRequest, request: Request) -> GraphCreateResponse:
    component_registry = _component_registry(request)
    record = service.create_graph(
        payload.graph,
        workspace=payload.workspace,
        workspace_document=payload.workspace_document,
        component_registry=component_registry,
    )
    return GraphCreateResponse(data={"id": record.graph_id, "metadata": record.project.metadata})


@router.get("/{graph_id}", response_model=GraphDetailResponse)
async def get_graph(graph_id: str, request: Request, response: Response) -> GraphDetailResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        record = service.get_graph(
            graph_id,
            component_registry=_component_registry(request),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Graph not found") from exc
    return GraphDetailResponse(
        data={
            "graph": record.project.graph,
            "workspace_document": record.project.workspace_document,
            "demo_training_data": record.project.demo_training_data,
            "metadata": record.project.metadata,
            "workspace": record.project.workspace,
            "compile_reports": record.project.compile_reports,
        }
    )


@router.put("/{graph_id}", response_model=GraphUpdateResponse)
async def update_graph(
    graph_id: str,
    payload: GraphUpdateRequest,
    request: Request,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> GraphUpdateResponse:
    expected_revision = _parse_if_match_revision(if_match)
    if expected_revision is None:
        expected_revision = payload.expected_save_revision
    try:
        record = service.update_graph(
            graph_id,
            payload.graph,
            workspace=payload.workspace,
            workspace_document=payload.workspace_document,
            expected_save_revision=expected_revision,
            require_save_revision=True,
            component_registry=_component_registry(request),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Graph not found") from exc
    except GraphSaveConflictError as exc:
        raise HTTPException(status_code=409, detail=_conflict_detail(exc)) from exc
    return GraphUpdateResponse(data={"success": True, "metadata": record.project.metadata})


@router.post("/{graph_id}/beacon")
async def beacon_update_graph(
    graph_id: str,
    payload: GraphUpdateRequest,
    request: Request,
):
    """sendBeacon endpoint for pagehide saves; returns 204 No Content."""
    try:
        service.update_graph(
            graph_id,
            payload.graph,
            workspace=payload.workspace,
            workspace_document=payload.workspace_document,
            expected_save_revision=payload.expected_save_revision,
            require_save_revision=True,
            component_registry=_component_registry(request),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Graph not found") from exc
    except GraphSaveConflictError as exc:
        raise HTTPException(status_code=409, detail=_conflict_detail(exc)) from exc
    return Response(status_code=204)


@router.delete("/{graph_id}", response_model=SuccessResponse)
async def delete_graph(graph_id: str) -> SuccessResponse:
    service.delete_graph(graph_id)
    return SuccessResponse(data=SuccessPayload(success=True))


@router.post("/{graph_id}/validate", response_model=GraphValidationResponse)
async def validate_graph(graph_id: str, graph: GraphSpec) -> GraphValidationResponse:
    return GraphValidationResponse(data=service.validate_graph(graph))


@router.post("/{graph_id}/nodes/compile", response_model=DomainCompileReport)
async def compile_graph_node(
    graph_id: str,
    payload: GraphNodeCompileRequest,
    request: Request,
) -> DomainCompileReport:
    try:
        return service.compile_node(
            graph_id,
            node_path=payload.node_path,
            interior=payload.interior,
            component_registry=request.app.state.bootstrap_state.bundle.components,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Graph not found") from exc


@router.post("/{graph_id}/nodes/penzai/compile", response_model=DomainCompileReport)
async def compile_penzai_graph_node(
    graph_id: str,
    payload: PenzaiNodeRequest,
) -> DomainCompileReport:
    try:
        return service.compile_penzai_node(
            graph_id,
            node_path=payload.node_path,
            builder_name=payload.builder_name,
            params=payload.params,
            input_port=payload.input_port,
            output_port=payload.output_port,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Graph not found") from exc


class ExportRequest(BaseModel):
    format: str


@router.post("/{graph_id}/export", response_model=GraphExportResponse)
async def export_graph(graph_id: str, payload: ExportRequest) -> GraphExportResponse:
    try:
        return GraphExportResponse(data=service.export_graph(graph_id, payload.format))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Graph not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
