from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel
from typing import Optional

from feedbax.contracts.graph import (
    AnalysisPageSpec,
    GraphSpec,
    GraphUIState,
    StudioWorkspaceSpec,
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
)
from feedbax.web.services.graph_service import GraphSaveConflictError, GraphService

router = APIRouter()
service = GraphService()


class GraphCreateRequest(BaseModel):
    graph: GraphSpec
    ui_state: Optional[GraphUIState] = None
    workspace: Optional[StudioWorkspaceSpec] = None


class GraphUpdateRequest(BaseModel):
    graph: Optional[GraphSpec] = None
    ui_state: Optional[GraphUIState] = None
    analysis_pages: Optional[list[AnalysisPageSpec]] = None
    active_analysis_page_id: Optional[str] = None
    workspace: Optional[StudioWorkspaceSpec] = None
    expected_save_revision: Optional[int] = None


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


@router.get('', response_model=GraphListResponse)
async def list_graphs(response: Response) -> GraphListResponse:
    response.headers['Cache-Control'] = 'no-store'
    return GraphListResponse(data={'graphs': service.list_graphs()})


@router.post('', response_model=GraphCreateResponse)
async def create_graph(payload: GraphCreateRequest) -> GraphCreateResponse:
    record = service.create_graph(payload.graph, payload.ui_state)
    if payload.workspace is not None:
        record = service.update_graph(
            record.graph_id,
            None,
            None,
            workspace=payload.workspace,
            expected_save_revision=record.project.metadata.save_revision,
            require_save_revision=True,
        )
    return GraphCreateResponse(data={'id': record.graph_id, 'metadata': record.project.metadata})


@router.get('/{graph_id}', response_model=GraphDetailResponse)
async def get_graph(graph_id: str, response: Response) -> GraphDetailResponse:
    response.headers['Cache-Control'] = 'no-store'
    try:
        record = service.get_graph(graph_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    return GraphDetailResponse(
        data={
            'graph': record.project.graph,
            'ui_state': record.project.ui_state,
            'demo_training_data': record.project.demo_training_data,
            'metadata': record.project.metadata,
            'analysis_pages': record.project.analysis_pages,
            'active_analysis_page_id': record.project.active_analysis_page_id,
            'workspace': record.project.workspace,
        }
    )


@router.put('/{graph_id}', response_model=GraphUpdateResponse)
async def update_graph(
    graph_id: str,
    payload: GraphUpdateRequest,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> GraphUpdateResponse:
    expected_revision = _parse_if_match_revision(if_match)
    if expected_revision is None:
        expected_revision = payload.expected_save_revision
    try:
        record = service.update_graph(
            graph_id, payload.graph, payload.ui_state, payload.analysis_pages,
            payload.active_analysis_page_id, payload.workspace,
            expected_save_revision=expected_revision,
            require_save_revision=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    except GraphSaveConflictError as exc:
        raise HTTPException(status_code=409, detail=_conflict_detail(exc)) from exc
    return GraphUpdateResponse(data={'success': True, 'metadata': record.project.metadata})


@router.post('/{graph_id}/beacon')
async def beacon_update_graph(graph_id: str, payload: GraphUpdateRequest):
    """sendBeacon endpoint for pagehide saves; returns 204 No Content."""
    try:
        service.update_graph(
            graph_id, payload.graph, payload.ui_state, payload.analysis_pages,
            payload.active_analysis_page_id, payload.workspace,
            expected_save_revision=payload.expected_save_revision,
            require_save_revision=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    except GraphSaveConflictError as exc:
        raise HTTPException(status_code=409, detail=_conflict_detail(exc)) from exc
    return Response(status_code=204)


@router.delete('/{graph_id}', response_model=SuccessResponse)
async def delete_graph(graph_id: str) -> SuccessResponse:
    service.delete_graph(graph_id)
    return SuccessResponse(data=SuccessPayload(success=True))


@router.post('/{graph_id}/validate', response_model=GraphValidationResponse)
async def validate_graph(graph_id: str, graph: GraphSpec) -> GraphValidationResponse:
    return GraphValidationResponse(data=service.validate_graph(graph))


class ExportRequest(BaseModel):
    format: str


@router.post('/{graph_id}/export', response_model=GraphExportResponse)
async def export_graph(graph_id: str, payload: ExportRequest) -> GraphExportResponse:
    try:
        return GraphExportResponse(data=service.export_graph(graph_id, payload.format))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
