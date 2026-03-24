from __future__ import annotations
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional

from feedbax.web.models.graph import AnalysisPageSpec, GraphSpec, GraphUIState
from feedbax.web.services.graph_service import GraphService

router = APIRouter()
service = GraphService()


class GraphCreateRequest(BaseModel):
    graph: GraphSpec
    ui_state: Optional[GraphUIState] = None


class GraphUpdateRequest(BaseModel):
    graph: Optional[GraphSpec] = None
    ui_state: Optional[GraphUIState] = None
    analysis_pages: Optional[list[AnalysisPageSpec]] = None
    active_analysis_page_id: Optional[str] = None


@router.get('')
async def list_graphs(response: Response):
    response.headers['Cache-Control'] = 'no-store'
    return {'graphs': service.list_graphs()}


@router.post('')
async def create_graph(payload: GraphCreateRequest):
    record = service.create_graph(payload.graph, payload.ui_state)
    return {'id': record.graph_id, 'metadata': record.project.metadata}


@router.get('/{graph_id}')
async def get_graph(graph_id: str, response: Response):
    response.headers['Cache-Control'] = 'no-store'
    try:
        record = service.get_graph(graph_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    return {
        'graph': record.project.graph,
        'ui_state': record.project.ui_state,
        'demo_training_data': record.project.demo_training_data,
        'metadata': record.project.metadata,
        'analysis_pages': record.project.analysis_pages,
        'active_analysis_page_id': record.project.active_analysis_page_id,
    }


@router.put('/{graph_id}')
async def update_graph(graph_id: str, payload: GraphUpdateRequest):
    try:
        service.update_graph(
            graph_id, payload.graph, payload.ui_state, payload.analysis_pages,
            payload.active_analysis_page_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    return {'success': True}


@router.post('/{graph_id}/beacon')
async def beacon_update_graph(graph_id: str, payload: GraphUpdateRequest):
    """sendBeacon endpoint for pagehide saves; returns 204 No Content."""
    try:
        service.update_graph(
            graph_id, payload.graph, payload.ui_state, payload.analysis_pages,
            payload.active_analysis_page_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    return Response(status_code=204)


@router.delete('/{graph_id}')
async def delete_graph(graph_id: str):
    service.delete_graph(graph_id)
    return {'success': True}


@router.post('/{graph_id}/validate')
async def validate_graph(graph_id: str, graph: GraphSpec):
    return service.validate_graph(graph)


class ExportRequest(BaseModel):
    format: str


@router.post('/{graph_id}/export')
async def export_graph(graph_id: str, payload: ExportRequest):
    try:
        return service.export_graph(graph_id, payload.format)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
