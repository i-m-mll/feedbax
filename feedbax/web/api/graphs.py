from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from feedbax.web.models.graph import GraphSpec, GraphUIState
from feedbax.web.services.graph_service import GraphService

router = APIRouter()
service = GraphService()


class GraphCreateRequest(BaseModel):
    graph: GraphSpec
    ui_state: Optional[GraphUIState] = None


class GraphUpdateRequest(BaseModel):
    graph: Optional[GraphSpec] = None
    ui_state: Optional[GraphUIState] = None


@router.get('')
async def list_graphs():
    return {'graphs': service.list_graphs()}


@router.post('')
async def create_graph(payload: GraphCreateRequest):
    record = service.create_graph(payload.graph, payload.ui_state)
    return {'id': record.graph_id, 'metadata': record.project.metadata}


@router.get('/{graph_id}')
async def get_graph(graph_id: str):
    try:
        record = service.get_graph(graph_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    return {'graph': record.project.graph, 'ui_state': record.project.ui_state}


@router.put('/{graph_id}')
async def update_graph(graph_id: str, payload: GraphUpdateRequest):
    try:
        service.update_graph(graph_id, payload.graph, payload.ui_state)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc
    return {'success': True}


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
