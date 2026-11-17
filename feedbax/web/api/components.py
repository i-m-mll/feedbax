from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request

from feedbax.plugins.declarations import missing_declaration
from feedbax.contracts.studio_api import (
    ComponentDetailResponse,
    ComponentListResponse,
)

router = APIRouter()


@router.get("", response_model=ComponentListResponse)
async def list_components(request: Request) -> ComponentListResponse:
    registry = request.app.state.bootstrap_state.bundle.components
    return ComponentListResponse(data={"components": registry.list_all()})


@router.get("/{name}", response_model=ComponentDetailResponse)
async def get_component(name: str, request: Request) -> ComponentDetailResponse:
    registry = request.app.state.bootstrap_state.bundle.components
    component = registry.get(name)
    if component is None:
        raise HTTPException(status_code=404, detail=missing_declaration("component", name))
    return ComponentDetailResponse(data=registry._to_definition(component))
