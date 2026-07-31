from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request

from feedbax.plugins.composition import compose_application
from feedbax.contracts.studio_api import (
    ComponentDetailResponse,
    ComponentListResponse,
    ComponentRefreshResponse,
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
        raise HTTPException(status_code=404, detail="Component not found")
    return ComponentDetailResponse(data=registry._to_definition(component))


@router.post("/refresh", response_model=ComponentRefreshResponse)
async def refresh_components(request: Request) -> ComponentRefreshResponse:
    registry = request.app.state.bootstrap_state.bundle.components
    before = {component.name for component in registry.list_all()}
    request.app.state.bootstrap_state = await compose_application(
        modules=request.app.state.bootstrap_modules
    )
    registry = request.app.state.bootstrap_state.bundle.components
    after = {component.name for component in registry.list_all()}
    return ComponentRefreshResponse(
        data={"added": sorted(after - before), "removed": sorted(before - after)}
    )
