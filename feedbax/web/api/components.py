from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pathlib import Path

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.studio_api import (
    ComponentDetailResponse,
    ComponentListResponse,
    ComponentRefreshResponse,
)

router = APIRouter()
registry = ComponentRegistry()


@router.get('', response_model=ComponentListResponse)
async def list_components() -> ComponentListResponse:
    return ComponentListResponse(data={'components': registry.list_all()})


@router.get('/{name}', response_model=ComponentDetailResponse)
async def get_component(name: str) -> ComponentDetailResponse:
    component = registry.get(name)
    if component is None:
        raise HTTPException(status_code=404, detail='Component not found')
    return ComponentDetailResponse(data=registry._to_definition(component))


@router.post('/refresh', response_model=ComponentRefreshResponse)
async def refresh_components() -> ComponentRefreshResponse:
    before = {component.name for component in registry.list_all()}
    registry.load_user_components(Path.home() / '.feedbax' / 'components')
    after = {component.name for component in registry.list_all()}
    return ComponentRefreshResponse(
        data={'added': sorted(after - before), 'removed': sorted(before - after)}
    )
