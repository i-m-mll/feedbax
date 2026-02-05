from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pathlib import Path

from feedbax.web.services.component_registry import ComponentRegistry
from feedbax.web.models.component import ComponentDefinition

router = APIRouter()
registry = ComponentRegistry()


@router.get('')
async def list_components():
    return {'components': registry.list_all()}


@router.get('/{name}')
async def get_component(name: str):
    component = registry.get(name)
    if component is None:
        raise HTTPException(status_code=404, detail='Component not found')
    return ComponentDefinition(
        name=component.name,
        category=component.category,
        description=component.description,
        param_schema=component.param_schema,
        input_ports=component.input_ports,
        output_ports=component.output_ports,
        icon=component.icon,
        default_params=component.default_params,
    )


@router.post('/refresh')
async def refresh_components():
    before = {component.name for component in registry.list_all()}
    registry.load_user_components(Path.home() / '.feedbax' / 'components')
    after = {component.name for component in registry.list_all()}
    return {'added': sorted(after - before), 'removed': sorted(before - after)}
