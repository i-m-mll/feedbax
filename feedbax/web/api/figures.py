"""Studio catalog for installed figure templates."""

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()


class RegistryItem(BaseModel):
    """Studio-enumerable figure registry item."""

    name: str
    description: str
    metadata: dict[str, Any] = {}


@router.get("/templates", response_model=list[RegistryItem])
async def list_figure_templates(request: Request) -> list[RegistryItem]:
    """List registered figure templates."""
    registry = request.app.state.bootstrap_state.bundle.figure_templates
    return [
        RegistryItem(
            name=template.key,
            description=template.description,
            metadata={
                "label": template.label,
                "renamed_from": list(template.renamed_from),
                "settings_schema": template.settings_model.model_json_schema(),
                "package": template.source.package,
                "distribution": template.source.distribution,
                "version": template.source.version,
            },
        )
        for key in registry.keys()
        for template in (registry.resolve(key),)
    ]
