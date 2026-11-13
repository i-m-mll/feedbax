"""App-local window preferences served only by the desktop service."""

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import Field, FiniteFloat

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelElementId
from feedbax.contracts.project_inputs import ProjectName
from feedbax.desktop.control import _read_private_json, write_private_json

_MAX_BYTES = 128 * 1024
type _Slot = Literal["primary", "secondary"]
type _View = Literal[
    "runs", "model", "task", "objective", "training", "scene", "results", "analysis", "figures", "document",
]


class _ModelPath(StrictModel):
    document: ProjectName
    path: list[ModelElementId]


class WindowArrangement(StrictModel):
    mode: Literal["map", "both", "details"]
    primaryView: _View
    secondaryView: _View | None
    mapDrawerOpen: bool
    detailsDrawerOpen: bool
    inspectorOpen: bool
    mapHeight: FiniteFloat = Field(gt=0)
    mapDrawerWidth: FiniteFloat = Field(gt=0)
    detailsDrawerWidth: FiniteFloat = Field(gt=0)
    inspectorWidth: FiniteFloat = Field(gt=0)
    secondaryWidth: FiniteFloat = Field(gt=0)
    modelPaths: dict[_Slot, _ModelPath] = Field(default_factory=dict)


def register_preferences(app: FastAPI, path: Path) -> None:
    def read() -> dict:
        try:
            return _read_private_json(path, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return {}

    @app.get("/api/desktop/preferences/windows/{project_id}")
    async def get_arrangement(project_id: ProjectName) -> WindowArrangement | None:
        value = read().get("windows", {}).get(project_id)
        return WindowArrangement.model_validate(value) if value is not None else None

    @app.put("/api/desktop/preferences/windows/{project_id}", status_code=204)
    async def set_arrangement(project_id: ProjectName, arrangement: WindowArrangement) -> None:
        preferences = read()
        preferences.setdefault("windows", {})[project_id] = arrangement.model_dump(mode="json")
        if len(json.dumps(preferences, separators=(",", ":")).encode()) > _MAX_BYTES:
            raise HTTPException(413, "Studio preferences exceed their storage limit")
        write_private_json(path, preferences)
