"""App-local settings and window preferences served only by the desktop service."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import Field, FiniteFloat, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelElementId
from feedbax.contracts.project_inputs import ProjectName
from feedbax.desktop.control import _read_private_json, _read_private_json_value, write_private_json
from feedbax.project_folder import ProjectFolderLocation

_MAX_BYTES = 128 * 1024
_MAX_WINDOW_SESSION_BYTES = 1024 * 1024
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


def _legacy_bindings(settings: Path, alias: str) -> list[object]:
    bindings = []
    try:
        windows = _read_private_json(settings / "studio-windows.json", max_bytes=_MAX_WINDOW_SESSION_BYTES)
        if alias in windows:
            window = windows[alias]
            bindings.append(window.get("folder") if isinstance(window, dict) else None)
    except FileNotFoundError:
        pass
    try:
        recents = _read_private_json_value(settings / "studio-recents.json", max_bytes=_MAX_WINDOW_SESSION_BYTES)
        if not isinstance(recents, list):
            raise ValueError("invalid Studio recent projects")
        bindings.extend(entry.get("folder") for entry in recents if isinstance(entry, dict) and entry.get("id") == alias)
    except FileNotFoundError:
        pass
    return bindings


def register_preferences(app: FastAPI, path: Path) -> Callable[[str, ProjectFolderLocation], None]:
    def read() -> dict:
        try:
            return _read_private_json(path, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return {}

    def save(preferences: dict) -> None:
        if len(json.dumps(preferences, separators=(",", ":")).encode()) > _MAX_BYTES:
            raise HTTPException(413, "Studio preferences exceed their storage limit")
        write_private_json(path, preferences)

    def rebind_arrangement(project_id: str, location: ProjectFolderLocation) -> None:
        alias = location.path.name
        if alias == project_id:
            return
        preferences = read()
        windows = preferences.get("windows", {})
        if alias not in windows:
            return
        bindings = _legacy_bindings(path.parent, alias)
        expected = {"path": str(location.path), "identity": list(location.identity)}
        if bindings and all(folder == expected for folder in bindings):
            windows.setdefault(project_id, windows.pop(alias))
            save(preferences)

    @app.get("/api/desktop/preferences/app")
    async def get_settings() -> dict[str, JsonValue]:
        return read().get("app", {})

    @app.put("/api/desktop/preferences/app", status_code=204)
    async def set_settings(settings: dict[str, JsonValue]) -> None:
        preferences = read()
        preferences["app"] = settings
        save(preferences)

    @app.get("/api/desktop/preferences/windows/{project_id}")
    async def get_arrangement(project_id: ProjectName) -> WindowArrangement | None:
        value = read().get("windows", {}).get(project_id)
        return WindowArrangement.model_validate(value) if value is not None else None

    @app.put("/api/desktop/preferences/windows/{project_id}", status_code=204)
    async def set_arrangement(project_id: ProjectName, arrangement: WindowArrangement) -> None:
        preferences = read()
        preferences.setdefault("windows", {})[project_id] = arrangement.model_dump(mode="json")
        save(preferences)

    return rebind_arrangement
