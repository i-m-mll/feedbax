"""Ordered composition deltas for managed project documents."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import re
from typing import Any

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_override_patch_in_place


_PATH_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class MatrixCompositionDelta(StrictModel):
    """One ordered authored layer with explicit ancestor-override acknowledgement."""

    layer_id: str
    patches: list[OverridePatch] = Field(default_factory=list)
    acknowledges_ancestor_paths: list[str] = Field(default_factory=list)
    schema_id: str | None = None
    schema_version: str | None = None

    @model_validator(mode="after")
    def _validate_delta(self) -> "MatrixCompositionDelta":
        if not _PATH_SAFE_RE.match(self.layer_id):
            raise ValueError(f"/deltas/layer_id is not path-safe: {self.layer_id!r}")
        if (self.schema_id is None) != (self.schema_version is None):
            raise ValueError(
                "/deltas schema_id and schema_version must be declared together at a boundary"
            )
        for path in self.acknowledges_ancestor_paths:
            _validate_dotted_path(path, "/deltas/acknowledges_ancestor_paths")
        return self


class CompositionDeltaError(ValueError):
    """A patch error with its authored layer and path."""

    def __init__(self, layer: str, path: str, message: str):
        self.layer = layer
        self.path = path
        super().__init__(message)


def apply_composition_deltas(
    payload: dict[str, Any],
    deltas: list[MatrixCompositionDelta],
    *,
    ancestor_written_paths: set[str] | None = None,
    apply_patch: Callable[[dict[str, Any], OverridePatch], None] = apply_override_patch_in_place,
) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    """Apply ordered layers and refuse unacknowledged ancestor overrides."""
    result = deepcopy(payload)
    written = set(ancestor_written_paths or ())
    attribution: dict[str, str] = {}
    current_schema = _payload_schema_identity(result)
    for delta in deltas:
        acknowledged = set(delta.acknowledges_ancestor_paths)
        same_layer_appends: set[str] = set()
        declared_boundary = (delta.schema_id, delta.schema_version)
        prior_identities = _schema_identities(result)
        for patch in delta.patches:
            conflicts = sorted(
                path
                for path in written
                if _paths_overlap(path, patch.path)
                and not (
                    patch.op == "add"
                    and patch.path.endswith(".-")
                    and path == patch.path
                    and path in same_layer_appends
                )
            )
            unacknowledged = [
                path
                for path in conflicts
                if not any(
                    _paths_overlap(path, acknowledgement)
                    and _paths_overlap(patch.path, acknowledgement)
                    for acknowledgement in acknowledged
                )
            ]
            if unacknowledged:
                raise CompositionDeltaError(
                    delta.layer_id,
                    patch.path,
                    f"/deltas/{delta.layer_id}/{patch.path} overlaps ancestor-written "
                    f"paths {unacknowledged!r} without explicit acknowledgement",
                )
            try:
                apply_patch(result, patch)
            except ValueError as error:
                raise CompositionDeltaError(
                    delta.layer_id, patch.path, f"/deltas/{delta.layer_id}: {error}"
                ) from error
            written.add(patch.path)
            if patch.op == "add" and patch.path.endswith(".-"):
                same_layer_appends.add(patch.path)
            attribution[patch.path] = delta.layer_id
        resulting_schema = _payload_schema_identity(result)
        resulting_identities = _schema_identities(result)
        if declared_boundary != (None, None):
            if declared_boundary == current_schema:
                raise ValueError(
                    f"/deltas/{delta.layer_id} declares schema boundary "
                    f"{declared_boundary!r} but does not change the active identity"
                )
            if resulting_schema != declared_boundary:
                raise ValueError(
                    f"/deltas/{delta.layer_id} declares schema boundary {declared_boundary!r} "
                    f"but flattened payload has identity {resulting_schema!r}"
                )
        elif resulting_identities != prior_identities:
            raise ValueError(
                f"/deltas/{delta.layer_id} changes schema identity from "
                f"{prior_identities!r} to {resulting_identities!r} without a declared "
                "schema_id/schema_version boundary"
            )
        current_schema = resulting_schema
    return result, attribution, written


def _paths_overlap(left: str, right: str) -> bool:
    """Return whether either dotted path denotes the other's subtree."""
    return left == right or left.startswith(f"{right}.") or right.startswith(f"{left}.")


def _payload_schema_identity(payload: dict[str, Any]) -> tuple[Any, Any]:
    return payload.get("schema_id"), payload.get("schema_version")


def _schema_identities(value: Any, path: str = "") -> dict[str, tuple[Any, Any]]:
    identities: dict[str, tuple[Any, Any]] = {}
    if isinstance(value, dict):
        if "schema_id" in value or "schema_version" in value:
            identities[path or "/"] = (value.get("schema_id"), value.get("schema_version"))
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            identities.update(_schema_identities(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}.{index}" if path else str(index)
            identities.update(_schema_identities(child, child_path))
    return identities


def _validate_dotted_path(path: str, field_path: str) -> None:
    if not path.strip() or any(not part for part in path.split(".")):
        raise ValueError(f"{field_path} is not dotted-path-like: {path!r}")
