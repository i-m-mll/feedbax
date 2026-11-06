"""Shared representation declaration helpers."""

from collections.abc import Mapping
from typing import Any

from feedbax.contracts.representation import RepresentationSpec


def _representation(data: Mapping[str, Any]) -> RepresentationSpec:
    return RepresentationSpec.model_validate(data)


def _selector_binding(
    namespace: str,
    compact: str,
    *,
    target_id: str | None = None,
    path: str | None = None,
    anchor_subpath: str | None = None,
    dim: int | None = None,
    role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selector: dict[str, Any] = {
        "namespace": namespace,
        "compact": compact,
    }
    if target_id is not None:
        selector["target_id"] = target_id
    if path is not None:
        selector["path"] = path
    if role is not None:
        selector["role"] = role
    if metadata:
        selector["metadata"] = dict(metadata)
    binding: dict[str, Any] = {
        "kind": "selector",
        "selector": selector,
    }
    if anchor_subpath is not None:
        binding["anchor_subpath"] = anchor_subpath
    if dim is not None:
        binding["dim"] = dim
    return binding


def _literal_binding(value: Any, *, dim: int | None = None) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "literal", "value": value}
    if dim is not None:
        binding["dim"] = dim
    return binding


def _param_binding(
    path: str,
    *,
    expected_type: str | None = None,
    dim: int | None = None,
) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "param_path", "path": path}
    if expected_type is not None:
        binding["expected_type"] = expected_type
    if dim is not None:
        binding["dim"] = dim
    return binding


def _trial_binding(path: str, *, dim: int | None = None) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "trial_spec_path", "path": path}
    if dim is not None:
        binding["dim"] = dim
    return binding


def _compose_children_metadata() -> dict[str, Any]:
    return {
        "composition_rule": {
            "kind": "subgraph_children",
            "allow_outer_geometry_fallback": False,
        }
    }
