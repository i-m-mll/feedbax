"""Task-data role classification vocabulary.

A scenario owns both ``task_spec`` and ``task_binding_spec``; this module
answers which task data can be wired into a graph and which data stays
protocol-owned by the task/environment boundary.

The classification is pure vocabulary over duck-typed records, so it lives with
the graph runtime that consumes it rather than in the Studio integration layer.
``feedbax.studio.protocol`` re-exports these names for Studio-facing callers.
"""

from __future__ import annotations

from typing import Any

GRAPH_BINDABLE_TASK_DATA_ROLES = frozenset(
    {"model_input", "graph_input", "component_parameter"}
)
PROTOCOL_TASK_DATA_ROLES = frozenset(
    {
        "target",
        "initial_state",
        "intervention",
        "eval_control",
        "trial_control",
        "compact_task_trajectory",
        "materialized_task_trajectory",
        "protocol_value",
    }
)
TASK_DATA_ROLES = GRAPH_BINDABLE_TASK_DATA_ROLES | PROTOCOL_TASK_DATA_ROLES

PROTOCOL_TASK_DATA_KINDS = frozenset(
    {
        "target",
        "initial_state",
        "intervention",
        "eval_control",
        "trial_control",
        "validation",
        "protocol_value",
    }
)
PROTOCOL_TASK_DATA_PATH_PREFIXES = (
    "targets",
    "inits",
    "intervene",
    "validation_trials",
    "eval",
    "trial_controls",
    "task.validation_trials",
)

__all__ = [
    "GRAPH_BINDABLE_TASK_DATA_ROLES",
    "PROTOCOL_TASK_DATA_KINDS",
    "PROTOCOL_TASK_DATA_PATH_PREFIXES",
    "PROTOCOL_TASK_DATA_ROLES",
    "TASK_DATA_ROLES",
    "is_bindable_task_data",
    "spec_field",
    "spec_metadata",
    "task_data_role",
    "task_data_surface",
    "task_data_uses_protocol_path",
]


def spec_field(value: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping-shaped or attribute-shaped spec record."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def spec_metadata(value: Any) -> dict[str, Any]:
    """Return the ``metadata`` mapping of a spec record, or an empty mapping."""

    metadata = spec_field(value, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def task_data_role(data: Any) -> str:
    """Return the normalized protocol role for a Studio Task Data record."""

    explicit = spec_field(data, "role") or spec_metadata(data).get("task_data_role")
    if isinstance(explicit, str) and explicit:
        return explicit

    kind = spec_field(data, "kind", "")
    if kind in {"signal", "input", "model_input", "graph_input"}:
        return "model_input" if bool(spec_field(data, "bindable", False)) else "protocol_value"
    if kind in PROTOCOL_TASK_DATA_ROLES:
        return str(kind)
    if kind == "trial_param":
        return "trial_control"
    return "protocol_value"


def task_data_surface(data: Any) -> str:
    """Return ``graph_input`` for bindable data, otherwise ``protocol``."""

    return "graph_input" if is_bindable_task_data(data) else "protocol"


def is_bindable_task_data(data: Any) -> bool:
    """Return whether Task Data is allowed to bind into graph input ports."""

    return bool(spec_field(data, "bindable", False)) and (
        task_data_role(data) in GRAPH_BINDABLE_TASK_DATA_ROLES
    )


def task_data_uses_protocol_path(data: Any) -> bool:
    """Return whether a Task Data path is reserved for protocol-only data."""

    path = spec_field(data, "path", "")
    return isinstance(path, str) and any(
        path == prefix or path.startswith(f"{prefix}.")
        for prefix in PROTOCOL_TASK_DATA_PATH_PREFIXES
    )
