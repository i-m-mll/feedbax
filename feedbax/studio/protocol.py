"""Studio task/environment protocol helpers.

These helpers keep Studio task data classification separate from concrete
training-loop code. A scenario owns both ``task_spec`` and
``task_binding_spec``; this module answers which task data can be wired into a
graph and which data stays protocol-owned by the task/environment boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from feedbax.contracts.training import TaskSpec

GRAPH_BINDABLE_TASK_DATA_ROLES = frozenset({"model_input", "graph_input"})
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


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata(value: Any) -> dict[str, Any]:
    metadata = _field(value, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def task_data_role(data: Any) -> str:
    """Return the normalized protocol role for a Studio Task Data record."""

    explicit = _field(data, "role") or _metadata(data).get("task_data_role")
    if isinstance(explicit, str) and explicit:
        return explicit

    kind = _field(data, "kind", "")
    if kind in {"signal", "input", "model_input", "graph_input"}:
        return "model_input" if bool(_field(data, "bindable", False)) else "protocol_value"
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

    return bool(_field(data, "bindable", False)) and (
        task_data_role(data) in GRAPH_BINDABLE_TASK_DATA_ROLES
    )


def bindable_task_data(items: Iterable[Any]) -> list[Any]:
    """Filter Task Data records to graph-facing bindable inputs."""

    return [item for item in items if is_bindable_task_data(item)]


def protocol_task_data(items: Iterable[Any]) -> list[Any]:
    """Filter Task Data records to protocol-owned, non-graph data."""

    return [item for item in items if not is_bindable_task_data(item)]


def task_data_role_map(items: Iterable[Any]) -> dict[str, str]:
    """Return a mapping from Task Data id to normalized protocol role."""

    return {
        str(_field(item, "id")): task_data_role(item)
        for item in items
        if _field(item, "id") is not None
    }


def task_data_uses_protocol_path(data: Any) -> bool:
    """Return whether a Task Data path is reserved for protocol-only data."""

    path = _field(data, "path", "")
    return isinstance(path, str) and any(
        path == prefix or path.startswith(f"{prefix}.")
        for prefix in PROTOCOL_TASK_DATA_PATH_PREFIXES
    )


def _task_params(task_spec: dict[str, Any] | TaskSpec | None) -> dict[str, Any]:
    if task_spec is None:
        return {}
    params = _field(task_spec, "params", {})
    return params if isinstance(params, dict) else {}


def _task_timeline(task_spec: dict[str, Any] | TaskSpec | None) -> dict[str, Any]:
    if task_spec is None:
        return {}
    timeline = _field(task_spec, "timeline", {})
    return timeline if isinstance(timeline, dict) else {}


def task_n_steps_values(
    task_spec: dict[str, Any] | TaskSpec | None,
) -> list[tuple[str, Any]]:
    """Return declared Studio task step-count candidates with spec paths."""

    params = _task_params(task_spec)
    timeline = _task_timeline(task_spec)
    values: list[tuple[str, Any]] = []
    if "n_steps" in timeline:
        values.append(("/timeline/n_steps", timeline["n_steps"]))
    if "n_steps" in params:
        values.append(("/params/n_steps", params["n_steps"]))
    if "n_reach_steps" in params:
        values.append(("/params/n_reach_steps", params["n_reach_steps"]))
    return values


def parse_positive_n_steps(value: Any) -> Optional[int]:
    """Parse a positive integer step count, returning ``None`` on invalid input."""

    try:
        n_steps = int(value)
    except (TypeError, ValueError):
        return None
    return n_steps if n_steps > 0 else None


def infer_task_n_steps(
    task_spec: dict[str, Any] | TaskSpec | None,
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    """Infer the scenario-owned task step count from ``task_spec``.

    Top-level task timelines are preferred over task params, then compact
    ``n_steps``/``n_reach_steps`` task params are used. Invalid candidates are
    ignored here so callers can use provider validation for pathful errors.
    """

    params = _task_params(task_spec)
    if "n_control_stages" in params:
        parsed = parse_positive_n_steps(params["n_control_stages"])
        if parsed is not None:
            return parsed

    for _path, value in task_n_steps_values(task_spec):
        parsed = parse_positive_n_steps(value)
        if parsed is not None:
            return parsed
    return default
