"""Studio task/environment protocol helpers.

These helpers keep Studio task data classification separate from concrete
training-loop code. A scenario owns both ``task_spec`` and
``task_binding_spec``; this module answers which task data can be wired into a
graph and which data stays protocol-owned by the task/environment boundary.

The task-data role vocabulary itself lives in
``feedbax.runtime.task_data_roles`` so the graph runtime can classify task data
without importing the Studio integration layer. It is re-exported here for
Studio-facing callers.
"""

from __future__ import annotations

from typing import Any, Optional

from feedbax.contracts.training import TaskSpec
from feedbax.runtime.task_data_roles import (
    GRAPH_BINDABLE_TASK_DATA_ROLES,
    PROTOCOL_TASK_DATA_KINDS,
    PROTOCOL_TASK_DATA_PATH_PREFIXES,
    PROTOCOL_TASK_DATA_ROLES,
    TASK_DATA_ROLES,
    is_bindable_task_data,
    spec_field,
    task_data_role,
    task_data_surface,
    task_data_uses_protocol_path,
)

__all__ = [
    "GRAPH_BINDABLE_TASK_DATA_ROLES",
    "PROTOCOL_TASK_DATA_KINDS",
    "PROTOCOL_TASK_DATA_PATH_PREFIXES",
    "PROTOCOL_TASK_DATA_ROLES",
    "TASK_DATA_ROLES",
    "infer_task_n_steps",
    "is_bindable_task_data",
    "parse_positive_n_steps",
    "task_data_role",
    "task_data_surface",
    "task_data_uses_protocol_path",
    "task_n_steps_values",
]


def _task_params(task_spec: dict[str, Any] | TaskSpec | None) -> dict[str, Any]:
    if task_spec is None:
        return {}
    params = spec_field(task_spec, "params", {})
    return params if isinstance(params, dict) else {}


def task_n_steps_values(
    task_spec: dict[str, Any] | TaskSpec | None,
) -> list[tuple[str, Any]]:
    """Return declared Studio task step-count candidates with spec paths."""

    params = _task_params(task_spec)
    values: list[tuple[str, Any]] = []
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

    Compact ``n_steps``/``n_reach_steps`` task params are used. The typed task
    timeline describes epochs inside that fixed runtime length; it does not own
    the rollout length. Invalid candidates are ignored here so callers can use
    provider validation for pathful errors.
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
