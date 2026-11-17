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

from feedbax.runtime.task_data_roles import (
    GRAPH_BINDABLE_TASK_DATA_ROLES,
    PROTOCOL_TASK_DATA_KINDS,
    PROTOCOL_TASK_DATA_PATH_PREFIXES,
    PROTOCOL_TASK_DATA_ROLES,
    TASK_DATA_ROLES,
    is_bindable_task_data,
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
    "is_bindable_task_data",
    "task_binding_id",
    "task_data_role",
    "task_data_surface",
    "task_data_uses_protocol_path",
]


def task_binding_id(source_data_id: str, node_id: str, port: str) -> str:
    return f"task:{source_data_id}->{node_id}:{port}"
