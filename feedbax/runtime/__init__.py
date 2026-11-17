"""Graph execution runtime primitives."""

from feedbax.runtime.graph import (
    Component,
    GraphNode,
    Graph,
    GraphState,
    GraphTraceRequest,
    RolloutStepContext,
    RolloutStepHook,
    RolloutStepHookResult,
    Wire,
    init_state_from_component,
)
from feedbax.runtime.model import (
    AbstractModel,
    wrap_stateless_callable,
    wrap_stateless_keyless_callable,
)
from feedbax.runtime.selectors import Selection, select, tree_at
from feedbax.runtime.state import CartesianState, State, StateBounds, StateT, clip_state
from feedbax.runtime.state_indices import align_state_indices_like
from feedbax.runtime.task_bindings import (
    TaskInputPlan,
    expose_task_inputs,
)

__all__ = [
    "AbstractModel",
    "CartesianState",
    "Component",
    "GraphNode",
    "Graph",
    "GraphState",
    "GraphTraceRequest",
    "RolloutStepContext",
    "RolloutStepHook",
    "RolloutStepHookResult",
    "Selection",
    "State",
    "StateBounds",
    "StateT",
    "TaskInputPlan",
    "Wire",
    "align_state_indices_like",
    "clip_state",
    "expose_task_inputs",
    "init_state_from_component",
    "select",
    "tree_at",
    "wrap_stateless_callable",
    "wrap_stateless_keyless_callable",
]
