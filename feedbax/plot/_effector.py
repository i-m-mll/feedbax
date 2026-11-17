"""Shared effector trajectory plot preparation."""

from collections.abc import Callable, Sequence
from typing import Any, Optional

from jaxtyping import Array, Float, PyTree, Shaped

from feedbax.runtime.where_selectors import where_func_to_attr_str_tree
from feedbax.models.feedback import SimpleFeedbackState


def resolve_effector_trajectory_vars(
    states: SimpleFeedbackState | PyTree[Float[Array, "trial time ..."] | Any],
    where_data: Optional[
        Callable[[PyTree[Array]], Sequence[Shaped[Array, "*batch trial time xy=2"]]]
    ] = None,
    var_labels: Optional[tuple[str, ...]] = None,
) -> tuple[tuple[Shaped[Array, "*batch trial time xy=2"], ...], Optional[tuple[str, ...]]]:
    """Return effector trajectory arrays and labels for plot backends."""
    if where_data is not None:
        vars_tuple = tuple(where_data(states))
        if var_labels is None:
            var_labels = where_func_to_attr_str_tree(where_data)
        return vars_tuple, var_labels

    if isinstance(states, SimpleFeedbackState):
        vars_tuple = (
            states.mechanics.effector.pos,
            states.mechanics.effector.vel,
            states.efferent.output,
        )
        if var_labels is None:
            var_labels = ("Position", "Velocity", "Force")
        return vars_tuple, var_labels

    raise ValueError(
        "If `states` is not a `SimpleFeedbackState`, `where_data` must be provided."
    )
