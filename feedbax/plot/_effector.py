"""Shared effector trajectory plot preparation."""

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from jaxtyping import Array, Float, PyTree, Shaped

from feedbax.config.selectors import where_func_to_attr_str_tree
from feedbax.models.feedback import SimpleFeedbackState

if TYPE_CHECKING:
    from feedbax.tasks import TaskTrialSpec


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


def resolve_effector_endpoints(
    trial_specs: Optional["TaskTrialSpec"],
) -> np.ndarray | None:
    """Return ``[init, goal]`` endpoint coordinates from task trial specs."""
    if trial_specs is None:
        return None

    target_specs = trial_specs.targets["mechanics.effector.pos"]
    if isinstance(target_specs, Mapping):
        target_specs = next(iter(target_specs.values()))
    if target_specs.value is None:
        return None

    return np.array(
        [
            trial_specs.inits["mechanics.effector"].pos,
            target_specs.value[:, -1],
        ]
    )
