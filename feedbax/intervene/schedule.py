"""Intervention scheduling utilities for graph-based models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, TypeAlias

import equinox as eqx
import jax.tree as jt
import jax_cookbook.tree as jtree
from jax_cookbook import is_module
from jaxtyping import PRNGKeyArray, PyTree

from feedbax.runtime.batch import BatchInfo
from feedbax.runtime.naming import get_unique_label

if TYPE_CHECKING:
    from feedbax.tasks import AbstractTask


IntervenorLabelStr: TypeAlias = str


class TimeSeriesParam(eqx.Module):
    """Wrapper to mark per-timestep parameters."""

    value: Any


class InterventionSpec(eqx.Module):
    """Specifies how to generate intervention parameters for a task."""

    params: PyTree
    default_active: bool = False


def evaluate_intervenor_params(
    params: PyTree,
    trial_spec,
    batch_info: BatchInfo,
    key: PRNGKeyArray,
):
    """Evaluate callable intervention parameter leaves with independent keys."""

    def is_timeseries_param(leaf):
        return isinstance(leaf, TimeSeriesParam)

    callables, other_values = eqx.partition(
        params,
        lambda leaf: callable(leaf) and not is_timeseries_param(leaf),
        is_leaf=is_timeseries_param,
    )
    keys = jtree.random_split_like_tree(key, callables, is_leaf=is_timeseries_param)
    values = jt.map(
        lambda leaf, leaf_key: leaf(trial_spec, batch_info, key=leaf_key),
        callables,
        keys,
        is_leaf=is_timeseries_param,
    )
    return eqx.combine(values, other_values, is_leaf=is_timeseries_param)


def schedule_intervenor(
    tasks: PyTree["AbstractTask"],
    models: PyTree[Any],
    label: Optional[str] = None,
    *,
    intervenor_params: PyTree,
    default_active: bool = False,
    validation_same_schedule: bool = True,
    intervenor_params_validation: Optional[PyTree] = None,
) -> Tuple[PyTree["AbstractTask"], PyTree[Any]]:
    """Attach intervention parameter schedules to tasks.

    This does not modify models. Models should already include intervention
    components whose parameter StateIndex labels match `label`.
    """
    invalid_labels_tasks = jt.reduce(
        lambda x, y: x + y,
        jt.map(
            lambda task: tuple(task.intervention_specs.all.keys()),
            tasks,
            is_leaf=is_module,
        ),
        is_leaf=lambda x: isinstance(x, tuple),
    )
    invalid_labels = set(invalid_labels_tasks)

    if label is None:
        label = get_unique_label("intervention", invalid_labels)

    intervention_specs = {label: InterventionSpec(intervenor_params, default_active)}
    if intervenor_params_validation is not None:
        intervention_specs_validation = {
            label: InterventionSpec(intervenor_params_validation, default_active)
        }
    elif validation_same_schedule:
        intervention_specs_validation = intervention_specs
    else:
        intervention_specs_validation = {}

    tasks = jt.map(
        lambda task: eqx.tree_at(
            lambda task: (
                task.intervention_specs.training,
                task.intervention_specs.validation,
            ),
            task,
            (
                task.intervention_specs.training | intervention_specs,
                task.intervention_specs.validation | intervention_specs_validation,
            ),
        ),
        tasks,
        is_leaf=is_module,
    )

    return tasks, models
