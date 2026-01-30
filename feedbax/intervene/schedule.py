"""Intervention scheduling utilities for graph-based models."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, TypeAlias

import equinox as eqx
from equinox import Module
import jax
import jax.tree as jt
from jaxtyping import PRNGKeyArray, PyTree

from feedbax.misc import BatchInfo, get_unique_label, is_module


logger = logging.getLogger(__name__)


IntervenorLabelStr: TypeAlias = str


class TimeSeriesParam(eqx.Module):
    """Wrapper to mark per-timestep parameters."""

    value: Any


class InterventionSpec(eqx.Module):
    """Specifies how to generate intervention parameters for a task."""

    params: PyTree
    default_active: bool = False


def _eval_intervenor_param_spec(
    intervention_spec: InterventionSpec,
    trial_spec,
    batch_info: BatchInfo,
    key: PRNGKeyArray,
):
    def is_timeseries_param(x):
        return isinstance(x, TimeSeriesParam)

    return jt.map(
        lambda leaf: leaf(trial_spec, batch_info, key=key)
        if callable(leaf)
        else leaf,
        intervention_spec.params,
        is_leaf=is_timeseries_param,
    )


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
    invalid_labels_tasks = jax.tree_util.tree_reduce(
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
