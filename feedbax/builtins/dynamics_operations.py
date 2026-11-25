"""Analyze retained activity and the exact step dynamics of a saved Model."""

from __future__ import annotations

from typing import Annotated

import jax
from jax.flatten_util import ravel_pytree
from jax.tree_util import keystr
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from feedbax.analysis.fp_finder import FixedPointFinder, fp_adam_optimizer
from feedbax.analysis.operations import DatasetOutput, operation
from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.operation import DatasetInputPort, DatasetPort, JaxOperation, ModelPort, TaskPort
from feedbax.contracts.rollout import RAW_STATES_SCHEMA
from feedbax.contracts.rollout_dataset import rollout_coordinates
from feedbax.dataset import Dataset, Variable
from feedbax.execution.trial_execution import TrialExecution, sample_evaluation_trials
from feedbax.pca import PCACentering, PCANormalization, fit_pca_samples
from feedbax.runtime._graph_key_schedule import derive_key
from feedbax.runtime.graph import Graph
from feedbax.tasks.document import TaskParameters


_CPU = JaxOperation(backend="cpu", floating_point_bits=32)
_ROLLOUT_AXES = ("checkpoint", "update", "condition", "trial", "state_time")
_PCA = DatasetPort(
    schema_ids=("feedbax.analysis.hidden_state_pca.v1",),
    optional_dimensions=("replicate", "checkpoint", "update"),
    variables={
        "basis": FieldDescriptor(dims=("component", "hidden_unit"), units="1"),
        "mean": FieldDescriptor(dims=("hidden_unit",), units="1"),
        "scale": FieldDescriptor(dims=("hidden_unit",), units="1"),
        "explained_variance": FieldDescriptor(dims=("component",), units="1"),
        "explained_variance_ratio": FieldDescriptor(dims=("component",), units="1"),
        "scores": FieldDescriptor(dims=(*_ROLLOUT_AXES, "component"), units="1"),
    },
    coordinates={
        name: FieldDescriptor(dims=(name,))
        for name in ("replicate", "checkpoint", "update", "condition", "trial", "component", "hidden_unit")
    } | {
        "state_time": rollout_coordinates(None)["state_time"],
    },
)
_FIXED_POINTS = DatasetPort(
    schema_ids=("feedbax.analysis.fixed_points.v1",),
    optional_dimensions=("replicate",),
    variables={
        "fixed_points": FieldDescriptor(dims=("condition", "candidate", "state_component")),
        "residual": FieldDescriptor(dims=("condition", "candidate"), label="Mean-square step residual"),
        "finite": FieldDescriptor(dims=("condition", "candidate")),
        "accepted": FieldDescriptor(dims=("condition", "candidate")),
        "accepted_count": FieldDescriptor(dims=("condition",), units="1"),
    },
    coordinates={
        "replicate": FieldDescriptor(dims=("replicate",)),
        "condition": FieldDescriptor(dims=("condition",)),
        "candidate": FieldDescriptor(dims=("candidate",)),
        "state_component": FieldDescriptor(dims=("state_component",)),
    },
)


class HiddenStatePCAParameters(StrictModel):
    state: str = Field(min_length=1, description="Address of the vector state field retained by Evaluate.")
    n_components: int = Field(default=3, ge=1, description="Number of principal components to retain.")
    centering: PCACentering = "mean"
    normalization: PCANormalization = "none"


@operation(settings=HiddenStatePCAParameters, label="Hidden-state PCA", numerical=_CPU)
def hidden_state_pca(
    states: Annotated[Dataset, DatasetInputPort(schema_ids=(RAW_STATES_SCHEMA,), takes="one")],
    params: HiddenStatePCAParameters,
) -> Annotated[Dataset, DatasetOutput("pca", _PCA)]:
    """Fit one basis on a saved rollout and project every retained trial into the same axes."""
    descriptor = states.descriptor
    if (field := descriptor.variables.get(params.state)) is None:
        raise ValueError(f"Evaluate did not retain the state field {params.state!r}.")
    axes = tuple(axis for axis in _ROLLOUT_AXES if axis in descriptor.dimensions)
    if field.dims[:-1] != axes or not params.state.startswith("/v1/model/state/") or field.units != "1":
        raise ValueError("PCA requires a dimensionless vector Model state over recorded trial time.")
    unit = field.dims[-1]
    values = jnp.asarray(states.data(params.state))
    fitted = fit_pca_samples(
        values.reshape((-1, values.shape[-1])),
        n_components=params.n_components,
        centering=params.centering,
        normalization=params.normalization,
    )
    port = _PCA.for_dimensions(descriptor.dimensions)
    arrays = {
        "basis": fitted.basis,
        "mean": fitted.mean,
        "scale": fitted.scale,
        "explained_variance": fitted.explained_variance,
        "explained_variance_ratio": fitted.explained_variance_ratio,
        "scores": fitted.transform(values),
    }
    coordinates = {
        name: Variable.from_field(states.data(name), coordinate, coordinate.dims)
        for name, coordinate in descriptor.coords.items()
        if all(axis in axes for axis in coordinate.dims)
    }
    coordinates["component"] = Variable(np.arange(1, params.n_components + 1), ("component",))
    coordinates["hidden_unit"] = Variable(states.data(unit), ("hidden_unit",))
    return Dataset.from_arrays(
        {name: Variable.from_field(arrays[name], out, out.dims) for name, out in port.variables.items()},
        coords=coordinates,
        attrs={"source_state": params.state, "sample_count": fitted.sample_count},
        schema_id=port.schema_ids[0],
    )


class FixedPointParameters(StrictModel):
    transition: int = Field(default=0, ge=0, description="Zero-based Task transition whose inputs are held fixed.")
    seed: int = Field(default=0, ge=0, lt=2**32, description="Random seed for the Task conditions.")
    max_updates: int = Field(default=1000, ge=1, le=10_000, description="Maximum optimizer updates per condition.")
    loss_tolerance: float = Field(default=1e-6, gt=0, allow_inf_nan=False, description="Maximum mean-square step residual.")
    merge_tolerance: float = Field(default=0.025, ge=0, allow_inf_nan=False, description="Distance for merging duplicate roots.")


@operation(settings=FixedPointParameters, label="Fixed points", numerical=_CPU)
def fixed_points(
    model: Annotated[Graph, ModelPort()],
    task: Annotated[TrialExecution, TaskPort()],
    params: FixedPointParameters,
) -> Annotated[Dataset, DatasetOutput("roots", _FIXED_POINTS)]:
    """Find equilibria of the Model's Step state with Task inputs and randomness held at one transition."""
    if params.transition >= task.task.n_steps:
        raise ValueError("The fixed-point transition is outside the Task.")
    trial_ids = task.task.condition_ids or ("0",)
    key = jax.random.key(params.seed)
    trials, keys = sample_evaluation_trials(task.task, TaskParameters(), key=key, trial_ids=trial_ids)
    _, history, evaluated = task.execute_batch(
        model, trials, keys, capture_state=lambda state, carry: (state, carry or {}),
    )
    assert history is not None
    # State sample k is the state immediately before transition k.
    states = [jax.tree.map(lambda value: value[index, params.transition], history) for index in range(len(trial_ids))]
    leaves = jax.tree.leaves_with_path(states[0])
    if not leaves or any(not jnp.issubdtype(value.dtype, jnp.floating) for _, value in leaves):
        raise ValueError("Fixed points require a Model with a nonempty floating-point Step state.")
    labels = np.asarray([f"{keystr(path)}[{index}]" for path, leaf in leaves for index in range(leaf.size)])
    candidates = jnp.stack([ravel_pytree(state)[0] for state in states])
    if not candidates.shape[-1]:
        raise ValueError("Fixed points require a nonempty Model Step state.")
    finder = FixedPointFinder(fp_adam_optimizer(), enable_log=False)
    fits = []
    for index, state in enumerate(states):
        if any(not bool(evaluated.present[signal][index, params.transition]) for _, signal in task.input_bindings):
            raise ValueError(f"Task condition {trial_ids[index]!r} has no input at transition {params.transition}.")
        held = {
            port: evaluated.inputs[signal][index, params.transition]
            for port, signal in task.input_bindings
        }
        _, unravel = ravel_pytree(state)
        step_key = derive_key(keys[index], "trial-model")

        def advance(vector):
            current, carry = unravel(vector)
            _, after, carry = model.step(held, current, carry, key=step_key, t=params.transition)
            return ravel_pytree((after, carry))[0]

        fits.append(finder.find_and_filter(
            advance, candidates, n_batches=params.max_updates,
            loss_tol=jnp.asarray(params.loss_tolerance, dtype=candidates.dtype),
            unique_tol=params.merge_tolerance, key=step_key,
        ))

    points = jnp.stack([fit.fps for fit in fits])
    losses = jnp.stack([fit.losses for fit in fits])
    finite = jnp.isfinite(losses) & jnp.all(jnp.isfinite(points), axis=-1)
    accepted = jnp.stack([fit.masks["meets_all_criteria"] for fit in fits])
    arrays = {
        "fixed_points": jnp.where(finite[..., None], points, 0),
        "residual": jnp.where(finite, losses, 0),
        "finite": finite,
        "accepted": accepted,
        "accepted_count": jnp.sum(accepted, axis=-1),
    }
    return Dataset.from_arrays(
        {name: Variable.from_field(arrays[name], out, out.dims) for name, out in _FIXED_POINTS.variables.items()},
        coords={
            "condition": Variable(np.asarray(trial_ids), ("condition",)),
            "candidate": Variable(np.arange(len(states)), ("candidate",)),
            "state_component": Variable(labels, ("state_component",)),
        },
        attrs={"transition": params.transition, "state_kind": "model_and_delay"},
        schema_id=_FIXED_POINTS.schema_ids[0],
    )
