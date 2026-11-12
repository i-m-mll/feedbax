"""The saved reaching evaluation's first two dataset operations."""

from __future__ import annotations

from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from feedbax.analysis.aligned import project_onto_direction
from feedbax.analysis.operations import DatasetOutput, OperationRegistry
from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.operation import DatasetPort, require_dataset_fields
from feedbax.dataset import Dataset, Variable
from feedbax.contracts.rollout import RAW_STATES_SCHEMA, STATES_SCHEMA


_MEMBER_AXES = ("checkpoint", "condition", "trial")
_STATE_AXES = (*_MEMBER_AXES, "state_time")
_SUMMARY_AXES = ("checkpoint", "condition", "state_time")
_REACH_FRAME = "reach-aligned"
_VELOCITY_SCHEMA = "feedbax.analysis.aligned_velocity.v1"
_SUMMARY_SCHEMA = "feedbax.analysis.velocity_summary.v1"
_VELOCITY_FIELDS = ("parallel_velocity", "lateral_velocity")


def _coordinate(name: str, *, units: str | None = None) -> FieldDescriptor:
    return FieldDescriptor(dims=(name,), units=units)


_STATE_COORDS = {name: _coordinate(name) for name in _MEMBER_AXES}
_STATE_COORDS["state_time"] = _coordinate("state_time", units="s")
_STATES = DatasetPort(
    schema_ids=(RAW_STATES_SCHEMA, STATES_SCHEMA),
    selected_variables={
        "position": FieldDescriptor(dims=(*_STATE_AXES, "xy"), units="m", frame="world"),
        "velocity": FieldDescriptor(dims=(*_STATE_AXES, "xy"), units="m/s", frame="world"),
        "target": FieldDescriptor(dims=(*_MEMBER_AXES, "transition", "xy"), units="m", frame="world"),
    },
    coordinates={
        **_STATE_COORDS,
        "transition_start": FieldDescriptor(dims=("transition",), units="s"),
        "transition_end": FieldDescriptor(dims=("transition",), units="s"),
    },
)
_VELOCITY = DatasetPort(
    schema_ids=(_VELOCITY_SCHEMA,),
    variables={
        name: FieldDescriptor(dims=_STATE_AXES, units="m/s", frame=_REACH_FRAME)
        for name in _VELOCITY_FIELDS
    },
    coordinates=_STATE_COORDS,
)
_SUMMARY = DatasetPort(
    schema_ids=(_SUMMARY_SCHEMA,),
    variables={
        **{
            name: FieldDescriptor(dims=_SUMMARY_AXES, units="m/s", frame=_REACH_FRAME)
            for name in ("mean", "standard_deviation", "lower", "upper")
        },
        "count": FieldDescriptor(dims=_SUMMARY_AXES, units="1"),
    },
    coordinates={name: _STATE_COORDS[name] for name in _SUMMARY_AXES},
)


class AlignedVelocityParameters(StrictModel):
    position: str = Field(
        strict=True, min_length=1,
        description="Retained reset-inclusive world position variable.",
        json_schema_extra={"feedbax": {"static": True}},
    )
    velocity: str = Field(
        strict=True, min_length=1,
        description="Retained reset-inclusive world velocity variable.",
        json_schema_extra={"feedbax": {"static": True}},
    )
    target: str = Field(
        strict=True, min_length=1,
        description="World target variable on the transition axis.",
        json_schema_extra={"feedbax": {"static": True}},
    )
    target_transition: int = Field(
        default=-1, strict=True, ge=-1,
        description="Reference target transition, zero-based. Use -1 for the final target.",
        json_schema_extra={"feedbax": {"static": True}},
    )


class VelocitySummaryParameters(StrictModel):
    variable: Literal["parallel_velocity", "lateral_velocity"] = Field(
        default="parallel_velocity",
        description="Reach-aligned velocity component to summarize over trials.",
        json_schema_extra={"feedbax": {"static": True}},
    )
    ddof: int = Field(strict=True, ge=0, description="Degrees of freedom deducted from trial count.")
    bounds_scale: float = Field(
        default=1.0, strict=True, ge=0, allow_inf_nan=False,
        description="Number of standard deviations from the mean to each bound.",
    )


def _require_dataset(dataset: Dataset, port: DatasetPort, params: StrictModel) -> None:
    descriptor = dataset.descriptor
    if descriptor.schema_id not in port.schema_ids:
        raise ValueError(f"Dataset schema {descriptor.schema_id!r} is not one of {port.schema_ids}.")
    require_dataset_fields(descriptor.variables, port.variables)
    for parameter, field in port.selected_variables.items():
        require_dataset_fields(descriptor.variables, {getattr(params, parameter): field})
    require_dataset_fields(descriptor.coords, port.coordinates)


def _product(port: DatasetPort, source: Dataset, arrays) -> Dataset:
    return Dataset.from_arrays(
        {
            name: Variable(arrays[name], field.dims, units=field.units, frame=field.frame)
            for name, field in port.variables.items()
        },
        coords={
            name: Variable(source.data(name), field.dims, units=field.units)
            for name, field in port.coordinates.items()
        },
        schema_id=port.schema_ids[0],
    )


def aligned_velocity(
    states: Annotated[Dataset, _STATES],
    params: AlignedVelocityParameters,
) -> Annotated[Dataset, DatasetOutput("velocity", _VELOCITY)]:
    """Project world velocity along and across the direction from reset to the selected target."""
    _require_dataset(states, _STATES, params)
    state_time = np.asarray(states.data("state_time"))
    if (
        len(state_time) < 2
        or not np.isfinite(state_time).all()
        or not np.all(np.diff(state_time) > 0)
        or not np.array_equal(states.data("transition_start"), state_time[:-1])
        or not np.array_equal(states.data("transition_end"), state_time[1:])
    ):
        raise ValueError("Transition times must span successive reset-inclusive state samples.")
    if params.target_transition >= len(state_time) - 1:
        raise ValueError("target_transition is outside the retained transitions.")
    members = (slice(None),) * len(_MEMBER_AXES)
    reset = np.asarray(states.data(params.position, (*members, 0)))
    goal = np.asarray(states.data(params.target, (*members, params.target_transition)))
    velocity = np.asarray(states.data(params.velocity))
    if reset.shape[-1] != 2:
        raise ValueError("Reach alignment requires two world coordinates.")
    direction = goal - reset
    length = np.linalg.norm(direction, axis=-1)
    if not np.all(np.isfinite(length) & (length > 0)) or not np.isfinite(velocity).all():
        raise ValueError("Reach direction must have finite nonzero length and velocity must be finite.")
    projected = project_onto_direction(velocity, direction)
    return _product(
        _VELOCITY, states,
        {"parallel_velocity": projected[..., 0], "lateral_velocity": projected[..., 1]},
    )


@jax.jit
def _summarize_trials(values, ddof, bounds_scale):
    trial_axis = _STATE_AXES.index("trial")
    count = values.shape[trial_axis]
    mean = jnp.mean(values, axis=trial_axis)
    sd = jnp.std(values, axis=trial_axis, ddof=ddof)
    half_width = bounds_scale * sd
    return {
        "mean": mean,
        "standard_deviation": sd,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "count": jnp.full(mean.shape, count),
    }


def summarize_velocity(
    velocity: Annotated[Dataset, _VELOCITY],
    params: VelocitySummaryParameters,
) -> Annotated[Dataset, DatasetOutput("summary", _SUMMARY)]:
    """Summarize one velocity component over trials at each checkpoint, condition, and sample."""
    _require_dataset(velocity, _VELOCITY, params)
    values = np.asarray(velocity.data(params.variable))
    if values.shape[_STATE_AXES.index("trial")] <= params.ddof:
        raise ValueError("Trial count must exceed ddof.")
    if not np.isfinite(values).all():
        raise ValueError("Trial summary requires finite velocity values.")
    result = _summarize_trials(values, jnp.asarray(params.ddof), jnp.asarray(params.bounds_scale))
    return _product(_SUMMARY, velocity, result)


def register_builtin_operations(registry: OperationRegistry) -> None:
    registry.register("feedbax.aligned_velocity", aligned_velocity, label="Aligned velocity", owner="feedbax")
    registry.register("feedbax.summarize_velocity", summarize_velocity, label="Trial summary", owner="feedbax")
