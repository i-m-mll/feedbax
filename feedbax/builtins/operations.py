"""Built-in operations over saved reaching evaluations."""

from __future__ import annotations

from functools import partial
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from feedbax.analysis.aligned import project_onto_direction
from feedbax.analysis.operations import DatasetOutput, operation
from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.analysis.operation_inputs import verify_dataset
from feedbax.contracts.operation import DatasetPort
from feedbax.dataset import Dataset, Variable
from feedbax.contracts.rollout import RAW_STATES_SCHEMA, STATES_SCHEMA


_MEMBER_AXES = ("checkpoint", "condition", "trial")
_STATE_AXES = (*_MEMBER_AXES, "state_time")
_SUMMARY_AXES = ("checkpoint", "condition", "state_time")
_OPTIONAL_AXES = ("checkpoint",)
_REACH_FRAME = "reach-aligned"
_VELOCITY_SCHEMA = "feedbax.analysis.aligned_velocity.v1"
_SPEED_SCHEMA = "feedbax.analysis.trial_speed.v1"
_SUMMARY_SCHEMA = "feedbax.analysis.velocity_summary.v1"
_PEAK_SUMMARY_SCHEMA = "feedbax.analysis.peak_velocity_summary.v1"
_VELOCITY_FIELDS = ("parallel_velocity", "lateral_velocity")


def _coordinate(name: str, *, units: str | None = None) -> FieldDescriptor:
    return FieldDescriptor(dims=(name,), units=units)


_STATE_COORDS = {name: _coordinate(name) for name in _MEMBER_AXES}
_STATE_COORDS["state_time"] = _coordinate("state_time", units="s")
_RECORDED_VELOCITY = FieldDescriptor(dims=(*_STATE_AXES, "xy"), units="m/s", frame="world")
_STATES = DatasetPort(
    schema_ids=(RAW_STATES_SCHEMA, STATES_SCHEMA),
    optional_dimensions=_OPTIONAL_AXES,
    selected_variables={
        "position": FieldDescriptor(dims=(*_STATE_AXES, "xy"), units="m", frame="world"),
        "velocity": _RECORDED_VELOCITY,
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
    optional_dimensions=_OPTIONAL_AXES,
    variables={
        name: FieldDescriptor(dims=_STATE_AXES, units="m/s", frame=_REACH_FRAME)
        for name in _VELOCITY_FIELDS
    },
    coordinates=_STATE_COORDS,
)
_SPEED_STATES = DatasetPort(
    schema_ids=_STATES.schema_ids,
    optional_dimensions=_OPTIONAL_AXES,
    selected_variables={"velocity": _RECORDED_VELOCITY},
    coordinates=_STATE_COORDS,
)
_SPEED = DatasetPort(
    schema_ids=(_SPEED_SCHEMA,),
    optional_dimensions=_OPTIONAL_AXES,
    variables={"speed": FieldDescriptor(dims=_STATE_AXES, units="m/s", frame="world")},
    coordinates=_STATE_COORDS,
)


def _summary_port(
    schema: str, dims: tuple[str, ...], *, peak_time: bool = False,
) -> DatasetPort:
    return DatasetPort(
        schema_ids=(schema,),
        optional_dimensions=_OPTIONAL_AXES,
        variables={
            **{
                name: FieldDescriptor(dims=dims, units="m/s", frame=_REACH_FRAME)
                for name in ("mean", "standard_deviation", "lower", "upper")
            },
            "count": FieldDescriptor(dims=dims, units="1"),
            **({"peak_time": FieldDescriptor(dims=dims, units="s")} if peak_time else {}),
        },
        coordinates={name: _STATE_COORDS[name] for name in dims},
    )


_SUMMARY = _summary_port(_SUMMARY_SCHEMA, _SUMMARY_AXES)
_PEAK_SUMMARY = _summary_port(_PEAK_SUMMARY_SCHEMA, ("checkpoint", "condition"), peak_time=True)


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


class TrialSpeedParameters(StrictModel):
    velocity: str = Field(
        strict=True, min_length=1,
        description="Retained reset-inclusive world velocity variable.",
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


def _product(port: DatasetPort, source: Dataset, stacking: tuple[str, ...], arrays) -> Dataset:
    """The declared variables after the source's stacking dimensions, with every source coordinate they span."""
    descriptor = source.descriptor
    port = port.for_dimensions(descriptor.dimensions)
    dims = {*stacking, *(dim for field in port.variables.values() for dim in field.dims)}
    coords = {
        name: Variable.from_field(source.data(name), field, field.dims)
        for name, field in descriptor.coords.items()
        if set(field.dims) <= dims
    }
    return Dataset.from_arrays(
        {
            name: Variable.from_field(arrays[name], field, (*stacking, *field.dims))
            for name, field in port.variables.items()
        },
        coords=coords,
        schema_id=port.schema_ids[0],
    )


@operation(settings=AlignedVelocityParameters, label="Aligned velocity")
def aligned_velocity(
    states: Annotated[Dataset, _STATES],
    params: AlignedVelocityParameters,
) -> Annotated[Dataset, DatasetOutput("velocity", _VELOCITY)]:
    """Project world velocity along and across the direction from reset to the selected target."""
    stacking = verify_dataset(states, _STATES, params)
    if any(states.descriptor.coords[name].dims != (dim,) for name, dim in (
        ("state_time", "state_time"), ("transition_start", "transition"), ("transition_end", "transition"),
    )):
        raise ValueError("Stacked members must share their state and transition times.")
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
    members = (slice(None),) * (
        len(stacking) + sum(dim in states.descriptor.dimensions for dim in _MEMBER_AXES)
    )
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
        _VELOCITY, states, stacking,
        {"parallel_velocity": projected[..., 0], "lateral_velocity": projected[..., 1]},
    )


@operation(settings=TrialSpeedParameters, label="Trial speed")
def trial_speed(
    states: Annotated[Dataset, _SPEED_STATES],
    params: TrialSpeedParameters,
) -> Annotated[Dataset, DatasetOutput("speed", _SPEED)]:
    """Take the magnitude of each trial's recorded world velocity at each state sample."""
    stacking = verify_dataset(states, _SPEED_STATES, params)
    speed = np.linalg.norm(np.asarray(states.data(params.velocity)), axis=-1)
    if not np.isfinite(speed).all():
        raise ValueError("Trial speed requires finite velocity magnitudes.")
    return _product(_SPEED, states, stacking, {"speed": speed})


@partial(jax.jit, static_argnums=3)
def _summarize_trials(values, ddof, bounds_scale, trial_axis):
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


def _velocity_samples(velocity: Dataset, params: VelocitySummaryParameters) -> tuple[np.ndarray, int]:
    trial_axis = velocity.descriptor.variables[params.variable].dims.index("trial")
    values = np.asarray(velocity.data(params.variable))
    if values.shape[trial_axis] <= params.ddof:
        raise ValueError("Trial count must exceed ddof.")
    if not np.isfinite(values).all():
        raise ValueError("Trial summary requires finite velocity values.")
    return values, trial_axis


@operation(settings=VelocitySummaryParameters, label="Trial summary")
def summarize_velocity(
    velocity: Annotated[Dataset, _VELOCITY],
    params: VelocitySummaryParameters,
) -> Annotated[Dataset, DatasetOutput("summary", _SUMMARY)]:
    """Summarize one velocity component over trials at each checkpoint, condition, and sample."""
    stacking = verify_dataset(velocity, _VELOCITY, params)
    values, trial_axis = _velocity_samples(velocity, params)
    result = _summarize_trials(values, jnp.asarray(params.ddof), jnp.asarray(params.bounds_scale), trial_axis)
    return _product(_SUMMARY, velocity, stacking, result)


@operation(settings=VelocitySummaryParameters, label="Peak velocity summary")
def summarize_peak_velocity(
    velocity: Annotated[Dataset, _VELOCITY],
    params: VelocitySummaryParameters,
) -> Annotated[Dataset, DatasetOutput("summary", _PEAK_SUMMARY)]:
    """Summarize each trial's maximum velocity and the time of its first occurrence."""
    stacking = verify_dataset(velocity, _VELOCITY, params)
    values, trial_axis = _velocity_samples(velocity, params)
    # The input port places state_time last, after trial and any stacking dimensions.
    peak_indices = values.argmax(axis=-1, keepdims=True)
    peaks = np.take_along_axis(values, peak_indices, axis=-1).squeeze(-1)
    result = _summarize_trials(peaks, jnp.asarray(params.ddof), jnp.asarray(params.bounds_scale), trial_axis)

    descriptor = velocity.descriptor
    time = np.asarray(velocity.data("state_time"))
    time_sizes = dict(zip(descriptor.coords["state_time"].dims, time.shape, strict=True))
    time = time.reshape(tuple(time_sizes.get(dim, 1) for dim in descriptor.variables[params.variable].dims))
    trial_peak_times = np.take_along_axis(time, peak_indices, axis=-1).squeeze(-1)
    return _product(
        _PEAK_SUMMARY, velocity, stacking, {**result, "peak_time": trial_peak_times.mean(axis=trial_axis)},
    )
