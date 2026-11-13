"""Declared values of the currently qualified model owner revisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelValue:
    symbol: str
    field: str
    kind: Literal["source", "learned"]


@dataclass(frozen=True)
class StateValue:
    symbol: str
    fields: tuple[str, ...]
    units: str
    axis: Literal["xy", "unit", "muscle"]
    frame: str | None = None
    reset: StateReset | None = None


@dataclass(frozen=True)
class StateReset:
    output_port: str
    output_path: tuple[str, ...] = ()
    aliases: tuple[tuple[str, ...], ...] = ()
    override_port: str | None = None


@dataclass(frozen=True)
class TrialParameter:
    name: str
    kind: Literal["real", "boolean"]
    units: str = "1"


@dataclass(frozen=True)
class ModelOwner:
    revision: int
    values: tuple[ModelValue, ...]
    states: tuple[StateValue, ...] | None = None
    clock: str | None = None
    checkpointable: bool = True
    scene_effector_port: str | None = None
    scene_position: str | None = None
    configuration_slots: tuple[str, ...] = ()
    fixed_configuration: bool = False
    trial_parameters: tuple[TrialParameter, ...] = ()
    trial_parameter_override_port: str | None = None


def _learned(*values: tuple[str, str]) -> tuple[ModelValue, ...]:
    return tuple(ModelValue(symbol, field, "learned") for symbol, field in values)


def _source(*names: str) -> tuple[ModelValue, ...]:
    return tuple(ModelValue(name, name, "source") for name in names)


def _cartesian(symbol: str, path: tuple[str, ...]) -> tuple[StateValue, ...]:
    writable = symbol == "effector"
    return (
        StateValue(
            f"{symbol}/position", (*path, "pos"), "m", "xy", "world",
            reset=StateReset("effector", ("pos",)) if writable else None,
        ),
        StateValue(
            f"{symbol}/velocity", (*path, "vel"), "m/s", "xy", "world",
            reset=StateReset("effector", ("vel",)) if writable else None,
        ),
        StateValue(f"{symbol}/force", (*path, "force"), "N", "xy", "world"),
    )


SPATIAL_OWNER = "feedbax.AcausalSystem"
MODEL_OWNERS = {
    "feedbax.GRU": ModelOwner(
        1,
        _learned(
            ("input_kernel", "cell.weight_ih"),
            ("recurrent_kernel", "cell.weight_hh"),
            ("gate_bias", "cell.bias"),
            ("candidate_bias", "cell.bias_n"),
        ),
        states=(
            StateValue(
                "hidden", ("hidden",), "1", "unit",
                reset=StateReset("hidden", aliases=(("output",),), override_port="hidden"),
            ),
            StateValue("output", ("output",), "1", "unit"),
        ),
    ),
    "feedbax.Linear": ModelOwner(
        2, _learned(("kernel", "layer.weight"), ("bias", "layer.bias")), states=(),
    ),
    "feedbax.Input": ModelOwner(1, (), states=()),
    "feedbax.Constant": ModelOwner(1, (), states=(), checkpointable=False),
    "feedbax.Saturation": ModelOwner(1, (), states=()),
    "feedbax.PointMass": ModelOwner(
        1,
        (
            ModelValue("mass", "plant.skeleton.mass", "source"),
            ModelValue("damping", "plant.skeleton.damping", "source"),
            *_source("dt"),
        ),
        states=(
            *_cartesian("skeleton", ("plant", "skeleton")),
            *_cartesian("effector", ("effector",)),
        ),
        clock="dt",
        scene_effector_port="effector",
        scene_position="effector/position",
    ),
    "feedbax.RigidTendonHillMuscleThelen": ModelOwner(
        2,
        _source(
            "max_isometric_force", "optimal_muscle_length", "tendon_slack_length",
            "vmax_factor", "min_activation", "tau_activation", "tau_deactivation", "dt",
        ),
        states=(StateValue("activation", (), "1", "muscle"),),
        clock="dt",
    ),
    "feedbax.Scale": ModelOwner(
        1, _source("scale"), states=(), checkpointable=False, fixed_configuration=True,
    ),
    "feedbax.CurlField": ModelOwner(
        2, (), states=(), checkpointable=False, configuration_slots=("params_index",),
        trial_parameters=(
            TrialParameter("active", "boolean"), TrialParameter("scale", "real"),
            TrialParameter("amplitude", "real"),
        ),
        trial_parameter_override_port="params_override",
    ),
    SPATIAL_OWNER: ModelOwner(1, ()),
}
