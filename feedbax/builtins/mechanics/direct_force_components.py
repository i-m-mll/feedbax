"""Portable direct-force mechanics owners."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any
from functools import partial
from equinox.nn import State
import jax.numpy as jnp
from pydantic import Field
from feedbax.contracts.component_presentation import PresentationOverride
from feedbax.contracts.direct_force_presentation import (
    _point_mass_representation,
    _two_link_arm_representation,
)
from feedbax.contracts.component import PortType
from feedbax.contracts.model_values import ModelFacts, ModelValue, StateValue
from feedbax.contracts.portable import Ports, ConfigBinding, ComponentOptions
from feedbax.runtime.graph import Component, TrialStateProblem
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm as ArmSkeleton
from feedbax.mechanics.skeleton.pointmass import PointMass as PointMassSkeleton


_PRESENTATION_SOURCES = (
    Path(__file__).resolve().parents[2] / "studio" / "direct_force_presentation.py",
)
_MECHANICS_PORTS = Ports(
    inputs={"force": PortType(dtype="vector")},
    outputs={"effector": PortType(dtype="state"), "state": PortType(dtype="state")},
)
_ARM_ANGLE = ("plant", "skeleton", "angle")
_ARM_RATE = ("plant", "skeleton", "d_angle")
_EFFECTOR_POSITION = ("effector", "pos")
_EFFECTOR_VELOCITY = ("effector", "vel")


class TwoLinkArm(Mechanics, Component):
    """Two-link arm plant with direct force input."""

    component_presentation = PresentationOverride(representation=_two_link_arm_representation())
    component_sources = _PRESENTATION_SOURCES
    ports = _MECHANICS_PORTS
    model_facts = ModelFacts(
        values=(
            ModelValue("link_lengths", "plant.skeleton.l", "source"),
            ModelValue("dt", "dt", "source"),
        ),
        states=tuple(
            StateValue(
                symbol, storage, unit, axis, ("state", *storage), frame,
                configuration=symbol == "skeleton/angle", space=space,
            )
            for symbol, storage, unit, axis, frame, space in (
                ("skeleton/angle", _ARM_ANGLE, "rad", "joint", None, None),
                ("skeleton/angular_velocity", _ARM_RATE, "rad/s", "joint", None, None),
                ("skeleton/torque", ("plant", "skeleton", "torque"), "N*m", "joint", None, None),
                ("skeleton/effector_force", ("plant", "skeleton", "effector_force"), "N", "xy", "world", None),
                ("effector/position", _EFFECTOR_POSITION, "m", "xy", "world", "point"),
                ("effector/velocity", _EFFECTOR_VELOCITY, "m/s", "xy", "world", None),
                ("effector/force", ("effector", "force"), "N", "xy", "world", None),
            )
        ),
        scene_effector_port="effector",
        scene_position="effector/position",
    )
    config_bindings = (ConfigBinding(path="plant.skeleton.l", name="link_lengths"),)

    def __init__(
        self,
        dt: Annotated[float, Field(ge=0.001)] = 0.01,
        link_lengths: Annotated[Any, Field(json_schema_extra={"feedbax": {"type": "array"}})] = (
            0.30,
            0.33,
        ),
    ):
        super().__init__(plant=DirectForceInput(ArmSkeleton(l=link_lengths)), dt=dt)

    def finalize_trial_state(
        self, state: State, explicit: frozenset[tuple[str, ...]],
    ) -> tuple[State, tuple[TrialStateProblem, ...]]:
        q_set, dq_set = _ARM_ANGLE in explicit, _ARM_RATE in explicit
        x_set, v_set = _EFFECTOR_POSITION in explicit, _EFFECTOR_VELOCITY in explicit
        if not (q_set or dq_set or x_set or v_set):
            return state, ()

        current = self.state_view(state)
        joints = current.plant.skeleton
        effector = current.effector
        arm = self.plant.skeleton
        problems = []
        pose_reachable = True
        if x_set and not q_set:
            q, pose_reachable = arm.inverse_position(effector.pos)
            joints = replace(joints, angle=q.astype(joints.angle.dtype))
            problems.append(TrialStateProblem(
                _EFFECTOR_POSITION, ~pose_reachable, "The arm cannot reach this position.",
            ))
        elif q_set and not x_set:
            pos = arm.effector(joints).pos
            effector = replace(effector, pos=pos.astype(effector.pos.dtype))

        if not v_set:
            vel = arm.effector_jac(joints.angle) @ joints.d_angle
            effector = replace(effector, vel=vel.astype(effector.vel.dtype))
        elif not dq_set:
            jacobian = arm.effector_jac(joints.angle)
            v = effector.vel
            dq = (jnp.linalg.pinv(jacobian) @ v).astype(joints.d_angle.dtype)
            residual = jnp.linalg.norm(jacobian @ dq - v)
            tolerance = 32 * jnp.finfo(v.dtype).eps * jnp.linalg.norm(v)
            reachable = jnp.isfinite(residual) & (residual <= tolerance)
            joints = replace(joints, d_angle=dq)
            problems.append(TrialStateProblem(
                _EFFECTOR_VELOCITY, pose_reachable & ~reachable,
                "The arm cannot attain this velocity at its initial joint angles.",
            ))

        plant = replace(current.plant, skeleton=joints)
        return state.set(self.state_index, replace(current, plant=plant, effector=effector)), tuple(problems)


class PointMass(Mechanics, Component):
    """Point-mass plant with direct force input."""

    component_presentation = PresentationOverride(representation=_point_mass_representation())
    component_sources = _PRESENTATION_SOURCES
    model_facts = ModelFacts(
        values=(
            ModelValue("mass", "plant.skeleton.mass", "source"),
            ModelValue("damping", "plant.skeleton.damping", "source"),
            ModelValue("dt", "dt", "source"),
        ),
        states=tuple(
            StateValue(
                f"{part}/{quantity}", (*storage, attribute), unit, "xy",
                ("state", *storage, attribute), "world",
                aliases=((*companion, attribute),) if quantity != "force" else (),
                space="point" if quantity == "position" else None,
            )
            for part, storage, companion in (
                ("skeleton", ("plant", "skeleton"), ("effector",)),
                ("effector", ("effector",), ("plant", "skeleton")),
            )
            for quantity, attribute, unit in (
                ("position", "pos", "m"), ("velocity", "vel", "m/s"), ("force", "force", "N"),
            )
        ),
        scene_effector_port="effector",
        scene_position="effector/position",
    )

    ports = _MECHANICS_PORTS
    config_bindings = (
        ConfigBinding(path="plant.skeleton.mass", name="mass"),
        ConfigBinding(path="plant.skeleton.damping", name="damping"),
    )

    def __init__(
        self,
        dt: Annotated[float, Field(ge=0.001)] = 0.01,
        mass: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0,
        damping: Annotated[float, Field(ge=0)] = 0.0,
    ):
        super().__init__(
            plant=DirectForceInput(PointMassSkeleton(mass=mass, damping=damping)), dt=dt
        )


def mechanics_state_output_prototype(owner_type, params, inputs):
    """Publish the same port trees as the initialized mechanics owner."""
    owner = owner_type(**params)
    return owner.initial_outputs(owner.state_index.init)


for _owner in (PointMass, TwoLinkArm):
    _owner.component_options = ComponentOptions(
        output_prototype_fn=partial(mechanics_state_output_prototype, _owner),
        state_only_outputs=("effector", "state"),
        reset_outputs=("effector", "state"),
    )


def _point_mass_template():
    from feedbax.builtins.templates.mechanics_templates import point_mass_template_graph
    from feedbax.contracts.portable import configuration_model

    return point_mass_template_graph(configuration_model(PointMass)().model_dump())


PointMass.component_options = replace(
    PointMass.component_options,
    template_graph=_point_mass_template,
    template_id="feedbax.templates.mechanics.point_mass",
    template_kind="executable",
)
