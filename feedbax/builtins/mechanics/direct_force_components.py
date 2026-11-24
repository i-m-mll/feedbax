"""Portable direct-force mechanics owners."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any
from functools import partial
from pydantic import Field
from feedbax.contracts.component_presentation import PresentationOverride
from feedbax.contracts.direct_force_presentation import (
    _point_mass_representation,
    _two_link_arm_representation,
)
from feedbax.contracts.component import PortType
from feedbax.contracts.model_values import ModelFacts, ModelValue, StateReset, StateValue
from feedbax.contracts.portable import Ports, ConfigBinding, ComponentOptions
from feedbax.runtime.graph import Component
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
        states=(
            StateValue("skeleton/angle", ("plant", "skeleton", "angle"), "rad", "joint", configuration=True),
            StateValue("skeleton/angular_velocity", ("plant", "skeleton", "d_angle"), "rad/s", "joint"),
            StateValue("skeleton/torque", ("plant", "skeleton", "torque"), "N*m", "joint"),
            StateValue("skeleton/effector_force", ("plant", "skeleton", "effector_force"), "N", "xy", "world"),
            StateValue(
                "effector/position", ("effector", "pos"), "m", "xy", "world",
                reset=StateReset("effector", ("pos",)),
            ),
            StateValue(
                "effector/velocity", ("effector", "vel"), "m/s", "xy", "world",
                reset=StateReset("effector", ("vel",)),
            ),
            StateValue("effector/force", ("effector", "force"), "N", "xy", "world"),
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
        states=(
            StateValue("skeleton/position", ("plant", "skeleton", "pos"), "m", "xy", "world"),
            StateValue("skeleton/velocity", ("plant", "skeleton", "vel"), "m/s", "xy", "world"),
            StateValue("skeleton/force", ("plant", "skeleton", "force"), "N", "xy", "world"),
            StateValue(
                "effector/position", ("effector", "pos"), "m", "xy", "world",
                reset=StateReset("effector", ("pos",)),
            ),
            StateValue(
                "effector/velocity", ("effector", "vel"), "m/s", "xy", "world",
                reset=StateReset("effector", ("vel",)),
            ),
            StateValue("effector/force", ("effector", "force"), "N", "xy", "world"),
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
