"""Portable direct-force mechanics owners."""

from __future__ import annotations
from typing import Annotated, Any
from functools import partial
from pydantic import Field
from feedbax.contracts.component import PortType
from feedbax.contracts.portable import Ports, ConfigBinding, ComponentOptions
from feedbax.runtime.graph import Component
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm as ArmSkeleton
from feedbax.mechanics.skeleton.pointmass import PointMass as PointMassSkeleton

_MECHANICS_PORTS = Ports(
    inputs={"force": PortType(dtype="vector")},
    outputs={"effector": PortType(dtype="state"), "state": PortType(dtype="state")},
)


class TwoLinkArm(Mechanics, Component):
    """Two-link arm plant with direct force input."""

    ports = _MECHANICS_PORTS
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

    ports = _MECHANICS_PORTS
    config_bindings = (
        ConfigBinding(path="plant.skeleton.mass", name="mass"),
        ConfigBinding(path="plant.skeleton.damping", name="damping"),
    )

    def __init__(
        self,
        dt: Annotated[float, Field(ge=0.001)] = 0.01,
        mass: Annotated[float, Field(ge=0)] = 1.0,
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
        output_prototype_fn=partial(mechanics_state_output_prototype, _owner)
    )


def _point_mass_template():
    from feedbax.compiler.mechanics_templates import point_mass_template_graph
    from feedbax.contracts.portable import configuration_model

    return point_mass_template_graph(configuration_model(PointMass)().model_dump())


from dataclasses import replace

PointMass.component_options = replace(
    PointMass.component_options,
    template_graph=_point_mass_template,
    template_id="feedbax.templates.mechanics.point_mass",
    template_kind="executable",
)
