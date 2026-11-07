"""Planar multibody mechanics vocabulary.

These descriptors are authoring-time elements for editable mechanics interiors.
They compile through the planar multibody graph compiler rather than the scalar
translational/rotational acausal assembler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from feedbax.acausal._component import NonNegativeFloat
from feedbax.acausal.base import AcausalElement, AcausalPort, Domain
from feedbax.contracts.portable import ComponentOptions
from feedbax.contracts.domain import MECHANICS_DOMAIN_ID


_FRAME_ACROSS_VARS = ("x", "y", "theta", "vx", "vy", "omega")


def _frame_port(name: str) -> AcausalPort:
    return AcausalPort(
        name=name,
        domain=Domain.PLANAR_MULTIBODY,
        across_vars=_FRAME_ACROSS_VARS,
        through_var="wrench",
    )


@dataclass
class PlanarLink(AcausalElement):
    """Rigid planar link with proximal and distal attachment frames."""

    component_options = ComponentOptions(domain=MECHANICS_DOMAIN_ID, build_context_fields=("name",))

    def __init__(
        self,
        name: str,
        length: float = 0.3,
        mass: NonNegativeFloat = 1.0,
        com: float = 0.5,
        inertia: NonNegativeFloat = 0.03,
    ):
        super().__init__(name=name)
        self.ports["proximal"] = _frame_port("proximal")
        self.ports["distal"] = _frame_port("distal")
        self.params[f"{name}.length"] = length
        self.params[f"{name}.mass"] = mass
        self.params[f"{name}.com"] = com
        self.params[f"{name}.inertia"] = inertia


@dataclass
class RevoluteJoint(AcausalElement):
    """One-DOF hinge between two planar link frames."""

    component_options = ComponentOptions(domain=MECHANICS_DOMAIN_ID, build_context_fields=("name",))

    def __init__(
        self,
        name: str,
        parent_frame: str = "",
        child_frame: str = "",
        damping: NonNegativeFloat = 0.0,
        rest_angle: float = 0.0,
        limit_min: float | None = None,
        limit_max: float | None = None,
    ):
        super().__init__(name=name)
        self.ports["parent"] = _frame_port("parent")
        self.ports["child"] = _frame_port("child")
        self.params[f"{name}.damping"] = damping
        self.params[f"{name}.rest_angle"] = rest_angle
        if limit_min is not None:
            self.params[f"{name}.limit_min"] = float(limit_min)
        if limit_max is not None:
            self.params[f"{name}.limit_max"] = float(limit_max)
        self.element_type = "revolute_joint"


@dataclass
class WorldFrame(AcausalElement):
    """Fixed inertial frame for a planar multibody interior."""

    component_options = ComponentOptions(domain=MECHANICS_DOMAIN_ID, build_context_fields=("name",))

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["frame"] = _frame_port("frame")
        self.element_type = "world_frame"


@dataclass
class Anchor(AcausalElement):
    """Ground a body frame to the world frame."""

    component_options = ComponentOptions(domain=MECHANICS_DOMAIN_ID, build_context_fields=("name",))

    def __init__(self, name: str, frame: str = "", world_frame: str = "world.frame"):
        super().__init__(name=name)
        self.ports["world"] = _frame_port("world")
        self.ports["frame"] = _frame_port("frame")
        self.element_type = "anchor"


@dataclass
class MusclePath(AcausalElement):
    """Ordered rigid-tendon muscle path over named planar frames."""

    component_options = ComponentOptions(domain=MECHANICS_DOMAIN_ID, build_context_fields=("name",))

    def __init__(
        self,
        name: str,
        path_points: list[dict[str, Any]] | None = None,
        moment_arm: list[float] | None = None,
        reference_length: float = 0.2,
        max_isometric_force: float = 1.0,
    ):
        super().__init__(name=name)
        self.signal_inputs = ("activation",)
        self.params[f"{name}.reference_length"] = reference_length
        self.params[f"{name}.max_isometric_force"] = max_isometric_force
        self.element_type = "muscle_path"


@dataclass
class PointMarker(AcausalElement):
    """Named point attached to a planar frame for observations or objectives."""

    component_options = ComponentOptions(domain=MECHANICS_DOMAIN_ID, build_context_fields=("name",))

    def __init__(
        self,
        name: str,
        frame: str = "",
        offset: list[float] | None = None,
        marker_name: str | None = None,
    ):
        super().__init__(name=name)
        self.element_type = "point_marker"


__all__ = [
    "Anchor",
    "MusclePath",
    "PlanarLink",
    "PointMarker",
    "RevoluteJoint",
    "WorldFrame",
]
