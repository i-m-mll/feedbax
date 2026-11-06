"""Planar multibody mechanics vocabulary.

These descriptors are authoring-time elements for editable mechanics interiors.
They compile through the planar multibody graph compiler rather than the scalar
translational/rotational acausal assembler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from feedbax.acausal._component import NonNegativeFloat, element_options
from feedbax.acausal.base import AcausalElement, AcausalPort, Domain
from feedbax.component_registry.discovery import component
from feedbax.contracts.domain import MECHANICS_DOMAIN_ID
from feedbax.contracts.graph import ParamSchema
from feedbax.contracts.representation import RepresentationSpec


_FRAME_ACROSS_VARS = ("x", "y", "theta", "vx", "vy", "omega")


def _frame_port(name: str) -> AcausalPort:
    return AcausalPort(
        name=name,
        domain=Domain.PLANAR_MULTIBODY,
        across_vars=_FRAME_ACROSS_VARS,
        through_var="wrench",
    )


def _literal_binding(value: Any, *, dim: int | None = None) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "literal", "value": value}
    if dim is not None:
        binding["dim"] = dim
    return binding


def _param_binding(
    path: str, *, expected_type: str | None = None, dim: int | None = None
) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "param_path", "path": path}
    if expected_type is not None:
        binding["expected_type"] = expected_type
    if dim is not None:
        binding["dim"] = dim
    return binding


def _representation(data: dict[str, Any]) -> RepresentationSpec:
    return RepresentationSpec.model_validate(data)


def _planar_link_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "proximal",
                    "semantic_role": "joint",
                    "binding": _literal_binding([0.0, 0.0], dim=2),
                    "units": "m",
                    "dim": 2,
                },
                {
                    "id": "distal",
                    "semantic_role": "endpoint",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"computed_from": "length_and_orientation"},
                },
            ],
            "elements": [
                {
                    "id": "link",
                    "archetype": "planar_chain",
                    "anchors": ["proximal", "distal"],
                    "bindings": {
                        "link_lengths": _param_binding("length", expected_type="float", dim=1)
                    },
                    "metadata": {"chain_kind": "planar_link_segment"},
                }
            ],
        }
    )


def _muscle_path_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "path",
                    "semantic_role": "path",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"collection": "path_points"},
                }
            ],
            "elements": [
                {
                    "id": "muscle-path",
                    "archetype": "muscle_path",
                    "anchors": ["path"],
                    "bindings": {
                        "path_points": _param_binding("path_points", expected_type="array")
                    },
                    "metadata": {"activation_input": "activation"},
                }
            ],
        }
    )


def _point_marker_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "point",
                    "semantic_role": "endpoint",
                    "binding": _param_binding("offset", expected_type="array", dim=2),
                    "units": "m",
                    "dim": 2,
                }
            ],
            "elements": [
                {
                    "id": "marker",
                    "archetype": "marker",
                    "anchors": ["point"],
                    "metadata": {"frame_param": "frame"},
                }
            ],
        }
    )


def _planar_link_options(element_type: type[AcausalElement]) -> dict:
    return {
        **element_options(element_type),
        "representation": _planar_link_representation(),
    }


def _muscle_path_options(element_type: type[AcausalElement]) -> dict:
    return {
        **element_options(element_type),
        "representation": _muscle_path_representation(),
        "param_schema": [
            ParamSchema(name="path_points", type="array", default=[], required=True),
            ParamSchema(name="moment_arm", type="array", default=[], required=False),
            ParamSchema(name="reference_length", type="float", default=0.2, min=0.0),
            ParamSchema(name="max_isometric_force", type="float", default=1.0, min=0.0),
        ],
    }


def _point_marker_options(element_type: type[AcausalElement]) -> dict:
    return {
        **element_options(element_type),
        "representation": _point_marker_representation(),
        "param_schema": [
            ParamSchema(name="frame", type="str", default="", required=True),
            ParamSchema(name="offset", type="array", default=[0.0, 0.0], required=False),
            ParamSchema(name="marker_name", type="str", default=None, required=False),
        ],
    }


@component(
    "PlanarLink",
    configure=_planar_link_options,
    category="Acausal / Planar Multibody",
    icon="Orbit",
    domain=MECHANICS_DOMAIN_ID,
)
@dataclass
class PlanarLink(AcausalElement):
    """Rigid planar link with proximal and distal attachment frames."""

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


@component(
    "RevoluteJoint",
    configure=element_options,
    category="Acausal / Planar Multibody",
    icon="Orbit",
    domain=MECHANICS_DOMAIN_ID,
)
@dataclass
class RevoluteJoint(AcausalElement):
    """One-DOF hinge between two planar link frames."""

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


@component(
    "WorldFrame",
    configure=element_options,
    category="Acausal / Planar Multibody",
    icon="Orbit",
    domain=MECHANICS_DOMAIN_ID,
)
@dataclass
class WorldFrame(AcausalElement):
    """Fixed inertial frame for a planar multibody interior."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["frame"] = _frame_port("frame")
        self.element_type = "world_frame"


@component(
    "Anchor",
    configure=element_options,
    category="Acausal / Planar Multibody",
    icon="Orbit",
    domain=MECHANICS_DOMAIN_ID,
)
@dataclass
class Anchor(AcausalElement):
    """Ground a body frame to the world frame."""

    def __init__(self, name: str, frame: str = "", world_frame: str = "world.frame"):
        super().__init__(name=name)
        self.ports["world"] = _frame_port("world")
        self.ports["frame"] = _frame_port("frame")
        self.element_type = "anchor"


@component(
    "MusclePath",
    configure=_muscle_path_options,
    category="Acausal / Planar Multibody",
    icon="Orbit",
    domain=MECHANICS_DOMAIN_ID,
)
@dataclass
class MusclePath(AcausalElement):
    """Ordered rigid-tendon muscle path over named planar frames."""

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


@component(
    "PointMarker",
    configure=_point_marker_options,
    category="Acausal / Planar Multibody",
    icon="Orbit",
    domain=MECHANICS_DOMAIN_ID,
)
@dataclass
class PointMarker(AcausalElement):
    """Named point attached to a planar frame for observations or objectives."""

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
