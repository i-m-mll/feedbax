"""Executable mechanics-domain acausal graph templates."""

from __future__ import annotations

from typing import Any

from feedbax.contracts.acausal import ACAUSAL_GRAPH_SCHEMA_VERSION
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.acausal_interface import derive_acausal_interface
from feedbax.contracts.graph import ComponentSpec
from feedbax.mechanics.body import default_2link_bounds
from feedbax.mechanics.muscle_config import (
    default_6muscle_2link_moment_arms,
    default_6muscle_2link_reference_lengths,
)


def point_mass_template_graph(params: dict[str, Any] | None = None) -> AcausalGraphSpec:
    """Build a driven translational point-mass mechanics interior."""

    params = validate_authoring(PointMassTemplateConfig, params or {}).model_dump()
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="feedbax.MechanicsGround"),
            "mass": ComponentSpec(
                type="feedbax.MechanicsMass",
                params={"mass": float(params["mass"])},
            ),
            "damper": ComponentSpec(
                type="feedbax.MechanicsLinearDamper",
                params={"damping": float(params["damping"])},
            ),
            "force": ComponentSpec(
                type="feedbax.ActuationInput",
                params={"port_name": "force", "source_kind": "force"},
            ),
            "position": ComponentSpec(
                type="feedbax.SensorOutput",
                params={"port_name": "position", "quantity": "position"},
            ),
            "velocity": ComponentSpec(
                type="feedbax.SensorOutput",
                params={"port_name": "velocity", "quantity": "velocity"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("damper", "flange_b"), "b": ("mass", "flange")},
            {"a": ("force", "flange"), "b": ("mass", "flange")},
            {"a": ("position", "flange"), "b": ("mass", "flange")},
            {"a": ("velocity", "flange"), "b": ("mass", "flange")},
        ],
        solver={
            "solver_type": str(params["solver_type"]),
            "dt": float(params["dt"]),
        },
    )


def mass_spring_damper_template_graph(
    params: dict[str, Any] | None = None,
) -> AcausalGraphSpec:
    """Build a grounded mass-spring-damper mechanics interior."""

    params = validate_authoring(MassSpringDamperConfig, params or {}).model_dump()
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="feedbax.MechanicsGround"),
            "mass": ComponentSpec(
                type="feedbax.MechanicsMass",
                params={"mass": float(params["mass"])},
            ),
            "spring": ComponentSpec(
                type="feedbax.MechanicsLinearSpring",
                params={"stiffness": float(params["stiffness"])},
            ),
            "damper": ComponentSpec(
                type="feedbax.MechanicsLinearDamper",
                params={"damping": float(params["damping"])},
            ),
            "force": ComponentSpec(
                type="feedbax.ActuationInput",
                params={"port_name": "force", "source_kind": "force"},
            ),
            "position": ComponentSpec(
                type="feedbax.SensorOutput",
                params={"port_name": "position", "quantity": "position"},
            ),
            "velocity": ComponentSpec(
                type="feedbax.SensorOutput",
                params={"port_name": "velocity", "quantity": "velocity"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("spring", "flange_a")},
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("spring", "flange_b"), "b": ("mass", "flange")},
            {"a": ("damper", "flange_b"), "b": ("mass", "flange")},
            {"a": ("force", "flange"), "b": ("mass", "flange")},
            {"a": ("position", "flange"), "b": ("mass", "flange")},
            {"a": ("velocity", "flange"), "b": ("mass", "flange")},
        ],
        solver={
            "solver_type": str(params["solver_type"]),
            "dt": float(params["dt"]),
        },
    )


def point_mass_with_muscles_template_graph(
    params: dict[str, Any] | None = None,
) -> AcausalGraphSpec:
    """Build a point mass driven by one antagonistic muscle pair."""

    params = validate_authoring(PointMassWithMusclesConfig, params or {}).model_dump()
    fmax = float(params["max_isometric_force"])
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="feedbax.MechanicsGround"),
            "mass": ComponentSpec(
                type="feedbax.MechanicsMass",
                params={"mass": float(params["mass"])},
            ),
            "damper": ComponentSpec(
                type="feedbax.MechanicsLinearDamper",
                params={"damping": float(params["damping"])},
            ),
            "flexor": ComponentSpec(
                type="feedbax.RigidTendonHillMuscle",
                params={"max_isometric_force": fmax, "direction": 1.0},
            ),
            "extensor": ComponentSpec(
                type="feedbax.RigidTendonHillMuscle",
                params={"max_isometric_force": fmax, "direction": -1.0},
            ),
            "position": ComponentSpec(
                type="feedbax.SensorOutput",
                params={"port_name": "position", "quantity": "position"},
            ),
            "velocity": ComponentSpec(
                type="feedbax.SensorOutput",
                params={"port_name": "velocity", "quantity": "velocity"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("damper", "flange_b"), "b": ("mass", "flange")},
            {"a": ("flexor", "flange"), "b": ("mass", "flange")},
            {"a": ("extensor", "flange"), "b": ("mass", "flange")},
            {"a": ("position", "flange"), "b": ("mass", "flange")},
            {"a": ("velocity", "flange"), "b": ("mass", "flange")},
        ],
        solver={
            "solver_type": str(params["solver_type"]),
            "dt": float(params["dt"]),
        },
    )


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def two_link_arm_6muscle_template_graph(
    params: dict[str, Any] | None = None,
) -> AcausalGraphSpec:
    """Build an editable planar multibody interior for the canonical arm."""

    params = validate_authoring(TwoLinkArm6MuscleConfig, params or {}).model_dump()
    bounds = default_2link_bounds()
    lengths = (bounds.segment_lengths_min + bounds.segment_lengths_max) / 2.0
    masses = (bounds.segment_masses_min + bounds.segment_masses_max) / 2.0
    damping = (bounds.joint_damping_min + bounds.joint_damping_max) / 2.0
    moment_arms = default_6muscle_2link_moment_arms()
    reference_lengths = default_6muscle_2link_reference_lengths()
    fmax = float(params["max_isometric_force"])

    muscle_nodes = {
        f"muscle_{index}": ComponentSpec(
            type="feedbax.MusclePath",
            params={
                "path_points": [
                    {"frame": "world.frame", "offset": [0.0, 0.0]},
                    {"frame": "upper.distal", "offset": [0.0, 0.0]},
                    {"frame": "forearm.distal", "offset": [0.0, 0.0]},
                ],
                "moment_arm": _float_list(moment_arms[index]),
                "reference_length": float(reference_lengths[index]),
                "max_isometric_force": fmax,
            },
        )
        for index in range(6)
    }

    return AcausalGraphSpec(
        schema_version=ACAUSAL_GRAPH_SCHEMA_VERSION,
        physical_domain="planar_multibody",
        nodes={
            "world": ComponentSpec(type="feedbax.WorldFrame"),
            "anchor": ComponentSpec(
                type="feedbax.Anchor",
                params={"world_frame": "world.frame", "frame": "upper.proximal"},
            ),
            "shoulder": ComponentSpec(
                type="feedbax.RevoluteJoint",
                params={
                    "parent_frame": "world.frame",
                    "child_frame": "upper.proximal",
                    "damping": float(damping[0]),
                },
            ),
            "elbow": ComponentSpec(
                type="feedbax.RevoluteJoint",
                params={
                    "parent_frame": "upper.distal",
                    "child_frame": "forearm.proximal",
                    "damping": float(damping[1]),
                },
            ),
            "upper": ComponentSpec(
                type="feedbax.PlanarLink",
                params={
                    "length": float(lengths[0]),
                    "mass": float(masses[0]),
                    "com": 0.5,
                    "inertia": float((1.0 / 3.0) * masses[0] * lengths[0] ** 2),
                },
            ),
            "forearm": ComponentSpec(
                type="feedbax.PlanarLink",
                params={
                    "length": float(lengths[1]),
                    "mass": float(masses[1]),
                    "com": 0.5,
                    "inertia": float((1.0 / 3.0) * masses[1] * lengths[1] ** 2),
                },
            ),
            **muscle_nodes,
            "excitation": ComponentSpec(
                type="feedbax.ActuationInput",
                params={
                    "port_name": "excitation",
                    "source_kind": "muscle_excitation_vector",
                },
            ),
            "effector": ComponentSpec(
                type="feedbax.SensorOutput",
                params={"port_name": "effector", "quantity": "effector"},
            ),
            "state": ComponentSpec(
                type="feedbax.SensorOutput",
                params={"port_name": "state", "quantity": "state"},
            ),
            "effector_marker": ComponentSpec(
                type="feedbax.PointMarker",
                params={
                    "frame": "forearm.distal",
                    "offset": [0.0, 0.0],
                    "marker_name": "effector",
                },
            ),
        },
        connections=[
            {"a": ("world", "frame"), "b": ("anchor", "world")},
            {"a": ("anchor", "frame"), "b": ("upper", "proximal")},
            {"a": ("world", "frame"), "b": ("shoulder", "parent")},
            {"a": ("shoulder", "child"), "b": ("upper", "proximal")},
            {"a": ("upper", "distal"), "b": ("elbow", "parent")},
            {"a": ("elbow", "child"), "b": ("forearm", "proximal")},
        ],
        solver={
            "solver_type": str(params["solver_type"]),
            "dt": float(params["dt"]),
        },
    )


__all__ = [
    "mass_spring_damper_template_graph",
    "point_mass_template_graph",
    "point_mass_with_muscles_template_graph",
    "two_link_arm_6muscle_template_graph",
]


from pydantic import BaseModel, ConfigDict, Field
from feedbax.contracts.portable import GraphTemplate, ComponentOptions, validate_authoring
from feedbax.contracts.domain import MECHANICS_DOMAIN_ID


class _SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    solver_type: str = "euler"
    dt: float = Field(default=0.001, gt=0, allow_inf_nan=False)


class PointMassTemplateConfig(_SolverConfig):
    mass: float = Field(default=1.0, ge=0)
    damping: float = Field(default=0.0, ge=0)


class MassSpringDamperConfig(PointMassTemplateConfig):
    stiffness: float = Field(default=10.0, ge=0)
    damping: float = Field(default=0.5, ge=0)


class PointMassWithMusclesConfig(PointMassTemplateConfig):
    max_isometric_force: float = Field(default=1.0, ge=0)


class TwoLinkArm6MuscleConfig(_SolverConfig):
    dt: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    max_isometric_force: float = Field(default=1.0, ge=0)


class MassSpringDamper(GraphTemplate):
    """Executable grounded mass-spring-damper mechanics interior."""

    config = MassSpringDamperConfig
    component_options = ComponentOptions(
        domain=MECHANICS_DOMAIN_ID,
        interior_domain=MECHANICS_DOMAIN_ID,
        is_composite=True,
        template_id="feedbax.templates.mechanics.mass_spring_damper",
        template_kind="executable",
    )

    @classmethod
    def expand(cls, config):
        return mass_spring_damper_template_graph(config.model_dump())


class PointMassWithMuscles(GraphTemplate):
    """Point mass driven by one antagonistic muscle pair."""

    config = PointMassWithMusclesConfig
    component_options = ComponentOptions(
        domain=MECHANICS_DOMAIN_ID,
        interior_domain=MECHANICS_DOMAIN_ID,
        is_composite=True,
        template_id="feedbax.templates.mechanics.point_mass_with_muscles",
        template_kind="executable",
    )

    @classmethod
    def expand(cls, config):
        return point_mass_with_muscles_template_graph(config.model_dump())


class TwoLinkArm6Muscle(GraphTemplate):
    """Editable planar multibody arm with six rigid-tendon muscle paths."""

    config = TwoLinkArm6MuscleConfig
    component_options = ComponentOptions(
        domain=MECHANICS_DOMAIN_ID,
        interior_domain=MECHANICS_DOMAIN_ID,
        is_composite=True,
        template_id="feedbax.templates.mechanics.two_link_arm_6muscle",
        template_kind="executable",
    )

    @classmethod
    def expand(cls, config):
        return two_link_arm_6muscle_template_graph(config.model_dump())
