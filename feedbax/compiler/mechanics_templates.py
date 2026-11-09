"""Executable mechanics-domain acausal graph templates."""

from __future__ import annotations

from typing import Any

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import ComponentSpec
from feedbax.mechanics.body import default_2link_preset
from feedbax.mechanics.model_builder import ChainConfig, SimConfig, planar_chain_graph
from feedbax.mechanics.muscle_config import default_6muscle_2link_topology


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


def two_link_arm_6muscle_template_graph(
    params: dict[str, Any] | None = None,
) -> AcausalGraphSpec:
    """Build an editable planar multibody interior for the canonical arm."""
    params = validate_authoring(TwoLinkArm6MuscleConfig, params or {})
    graph = planar_chain_graph(
        default_2link_preset(),
        ChainConfig(n_joints=2, muscle_topology=default_6muscle_2link_topology()),
        SimConfig(dt=params.dt),
        solver_type=params.solver_type,
    )
    for node in graph.nodes.values():
        if node.type == "feedbax.MusclePath":
            node.params["max_isometric_force"] = params.max_isometric_force
    return graph


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
