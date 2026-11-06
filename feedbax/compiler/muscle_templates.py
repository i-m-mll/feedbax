"""Portable causal graphs for the built-in muscle force generators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.mechanics.geometry import TwoLinkArmMuscleGeometry
from feedbax.mechanics.muscle_graph import (
    JointMuscleGeometry,
    RadialMuscleGeometry,
    muscle_path_parameters,
)
from feedbax.mechanics.muscles.relu_muscle import ReluMuscle
from feedbax.mechanics.muscles.thelen_muscle import RigidTendonHillMuscleThelen
from feedbax.runtime.components import MatMul


if TYPE_CHECKING:
    from feedbax.component_registry.declarations import DeclaredComponent


def arm_6muscle_rigid_tendon_graph(
    dt: float = 0.01,
    max_isometric_force: float = 500.0,
    optimal_muscle_length: float = 0.1,
    tendon_slack_length: float = 0.1,
    geometry: TwoLinkArmMuscleGeometry | None = None,
) -> GraphSpec:
    """Build a joint-kinematics and excitation to muscle force and torque graph."""
    if geometry is None:
        geometry = TwoLinkArmMuscleGeometry.default_six_muscle()
    return _muscle_force_graph(
        geometry=ComponentSpec(
            type="JointMuscleGeometry",
            params={"paths": muscle_path_parameters(geometry)},
            input_ports=list(JointMuscleGeometry.input_ports),
            output_ports=list(JointMuscleGeometry.output_ports),
        ),
        muscle_count=geometry.n_muscles,
        actuator=ComponentSpec(
            type="RigidTendonHillMuscleThelen",
            params={
                "dt": dt,
                "max_isometric_force": max_isometric_force,
                "optimal_muscle_length": optimal_muscle_length,
                "tendon_slack_length": tendon_slack_length,
                "n_muscles": 1,
            },
            input_ports=list(RigidTendonHillMuscleThelen.input_ports),
            output_ports=list(RigidTendonHillMuscleThelen.output_ports),
        ),
        kinematics={
            "lengths": "musculotendon_length",
            "velocities": "musculotendon_velocity",
        },
        force_port="torques",
    )


def point_mass_8muscle_relu_graph(
    n_pairs: int = 4,
    max_isometric_force: float = 500.0,
    dt: float = 0.01,
    tau_activation: float = 0.015,
    tau_deactivation: float = 0.05,
) -> GraphSpec:
    """Build an excitation to muscle force and net Cartesian force graph."""
    return _muscle_force_graph(
        geometry=ComponentSpec(
            type="RadialMuscleGeometry",
            params={"n_pairs": n_pairs},
            output_ports=list(RadialMuscleGeometry.output_ports),
        ),
        muscle_count=2 * n_pairs,
        actuator=ComponentSpec(
            type="ReluMuscle",
            params={
                "max_isometric_force": max_isometric_force,
                "dt": dt,
                "tau_activation": tau_activation,
                "tau_deactivation": tau_deactivation,
            },
            input_ports=list(ReluMuscle.input_ports),
            output_ports=list(ReluMuscle.output_ports),
        ),
        kinematics={},
        force_port="force_2d",
    )


def _muscle_force_graph(
    *,
    geometry: ComponentSpec,
    muscle_count: int,
    actuator: ComponentSpec,
    kinematics: dict[str, str],
    force_port: str,
) -> GraphSpec:
    if muscle_count <= 0:
        raise ValueError("Muscle force graphs require at least one muscle")

    scalar_ports = [f"out_{index}" for index in range(muscle_count)]
    mux_ports = [f"in_{index}" for index in range(muscle_count)]
    nodes = {
        "geometry": geometry,
        "reduction": ComponentSpec(
            type="MatMul",
            input_ports=list(MatMul.input_ports),
            output_ports=list(MatMul.output_ports),
        ),
    }
    wires = [
        WireSpec(
            source_node="geometry",
            source_port="directions",
            target_node="reduction",
            target_port="b",
        ),
        WireSpec(
            source_node="forces",
            source_port="output",
            target_node="reduction",
            target_port="a",
        ),
    ]
    for channel in ("excitation", *kinematics):
        nodes[channel] = ComponentSpec(
            type="VectorElements",
            params={"size": muscle_count},
            input_ports=["input"],
            output_ports=scalar_ports,
        )
        if channel in kinematics:
            wires.append(
                WireSpec(
                    source_node="geometry",
                    source_port=channel,
                    target_node=channel,
                    target_port="input",
                )
            )
    for channel in ("forces", "activations"):
        nodes[channel] = ComponentSpec(
            type="Mux",
            params={"n_inputs": muscle_count},
            input_ports=mux_ports,
            output_ports=["output"],
        )
    for index, (scalar_port, mux_port) in enumerate(zip(scalar_ports, mux_ports, strict=True)):
        muscle = f"muscle_{index}"
        nodes[muscle] = actuator.model_copy(deep=True)
        for source, target in {"excitation": "excitation", **kinematics}.items():
            wires.append(
                WireSpec(
                    source_node=source,
                    source_port=scalar_port,
                    target_node=muscle,
                    target_port=target,
                )
            )
        for source, target in (("force", "forces"), ("activation", "activations")):
            wires.append(
                WireSpec(
                    source_node=muscle,
                    source_port=source,
                    target_node=target,
                    target_port=mux_port,
                )
            )

    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=["excitation", *geometry.input_ports],
        output_ports=[force_port, "forces", "activations"],
        input_bindings={
            "excitation": ("excitation", "input"),
            **{port: ("geometry", port) for port in geometry.input_ports},
        },
        output_bindings={
            force_port: ("reduction", "out"),
            "forces": ("forces", "output"),
            "activations": ("activations", "output"),
        },
    )


def arm_6muscle_rigid_tendon_template_definition() -> DeclaredComponent:
    from feedbax.component_registry.declarations import ParameterField, declare_component
    from feedbax.contracts.component import PortType, PortTypeSpec
    from feedbax.mechanics.muscle_representation import (
        _default_two_link_muscle_path_geometry,
        _two_link_muscle_representation,
    )

    graph = arm_6muscle_rigid_tendon_graph()
    return declare_component(
        name="Arm6MuscleRigidTendon",
        template_graph=graph,
        template_id="feedbax.templates.mechanics.arm_6muscle_rigid_tendon",
        template_kind="executable",
        category="Mechanics",
        description="6-muscle arm with Thelen rigid tendon.",
        parameter_fields=[
            ParameterField(
                name="dt",
                type="float",
                default=0.01,
                min=0.001,
                required=True,
            ),
            ParameterField(
                name="max_isometric_force",
                type="float",
                default=500.0,
                min=0.0,
                required=False,
            ),
            ParameterField(
                name="optimal_muscle_length",
                type="float",
                default=0.1,
                min=0.001,
                required=False,
            ),
            ParameterField(
                name="tendon_slack_length",
                type="float",
                default=0.1,
                min=0.0,
                required=False,
            ),
        ],
        input_ports=graph.input_ports,
        output_ports=graph.output_ports,
        icon="Activity",
        is_composite=True,
        port_types=PortTypeSpec(
            inputs={
                "excitation": PortType(dtype="vector"),
                "angles": PortType(dtype="vector"),
                "angular_velocities": PortType(dtype="vector"),
            },
            outputs={
                "torques": PortType(dtype="vector"),
                "forces": PortType(dtype="vector"),
                "activations": PortType(dtype="vector"),
            },
        ),
        representation=_two_link_muscle_representation(
            source="feedbax.mechanics.geometry.TwoLinkArmMuscleGeometry.default_six_muscle",
            composite=True,
            muscle_path_geometry=_default_two_link_muscle_path_geometry(),
            frame_input_port="angles",
        ),
    )


def point_mass_8muscle_relu_template_definition() -> DeclaredComponent:
    from feedbax.component_registry.declarations import ParameterField, declare_component
    from feedbax.contracts.component import PortType, PortTypeSpec
    from feedbax.mechanics.muscle_representation import _point_mass_muscle_representation

    graph = point_mass_8muscle_relu_graph()
    return declare_component(
        name="PointMass8MuscleRelu",
        template_graph=graph,
        template_id="feedbax.templates.mechanics.point_mass_8muscle_relu",
        template_kind="executable",
        category="Mechanics",
        description="8-muscle point mass with ReLU actuators.",
        parameter_fields=[
            ParameterField(
                name="n_pairs",
                type="int",
                default=4,
                min=1,
                required=False,
            ),
            ParameterField(
                name="max_isometric_force",
                type="float",
                default=500.0,
                min=0.0,
                required=False,
            ),
            ParameterField(
                name="dt",
                type="float",
                default=0.01,
                min=0.001,
                required=True,
            ),
        ],
        input_ports=graph.input_ports,
        output_ports=graph.output_ports,
        icon="Activity",
        is_composite=True,
        port_types=PortTypeSpec(
            inputs={"excitation": PortType(dtype="vector")},
            outputs={
                "force_2d": PortType(dtype="vector"),
                "forces": PortType(dtype="vector"),
                "activations": PortType(dtype="vector"),
            },
        ),
        representation=_point_mass_muscle_representation(),
    )
