"""Production planar fixtures for the closed-loop muscle qualification."""

from __future__ import annotations

from math import cos, sin, sqrt
from typing import Literal

import diffrax

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec, JointDampingMode, PlanarDynamicsSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.mechanics.analytical_plant import TwoLinkMusclePlant
from feedbax.mechanics.articulated_muscles import build_muscle_law
from feedbax.mechanics.authored_model import compile_planar_model
from feedbax.mechanics.backend import DiffraxBackend
from feedbax.mechanics.geometry import TwoLinkArmMuscleGeometry, TwoLinkPointPath
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.muscle_law import MuscleLawBank
from feedbax.mechanics.skeleton.arm import TwoLinkArm

CONTROL_DT = 0.01
START_QPOS = (-0.55, 1.30)
LENGTHS = (0.30, 0.33)
MASSES = (1.4, 1.0)
COM_OFFSETS = (0.11, 0.16)
PIVOT_INERTIAS = (0.025, 0.045)
DAMPING_MODES = ((1.0, 0.0375), (-1.0, 0.0125))
DAMPING = ((0.05, 0.025), (0.025, 0.05))


def _point(frame: str, x: float, y: float = 0.0) -> dict:
    return {"frame": frame, "offset": [x, y]}


def _origin(frame: str, pivot_x: float, radius: float, angle: float) -> dict:
    return _point(frame, pivot_x + radius * cos(angle), radius * sin(angle))


def _path_specs() -> list[tuple[list[dict], float, float]]:
    """Author balanced antagonists at the task's initial pose."""
    shoulder, elbow = START_QPOS
    upper_length = LENGTHS[0]
    separation = 1.15

    def length(origin_radius: float, insertion_radius: float) -> float:
        return sqrt(
            origin_radius**2 + insertion_radius**2
            - 2 * origin_radius * insertion_radius * cos(separation)
        )

    shoulder_pair = [
        ([_origin("world.frame", 0.0, 0.07, shoulder + sign * separation),
          _point("link0.proximal", 0.12)], length(0.07, 0.12), 80.0)
        for sign in (1, -1)
    ]
    elbow_pair = [
        ([_origin("link0.proximal", upper_length, 0.05, elbow + sign * separation),
          _point("link1.proximal", 0.10)], length(0.05, 0.10), 50.0)
        for sign in (1, -1)
    ]
    biarticular_pair = []
    for sign in (1, -1):
        shoulder_origin = _origin("world.frame", 0.0, 0.05, shoulder + sign * separation)
        upper_guide = _point("link0.proximal", 0.08)
        elbow_guide = _origin("link0.proximal", upper_length, 0.035, elbow + sign * separation)
        guide_x, guide_y = elbow_guide["offset"]
        rest_length = length(0.05, 0.08) + sqrt((guide_x - 0.08)**2 + guide_y**2) + length(
            0.035, 0.08,
        )
        biarticular_pair.append((
            [shoulder_origin, upper_guide, elbow_guide, _point("link1.proximal", 0.08)],
            rest_length, 35.0,
        ))
    return shoulder_pair + elbow_pair + biarticular_pair


def planar_interior(
    physics_substeps: int,
    *,
    law_profile: Literal["rigid", "activation"] = "rigid",
    direct_torque: bool = False,
    solver: Literal["euler", "tsit5"] = "euler",
) -> AcausalGraphSpec:
    """Author a horizontal two-link, six-muscle arm inspired by Li and Todorov."""
    def node(kind: str, **params) -> ComponentSpec:
        return ComponentSpec(type=f"feedbax.{kind}", params=params)

    nodes = {
        "world": node("WorldFrame"),
        "anchor": node("Anchor", frame="link0.proximal", world_frame="world.frame"),
    }
    for i, (length, mass, com, inertia) in enumerate(
        zip(LENGTHS, MASSES, COM_OFFSETS, PIVOT_INERTIAS, strict=True)
    ):
        nodes[f"link{i}"] = node(
            "PlanarLink", length=length, mass=mass, com=com / length,
            inertia=inertia - mass * com**2,
        )
        parent_frame = "world.frame" if i == 0 else f"link{i-1}.distal"
        nodes[f"joint{i}"] = node(
            "RevoluteJoint", parent_frame=parent_frame, child_frame=f"link{i}.proximal",
        )

    for i, (points, rest_length, max_force) in enumerate(_path_specs()):
        optimal_length = 0.065
        params = {"path_points": points, "state_size": 1}
        if law_profile == "rigid":
            params.update(
                law="feedbax.RigidTendonHillMuscle",
                law_parameters={
                    "max_isometric_force": max_force,
                    "optimal_muscle_length": optimal_length,
                    "tendon_slack_length": rest_length - optimal_length,
                    "tau_act": 0.01,
                    "tau_deact": 0.04,
                    "initial_activation": 0.05,
                },
            )
        else:
            params.update(
                law="feedbax.ReluMuscle",
                law_parameters={
                    "max_isometric_force": max_force,
                    "tau_activation": 0.01,
                    "tau_deactivation": 0.04,
                    "initial_activation": 0.05,
                },
            )
        nodes[f"muscle_{i}"] = node("MusclePath", **params)

    nodes.update({
        "drive": node(
            "ActuationInput", port_name="force",
            source_kind="actuator_control" if direct_torque else "muscle_excitation_vector",
            size=2 if direct_torque else 6,
        ),
        "effector": node("SensorOutput", port_name="effector", quantity="effector"),
        "state": node("SensorOutput", port_name="state", quantity="state"),
        "tip": node("PointMarker", frame="link1.distal", marker_name="effector"),
    })
    return AcausalGraphSpec(
        physical_domain="planar_multibody",
        nodes=nodes,
        solver={"solver_type": solver, "dt": CONTROL_DT, "planar_substeps": physics_substeps},
        planar_dynamics=PlanarDynamicsSpec(
            gravity=(0.0, 0.0),
            joint_dampers={
                f"mode_{i}": JointDampingMode(
                    coefficients={"joint0": 1.0, "joint1": sign}, damping=damping,
                )
                for i, (sign, damping) in enumerate(DAMPING_MODES)
            },
        ),
    )


def graph_document(interior: AcausalGraphSpec) -> GraphDocument:
    return GraphDocument(graph=GraphSpec(
        nodes={"mechanics": ComponentSpec(
            type="feedbax.AcausalSystem",
            input_ports=["force"],
            output_ports=["effector", "state"],
        )},
        subgraphs={"mechanics": interior},
        input_ports=["force"],
        input_bindings={"force": ("mechanics", "force")},
        output_ports=["effector", "state"],
        output_bindings={port: ("mechanics", port) for port in ("effector", "state")},
    ))


def build_mechanics(
    physics_substeps: int,
    *,
    law_profile: Literal["rigid", "activation"] = "rigid",
    solver: Literal["euler", "tsit5"] = "euler",
    engine: Literal["mjx", "analytical"] = "mjx",
) -> Mechanics:
    """Build the benchmark using the production graph or analytical plant."""
    interior = planar_interior(physics_substeps, law_profile=law_profile, solver=solver)
    registry = ComponentRegistry()
    if engine == "mjx":
        return compile_graph(graph_document(interior), registry).graph.nodes["mechanics"]
    if engine != "analytical":
        raise ValueError(f"Unknown benchmark engine {engine!r}")

    descriptor = compile_planar_model(interior)
    frames = {"world.frame": 0, "link0.proximal": 1, "link1.proximal": 2}
    paths = []
    for path in descriptor.paths:
        points = interior.nodes[path.name].params["path_points"]
        paths.append(TwoLinkPointPath(
            LENGTHS[0], tuple(frames[point["frame"]] for point in points),
            [point["offset"] for point in points],
        ))
    law_bank = MuscleLawBank(tuple(build_muscle_law(path, registry)[0] for path in descriptor.paths))
    plant = TwoLinkMusclePlant(
        TwoLinkArm(LENGTHS, MASSES, PIVOT_INERTIAS, COM_OFFSETS, DAMPING),
        TwoLinkArmMuscleGeometry(tuple(paths)),
        law_bank,
        START_QPOS,
    )
    backend = DiffraxBackend(
        control_dt=CONTROL_DT,
        sub_dt=CONTROL_DT / physics_substeps,
        solver={"euler": diffrax.Euler, "tsit5": diffrax.Tsit5}[solver](),
    )
    return Mechanics(plant=plant, backend=backend)
