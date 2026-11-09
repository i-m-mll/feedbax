"""Production planar fixtures for the closed-loop muscle qualification."""

from __future__ import annotations

from math import acos, cos, sin, sqrt
from typing import Literal

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec, JointDampingMode, PlanarDynamicsSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.mechanics.analytical_plant import TwoLinkMusclePlant
from feedbax.mechanics.articulated_muscles import build_muscle_law
from feedbax.mechanics.authored_model import compile_planar_model
from feedbax.mechanics.backend import DiffraxBackend
from feedbax.mechanics.geometry import TwoLinkArmMuscleGeometry, TwoLinkPointPath
from feedbax.mechanics.hill_muscles import CompliantTendonHillLaw
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.muscle_law import MuscleLawBank
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.mechanics.skeleton._planar import two_link_inertia

CONTROL_DT = 0.01
START_QPOS = (-0.55, 1.30)
LENGTHS = (0.30, 0.33)
MASSES = (1.4, 1.0)
COM_OFFSETS = (0.11, 0.16)
PIVOT_INERTIAS = (0.025, 0.045)
DAMPING_MODES = ((1.0, 0.0375), (-1.0, 0.0125))
DAMPING = ((0.05, 0.025), (0.025, 0.05))
LawProfile = Literal["rigid", "activation", "compliant"]
SHOULDER_WRAP_MAX_DISPLACEMENT_RAD = 0.45
_SHOULDER_WRAP_RADIUS = 0.062
_SHOULDER_WRAP_SEPARATION = 2.10
_SHOULDER_WRAP_SIDE_RADIUS = 0.095


def _point(frame: str, x: float, y: float = 0.0) -> dict:
    return {"frame": frame, "offset": [x, y]}


def _origin(frame: str, pivot_x: float, radius: float, angle: float) -> dict:
    return _point(frame, pivot_x + radius * cos(angle), radius * sin(angle))


def _path_specs(wrap_shoulder: bool) -> list[tuple[list[dict], float, float]]:
    """Author balanced antagonists at the task's initial pose."""
    shoulder, elbow = START_QPOS
    upper_length = LENGTHS[0]
    separation = 1.15

    def length(origin_radius: float, insertion_radius: float) -> float:
        return sqrt(
            origin_radius**2 + insertion_radius**2
            - 2 * origin_radius * insertion_radius * cos(separation)
        )

    origin_radius, insertion_radius = 0.07, 0.12
    shoulder_separation = separation
    shoulder_length = length(origin_radius, insertion_radius)
    if wrap_shoulder:
        radius = _SHOULDER_WRAP_RADIUS
        shoulder_separation = _SHOULDER_WRAP_SEPARATION
        tangent_angle = acos(radius / origin_radius) + acos(radius / insertion_radius)
        shoulder_length = (
            sqrt(origin_radius**2 - radius**2) + sqrt(insertion_radius**2 - radius**2)
            + radius * (shoulder_separation - tangent_angle)
        )
    shoulder_pair = [
        ([_origin("world.frame", 0.0, origin_radius, shoulder + sign * shoulder_separation),
          _point("link0.proximal", insertion_radius)], shoulder_length, 80.0)
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
    law_profile: LawProfile = "rigid",
    direct_torque: bool = False,
    wrap_shoulder: bool = False,
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

    for i, (points, rest_length, max_force) in enumerate(_path_specs(wrap_shoulder)):
        optimal_length = 0.065
        params = {"path_points": points, "state_size": 1}
        if law_profile in ("rigid", "compliant"):
            compliant = law_profile == "compliant" and i >= 4
            params.update(
                law=(
                    "feedbax.CompliantTendonHillMuscle" if compliant
                    else "feedbax.RigidTendonHillMuscle"
                ),
                state_size=2 if compliant else 1,
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

    wrap_geometries = {}
    if wrap_shoulder:
        wrap_geometries["shoulder_wrap"] = {
            "kind": "cylinder", "frame": "world.frame", "radius": _SHOULDER_WRAP_RADIUS,
        }
        for index, sign in enumerate((1, -1)):
            side = f"shoulder_side_{index}"
            nodes[side] = node("PointMarker", **_origin(
                "world.frame", 0.0, _SHOULDER_WRAP_SIDE_RADIUS,
                START_QPOS[0] + sign * _SHOULDER_WRAP_SEPARATION / 2,
            ))
            nodes[f"muscle_{index}"].params["wraps"] = [
                {"after_site": 0, "geom": "shoulder_wrap", "side_site": side},
            ]

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
        wrap_geometries=wrap_geometries,
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
    law_profile: LawProfile = "rigid",
    solver: Literal["euler", "tsit5"] = "euler",
    engine: Literal["mjx", "analytical"] = "mjx",
    wrap_shoulder: bool = False,
) -> Mechanics:
    """Build the benchmark using the production graph or analytical plant."""
    if wrap_shoulder and engine != "mjx":
        raise ValueError("The analytical benchmark only supports fixed guides")
    interior = planar_interior(
        physics_substeps, law_profile=law_profile, solver=solver, wrap_shoulder=wrap_shoulder,
    )
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


def matched_physics(
    physics_substeps: int, *, law_profile: LawProfile, solver: Literal["euler", "tsit5"],
) -> dict[str, float]:
    """Compare independent two-link dynamics and matched production trajectories."""
    import mujoco

    if not jax.config.x64_enabled:
        raise ValueError("The physical reference comparison requires JAX x64")
    analytical = build_mechanics(
        physics_substeps, law_profile=law_profile, solver=solver, engine="analytical",
    )
    mjx_mechanics = build_mechanics(physics_substeps, law_profile=law_profile, solver=solver)
    arm = analytical.plant.skeleton
    model = mjx_mechanics.descriptor.model
    native = mujoco.MjData(model)
    key = jax.random.key(30)
    zero = jnp.zeros(2)
    bare_arm = eqx.tree_at(lambda s: s.B, arm, jnp.zeros((2, 2)))
    direct = {
        "mass_matrix_kg_m2": 0.0,
        "coriolis_nm": 0.0,
        "damping_nm": 0.0,
        "path_length_m": 0.0,
        "path_velocity_m_s": 0.0,
        "path_jacobian_m": 0.0,
        "coupled_accel_rad_s2": 0.0,
    }

    @jax.jit
    def stages(qpos, qvel, excitation):
        native_state = analytical.plant.init(key=key)
        native_state = eqx.tree_at(
            lambda state: (state.skeleton.angle, state.skeleton.d_angle), native_state,
            (qpos, qvel),
        )
        engine_state = mjx_mechanics.plant.init(key=key)
        engine_state = eqx.tree_at(
            lambda state: (state.skeleton.qpos, state.skeleton.qvel), engine_state,
            (qpos, qvel),
        )
        skel = mjx_mechanics.plant.skeleton
        mjx_path = skel.muscle_path_kinematics(skel.continuous_stage(engine_state.skeleton))
        ref_path = analytical.plant.geometry.path_kinematics(qpos, qvel)
        if law_profile == "compliant":
            native_state = eqx.tree_at(
                lambda state: state.muscles, native_state,
                analytical.plant.law_bank.init_state(*ref_path[:2]),
            )
            engine_state = mjx_mechanics.plant.equilibrate_muscles(engine_state)
        mjx_acc = mjx_mechanics.plant.vector_field(0.0, engine_state, excitation).skeleton.qvel
        ref_acc = analytical.plant.vector_field(0.0, native_state, excitation).skeleton.d_angle
        return mjx_path, ref_path, mjx_acc, ref_acc

    for qpos, qvel in (
        (START_QPOS, (0.35, -0.72)), ((-0.38, 1.12), (-0.40, 0.85)),
        ((-0.73, 1.53), (0.83, -0.53)),
    ):
        native.qpos[:], native.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, native)
        inertia = np.empty((2, 2))
        mujoco.mj_fullM(model, native, inertia)
        qpos, qvel = jnp.asarray(qpos), jnp.asarray(qvel)
        expected_inertia, _ = two_link_inertia(arm.l, arm.m, arm.I, arm.s, qpos[1])
        arm_state = eqx.tree_at(
            lambda state: (state.angle, state.d_angle), arm.init(), (qpos, qvel),
        )
        coriolis = -expected_inertia @ bare_arm.vector_field(0.0, arm_state, zero).d_angle
        observed, reference, mjx_acc, reference_acc = stages(
            qpos, qvel, jnp.linspace(0.2, 0.8, 6),
        )
        errors = (
            inertia - expected_inertia, native.qfrc_bias - coriolis,
            native.qfrc_passive + arm.B @ qvel,
            *(left - right for left, right in zip(observed, reference, strict=True)),
            mjx_acc - reference_acc,
        )
        for name, error in zip(direct, errors, strict=True):
            direct[name] = max(direct[name], float(jnp.max(jnp.abs(error))))

    phase = jnp.arange(60)[:, None] * (2 * jnp.pi / 60)
    commands = 0.2 + 0.15 * jnp.sin(phase + jnp.arange(6)[None, :])

    def rollout(mechanics, coordinate):
        mechanics = mechanics.prepare()
        state = mechanics.init_state(key=key)
        state = mechanics.replace_state_value(
            state, ("plant", "skeleton", coordinate), jnp.asarray(START_QPOS),
        )

        @jax.jit
        def run(initial):
            def step(carry, command):
                out, carry = mechanics({"force": command}, carry, key=key)
                return carry, (out["effector"].pos, out["effector"].vel)

            return jax.lax.scan(step, initial, commands)[1]

        return run(state)

    actual_pos, actual_vel = rollout(mjx_mechanics, "qpos")
    ref_pos, ref_vel = rollout(analytical, "angle")
    direct["rollout_position_m"] = float(jnp.max(jnp.linalg.norm(actual_pos - ref_pos, axis=-1)))
    direct["rollout_velocity_m_s"] = float(jnp.max(jnp.linalg.norm(actual_vel - ref_vel, axis=-1)))
    if law_profile == "compliant":
        plant = analytical.plant
        initial = plant.init(key=key)
        lengths = plant.geometry.musculotendon_lengths(initial.skeleton.angle)
        residuals = []
        for law, state, length in zip(plant.law_bank.laws, initial.muscles, lengths, strict=True):
            if isinstance(law, CompliantTendonHillLaw):
                activation, fiber = state
                projected_fiber = law.fiber_force(activation, fiber, jnp.zeros_like(fiber)) * jnp.cos(
                    law.pennation(fiber)
                )
                spring_force = law.tendon_force(law.tendon_length(fiber, length))
                residuals.append(float(jnp.abs(projected_fiber - spring_force)))
        direct["initial_compliant_equilibrium_n"] = max(residuals)
    return direct
