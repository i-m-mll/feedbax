"""Qualify fixed-guide external muscle graphs with explicit Euler on MJX-Warp.

Use --backend jax --device cpu for the fixture smoke or --device cpu for a Warp
diagnostic. The default Warp CUDA run is the hardware qualification.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from feedbax.compiler import GraphCompilationError, GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from qualify_mjx_warp import (
    assert_single_cuda_device,
    assert_warp_ffi,
    exact_tree,
    select_lane,
    stack_lanes,
)

STEPS = (0.001, 0.0005, 0.00025)
HORIZON = 0.008
REFINEMENT_RATIO = 0.8
REFINEMENT_FLOOR = 2e-6
ENDPOINT_ATOL = {"qvel": 3e-3, "activation": 5e-3, "fiber": 1e-4}
KINEMATIC_TOL = {"rtol": 2e-4, "atol": 2e-5}
ACCELERATION_TOL = {"rtol": 5e-4, "atol": 2e-4}

PLANAR_LAW = {
    "initial_activation": 0.23,
    "max_isometric_force": 22.0,
    "optimal_muscle_length": 0.25,
    "tendon_slack_length": 0.18,
}
SPATIAL_COMPLIANT = {
    "initial_activation": 0.25,
    "max_isometric_force": 25.0,
    "optimal_muscle_length": 0.10,
    "tendon_slack_length": 0.17,
    "pennation_angle": 0.30,
}
SPATIAL_RIGID = {
    "initial_activation": 0.17,
    "max_isometric_force": 15.0,
    "optimal_muscle_length": 0.18,
    "tendon_slack_length": 0.14,
}
SPATIAL_XML = """
<mujoco>
  <option gravity="0 0 0" timestep="{dt}"/>
  <worldbody>
    <site name="origin" pos=".15 -.18 .14"/>
    <site name="guide" pos=".17 -.09 .07"/>
    <site name="antagonist" pos="-.13 .19 -.02"/>
    <site name="return_guide" pos=".02 .13 -.015"/>
    <body name="slider">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass=".5" diaginertia=".03 .04 .05"/>
    </body>
    <body name="arm">
      <joint name="pivot" type="ball" damping=".05"/>
      <inertial pos="0 0 0" mass="1" diaginertia=".11 .14 .19"/>
      <site name="insertion" pos=".17 .04 -.04" euler="20 30 -15"/>
    </body>
  </worldbody>
</mujoco>
"""


def component(name, **params):
    return {"type": f"feedbax.{name}", "params": params}


def ports():
    return {
        "drive": component(
            "ActuationInput",
            port_name="excitation",
            source_kind="muscle_excitation_vector",
        ),
        "tip": component("SensorOutput", port_name="effector", quantity="effector"),
    }


def planar_interior(provider, dt):
    return AcausalGraphSpec.model_validate(
        {
            "physical_domain": "planar_multibody",
            "solver": {"solver_type": "euler", "dt": dt},
            "nodes": {
                **ports(),
                "world": component("WorldFrame"),
                "anchor": component("Anchor", frame="upper.proximal"),
                "upper": component("PlanarLink", length=0.25, mass=1.0, inertia=0.025),
                "lower": component("PlanarLink", length=0.22, mass=0.8, inertia=0.020),
                "shoulder": component(
                    "RevoluteJoint",
                    parent_frame="world.frame",
                    child_frame="upper.proximal",
                    damping=0.03,
                ),
                "elbow": component(
                    "RevoluteJoint",
                    parent_frame="upper.distal",
                    child_frame="lower.proximal",
                    damping=0.02,
                ),
                "marker": component("PointMarker", frame="lower.distal", marker_name="effector"),
                "flexor": component(
                    "MusclePath",
                    law="feedbax.RigidTendonHillMuscle",
                    state_size=1,
                    law_parameters=PLANAR_LAW,
                    path_points=[
                        {"frame": "world.frame", "offset": [-0.06, 0.09]},
                        {"frame": "upper.proximal", "offset": [0.13, 0.06]},
                        {"frame": "lower.proximal", "offset": [0.13, -0.02]},
                    ],
                ),
            },
        }
    )


def spatial_interior(provider, dt):
    asset = provider.store_object(SPATIAL_XML.format(dt=dt).encode())
    return AcausalGraphSpec.model_validate(
        {
            "physical_domain": "spatial_multibody",
            "solver": {"solver_type": "euler", "dt": dt},
            "nodes": ports(),
            "mujoco_asset": {
                "asset": asset.model_dump(),
                "effector_site": "insertion",
                "paths": {
                    "compliant": {
                        "sites": ["origin", "guide", "insertion"],
                        "law": "feedbax.CompliantTendonHillMuscle",
                        "state_size": 2,
                        "parameters": SPATIAL_COMPLIANT,
                    },
                    "rigid": {
                        "sites": ["antagonist", "return_guide", "insertion"],
                        "law": "feedbax.RigidTendonHillMuscle",
                        "state_size": 1,
                        "parameters": SPATIAL_RIGID,
                    },
                },
            },
        }
    )


def executable(interior, provider, *, warp, solver="euler"):
    solver_update = {"solver_type": solver}
    if warp:
        solver_update["mjx_backend"] = "warp"
    interior = interior.model_copy(
        update={
            "solver": interior.solver.model_copy(update=solver_update),
        }
    )
    document = GraphDocument(
        graph=GraphSpec(
            nodes={
                "physics": ComponentSpec(
                    type="feedbax.AcausalSystem",
                    input_ports=["excitation"],
                    output_ports=["effector"],
                )
            },
            subgraphs={"physics": interior},
            input_ports=["excitation"],
            input_bindings={"excitation": ("physics", "excitation")},
            output_ports=["effector"],
            output_bindings={"effector": ("physics", "effector")},
        )
    )
    return compile_graph(document, ComponentRegistry(), artifact_provider=provider).graph


def isometric_multiplier(length, activation):
    active = activation * math.exp(-(((length - 1.0) / 0.56) ** 2))
    passive = math.expm1(4 * max(length - 1.0, 0.0) / 0.7) / math.expm1(4)
    return active + passive


def tension_at_rest(binding, state, length):
    params = binding.parameters
    activation = float(state[0])
    np.testing.assert_allclose(activation, params["initial_activation"], atol=1e-7)
    optimal, slack = params["optimal_muscle_length"], params["tendon_slack_length"]
    if binding.state_size == 2:
        fiber = float(state[1])
        thickness = optimal * math.sin(params["pennation_angle"])
        projection = math.sqrt(fiber**2 - thickness**2)
        strain = max((length - projection) / slack - 1, 0.0)
        spring = math.expm1(35 * strain / 0.033) / math.expm1(35)
        equilibrium = isometric_multiplier(fiber / optimal, activation) * projection / fiber
        np.testing.assert_allclose(spring, equilibrium, rtol=5e-4, atol=2e-5)
        multiplier = spring
    else:
        multiplier = isometric_multiplier((length - slack) / optimal, activation)
    return params["max_isometric_force"] * multiplier


def polyline_length(data, sites):
    return np.linalg.norm(np.diff(data.site_xpos[sites], axis=0), axis=1).sum()


def check_native(node, initial, result, dt):
    descriptor = node.descriptor
    model = descriptor.model
    paths = descriptor.paths
    assert len(paths) == (1 if descriptor.dimension == 2 else 2)
    assert sorted(path.state_size for path in paths) == list(range(1, len(paths) + 1))
    site = descriptor.site_ids[descriptor.effector_site]
    outputs, final, history, final_geometry = result
    qvel = history.physics.plant.skeleton.qvel
    initial = node.state_view(initial).plant
    final = node.state_view(final).plant
    max_qacc_error = 0.0
    for lane in range(qvel.shape[0]):
        start, finish = select_lane(initial, lane), select_lane(final, lane)
        data = mujoco.MjData(model)
        data.qpos[:] = start.skeleton.qpos
        np.testing.assert_array_equal(start.skeleton.qvel, data.qvel)
        mujoco.mj_forward(model, data)
        unforced = data.qacc.copy()
        expected_force = np.zeros(model.nv)
        for binding, state in zip(paths, start.muscles, strict=True):
            actuator = binding.actuator_id
            assert actuator is not None
            assert model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_TENDON
            assert model.actuator_trnid[actuator, 0] == binding.tendon_id
            assert model.actuator_dyntype[actuator] == mujoco.mjtDyn.mjDYN_NONE
            assert model.actuator_gaintype[actuator] == mujoco.mjtGain.mjGAIN_FIXED
            assert (
                not model.actuator_ctrllimited[actuator]
                and not model.actuator_forcelimited[actuator]
            )
            np.testing.assert_array_equal(model.actuator_gear[actuator], [-1, 0, 0, 0, 0, 0])
            assert model.actuator_gainprm[actuator, 0] == 1
            sites = [descriptor.site_ids[name] for name in binding.sites]
            assert len(sites) == 3
            length = polyline_length(data, sites)
            np.testing.assert_allclose(data.ten_length[binding.tendon_id], length, **KINEMATIC_TOL)
            tension = tension_at_rest(binding, state, length)
            assert tension > 0.1, (binding.name, tension)
            data.ctrl[actuator] = tension
            derivative = np.empty(model.nv)
            shifted = mujoco.MjData(model)
            for axis, velocity in enumerate(np.eye(model.nv)):
                lengths = []
                for epsilon in (-1e-6, 1e-6):
                    shifted.qpos[:] = data.qpos
                    mujoco.mj_integratePos(model, shifted.qpos, velocity, epsilon)
                    mujoco.mj_kinematics(model, shifted)
                    lengths.append(polyline_length(shifted, sites))
                derivative[axis] = (lengths[1] - lengths[0]) / 2e-6
            expected_force -= tension * derivative
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.qfrc_actuator, expected_force, **KINEMATIC_TOL)
        assert np.linalg.norm(data.qacc - unforced) > 0.1
        measured = np.asarray(qvel[lane, 1]) / dt
        np.testing.assert_allclose(measured, data.qacc, **ACCELERATION_TOL)
        max_qacc_error = max(max_qacc_error, float(np.max(np.abs(measured - data.qacc))))

        data.qpos[:], data.qvel[:] = finish.skeleton.qpos, finish.skeleton.qvel
        mujoco.mj_forward(model, data)
        tendons = [binding.tendon_id for binding in paths]
        native_jacobian = np.zeros((model.ntendon, model.nv))
        mujoco.mju_sparse2dense(
            native_jacobian, data.ten_J,
            model.ten_J_rownnz, model.ten_J_rowadr, model.ten_J_colind,
        )
        assert np.any(np.abs(data.ten_velocity[tendons]) > 1e-6)
        geometry = select_lane(final_geometry, lane)
        np.testing.assert_allclose(geometry.length, data.ten_length[tendons], **KINEMATIC_TOL)
        np.testing.assert_allclose(geometry.velocity, data.ten_velocity[tendons], **KINEMATIC_TOL)
        np.testing.assert_allclose(geometry.jacobian, native_jacobian[tendons], **KINEMATIC_TOL)
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian, None, site)
        tip = select_lane(outputs["effector"], lane)
        dimension = descriptor.dimension
        np.testing.assert_allclose(tip.pos[-1], data.site_xpos[site, :dimension], **KINEMATIC_TOL)
        np.testing.assert_allclose(tip.vel[-1], (jacobian @ data.qvel)[:dimension], **KINEMATIC_TOL)
        if dimension == 3:
            np.testing.assert_allclose(
                tip.orientation[-1], data.site_xmat[site].reshape(3, 3), **KINEMATIC_TOL
            )
    print(f"Native first-stage acceleration max absolute error {max_qacc_error:.3g}", flush=True)


def endpoints(node, state):
    plant = node.state_view(state).plant
    return {
        "qvel": np.asarray(plant.skeleton.qvel),
        "activation": np.stack([np.asarray(muscle[:, 0]) for muscle in plant.muscles], axis=1),
        "fiber": np.concatenate([np.asarray(muscle[:, 1:]) for muscle in plant.muscles], axis=1),
    }


def check_refinement(rows):
    coarse, medium, fine = rows
    coarse_scaled, medium_scaled = [], []
    for field, limit in ENDPOINT_ATOL.items():
        for values in (coarse, medium, fine):
            assert values[field].dtype == np.float32 and np.isfinite(values[field]).all()
        delta = coarse[field] - fine[field]
        np.testing.assert_allclose(coarse[field], fine[field], rtol=0, atol=limit)
        coarse_scaled.append(delta / limit)
        medium_scaled.append((medium[field] - fine[field]) / limit)
    before = np.linalg.norm(np.concatenate(coarse_scaled, axis=1), axis=1)
    after = np.linalg.norm(np.concatenate(medium_scaled, axis=1), axis=1)
    assert np.all(after <= REFINEMENT_RATIO * before + REFINEMENT_FLOOR), (before, after)
    print(f"Scaled endpoint errors coarse/fine {before}; medium/fine {after}", flush=True)


def check_restart(node, run, final, fresh, commands, keys, path):
    reset = jax.jit(
        jax.vmap(
            lambda old, new, done: jax.tree.map(lambda a, b: jnp.where(done, b, a), old, new),
        )
    )(final, fresh, jnp.array([True, False]))
    exact_tree(select_lane(reset, 0), select_lane(fresh, 0))
    exact_tree(select_lane(reset, 1), select_lane(final, 1))
    reset_result = run(reset, commands, keys)
    fresh_result = run(fresh, commands, keys)
    continued = run(final, commands, keys)
    for actual, fresh_value, continued_value in zip(
        reset_result, fresh_result, continued, strict=True
    ):
        exact_tree(select_lane(actual, 0), select_lane(fresh_value, 0))
        exact_tree(select_lane(actual, 1), select_lane(continued_value, 1))

    lanes = []
    for lane in range(len(keys)):
        snapshot = path / f"muscle-{lane}.eqx"
        node.save_state(snapshot, select_lane(final, lane))
        lanes.append(node.resume_state(snapshot, select_lane(fresh, lane)))
    restored = stack_lanes(lanes)
    exact_tree(restored, final)
    exact_tree(run(restored, commands, keys), continued)


def qualify(name, factory, poses, excitation, provider, path, *, warp, device):
    keys = jax.random.split(jax.random.PRNGKey(33), 2)
    poses, excitation = (jnp.asarray(value, dtype=jnp.float32) for value in (poses, excitation))
    results = []
    for index, dt in enumerate(STEPS):
        interior = factory(provider, dt)
        graph = executable(interior, provider, warp=warp)
        node = graph.nodes["physics"]
        if warp:
            assert not node.supports_reverse_mode
            if index == 0:
                for solver in ("implicit_euler", "kvaerno5", "tsit5"):
                    try:
                        executable(interior, provider, warp=True, solver=solver)
                    except GraphCompilationError as error:
                        message = str(error)
                        assert (
                            "MJX-Warp supports native stepping and explicit Euler" in message
                            or (solver in message and "requires an explicit solver" in message)
                        ), message
                    else:
                        raise AssertionError(f"Warp accepted unsupported solver {solver}")

        def initialize(key, position):
            state = graph.init_state(key=key)
            return node.replace_state_value(state, ("plant", "skeleton", "qpos"), position)

        def rollout(state, command, key):
            outputs, final, history = graph(
                {"excitation": command},
                state,
                key=key,
                n_steps=command.shape[0],
                return_state_history=True,
            )
            skeleton = node.plant.skeleton
            stage = skeleton.continuous_stage(node.state_view(final).plant.skeleton)
            return outputs, final, history, skeleton.muscle_path_kinematics(stage)

        initial = jax.jit(jax.vmap(initialize))(keys, poses)
        qpos = node.state_view(initial).plant.skeleton.qpos
        np.testing.assert_allclose(qpos, poses, rtol=0, atol=5e-7)
        count = round(HORIZON / dt)
        commands = jnp.broadcast_to(excitation[:, None, :], (len(keys), count, excitation.shape[1]))
        lowered = jax.jit(jax.vmap(rollout)).lower(initial, commands, keys)
        run = lowered.compile()
        if warp:
            assert_warp_ffi(lowered, run, device)
        result = run(initial, commands, keys)
        final = node.state_view(result[1]).plant
        np.testing.assert_allclose(final.skeleton.time, HORIZON, rtol=0, atol=1e-7)
        if node.descriptor.dimension == 3:
            pivot = next(joint for joint in node.descriptor.joints if joint.name == "pivot")
            quaternion = final.skeleton.qpos[:, pivot.qpos]
            np.testing.assert_allclose(np.linalg.norm(quaternion, axis=-1), 1, atol=5e-7)
            assert np.all(
                np.abs(
                    final.muscles[0][:, 1] - node.state_view(initial).plant.muscles[0][:, 1],
                )
                > 1e-6
            )
        results.append(endpoints(node, result[1]))
        if index == 1:
            check_native(node, initial, result, dt)
            fresh = jax.jit(jax.vmap(lambda key: graph.init_state(key=key)))(keys)
            check_restart(node, run, result[1], fresh, commands, keys, path)
    check_refinement(results)
    print(
        f"{name}: Euler rollout, native force, refinement, reset and continuation passed",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "warp"), default="warp")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fixture", choices=("all", "planar", "spatial"), default="all")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_platforms", args.device)
    devices = jax.devices()
    assert devices and all(
        device.platform == ("cpu" if args.device == "cpu" else "gpu") for device in devices
    ), devices
    print(f"{args.backend} {args.device}: {[(str(d), d.device_kind) for d in devices]}", flush=True)
    if (args.backend, args.device) == ("warp", "cuda"):
        assert_single_cuda_device(devices)
    print(
        f"Euler dt={STEPS}, horizon={HORIZON}, endpoint atol={ENDPOINT_ATOL}, "
        f"refinement ratio={REFINEMENT_RATIO}, floor={REFINEMENT_FLOOR}",
        flush=True,
    )
    with TemporaryDirectory(prefix="feedbax-qualify-warp-muscles-") as temporary:
        path = Path(temporary)
        provider = ImmutableArtifactBlobProvider(path / "content")
        if args.fixture in ("all", "planar"):
            qualify(
                "planar rigid",
                planar_interior,
                [[0.14, -0.22], [-0.12, 0.18]],
                [[0.35], [0.31]],
                provider,
                path,
                warp=args.backend == "warp",
                device=args.device,
            )
        if args.fixture in ("all", "spatial"):
            poses = [
                [0.03, math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
                for angle in (0.18, -0.12)
            ]
            qualify(
                "spatial mixed",
                spatial_interior,
                poses,
                [[0.34, 0.29], [0.31, 0.24]],
                provider,
                path,
                warp=args.backend == "warp",
                device=args.device,
            )
    if (args.backend, args.device) == ("warp", "cuda"):
        print(f"CUDA MJX-Warp external muscle qualification passed ({args.fixture})", flush=True)
    else:
        print(f"{args.backend} {args.device} external muscle diagnostic passed", flush=True)


if __name__ == "__main__":
    main()
