"""Qualify MJX-Warp on NVIDIA CUDA through Feedbax's authored graph API.

Run with --backend jax --device cpu for a fixture smoke, or --device cpu
for a Warp CPU diagnostic. CUDA qualification requires the default Warp CUDA run.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.mechanics.body import default_3link_bounds, sample_preset
from feedbax.mechanics.model_builder import ChainConfig, SimConfig, planar_chain_graph
from feedbax.mechanics.skeleton.mjx_skeleton import AppliedWrench
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

DT = 0.002
ATOL = 2e-5
RTOL = 2e-4
TIME_ATOL = 1e-7
COMMON_FIELDS = ("qpos", "qvel", "act", "time", "effector_force", "wrenches")
SPATIAL_XML = b"""
<mujoco>
  <option timestep="0.002" gravity="0 0 0" integrator="Euler"/>
  <worldbody>
    <site name="muscle_anchor" pos="-.8 .4 .3"/>
    <body name="tool" pos="0 .2 .1">
      <freejoint name="joint"/>
      <inertial pos=".06 .03 -.04" mass="2" diaginertia=".2 .3 .4"/>
      <site name="tip" pos=".2 .1 .3" quat=".9238795325 0 .3826834324 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="native_path">
      <site site="muscle_anchor"/>
      <site site="tip"/>
    </spatial>
  </tendon>
  <actuator>
    <general name="drive" joint="joint" gear=".2 -.1 1 .3 1 -.2"
             dyntype="filter" dynprm=".02"/>
    <muscle name="native_muscle" tendon="native_path" force="2" lengthrange=".2 1.6"/>
  </actuator>
</mujoco>
"""


def planar_interior():
    graph = planar_chain_graph(
        sample_preset(default_3link_bounds(), jax.random.PRNGKey(10)),
        ChainConfig(),
        SimConfig(dt=DT),
        solver_type="mujoco",
    )
    graph.nodes = {
        name: node
        for name, node in graph.nodes.items()
        if node.type != "feedbax.MusclePath" and name != "state"
    }
    graph.nodes["excitation"].params.update(
        port_name="control",
        source_kind="actuator_control",
        size=3,
    )
    graph.nodes["joint0"].params.update(limit_min=-0.4, limit_max=0.19)
    return graph


def spatial_interior(provider):
    asset = provider.store_object(SPATIAL_XML)
    return AcausalGraphSpec.model_validate(
        {
            "physical_domain": "spatial_multibody",
            "mujoco_asset": {"asset": asset.model_dump(), "effector_site": "tip"},
            "solver": {"solver_type": "mujoco", "dt": DT},
            "nodes": {
                "drive": {
                    "type": "feedbax.ActuationInput",
                    "params": {
                        "port_name": "control",
                        "source_kind": "actuator_control",
                        "size": 2,
                    },
                },
                "sense": {
                    "type": "feedbax.SensorOutput",
                    "params": {
                        "port_name": "effector",
                        "quantity": "effector",
                    },
                },
            },
        }
    )


def executable(interior, provider, *, warp):
    if warp:
        interior = interior.model_copy(
            update={
                "solver": interior.solver.model_copy(update={"mjx_backend": "warp"}),
            }
        )
    document = GraphDocument(
        graph=GraphSpec(
            nodes={
                "physics": ComponentSpec(
                    type="feedbax.AcausalSystem",
                    input_ports=["control"],
                    output_ports=["effector"],
                )
            },
            subgraphs={"physics": interior},
            input_ports=["control"],
            input_bindings={"control": ("physics", "control")},
            output_ports=["effector"],
            output_bindings={"effector": ("physics", "effector")},
        )
    )
    return compile_graph(document, ComponentRegistry(), artifact_provider=provider).graph


def select_lane(tree, lane):
    return jax.tree.map(lambda array: array[lane], tree)


def stack_lanes(lanes):
    return jax.tree.map(lambda *arrays: jnp.stack(arrays), *lanes)


def exact_tree(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(value, reference)


def assert_warp_ffi(lowered, compiled, device):
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert "stablehlo.while" in stablehlo, "Expected a compiled scan in the authored graph"
    # MJX's variadic-tuple adapter registers its Warp callable under its wrapper name.
    hlo = compiled.as_text()
    targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo)))
    warp_targets = [target for target in targets if "warp" in target and "ffi" in target]
    assert warp_targets, f"No Warp FFI survived {device} compilation. Custom calls: {targets}"
    print(f"{device} Warp FFI targets: {warp_targets}", flush=True)


def native_trace(model, initial, controls, body, site, dimension, wrench):
    """Advance independent MuJoCo MjData at matched timestep and held body loads."""
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:], data.time = (
        np.asarray(initial.qpos),
        np.asarray(initial.qvel),
        np.asarray(initial.act),
        float(initial.time),
    )
    continuation = (
        mujoco.mjtState.mjSTATE_WARMSTART | mujoco.mjtState.mjSTATE_EQ_ACTIVE
        | mujoco.mjtState.mjSTATE_MOCAP_POS | mujoco.mjtState.mjSTATE_MOCAP_QUAT
    )
    mujoco.mj_setState(model, data, np.asarray(initial.continuation, np.float64), continuation)
    local_force = np.zeros(3)
    local_moment = np.zeros(3)
    local_point = np.zeros(3)
    if wrench is not None:
        local_force[:dimension] = wrench.force
        local_moment[-len(wrench.moment) :] = wrench.moment
        local_point[:dimension] = wrench.point
    jac = np.zeros((3, model.nv))
    body_jac = np.zeros_like(jac)
    records = []
    mujoco.mj_forward(model, data)
    for command in controls:
        rotation = data.xmat[body].reshape(3, 3)
        force = rotation @ local_force
        offset = data.xpos[body] + rotation @ local_point - data.xipos[body]
        moment = rotation @ local_moment + np.cross(offset, force)
        data.xfrc_applied[body] = np.concatenate((force, moment))
        data.ctrl[:] = command
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jac, None, site)
        mujoco.mj_jacBody(model, data, body_jac, None, body)
        record = {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "act": data.act.copy(),
            "time": data.time,
            "position": data.site_xpos[site, :dimension].copy(),
            "velocity": (jac @ data.qvel)[:dimension],
            "body_position": data.xpos[body, :dimension].copy(),
            "body_velocity": (body_jac @ data.qvel)[:dimension],
        }
        if dimension == 3:
            record["orientation"] = data.site_xmat[site].reshape(3, 3).copy()
        records.append(record)
    return {name: np.stack([record[name] for record in records]) for name in records[0]}


def compare_native(node, result, initial, commands, body_name, wrenches, label):
    outputs, _, history, bodies = result
    physical = history.physics.plant.skeleton
    site_state = outputs["effector"]
    actual = {name: getattr(physical, name)[:, 1:] for name in ("qpos", "qvel", "act", "time")}
    actual.update(position=site_state.pos, velocity=site_state.vel)
    descriptor = node.descriptor
    body = descriptor.body_ids[body_name]
    actual.update(
        body_position=bodies.pos[:, body, None, :],
        body_velocity=bodies.vel[:, body, None, :],
    )
    if descriptor.dimension == 3:
        actual["orientation"] = site_state.orientation
    max_errors = dict.fromkeys(actual, 0.0)
    for lane, control in enumerate(np.asarray(commands)):
        reference = native_trace(
            descriptor.model,
            select_lane(initial, lane),
            control,
            descriptor.body_ids[body_name],
            descriptor.site_ids[descriptor.effector_site],
            descriptor.dimension,
            wrenches[lane],
        )
        for field, expected in reference.items():
            value = np.asarray(actual[field][lane])
            if field in {"body_position", "body_velocity"}:
                expected = expected[-1:]
            assert value.dtype == np.float32, (label, field, value.dtype)
            np.testing.assert_allclose(
                value,
                expected,
                rtol=0 if field == "time" else RTOL,
                atol=TIME_ATOL if field == "time" else ATOL,
                equal_nan=False,
                err_msg=f"{label}, world {lane}, {field}",
            )
            max_errors[field] = max(
                max_errors[field],
                float(np.max(np.abs(value - expected), initial=0.0)),
            )
    print(
        f"{label} max absolute errors: "
        + ", ".join(f"{field}={error:.3g}" for field, error in max_errors.items()),
        flush=True,
    )


def qualify(interior, provider, path, *, warp, device, body, qpos, qvel, wrench, commands):
    graph = executable(interior, provider, warp=warp)
    node = graph.nodes["physics"]
    assert not warp or not node.supports_reverse_mode
    assert node.backend.n_substeps == 1
    assert node.descriptor.model.opt.timestep == DT
    keys = jax.random.split(jax.random.PRNGKey(0), len(qpos))
    commands = jnp.asarray(commands, dtype=jnp.float32)
    qpos, qvel = jnp.asarray(qpos, dtype=jnp.float32), jnp.asarray(qvel, dtype=jnp.float32)

    def initialize(key, position, velocity):
        state = graph.init_state(key=key)
        for field, value in (("qpos", position), ("qvel", velocity)):
            state = node.replace_state_value(state, ("plant", "skeleton", field), value)
        return node.apply_wrench(state, body, wrench)

    def batched_rollout(graph):
        physics = graph.nodes["physics"]

        def rollout(state, control, key):
            outputs, final, history = graph(
                {"control": control},
                state,
                key=key,
                n_steps=control.shape[0],
                return_state_history=True,
            )
            bodies = physics.plant.skeleton.forward_kinematics(
                physics.state_view(final).plant.skeleton,
            )
            return outputs, final, history, bodies

        return jax.jit(jax.vmap(rollout))

    def check(result, start, label, node=node, wrenches=(wrench, wrench)):
        compare_native(
            node,
            result,
            node.state_view(start).plant.skeleton,
            commands,
            body,
            wrenches,
            label,
        )

    initial = jax.jit(jax.vmap(initialize))(keys, qpos, qvel)
    initial_physical = node.state_view(initial).plant.skeleton
    expected_qpos = np.asarray(qpos).copy()
    if node.descriptor.dimension == 3:
        expected_qpos[:, 3:] /= np.linalg.norm(expected_qpos[:, 3:], axis=1, keepdims=True)
    np.testing.assert_allclose(initial_physical.qpos, expected_qpos, rtol=0, atol=5e-7)
    np.testing.assert_array_equal(initial_physical.qvel, qvel)
    print(f"Compiling {interior.physical_domain} graph", flush=True)
    lowered = batched_rollout(graph).lower(initial, commands, keys)
    run = lowered.compile()
    if warp:
        assert_warp_ffi(lowered, run, device)
    result = run(initial, commands, keys)
    check(result, initial, f"{interior.physical_domain} rollout")
    final = result[1]
    if node.descriptor.model.na:
        activation = np.asarray(node.state_view(final).plant.skeleton.act)
        assert np.all(np.abs(activation) > 0.05), activation

    fresh = jax.jit(jax.vmap(lambda key: graph.init_state(key=key)))(keys)

    def reset_if_done(old, new, done):
        return jax.tree.map(lambda a, b: jnp.where(done, b, a), old, new)

    reset = jax.jit(jax.vmap(reset_if_done))(final, fresh, jnp.array([True, False]))
    exact_tree(select_lane(reset, 0), select_lane(fresh, 0))
    exact_tree(select_lane(reset, 1), select_lane(final, 1))
    reset_rollout = run(reset, commands, keys)
    check(
        reset_rollout, reset, f"{interior.physical_domain} selective reset", wrenches=(None, wrench)
    )

    resumed_lanes = []
    for lane in range(len(qpos)):
        snapshot = path / f"{interior.physical_domain}-{lane}.eqx"
        node.save_state(snapshot, select_lane(final, lane))
        resumed_lanes.append(node.resume_state(snapshot, select_lane(fresh, lane)))
    resumed = stack_lanes(resumed_lanes)
    exact_tree(resumed, final)
    continuation = run(final, commands, keys)
    exact_tree(run(resumed, commands, keys), continuation)
    check(continuation, final, f"{interior.physical_domain} file continuation")

    if warp:
        alternate_graph = executable(interior, provider, warp=False)
        alternate_node = alternate_graph.nodes["physics"]
        alternate_fresh = jax.jit(jax.vmap(lambda key: alternate_graph.init_state(key=key)))(keys)
    else:
        alternate_graph, alternate_node, alternate_fresh = graph, node, fresh

    def transfer(source, source_state, destination, destination_fresh):
        moved = jax.jit(
            jax.vmap(
                lambda old, fresh: destination.transfer_state(source, old, fresh),
            )
        )(source_state, destination_fresh)
        source_physics = source.state_view(source_state).plant.skeleton
        moved_physics = destination.state_view(moved).plant.skeleton
        for field in COMMON_FIELDS:
            exact_tree(getattr(moved_physics, field), getattr(source_physics, field))
        return moved

    alternate_run = batched_rollout(alternate_graph)
    if warp:
        shared_initial = transfer(node, initial, alternate_node, alternate_fresh)
        reference = alternate_run(shared_initial, commands, keys)
        check(reference, shared_initial, f"{interior.physical_domain} JAX reference", alternate_node)
        for actual, expected in zip(
            jax.tree.leaves(result[0]["effector"]),
            jax.tree.leaves(reference[0]["effector"]),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
    moved = transfer(node, final, alternate_node, alternate_fresh)
    other_result = alternate_run(moved, commands, keys)
    check(other_result, moved, f"{interior.physical_domain} transfer to JAX", alternate_node)
    returned = transfer(alternate_node, other_result[1], node, fresh)
    check(run(returned, commands, keys), returned, f"{interior.physical_domain} transfer from JAX")
    print(
        f"{interior.physical_domain}: rollout, reset, file continuation, and transfer passed",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "warp"), default="warp")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fixture", choices=("planar", "spatial", "all"), default="all")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_platforms", args.device)
    for package in (
        "mujoco-core",
        "mujoco-mjx",
        "jax",
        "jaxlib",
        "warp-lang",
        "jax-cuda13-plugin",
        "jax-cuda13-pjrt",
        "numpy",
    ):
        try:
            print(f"{package} {version(package)}", flush=True)
        except PackageNotFoundError:
            continue
    devices = jax.devices()
    assert devices and all(
        device.platform == ("cpu" if args.device == "cpu" else "gpu") for device in devices
    ), devices
    print(f"Devices: {[(str(device), device.device_kind) for device in devices]}", flush=True)
    print(f"Float32 acceptance: rtol={RTOL}, atol={ATOL}, time atol={TIME_ATOL}", flush=True)
    with TemporaryDirectory(prefix="feedbax-qualify-mjx-warp-") as temporary:
        path = Path(temporary)
        provider = ImmutableArtifactBlobProvider(path / "content")
        if args.fixture in ("all", "planar"):
            qualify(
                planar_interior(),
                provider,
                path,
                warp=args.backend == "warp",
                device=args.device,
                body="link2",
                qpos=[[0.2, -0.4, 0.3], [-0.3, 0.35, -0.2]],
                qvel=[[0.3, -0.2, 0.1], [-0.1, 0.2, -0.3]],
                wrench=AppliedWrench(
                    force=jnp.array([0.4, -0.7]),
                    moment=jnp.array([0.05]),
                    point=jnp.array([0.13, -0.04]),
                    frame="body",
                ),
                commands=[
                    [[0.12, -0.14, 0.03], [0.18, -0.1, 0.05], [0.2, -0.15, 0.01], [0.14, -0.12, 0.07]],
                    [
                        [-0.16, 0.11, -0.02],
                        [-0.11, 0.17, -0.03],
                        [-0.2, 0.13, -0.02],
                        [-0.15, 0.18, -0.03],
                    ],
                ],
            )
        if args.fixture in ("all", "spatial"):
            qualify(
                spatial_interior(provider),
                provider,
                path,
                warp=args.backend == "warp",
                device=args.device,
                body="tool",
                qpos=[
                    [0.08, 0.21, 0.11, 0.9, 0.18, -0.27, 0.29],
                    [0.04, 0.17, 0.15, 0.85, -0.21, 0.33, 0.32],
                ],
                qvel=[[0.07, -0.03, 0.11, 0.3, -0.4, 0.2], [-0.05, 0.08, -0.13, -0.2, 0.3, 0.4]],
                wrench=AppliedWrench(
                    force=jnp.array([0.2, 0.1, 1.1]),
                    moment=jnp.array([0.4, -0.3, 0.1]),
                    point=jnp.array([0.15, 0.03, 0.08]),
                    frame="body",
                ),
                commands=[
                    [[0.9, 0.55], [0.6, 0.42], [0.7, 0.49], [0.8, 0.52]],
                    [[-0.8, 0.44], [-0.5, 0.61], [-0.7, 0.39], [-0.6, 0.56]],
                ],
            )
    if (args.backend, args.device) == ("warp", "cuda"):
        print(f"CUDA MJX-Warp qualification passed ({args.fixture})")
    else:
        print(f"{args.backend} {args.device} fixture diagnostic passed")


if __name__ == "__main__":
    main()
