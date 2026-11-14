"""Qualification of the compiled MuJoCo boundary through the public graph route."""

from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.compiler import GraphCompilationError, GraphDocument, compile_graph
from feedbax.compiler.serialization import graph_to_spec
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.mechanics.body import default_3link_bounds, sample_preset
from feedbax.mechanics.model_builder import ChainConfig, SimConfig, planar_chain_graph
from feedbax.mechanics.skeleton.mjx_skeleton import AppliedWrench
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


mujoco = pytest.importorskip("mujoco")
pytestmark = [pytest.mark.optional_mjx, pytest.mark.usefixtures("enable_jax_x64")]


def _document(interior: AcausalGraphSpec) -> GraphDocument:
    return GraphDocument(graph=GraphSpec(
        nodes={"physics": ComponentSpec(
            type="feedbax.AcausalSystem", input_ports=["control"], output_ports=["effector"],
        )},
        subgraphs={"physics": interior},
        input_ports=["control"], input_bindings={"control": ("physics", "control")},
        output_ports=["effector"], output_bindings={"effector": ("physics", "effector")},
    ))


def _spatial(provider, solver="mujoco", dt=0.005):
    asset = provider.store_object(f"""
      <mujoco>
        <option timestep="{dt}" gravity="0 0 0" integrator="RK4"/>
        <worldbody>
          <body name="tool" pos="0 .2 .1">
            <freejoint name="joint"/>
            <inertial pos=".06 .03 -.04" mass="2" diaginertia=".2 .3 .4"/>
            <site name="tip" pos=".2 .1 .3" quat=".9238795325 0 .3826834324 0"/>
          </body>
        </worldbody>
        <actuator>
          <general name="drive" joint="joint" gear="0 0 1 0 1 0" dyntype="filter" dynprm=".02"/>
        </actuator>
      </mujoco>
    """.encode())
    return AcausalGraphSpec.model_validate({
        "physical_domain": "spatial_multibody",
        "mujoco_asset": {"asset": asset.model_dump(), "effector_site": "tip"},
        "solver": {"solver_type": solver, "dt": dt},
        "nodes": {
            "drive": {"type": "feedbax.ActuationInput", "params": {
                "port_name": "control", "source_kind": "actuator_control", "size": 1,
            }},
            "sense": {"type": "feedbax.SensorOutput", "params": {
                "port_name": "effector", "quantity": "effector",
            }},
        },
    })


def test_planar_graph_executes_three_authored_hinges_and_rejects_spatial_forces():
    key = jax.random.PRNGKey(0)
    interior = planar_chain_graph(
        sample_preset(default_3link_bounds(), key), ChainConfig(), SimConfig(),
    )
    interior.nodes["excitation"].params.update(
        port_name="control", source_kind="actuator_control", size=3,
    )
    graph = compile_graph(_document(interior), ComponentRegistry()).graph
    node = graph.nodes["physics"]
    initial = graph.init_state(key=key)
    outputs, state = graph({"control": jnp.ones(3)}, initial, key=key)
    assert outputs["effector"].pos.shape == (2,)
    assert node.state_view(state).plant.skeleton.qpos.shape == (3,)
    assert node.state_view(state).plant.skeleton.time > 0
    with pytest.raises(ValueError, match="Effector force requires shape"):
        node.plant.skeleton.update_state_given_effector_force(
            jnp.ones(3), node.state_view(state).plant.skeleton,
        )

    interior.nodes["joint0"].params.update(limit_min=-1.0, limit_max=1.0)
    with pytest.raises(GraphCompilationError, match="zero compiled constraint rows"):
        compile_graph(_document(interior), ComponentRegistry())


def test_spatial_native_wrench_and_checkpoint_match_continuation(tmp_path):
    key = jax.random.PRNGKey(0)
    registry = ComponentRegistry()
    provider = ImmutableArtifactBlobProvider(tmp_path / "content")
    document = _document(_spatial(provider))
    graph = compile_graph(document, registry, artifact_provider=provider).graph
    node = graph.nodes["physics"]
    state = graph.init_state(key=key)
    control = jnp.array([0.9])
    for _ in range(3):
        _, state = graph({"control": control}, state, key=key)
    saved = node.state_view(state).plant.skeleton
    assert saved.act[0] > 0.1
    assert saved.time == pytest.approx(0.015)

    wrench = AppliedWrench(
        force=jnp.array([0.2, 0.1, 1.1]), moment=jnp.array([0.4, -0.3, 0.1]),
        point=jnp.array([0.15, 0.03, 0.08]), frame="body",
    )
    state = node.apply_wrench(state, "tool", wrench)
    path = tmp_path / "state.eqx"
    node.save_state(path, state)
    transfer_source = jax.tree.map(lambda value: value, state)
    out, expected = graph({"control": control}, state, key=key)
    ref_model = node.descriptor.model
    native = mujoco.MjData(ref_model)
    native.qpos[:], native.qvel[:], native.act[:], native.time = (
        saved.qpos, saved.qvel, saved.act, float(saved.time),
    )
    native.ctrl[:] = control
    mujoco.mj_forward(ref_model, native)
    rotation = native.xmat[1].reshape(3, 3)
    force = rotation @ wrench.force
    application = native.xpos[1] + rotation @ wrench.point
    moment = rotation @ wrench.moment + np.cross(application - native.xipos[1], force)
    native.xfrc_applied[1] = np.concatenate((force, moment))
    mujoco.mj_step(ref_model, native)
    mujoco.mj_forward(ref_model, native)
    np.testing.assert_allclose(out["effector"].pos, native.site_xpos[0], atol=2e-8)
    np.testing.assert_allclose(out["effector"].orientation, native.site_xmat[0].reshape(3, 3), atol=2e-8)
    jacobian = np.zeros((3, ref_model.nv))
    mujoco.mj_jacSite(ref_model, native, jacobian, None, 0)
    np.testing.assert_allclose(out["effector"].vel, jacobian @ native.qvel, atol=2e-8)

    reauthored = GraphDocument.model_validate_json(
        GraphDocument(graph=graph_to_spec(graph, registry)).model_dump_json(),
    )
    resumed = compile_graph(reauthored, registry, artifact_provider=provider).graph
    resumed_node = resumed.nodes["physics"]
    reset = resumed.init_state(key=key)
    assert resumed_node.state_view(reset).plant.skeleton.time == 0
    restored = resumed_node.resume_state(path, reset)
    _, actual = resumed({"control": control}, restored, key=key)
    for left, right in zip(
        jax.tree.leaves(node.state_view(expected)),
        jax.tree.leaves(resumed_node.state_view(actual)),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-12)

    alternate = _spatial(provider, solver="euler")
    continuous = compile_graph(_document(alternate), registry, artifact_provider=provider).graph
    continuous_node = continuous.nodes["physics"]
    new_state = continuous.init_state(key=key)
    with pytest.raises(ValueError, match="checkpoint model, runtime or solver"):
        continuous_node.resume_state(path, new_state)
    transferred = continuous_node.transfer_state(node, transfer_source, new_state)
    destination = continuous_node.state_view(transferred).plant.skeleton
    np.testing.assert_array_equal(destination.act, saved.act)
    np.testing.assert_array_equal(destination.qpos, saved.qpos)
    assert destination.time == saved.time


def test_spatial_continuous_configuration_and_trial_loss_gradient(tmp_path):
    from feedbax.plot.constructors import (
        FigureRegistry, get_figure_constructor, register_default_figure_constructors,
    )

    key = jax.random.PRNGKey(0)
    provider = ImmutableArtifactBlobProvider(tmp_path / "content")
    interior = _spatial(provider, solver="euler")
    graph = compile_graph(_document(interior), ComponentRegistry(), artifact_provider=provider).graph
    node = graph.nodes["physics"]
    skeleton = node.plant.skeleton
    assert node.supports_reverse_mode
    qpos = jnp.array([0.1, 0.2, 0.3, 0.8, 0.6, 0.0, 0.0])
    qvel = jnp.array([0.2, -0.1, 0.3, 0.4, 0.2, -0.1])
    velocity = skeleton.configuration_derivative(qpos, qvel)
    plus, minus = np.asarray(qpos).copy(), np.asarray(qpos).copy()
    mujoco.mj_integratePos(node.descriptor.model, plus, np.asarray(qvel), 1e-5)
    mujoco.mj_integratePos(node.descriptor.model, minus, np.asarray(qvel), -1e-5)
    np.testing.assert_allclose(velocity, (plus - minus) / 2e-5, atol=1e-11)

    n_steps = 5
    target = jnp.array([0.3, 0.2, 0.8])

    def initial(model):
        owner = model.nodes["physics"]
        state = model.init_state(key=key)
        state = owner.replace_state_value(state, ("plant", "skeleton", "qpos"), qpos)
        return owner.replace_state_value(state, ("plant", "skeleton", "qvel"), qvel)

    start = initial(graph)

    def rollout(command):
        return graph(
            {"control": jnp.broadcast_to(command, (n_steps, 1))},
            start, key=key, n_steps=n_steps,
        )

    @eqx.filter_jit
    def objective(command):
        outputs, _ = rollout(command)
        return jnp.mean(jnp.sum((outputs["effector"].pos - target) ** 2, axis=-1))

    control = jnp.array(0.8)
    epsilon = 1e-4
    reverse = jax.grad(objective)(control)
    finite_difference = (objective(control + epsilon) - objective(control - epsilon)) / (2 * epsilon)
    assert abs(reverse) > 1e-7
    np.testing.assert_allclose(reverse, finite_difference, rtol=2e-4, atol=1e-9)

    outputs, state = rollout(control)
    positions = outputs["effector"].pos
    assert abs(positions[-1, 2] - positions[0, 2]) > 1e-4
    np.testing.assert_allclose(
        jnp.linalg.norm(node.state_view(state).plant.skeleton.qpos[3:]), 1, atol=1e-12,
    )
    registry = FigureRegistry()
    register_default_figure_constructors(registry)
    renderer = get_figure_constructor("feedbax.trajectory_3d", registry=registry, tier="trace")
    traces = renderer.callable({"trajectories": np.asarray(positions)}, renderer.params())
    assert traces[0].type == "scatter3d"
    np.testing.assert_allclose(traces[0].z, positions[:, 2])

    native_model = mujoco.MjModel.from_xml_string(
        provider.get_object(interior.mujoco_asset.asset).decode()
    )
    native_model.opt.timestep = 0.00025
    reference = mujoco.MjData(native_model)
    reference.qpos[:], reference.qvel[:], reference.ctrl[:] = qpos, qvel, [float(control)]
    for _ in range(100):
        mujoco.mj_step(native_model, reference)
    refined_graph = compile_graph(
        _document(_spatial(provider, solver="euler", dt=0.0025)),
        ComponentRegistry(), artifact_provider=provider,
    ).graph
    _, refined_state = refined_graph(
        {"control": jnp.full((10, 1), control)}, initial(refined_graph), key=key, n_steps=10,
    )
    coarse_qpos = node.state_view(state).plant.skeleton.qpos
    refined_qpos = refined_graph.nodes["physics"].state_view(refined_state).plant.skeleton.qpos
    coarse_error = np.linalg.norm(coarse_qpos - reference.qpos)
    fine_error = np.linalg.norm(refined_qpos - reference.qpos)
    assert fine_error < 0.7 * coarse_error

@pytest.fixture
def planar_fixed_muscle_paths():
    lengths = (0.40, 0.30, 0.25)
    points = {
        "elbow_flexor": [(0, 0.33, 0.055), (1, 0.07, 0.05)],
        "shoulder_flexor": [(-1, -0.04, 0.10), (0, 0.14, 0.06)],
        "whole_chain": [(-1, 0.02, -0.14), (1, 0.09, -0.07), (2, 0.19, -0.05)],
        "wrist_extensor": [(1, 0.22, -0.04), (2, 0.11, -0.045)],
    }
    nodes = {
        "world": ComponentSpec(type="feedbax.WorldFrame"),
        "anchor": ComponentSpec(type="feedbax.Anchor", params={"frame": "link0.proximal"}),
        "drive": ComponentSpec(type="feedbax.ActuationInput", params={
            "port_name": "control", "source_kind": "actuator_control", "size": 3,
        }),
        "sense": ComponentSpec(type="feedbax.SensorOutput", params={
            "port_name": "effector", "quantity": "effector",
        }),
        "marker": ComponentSpec(type="feedbax.PointMarker", params={
            "frame": "link2.distal", "marker_name": "effector",
        }),
    }
    for index, length in enumerate(lengths):
        nodes[f"link{index}"] = ComponentSpec(
            type="feedbax.PlanarLink", params={"length": length},
        )
        parent = "world.frame" if index == 0 else f"link{index - 1}.distal"
        nodes[f"joint{index}"] = ComponentSpec(type="feedbax.RevoluteJoint", params={
            "parent_frame": parent, "child_frame": f"link{index}.proximal",
        })
    for name, route in points.items():
        nodes[name] = ComponentSpec(type="feedbax.MusclePath", params={
            "path_points": [
                {"frame": "world.frame" if index == -1 else f"link{index}.proximal",
                 "offset": [x, y]}
                for index, x, y in route
            ],
        })
    interior = AcausalGraphSpec(
        physical_domain="planar_multibody", nodes=nodes,
        solver={"solver_type": "euler", "dt": 0.002},
    )
    return interior, lengths, points


def _saved_path_graph(interior, provider=None):
    registry = ComponentRegistry()
    original = compile_graph(_document(interior), registry, artifact_provider=provider).graph
    saved = GraphDocument(graph=graph_to_spec(original, registry)).model_dump_json()
    graph = compile_graph(
        GraphDocument.model_validate_json(saved), registry, artifact_provider=provider,
    ).graph
    assert graph.nodes["physics"].descriptor.identity == original.nodes["physics"].descriptor.identity
    return graph


def _assert_path_stage(
    skeleton, state, independent_lengths, integrate, expected_tension,
):
    stage = skeleton.continuous_stage(state)
    geometry = skeleton.muscle_path_kinematics(stage)
    qpos = np.asarray(state.qpos)
    qvel = np.asarray(state.qvel)
    epsilon = 1e-5
    directions = np.eye(skeleton.nv)
    expected_jacobian = np.stack([
        (independent_lengths(integrate(qpos, direction, epsilon))
         - independent_lengths(integrate(qpos, direction, -epsilon))) / (2 * epsilon)
        for direction in directions
    ], axis=1)
    np.testing.assert_allclose(geometry.length, independent_lengths(qpos), atol=1e-12)
    np.testing.assert_allclose(geometry.jacobian, expected_jacobian, atol=2e-10)
    np.testing.assert_allclose(geometry.velocity, expected_jacobian @ qvel, atol=2e-10)

    ctrl = skeleton.tendon_controls(expected_tension)
    under_load = skeleton.continuous_derivative(state, stage, ctrl)
    unloaded = skeleton.continuous_derivative(state, stage, jnp.zeros_like(ctrl))
    model = skeleton.descriptor.model
    native = mujoco.MjData(model)
    native.qpos[:], native.qvel[:], native.act[:] = state.qpos, state.qvel, state.act
    mujoco.mj_forward(model, native)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, native, mass)
    actual_force = mass @ (under_load.qvel - unloaded.qvel)
    np.testing.assert_allclose(actual_force, -expected_jacobian.T @ expected_tension, atol=2e-9)
    np.testing.assert_allclose(
        actual_force @ qvel, -expected_tension @ geometry.velocity, atol=2e-9,
    )

    def local_lengths(delta):
        if skeleton.nq == skeleton.nv:
            qpos = state.qpos + delta
        else:
            qpos = state.qpos + skeleton.configuration_derivative(state.qpos, delta)
        stage = skeleton.continuous_stage(replace(state, qpos=qpos))
        return skeleton.muscle_path_kinematics(stage).length

    reverse = eqx.filter_jit(jax.jacrev(local_lengths))(jnp.zeros(skeleton.nv))
    np.testing.assert_allclose(reverse, expected_jacobian, atol=2e-10)
    return geometry


def test_planar_fixed_guides_and_multijoint_shortening_force(planar_fixed_muscle_paths):
    interior, lengths, points = planar_fixed_muscle_paths
    graph = _saved_path_graph(interior)
    key = jax.random.PRNGKey(0)
    node = graph.nodes["physics"]
    skeleton = node.plant.skeleton
    initial = graph.init_state(key=key)
    state = replace(
        node.state_view(initial).plant.skeleton,
        qpos=jnp.array([0.32, -0.48, 0.36]), qvel=jnp.array([0.7, -0.3, 0.2]),
    )
    effector, _ = graph({"control": jnp.array([0.1, 0.0, -0.2])}, initial, key=key)
    assert effector["effector"].pos.shape == (2,)

    def reference(qpos):
        orientation = np.exp(1j * np.cumsum(qpos))
        origins = np.concatenate(([0], np.cumsum(np.asarray(lengths) * orientation)[:-1]))
        return np.array([
            np.abs(np.diff([
                complex(x, y) if i == -1 else origins[i] + orientation[i] * complex(x, y)
                for i, x, y in points[path.name]
            ])).sum()
            for path in node.descriptor.paths
        ])

    geometry = _assert_path_stage(
        skeleton, state, reference, lambda q, v, step: q + step * v,
        jnp.array([1.2, 2.0, 1.7, 0.4]),
    )
    whole_chain = next(i for i, path in enumerate(node.descriptor.paths) if path.name == "whole_chain")
    assert (np.abs(geometry.jacobian[whole_chain]) > 0.01).sum() == 3

    invalid = interior.model_copy(deep=True)
    invalid.nodes["elbow_flexor"].params["path_points"] = [
        {"frame": "link0.proximal"}, {"frame": "link0.distal", "offset": [-lengths[0], 0]},
    ]
    with pytest.raises(GraphCompilationError, match="zero-length fixed segment"):
        compile_graph(_document(invalid), ComponentRegistry())


@pytest.fixture
def spatial_fixed_muscle_path(tmp_path):
    provider = ImmutableArtifactBlobProvider(tmp_path / "content")
    interior = _spatial(provider, solver="euler")
    asset = provider.get_object(interior.mujoco_asset.asset).replace(
        b"<worldbody>", b'<worldbody><site name="origin" pos=".12 -.11 .44"/>',
    )
    interior.mujoco_asset.asset = provider.store_object(asset)
    payload = interior.model_dump()
    payload["mujoco_asset"]["sites"] = {
        "guide": {"body": "world", "position": [-0.18, 0.16, 0.10]},
        "insertion": {"body": "tool", "position": [0.11, 0.09, 0.24]},
    }
    payload["mujoco_asset"]["paths"] = {"flexor": {
        "sites": ["origin", "guide", "insertion"], "law": "feedbax.RigidTendonHillMuscle",
        "state_size": 1,
    }}
    return AcausalGraphSpec.model_validate(payload), provider


def test_spatial_overlay_transmits_through_quaternion_velocity_map(spatial_fixed_muscle_path):
    from scipy.spatial.transform import Rotation

    interior, provider = spatial_fixed_muscle_path
    graph = _saved_path_graph(interior, provider)
    key = jax.random.PRNGKey(0)
    node = graph.nodes["physics"]
    skeleton = node.plant.skeleton
    initial = graph.init_state(key=key)
    state = replace(
        node.state_view(initial).plant.skeleton,
        qpos=jnp.array([0.10, 0.22, 0.37, 0.8, 0.0, 0.6, 0.0]),
        qvel=jnp.array([0.2, -0.1, 0.3, 0.4, 0.2, -0.1]),
    )

    def reference(qpos):
        origin, guide = np.array([[.12, -.11, .44], [-.18, .16, .10]])
        rotation = Rotation.from_quat(qpos[[4, 5, 6, 3]])
        insertion = qpos[:3] + rotation.apply([.11, .09, .24])
        return np.array([np.linalg.norm(guide - origin) + np.linalg.norm(insertion - guide)])

    def integrate(qpos, qvel, step):
        displaced = qpos.copy()
        mujoco.mj_integratePos(node.descriptor.model, displaced, qvel, step)
        return displaced

    assert node.workspace_dimension == 3 and (skeleton.nq, skeleton.nv) == (7, 6)
    assert skeleton.input_size == 1
    _assert_path_stage(skeleton, state, reference, integrate, jnp.array([2.1]))


def test_exterior_sphere_and_cylinder_wraps_match_native_and_tangency():
    radius, start_radius, end_radius = 0.062, 0.07, 0.12
    interior = AcausalGraphSpec.model_validate({
        "physical_domain": "planar_multibody",
        "solver": {"solver_type": "euler", "dt": 0.002},
        "wrap_geometries": {
            kind: {"kind": kind, "frame": "world.frame", "radius": radius}
            for kind in ("cylinder", "sphere")
        },
        "nodes": {
            "world": {"type": "feedbax.WorldFrame"},
            "anchor": {"type": "feedbax.Anchor", "params": {"frame": "link.proximal"}},
            "link": {"type": "feedbax.PlanarLink", "params": {"length": 0.30}},
            "joint": {"type": "feedbax.RevoluteJoint", "params": {
                "parent_frame": "world.frame", "child_frame": "link.proximal",
            }},
            "drive": {"type": "feedbax.ActuationInput", "params": {
                "port_name": "control", "source_kind": "actuator_control", "size": 1,
            }},
            "sense": {"type": "feedbax.SensorOutput", "params": {
                "port_name": "effector", "quantity": "effector",
            }},
            "tip": {"type": "feedbax.PointMarker", "params": {
                "frame": "link.distal", "marker_name": "effector",
            }},
            "side": {"type": "feedbax.PointMarker", "params": {
                "frame": "world.frame", "offset": [0.055, 0.08],
            }},
            **{
                kind: {"type": "feedbax.MusclePath", "params": {
                    "path_points": [
                        {"frame": "world.frame", "offset": [start_radius, 0]},
                        {"frame": "link.proximal", "offset": [end_radius, 0]},
                    ],
                    "wraps": [{"after_site": 0, "geom": kind, "side_site": "side"}],
                }}
                for kind in ("cylinder", "sphere")
            },
        },
    })
    graph = _saved_path_graph(interior)
    node = graph.nodes["physics"]
    assert node.supports_reverse_mode
    skeleton = node.plant.skeleton
    key = jax.random.PRNGKey(0)
    state = replace(
        node.state_view(graph.init_state(key=key)).plant.skeleton, qvel=jnp.array([0.35]),
    )

    @eqx.filter_jit
    def evaluate(q):
        current = replace(state, qpos=jnp.reshape(q, (1,)))
        stage = skeleton.continuous_stage(current)
        geometry = skeleton.muscle_path_kinematics(stage)
        tension = skeleton.tendon_controls(jnp.array([0.9, 1.3]))
        loaded = skeleton.continuous_derivative(current, stage, tension).qvel
        idle = skeleton.continuous_derivative(current, stage, jnp.zeros_like(tension)).qvel
        return geometry, skeleton.muscle_path_wrap_counts(stage), loaded - idle

    def reference_length(q):
        return evaluate(q)[0].length

    derivative = eqx.filter_jit(jax.jacrev(reference_length))
    native_model = node.descriptor.model
    native = mujoco.MjData(native_model)
    native.qvel[:] = state.qvel
    for path, tension in zip(node.descriptor.paths, [0.9, 1.3], strict=True):
        native.ctrl[path.actuator_id] = tension
    grazing = np.arccos(radius / start_radius) + np.arccos(radius / end_radius)
    tangent_lengths = np.sqrt(start_radius**2 - radius**2) + np.sqrt(end_radius**2 - radius**2)

    for q in [grazing - 1e-4, grazing + 1e-4, 1.65, 2.10, 2.55]:
        geometry, count, delta_accel = evaluate(jnp.asarray(q))
        native.qpos[:] = q
        mujoco.mj_forward(native_model, native)
        mass = np.empty((1, 1))
        mujoco.mj_fullM(native_model, native, mass)
        engaged = q > grazing
        expected = (
            tangent_lengths + radius * (q - grazing) if engaged
            else abs(start_radius - end_radius * np.exp(1j * q))
        )
        np.testing.assert_array_equal(count, [int(engaged)] * 2)
        np.testing.assert_array_equal(native.ten_wrapnum, 2 + 2 * np.asarray(count))
        np.testing.assert_allclose(geometry.length, expected, atol=2e-12)
        np.testing.assert_allclose(geometry.length, native.ten_length, atol=2e-12)
        np.testing.assert_allclose(geometry.velocity, native.ten_velocity, atol=2e-12)
        np.testing.assert_allclose(mass @ delta_accel, native.qfrc_actuator, atol=2e-10)
        if q >= 1.65:
            np.testing.assert_allclose(derivative(jnp.asarray(q)), radius, atol=2e-10)
            np.testing.assert_allclose(geometry.jacobian, radius, atol=2e-10)

    external = interior.model_copy(deep=True)
    external.nodes["drive"].params.update(source_kind="muscle_excitation_vector", size=2)
    gains = np.array([2.0, 3.0])
    activation_time = 0.015
    for name, gain in zip(("cylinder", "sphere"), gains, strict=True):
        external.nodes[name].params.update(
            law="feedbax.ReluMuscle",
            law_parameters={"max_isometric_force": float(gain), "tau_activation": activation_time},
        )
    coupled = compile_graph(_document(external), ComponentRegistry()).graph
    physics = coupled.nodes["physics"]

    def end_velocity(command):
        initial = physics.replace_state_value(
            coupled.init_state(key=key), ("plant", "skeleton", "qpos"), jnp.array([2.10]),
        )
        _, first = coupled({"control": command}, initial, key=key)
        _, last = coupled({"control": command}, first, key=key)
        return physics.state_view(last).plant.skeleton.qvel[0]

    native.qpos[:] = 2.10
    mujoco.mj_forward(native_model, native)
    mass = np.empty((1, 1))
    mujoco.mj_fullM(native_model, native, mass)
    sensitivity = eqx.filter_jit(jax.grad(end_velocity))(jnp.array([0.2, 0.3]))
    expected_sensitivity = -radius * gains * interior.solver.dt**2 / (mass[0, 0] * activation_time)
    np.testing.assert_allclose(sensitivity, expected_sensitivity, atol=2e-10)

    inside = interior.model_copy(deep=True)
    inside.nodes["side"].params["offset"] = [0.01, 0]
    forward_graph = compile_graph(_document(inside), ComponentRegistry()).graph
    assert not forward_graph.nodes["physics"].supports_reverse_mode

    def forward(command):
        outputs, _ = forward_graph(
            {"control": command}, forward_graph.init_state(key=key), key=key,
        )
        return outputs["effector"].pos[1]

    command = jnp.array([0.2])
    assert jnp.isfinite(forward(command))
    with pytest.raises(RuntimeError, match="fixed exterior side site"):
        jax.grad(forward)(command)
