"""Qualification of the compiled MuJoCo boundary through the public graph route."""

from __future__ import annotations

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
from feedbax.objectives.loss import TargetSpec, TargetStateLoss
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.runtime.mapping import WhereDict
from feedbax.tasks.task import TaskTrialSpec, prepare_trial


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

    position = lambda state: state.physics.effector.pos
    loss = TargetStateLoss(label="spatial reach", where=position)
    trial = TaskTrialSpec(
        inits=WhereDict({
            "physics.plant.skeleton.qpos": qpos,
            "physics.plant.skeleton.qvel": qvel,
        }),
        targets=WhereDict({"physics.effector.pos": TargetSpec(value=jnp.array([[.3, .2, .8]]))}),
        inputs=jnp.zeros((5, 1)),
    )
    prepared = prepare_trial(graph, trial)

    def rollout(command):
        return graph(
            {"control": jnp.broadcast_to(command, (prepared.n_steps, 1))},
            prepared.init_state, key=key, n_steps=prepared.n_steps, return_state_history=True,
        )

    @eqx.filter_jit
    def objective(command):
        _, _, history = rollout(command)
        batched = jax.tree.map(lambda value: value[None], history)
        return loss.term(batched, trial, graph)[0]

    control = jnp.array(0.8)
    epsilon = 1e-4
    reverse = jax.grad(objective)(control)
    finite_difference = (objective(control + epsilon) - objective(control - epsilon)) / (2 * epsilon)
    assert abs(reverse) > 1e-7
    np.testing.assert_allclose(reverse, finite_difference, rtol=2e-4, atol=1e-9)

    outputs, state, _ = rollout(control)
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
    fine_trial = eqx.tree_at(lambda spec: spec.inputs, trial, jnp.ones((10, 1)) * control)
    fine = prepare_trial(refined_graph, fine_trial)
    _, refined_state = refined_graph(fine.inputs, fine.init_state, key=key, n_steps=fine.n_steps)
    coarse_qpos = node.state_view(state).plant.skeleton.qpos
    refined_qpos = refined_graph.nodes["physics"].state_view(refined_state).plant.skeleton.qpos
    coarse_error = np.linalg.norm(coarse_qpos - reference.qpos)
    fine_error = np.linalg.norm(refined_qpos - reference.qpos)
    assert fine_error < 0.7 * coarse_error
