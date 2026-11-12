"""Production graph acceptance for mixed rigid and compliant integration."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.serialization import graph_to_spec
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.objectives.loss import TargetSpec, TargetStateLoss
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.runtime.mapping import WhereDict
from feedbax.tasks.task import TaskTrialSpec, prepare_trial


pytestmark = [pytest.mark.optional_mjx, pytest.mark.usefixtures("enable_jax_x64")]

ACTIVATION = 0.25
OPTIMAL = 0.10
SLACK = 0.17
PENNATION = 0.30
MAX_FORCE = 25.0
ORIGIN = np.array([0.15, -0.18, 0.14])
INSERTION = np.array([0.17, 0.04, -0.04])


def _document(provider, dt):
    asset = provider.store_object(f"""
      <mujoco>
        <option gravity="0 0 0" timestep="{dt}"/>
        <worldbody>
          <site name="origin" pos=".15 -.18 .14"/>
          <site name="antagonist" pos="-.13 .19 -.02"/>
          <body name="arm">
            <joint name="pivot" type="ball" damping=".05"/>
            <inertial pos="0 0 0" mass="1" diaginertia=".11 .14 .19"/>
            <site name="insertion" pos=".17 .04 -.04"/>
          </body>
        </worldbody>
      </mujoco>
    """.encode())
    interior = AcausalGraphSpec.model_validate({
        "physical_domain": "spatial_multibody",
        "mujoco_asset": {
            "asset": asset.model_dump(),
            "effector_site": "insertion",
            "paths": {
                "compliant": {
                    "sites": ["origin", "insertion"],
                    "law": "feedbax.CompliantTendonHillMuscle",
                    "state_size": 2,
                    "parameters": {
                        "initial_activation": ACTIVATION,
                        "max_isometric_force": MAX_FORCE,
                        "optimal_muscle_length": OPTIMAL,
                        "tendon_slack_length": SLACK,
                        "pennation_angle": PENNATION,
                    },
                },
                "rigid": {
                    "sites": ["antagonist", "insertion"],
                    "law": "feedbax.RigidTendonHillMuscle",
                    "state_size": 1,
                    "parameters": {
                        "initial_activation": 0.17, "max_isometric_force": 15.0,
                        "optimal_muscle_length": 0.18, "tendon_slack_length": 0.14,
                    },
                },
            },
        },
        "solver": {"solver_type": "euler", "dt": 0.002},
        "nodes": {
            "drive": {"type": "feedbax.ActuationInput", "params": {
                "port_name": "excitation", "source_kind": "muscle_excitation_vector",
            }},
            "tip": {"type": "feedbax.SensorOutput", "params": {
                "port_name": "effector", "quantity": "effector",
            }},
        },
    })
    return GraphDocument(graph=GraphSpec(
        nodes={"physics": ComponentSpec(
            type="feedbax.AcausalSystem", input_ports=["excitation"], output_ports=["effector"],
        )},
        subgraphs={"physics": interior},
        input_ports=["excitation"], input_bindings={"excitation": ("physics", "excitation")},
        output_ports=["effector"], output_bindings={"effector": ("physics", "effector")},
    ))


def test_studio_opens_spatial_project_with_imported_asset(tmp_path, monkeypatch):
    import pickle
    import subprocess
    import sys
    from types import SimpleNamespace

    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from feedbax.contracts.artifact_custody import ContentGcRequest
    from feedbax.training.checkpoint_custody import (
        _semantic_structural_abi_sha256,
        structural_abi_fingerprint,
    )
    from feedbax.web import config
    from feedbax.web.api import graphs, projects
    from feedbax.web.model_assets import model_artifact_provider
    from feedbax.web.services.project_folder import ProjectFolderService

    pytest.importorskip("mujoco.mjx")
    fixture = ImmutableArtifactBlobProvider(tmp_path / "fixture")
    graph = _document(fixture, 0.001).graph
    asset = graph.subgraphs["physics"].mujoco_asset.asset
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    app = FastAPI()
    app.state.bootstrap_state = SimpleNamespace(bundle=SimpleNamespace(components=ComponentRegistry()))
    app.include_router(graphs.router, prefix="/api/graphs")
    app.include_router(projects.router)
    app.state.project_folders = ProjectFolderService(tmp_path / "projects")

    with TestClient(app) as client:
        external = client.post("/api/graphs/assets/mujoco", content=b'<mujoco><include file="other.xml"/></mujoco>')
        assert external.status_code == 422
        imported = client.post("/api/graphs/assets/mujoco", content=fixture.get_object(asset))
        assert imported.status_code == 200
        assert imported.json() == asset.model_dump(mode="json")
        created = client.post("/api/projects").json()["snapshot"]
        project_id = created["project_id"]
        path = f"/api/projects/{project_id}"
        saved = client.post(path + "/files", json={
            "revision": created["revision"],
            "file": {"kind": "document", "id": "model", "document": {
                "schema": graph.schema_id, "payload": graph.model_dump(mode="json", exclude_none=True),
            }},
        })
        assert saved.status_code == 201, saved.text
        content = model_artifact_provider(graph)
        assert content is not None
        dry_run = content.collect_garbage()
        assert any(
            item.sha256 == asset.sha256 and item.decision == "live"
            for item in dry_run.decisions
        )
        content.collect_garbage(ContentGcRequest(
            destructive=True, expected_generation=dry_run.generation,
        ))
        project = client.get(path + "/open")
        assert project.status_code == 200, project.text
        payload = project.json()["snapshot"]["project"]["documents"]["model"]["payload"]
        binding = payload["subgraphs"]["physics"]["mujoco_asset"]
        assert binding["asset"] == imported.json()

    reopened = GraphSpec.model_validate(payload)
    compiled = compile_graph(
        GraphDocument(graph=reopened), app.state.bootstrap_state.bundle.components,
        artifact_provider=model_artifact_provider(reopened),
    ).graph
    restored = pickle.loads(pickle.dumps(compiled))
    assert jax.tree.structure(restored) == jax.tree.structure(compiled)
    assert structural_abi_fingerprint(restored) == structural_abi_fingerprint(compiled)

    model_file = tmp_path / "model.pkl"
    model_file.write_bytes(pickle.dumps(compiled))
    fingerprint_file = tmp_path / "fingerprint.txt"
    subprocess.run(
        [sys.executable, "-B", "-c", """
import pickle
import sys
from pathlib import Path

import jax
from feedbax.training.checkpoint_custody import (
    _semantic_structural_abi_sha256, structural_abi_fingerprint,
)

jax.config.update("jax_enable_x64", True)
model = pickle.loads(Path(sys.argv[1]).read_bytes())
fingerprint = _semantic_structural_abi_sha256(structural_abi_fingerprint(model), value=model)
Path(sys.argv[2]).write_text(fingerprint)
""", str(model_file), str(fingerprint_file)],
        check=True,
        timeout=25,
    )
    assert fingerprint_file.read_text() == _semantic_structural_abi_sha256(
        structural_abi_fingerprint(compiled), value=compiled,
    )

    mjx_model = compiled.nodes["physics"].plant.skeleton.mjx_model
    different_limits = np.logical_not(mjx_model.jnt_limited).astype(mjx_model.jnt_limited.dtype)
    changed_model = mjx_model.replace(jnt_limited=different_limits)
    assert structural_abi_fingerprint(changed_model) != structural_abi_fingerprint(mjx_model)


def _reference_equilibrium(angle):
    # Independent scalar force balance and world-space ball-joint rotation.
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = INSERTION
    length = np.linalg.norm(ORIGIN - np.array([c * x - s * y, s * x + c * y, z]))
    h = OPTIMAL * math.sin(PENNATION)

    def residual(fiber):
        projection = math.sqrt(fiber**2 - h**2)
        normalized = fiber / OPTIMAL
        active = ACTIVATION * math.exp(-((normalized - 1) / 0.56)**2)
        passive = math.expm1(4 * max(normalized - 1, 0.0) / 0.7) / math.expm1(4)
        strain = max((length - projection) / SLACK - 1.0, 0.0)
        tendon = math.expm1(35 * strain / 0.033) / math.expm1(35)
        return tendon - (active + passive) * projection / fiber

    return brentq(residual, math.hypot(h, length - 1.07 * SLACK), math.hypot(h, length))


def test_spatial_mixed_muscles_reset_refine_differentiate_and_resume(tmp_path):
    pytest.importorskip("mujoco.mjx")
    key = jax.random.PRNGKey(1)
    provider = ImmutableArtifactBlobProvider(tmp_path / "content")
    registry = ComponentRegistry()
    n_steps = 8
    angle = 0.18
    qpos = jnp.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
    def where(state):
        return state.physics.effector.pos

    loss = TargetStateLoss("spatial target", where=where)
    trial = TaskTrialSpec(
        inits=WhereDict({"physics.plant.skeleton.qpos": qpos}),
        targets=WhereDict({"physics.effector.pos": TargetSpec(
            value=jnp.array([[0.16, 0.05, -0.02]]),
        )}),
        inputs=jnp.zeros((n_steps, 2)),
    )

    def build(dt, *, roundtrip=False):
        graph = compile_graph(_document(provider, dt), registry, artifact_provider=provider).graph
        if roundtrip:
            saved = GraphDocument(graph=graph_to_spec(graph, registry)).model_dump_json()
            graph = compile_graph(
                GraphDocument.model_validate_json(saved), registry, artifact_provider=provider,
            ).graph
        node = graph.nodes["physics"]
        initial = prepare_trial(graph, trial).init_state
        @jax.jit
        def rollout(state, command):
            return graph(
                {"excitation": jnp.broadcast_to(command, (n_steps, 2))}, state,
                key=key, n_steps=n_steps, return_state_history=True,
            )

        return graph, node, initial, rollout

    middle, node, initial, rollout = build(0.001, roundtrip=True)
    initial_muscles = node.state_view(initial).plant.muscles
    np.testing.assert_allclose(
        initial_muscles[0], [ACTIVATION, _reference_equilibrium(angle)], atol=1e-12,
    )
    assert initial_muscles[1].shape == (1,)
    default = node.state_view(middle.init_state(key=key)).plant.muscles[0][1]
    assert abs(float(default - initial_muscles[0][1])) > 1e-3

    @jax.jit
    def objective(command):
        _, _, history = rollout(initial, command)
        batched = jax.tree.map(lambda value: value[None], history)
        return loss.term(batched, trial, middle)[0]

    command = jnp.array([0.34, 0.29])
    direction = jnp.array([0.8, -0.6])
    reverse = jax.grad(objective)(command) @ direction
    epsilon = 1e-5
    finite_difference = (
        objective(command + epsilon * direction) - objective(command - epsilon * direction)
    ) / (2 * epsilon)
    assert abs(reverse) > 1e-8
    np.testing.assert_allclose(reverse, finite_difference, rtol=3e-5, atol=1e-10)

    endpoints = []
    for dt in (0.002, 0.001, 0.0005):
        _, physics, start, run = (middle, node, initial, rollout) if dt == 0.001 else build(dt)
        _, state, _ = run(start, command)
        plant = physics.state_view(state).plant
        np.testing.assert_allclose(jnp.linalg.norm(plant.skeleton.qpos), 1.0, atol=1e-12)
        endpoints.append(jnp.concatenate((plant.skeleton.qvel, plant.muscles[0][1:])))
    coarse, medium, fine = endpoints
    assert jnp.linalg.norm(medium - fine) < 0.7 * jnp.linalg.norm(coarse - fine)

    _, checkpoint, _ = rollout(initial, command)
    assert abs(node.state_view(checkpoint).plant.muscles[0][1] - initial_muscles[0][1]) > 1e-6
    file = tmp_path / "mixed.eqx"
    node.save_state(file, checkpoint)
    restored = node.resume_state(file, middle.init_state(key=key))
    for actual, expected in zip(
        jax.tree.leaves(node.state_view(restored)),
        jax.tree.leaves(node.state_view(checkpoint)), strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    _, continued, _ = rollout(checkpoint, command)
    _, resumed, _ = rollout(restored, command)
    for actual, expected in zip(
        jax.tree.leaves(node.state_view(resumed)),
        jax.tree.leaves(node.state_view(continued)), strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
