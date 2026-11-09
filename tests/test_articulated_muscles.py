"""Production graph acceptance for heterogeneous stage-local muscle laws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.serialization import graph_to_spec
from feedbax.component_registry import ComponentRegistry
from feedbax.component_registry.discovery import ComponentPack
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.component import PortType
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.portable import Ports
from feedbax.mechanics.muscle_law import MuscleStage
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.runtime.graph import Component


pytestmark = [pytest.mark.optional_mjx, pytest.mark.usefixtures("enable_jax_x64")]


class InstantTension(Component):
    ports = Ports(
        inputs={"excitation": PortType(dtype="scalar")},
        outputs={"force": PortType(dtype="scalar")},
    )

    gain: float

    def __init__(self, gain: float):
        self.gain = gain

    @property
    def state_size(self):
        return 0

    def init_state(self, length, velocity):
        return jnp.empty((0,), dtype=length.dtype)

    def evaluate(self, state, excitation, length, velocity):
        return MuscleStage(state, self.gain * excitation)

    def __call__(self, inputs, state, *, key):
        return {"force": self.gain * inputs["excitation"]}, state


def _document(asset):
    sites = ["origin", "insertion"]
    interior = AcausalGraphSpec.model_validate({
        "physical_domain": "spatial_multibody",
        "mujoco_asset": {
            "asset": asset.model_dump(),
            "effector_site": "insertion",
            "paths": {
                "hill": {
                    "sites": sites, "law": "feedbax.RigidTendonHillMuscle", "state_size": 1,
                    "parameters": {
                        "max_isometric_force": 30.0, "optimal_muscle_length": 0.3,
                        "tendon_slack_length": 0.2, "initial_activation": 0.3,
                    },
                },
                "instant": {
                    "sites": sites, "law": "test.InstantTension", "law_revision": 1,
                    "state_size": 0, "parameters": {"gain": 12.0},
                },
                "linear": {
                    "sites": sites, "law": "feedbax.ReluMuscle", "state_size": 1,
                    "parameters": {"max_isometric_force": 20.0, "initial_activation": 0.2},
                },
            },
        },
        "solver": {"solver_type": "euler", "dt": 0.001},
        "nodes": {
            "excitation": {"type": "feedbax.ActuationInput", "params": {
                "port_name": "excitation", "source_kind": "muscle_excitation_vector",
            }},
            "state": {"type": "feedbax.SensorOutput", "params": {
                "port_name": "state", "quantity": "state",
            }},
        },
    })
    return GraphDocument(graph=GraphSpec(
        nodes={"physics": ComponentSpec(
            type="feedbax.AcausalSystem", input_ports=["excitation"], output_ports=["state"],
        )},
        subgraphs={"physics": interior},
        input_ports=["excitation"], input_bindings={"excitation": ("physics", "excitation")},
        output_ports=["state"], output_bindings={"state": ("physics", "state")},
    ))


def test_authored_laws_roundtrip_and_couple_at_the_same_stage(tmp_path):
    pytest.importorskip("mujoco.mjx")
    provider = ImmutableArtifactBlobProvider(tmp_path / "content")
    asset = provider.store_object(b"""
      <mujoco>
        <option gravity="0 0 0" timestep=".001"/>
        <worldbody>
          <site name="origin" pos="0 .4 0"/>
          <body name="limb">
            <joint axis="0 0 1" type="hinge"/>
            <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .2"/>
            <site name="insertion" pos=".3 0 0"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="test", components=(InstantTension,)))
    document = _document(asset)
    key = jax.random.PRNGKey(0)
    excitation = jnp.array([0.3, 0.5, 0.5])

    def compile_doc(value):
        return compile_graph(value, registry, artifact_provider=provider).graph

    def step(graph):
        return graph({"excitation": excitation}, graph.init_state(key=key), key=key)

    original = compile_doc(document)
    serialized = GraphDocument(graph=graph_to_spec(original, registry)).model_dump_json()
    reloaded_document = GraphDocument.model_validate_json(serialized)
    reloaded = compile_doc(reloaded_document)
    before, _ = step(original)
    after, continuation = step(reloaded)
    before_plant, after_plant = before["state"].plant, after["state"].plant
    np.testing.assert_array_equal(before_plant.skeleton.qvel, after_plant.skeleton.qvel)

    # At rest the three paths are 0.5 m long with dL/dq = -0.24 m/rad.
    # Hill tension is isometric, and Euler uses the initial ReLU activation.
    tension = 30.0 * 0.3 + 12.0 * 0.5 + 20.0 * 0.2
    np.testing.assert_allclose(after_plant.skeleton.qvel, [0.001 * 0.24 * tension / 0.2])
    np.testing.assert_allclose(after_plant.muscles[0], [0.3])
    assert after_plant.muscles[1].shape == (0,)
    np.testing.assert_allclose(after_plant.muscles[2], [0.2 + 0.001 * (0.5 - 0.2) / 0.015])

    physics = reloaded.nodes["physics"]
    checkpoint = tmp_path / "state.eqx"
    physics.save_state(checkpoint, continuation)
    resumed = physics.resume_state(checkpoint, reloaded.init_state(key=key))
    continued, _ = reloaded({"excitation": excitation}, continuation, key=key)
    restored, _ = reloaded({"excitation": excitation}, resumed, key=key)
    np.testing.assert_array_equal(
        restored["state"].plant.skeleton.qvel, continued["state"].plant.skeleton.qvel,
    )

    reloaded_document.graph.subgraphs["physics"].mujoco_asset.paths["instant"].parameters["gain"] = 22.0
    edited, _ = step(compile_doc(reloaded_document))
    np.testing.assert_allclose(
        edited["state"].plant.skeleton.qvel - after_plant.skeleton.qvel,
        [0.001 * 0.24 * (22.0 - 12.0) * 0.5 / 0.2],
    )
