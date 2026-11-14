"""Exercise normal checkpoint publication with constructor and asset-backed owners."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import rmtree
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.model_assets import model_asset_refs
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.checkpoints import CheckpointLatestPointer
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.training import GraphTopologySourceSpec
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.dataset import Dataset, Variable
from feedbax.mechanics.muscles import RigidTendonHillMuscleThelen
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.persistence.dataset import open_dataset, publish_dataset
from feedbax.runtime.raw_state import bind_raw_state
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training.checkpoint_custody import (
    authenticate_published_checkpoint_custody,
    open_published_model_slot,
    write_checkpoint_transaction,
)
from feedbax.training.model_checkpoint import (
    ModelCheckpointError,
    SemanticModelWriter,
    effective_model_graph,
)
from tests.support.managed_train_fixture import managed_train_spec


def _publish(root, graph, model, assets=None):
    spec, _ = managed_train_spec(batches=2)
    spec = spec.model_copy(update={
        "graph": GraphTopologySourceSpec(inline=graph.model_dump(mode="json")),
    })
    phase = spec.worker_execution.effective_phase.phase_program
    writer = SemanticModelWriter.for_graph(graph, assets)
    assert writer is not None
    return write_checkpoint_transaction(
        root,
        run_spec=spec,
        phase_program=phase,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="selected-model", phase="train_batch", program_step=1,
            completed_barrier="after_train_batch",
        ),
        slots={
            "model": model, "optimizer": (), "prng": jax.random.PRNGKey(0),
            "batch_counter": jnp.array(1, dtype=jnp.int32),
        },
        completed_training_batches=1,
        semantic_model=writer,
    )


def _thelen_graph(params):
    ports = RigidTendonHillMuscleThelen.ports
    inputs, outputs = list(ports.inputs), list(ports.outputs)
    return GraphSpec(
        nodes={"muscle": ComponentSpec(
            type="feedbax.RigidTendonHillMuscleThelen",
            params=params,
            input_ports=inputs,
            output_ports=outputs,
        )},
        input_ports=inputs,
        output_ports=outputs,
        input_bindings={name: ("muscle", name) for name in inputs},
        output_bindings={name: ("muscle", name) for name in outputs},
    )


def test_thelen_model_slot_rebuilds_owner_from_restored_source_values(tmp_path, monkeypatch):
    structural = {"n_muscles": 2, "initial_activation": 0.25}
    final_values = {
        "max_isometric_force": 710.0,
        "optimal_muscle_length": 0.24,
        "tendon_slack_length": 0.12,
        "vmax_factor": 7.5,
    }
    graph = _thelen_graph(structural)
    source = compile_graph(
        GraphDocument(graph=_thelen_graph(structural | final_values)), ComponentRegistry(),
    ).graph
    template = compile_graph(GraphDocument(graph=graph), ComponentRegistry()).graph
    written = _publish(tmp_path / "thelen", graph, source)
    selected = open_published_model_slot(written.root, written.latest_pointer)
    assert selected is not None
    restored = selected.restore(template)
    muscle = restored.nodes["muscle"]
    fresh = RigidTendonHillMuscleThelen(**structural, **final_values)
    assert template.nodes["muscle"].vmax != muscle.vmax
    for field in ("vmax", "pe_den", "ce_0", "ce_1", "ce_2", "ce_4"):
        assert getattr(muscle, field) == getattr(fresh, field)

    inputs = {
        "excitation": jnp.array([0.7, 0.4]),
        "musculotendon_length": jnp.array([0.36, 0.36]),
        "musculotendon_velocity": jnp.array([-0.31, 0.42]),
    }
    key = jax.random.PRNGKey(0)
    expected, _ = source(inputs, source.init_state(key=key), key=key)
    actual, _ = restored(inputs, restored.init_state(key=key), key=key)
    for name in ("force", "activation", "fiber_length", "fiber_velocity"):
        np.testing.assert_array_equal(expected[name], actual[name])

    capture = bind_raw_state(restored, selected.graph)
    step_inputs = {name: jnp.repeat(values[None], 4, axis=0) for name, values in inputs.items()}
    actual, _, _, history = restored.rollout_with_trace_and_history(
        step_inputs, restored.init_state(key=key), key=key, n_steps=4, trace=(),
        capture_state=capture.capture,
    )
    expected, _ = source(step_inputs, source.init_state(key=key), key=key, n_steps=4)
    assert not np.array_equal(actual["force"], template(
        step_inputs, template.init_state(key=key), key=key, n_steps=4,
    )[0]["force"])
    np.testing.assert_allclose(actual["force"], expected["force"], rtol=2e-7)
    variables, coordinates = capture.dataset_fields(
        [history], axes=("checkpoint", "condition", "trial"),
    )
    coordinates["state_time"] = Variable(np.arange(5) * muscle.dt, ("state_time",), units="s")
    coordinates["trial"] = Variable(np.array(["thelen"]), ("trial",))
    dataset = Dataset.from_arrays(variables, coords=coordinates)
    states = ImmutableArtifactBlobProvider(tmp_path / "states")
    ref = publish_dataset(states, dataset)
    reads = []
    read_tensor = ArrayContainerReader.read_tensor

    def selected_tensor(reader, name):
        reads.append(name)
        return read_tensor(reader, name)

    with monkeypatch.context() as patch:
        patch.setattr(ArrayContainerReader, "read_tensor", selected_tensor)
        with open_dataset(
            states.object_path(ref.object_ref), expected_descriptor_sha256=ref.descriptor_sha256,
        ) as opened:
            name = "/v1/model/state/muscle/activation"
            assert set(opened.descriptor.variables) == {name}
            field = opened.descriptor.variables[name]
            assert field.dims == ("checkpoint", "condition", "trial", "state_time", "muscle:muscle")
            assert field.units == "1"
            samples = opened.select(name, labels={"trial": "thelen"})[0, 0]
        assert reads == [name]
    np.testing.assert_array_equal(samples[0], [0.25, 0.25])
    np.testing.assert_allclose(samples[1:], expected["activation"], rtol=2e-7)

    evidence = authenticate_published_checkpoint_custody(written.root)
    assert evidence.manifest.transaction_id == written.manifest.transaction_id



def _spatial_step(model):
    key = jax.random.PRNGKey(11)
    outputs, _ = model({"control": jnp.array([0.9])}, model.init_state(key=key), key=key)
    return [np.asarray(leaf).tolist() for leaf in jax.tree.leaves(outputs["effector"])]


@pytest.mark.optional_mjx
@pytest.mark.usefixtures("enable_jax_x64")
def test_asset_backed_model_slot_round_trips_from_checkpoint_closure(tmp_path):
    from tests.test_stateful_mujoco_boundary import _document, _spatial

    imported = ImmutableArtifactBlobProvider(tmp_path / "imported")
    graph = _document(_spatial(imported)).graph
    model = compile_graph(
        GraphDocument(graph=graph), ComponentRegistry(), artifact_provider=imported,
    ).graph
    written = _publish(tmp_path / "checkpoint", graph, model, imported)
    pointer = tmp_path / "selected.json"
    pointer.write_text(written.latest_pointer.model_dump_json())
    expected = _spatial_step(model)
    rmtree(imported.root)
    selected = open_published_model_slot(written.root, written.latest_pointer)
    assert selected is not None and selected.graph == effective_model_graph(graph, model)
    output = tmp_path / "restored.json"
    subprocess.run(
        [
            sys.executable, "-m", "tests.test_semantic_model_checkpoint",
            str(written.root), str(pointer), str(output),
        ],
        capture_output=True, text=True, timeout=35, check=True,
    )
    for actual, reference in zip(json.loads(output.read_bytes()), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)

    authored_path, asset = next(model_asset_refs(graph))
    physical = selected.provider.object_path(asset)
    exact = physical.read_bytes()
    physical.unlink()
    with pytest.raises(ModelCheckpointError) as missing:
        open_published_model_slot(written.root, written.latest_pointer)
    assert str(missing.value).startswith(authored_path)
    physical.write_bytes(exact[:-1] + b"!")
    with pytest.raises(ModelCheckpointError) as changed:
        open_published_model_slot(written.root, written.latest_pointer)
    assert str(changed.value).startswith(authored_path)
    assert "sha256" in str(changed.value).lower()
    with pytest.raises(ValueError, match="semantic model member"):
        authenticate_published_checkpoint_custody(written.root)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    root, pointer_path, output = map(Path, sys.argv[1:])
    pointer = CheckpointLatestPointer.model_validate_json(pointer_path.read_bytes())
    selected = open_published_model_slot(root, pointer)
    assert selected is not None
    output.write_text(json.dumps(_spatial_step(selected.restore())))
