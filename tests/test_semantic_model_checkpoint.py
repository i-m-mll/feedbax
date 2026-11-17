"""Exercise normal checkpoint publication with constructor and asset-backed owners."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import rmtree
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.builtins.training import StandardTrainingSettings, _init_supervised
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.model_assets import model_asset_refs
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.method_checkpoint import (
    CheckpointLatestPointer,
    MethodCheckpointClock,
    MethodCheckpointIdentity,
    method_checkpoint_extra_path,
)
from feedbax.contracts.training import OptimizerSpec
from feedbax.dataset import Dataset, Variable
from feedbax.builtins.mechanics.muscles.thelen_muscle import RigidTendonHillMuscleThelen
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.persistence.dataset import open_dataset, publish_dataset
from feedbax.runtime.raw_state import bind_raw_state
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.method_checkpoint import (
    bind_method_checkpoint_content,
    materialize_checkpoint_content,
    verify_method_checkpoint_content,
)
from feedbax.training.checkpoint_errors import CheckpointCustodyError
from feedbax.training.method_checkpoint import (
    read_method_checkpoint,
    write_method_checkpoint,
)
from feedbax.training.method_runtime import stack_replicas
from feedbax.training.methods import TrainingState
from feedbax.training.model_checkpoint import (
    ModelCheckpointError,
    SemanticModelWriter,
)


def _publish(root, graph, models, templates, *, extra=(), assets=None, replica_ids=("0",)):
    writer = SemanticModelWriter.for_graph(graph, assets, ComponentRegistry())
    assert writer is not None
    return write_method_checkpoint(
        root,
        run_id="selected-model",
        identity=MethodCheckpointIdentity(
            request_id="train:" + "1" * 64, method_key="external.Adversary",
        ),
        clock=MethodCheckpointClock(completed_updates=2, segment_start=0, schedule_origin=0, seed=81),
        state=TrainingState(stack_replicas(models), extra),
        model_writer=writer,
        model_templates=templates,
        replica_ids=replica_ids,
    )


def _open(root, pointer):
    return read_method_checkpoint(root, pointer, components=ComponentRegistry())


class _AdaptiveState(eqx.Module):
    configuration: bytes = eqx.field(static=True)
    price: jax.Array
    noise: jax.Array
    history: jax.Array


def test_recover_replicated_method_state_and_read_model_without_extra(tmp_path):
    graph = GraphSpec(nodes={
        "linear": ComponentSpec(
            type="feedbax.Linear", params={"input_size": 1, "output_size": 1},
            input_ports=["input"], output_ports=["output"],
        ),
    })
    template = compile_graph(GraphDocument(graph=graph), ComponentRegistry()).graph
    extra = _AdaptiveState(
        b"adam", jnp.stack((jnp.asarray(3.5), jnp.asarray(-2.25))),
        jax.random.split(jax.random.key(912), 2),
        jnp.array([[0.25, 0.75], [1.5, 2]], dtype=jnp.bfloat16),
    )
    replicas = tuple(
        eqx.tree_at(lambda model: model.nodes["linear"].layer.weight, template, jnp.asarray([[weight]]))
        for weight in (0.25, -0.75)
    )
    written = _publish(
        tmp_path / "attempt", graph, replicas, (template, template),
        extra=extra, replica_ids=("left", "right"),
    )
    reader = _open(written.root, written.latest_pointer)
    fresh = _AdaptiveState(
        b"adam", jnp.stack((jnp.asarray(0.0), jnp.asarray(0.0))),
        jax.random.split(jax.random.key(0), 2),
        jnp.zeros((2, 2), dtype=jnp.bfloat16),
    )
    recovered = reader.restore_extra(fresh)
    np.testing.assert_array_equal(recovered.price, [3.5, -2.25])
    assert recovered.price.weak_type
    np.testing.assert_array_equal(jax.random.key_data(recovered.noise), jax.random.key_data(extra.noise))
    assert recovered.noise.dtype == extra.noise.dtype
    assert recovered.history.dtype == jnp.bfloat16
    np.testing.assert_array_equal(recovered.history.astype(jnp.float32), [[0.25, 0.75], [1.5, 2]])
    with pytest.raises(CheckpointCustodyError):
        reader.restore_extra(_AdaptiveState(b"sgd", fresh.price, fresh.noise, fresh.history))

    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    full = bind_method_checkpoint_content(written.root, written.latest_pointer, store)
    verify_method_checkpoint_content(store, written.latest_pointer, full)
    with materialize_checkpoint_content(full, store, owner="test") as materialized:
        moved = _open(materialized, written.latest_pointer)
        np.testing.assert_array_equal(moved.restore_extra(fresh).price, recovered.price)

    (written.manifest_path.parent / method_checkpoint_extra_path(written.manifest.extra)).unlink()
    model_only = bind_method_checkpoint_content(
        written.root, written.latest_pointer, store, include_extra=False,
    )
    verify_method_checkpoint_content(store, written.latest_pointer, model_only, include_extra=False)
    with materialize_checkpoint_content(model_only, store, owner="test", include_extra=False) as materialized:
        selected = _open(materialized, written.latest_pointer)
        assert selected.model.replica_ids == ("left", "right")
        left, right = selected.model.restore_all((template, template))
        np.testing.assert_array_equal(left.nodes["linear"].layer.weight, [[0.25]])
        np.testing.assert_array_equal(right.nodes["linear"].layer.weight, [[-0.75]])
        with pytest.raises(CheckpointCustodyError):
            selected.restore_extra(fresh)


def test_stateful_graph_adam_checkpoint_preserves_the_optimizer_abi(tmp_path):
    graph = GraphSpec(nodes={
        "cell": ComponentSpec(
            type="feedbax.GRU",
            input_ports=["input", "hidden"], output_ports=["output", "hidden"],
        ),
    })
    model = compile_graph(GraphDocument(graph=graph), ComponentRegistry()).graph

    def optimizer_state(**params):
        settings = StandardTrainingSettings(
            optimizer=OptimizerSpec(type="adam", params={"learning_rate": 0.001, **params}),
        )
        return stack_replicas((_init_supervised(settings, model, jax.random.key(0)).extra,))

    source = optimizer_state()
    written = _publish(tmp_path / "adam", graph, (model,), (model,), extra=source)
    selected = _open(written.root, written.latest_pointer)
    assert eqx.tree_equal(selected.restore_extra(optimizer_state(learning_rate=0.01)), source)
    with pytest.raises(CheckpointCustodyError):
        selected.restore_extra(optimizer_state(b1=0.8))


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
    inconsistent = eqx.tree_at(
        lambda model: model.nodes["muscle"].state_index.init,
        source,
        replace_fn=lambda leaf: jnp.full_like(leaf, 0.75),
    )
    rejected_root = tmp_path / "unreconstructible"
    with pytest.raises(ModelCheckpointError):
        _publish(rejected_root, graph, (inconsistent,), (template,))
    assert not (rejected_root / "latest.json").exists()

    written = _publish(tmp_path / "thelen", graph, (source,), (template,))
    selected = _open(written.root, written.latest_pointer).model
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
    written = _publish(tmp_path / "checkpoint", graph, (model,), (model,), assets=imported)
    pointer = tmp_path / "selected.json"
    pointer.write_text(written.latest_pointer.model_dump_json())
    expected = _spatial_step(model)
    rmtree(imported.root)
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
        _open(written.root, written.latest_pointer)
    assert str(missing.value).startswith(authored_path)
    physical.write_bytes(exact[:-1] + b"!")
    with pytest.raises(ModelCheckpointError) as changed:
        _open(written.root, written.latest_pointer)
    assert str(changed.value).startswith(authored_path)
    assert "sha256" in str(changed.value).lower()


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    root, pointer_path, output = map(Path, sys.argv[1:])
    pointer = CheckpointLatestPointer.model_validate_json(pointer_path.read_bytes())
    selected = _open(root, pointer).model
    output.write_text(json.dumps(_spatial_step(selected.restore())))
