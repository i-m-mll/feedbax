"""Exercise the typed checkpoint codec on the registered optimizer layouts."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.training import OptimizerSpec
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.training.checkpoint_custody import load_latest_checkpoint, write_checkpoint_transaction
from feedbax.training.invocation import prepare_training_invocation
from feedbax.training.model_checkpoint import SemanticModelWriter
from feedbax.training.standard_train_checkpoint import (
    STANDARD_TRAIN_STATE_MEDIA_TYPE,
    StandardTrainStateError,
    decode_standard_train_slot,
)
from tests.test_studio_training_host_worker import _authored_spec


@pytest.mark.parametrize(("optimizer_type", "params"), [
    ("adam", {}),
    ("adamw", {"weight_decay": 0.001}),
    ("sgd", {}),
    ("sgd", {"momentum": 0.9}),
    ("rmsprop", {"centered": True, "momentum": 0.9}),
])
def test_typed_standard_optimizer_state_reproduces_the_next_update(tmp_path, optimizer_type, params):
    spec = _authored_spec(batches=2).model_copy(deep=True)
    spec.method_payload.payload["optimizer"] = OptimizerSpec(
        type=optimizer_type, params={"learning_rate": spec.training_config.learning_rate, **params},
    ).model_dump(mode="json", exclude_none=True)
    registries = new_application_registry_bundle()
    preparation = prepare_training_invocation(spec, registries, run_id="optimizer-codec", resume=False)
    kernel = preparation.kernel_context["studio_graph_update"]
    state = kernel.reported_update(preparation.initial_slots)
    graph = GraphSpec.model_validate(spec.graph.inline)
    model_writer = SemanticModelWriter.for_graph(graph, None)
    assert model_writer is not None
    phase = spec.worker_execution.effective_phase.phase_program
    publication = write_checkpoint_transaction(
        tmp_path / "runs",
        run_spec=spec,
        phase_program=phase,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="optimizer-codec", phase="train_batch", program_step=1,
            completed_barrier="after_train_batch",
        ),
        slots=state,
        semantic_model=model_writer,
        train_state_codec="standard_train",
    )
    resumed = load_latest_checkpoint(
        publication.root,
        expected_run_spec=spec,
        expected_phase_program=phase,
        expected_slots=preparation.initial_slots,
        published_pointer=publication.latest_pointer,
    )
    for name in ("model", "optimizer", "prng", "batch_counter"):
        assert eqx.tree_equal(state[name], resumed.slots[name])
    expected_next = kernel.reported_update(state)
    actual_next = kernel.reported_update(resumed.slots)
    for name in ("model", "optimizer", "prng", "batch_counter", "train_loss", "loss_terms"):
        assert eqx.tree_equal(expected_next[name], actual_next[name])

    cursor = next(slot for slot in publication.manifest.slots if slot.slot == "batch_counter")
    assert cursor.media_type == STANDARD_TRAIN_STATE_MEDIA_TYPE
    with pytest.raises(StandardTrainStateError, match="/slots/batch_counter.*coordinate"):
        decode_standard_train_slot(
            cursor, (publication.transaction_dir / cursor.relative_path).read_bytes(),
            template=jnp.asarray(0, dtype=jnp.int32), completed_updates=2,
        )
