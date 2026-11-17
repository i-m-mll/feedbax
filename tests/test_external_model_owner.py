"""A package-owned model component participates in managed Train and Evaluate."""

import jax
import jax.numpy as jnp
import numpy as np

from feedbax.compiler.model_attachment import lower_evaluate_attachments
from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_attachment import MODEL_DIFFERENCE_BATCHES
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.plugins.composition import load_application
from feedbax.runtime.raw_state import bind_raw_state
from feedbax.studio.canonical_train import lower_canonical_train
from feedbax.training.checkpoint_custody import (
    authenticate_published_checkpoint_custody,
    open_published_model_slot,
    resolve_checkpoint_custody_ref,
    write_checkpoint_transaction,
)
from feedbax.training.evaluation_variant import resolve_evaluation_variant, validate_evaluation_source
from feedbax.training.managed_studio import prepare_managed_training
from feedbax.training.model_checkpoint import ModelRestoreTarget, SemanticModelWriter
from feedbax.training.rollout_scoring import RolloutScorer
from tests.support.installed_package import install_discovery_fixture
from tests.support.managed_train_fixture import managed_train_payload


def test_external_owner_trains_checkpoints_and_records_an_evaluate_difference(tmp_path, monkeypatch):
    installed_site = install_discovery_fixture(tmp_path / "installed")
    monkeypatch.syspath_prepend(str(installed_site))
    registry = load_application().bundle

    raw = managed_train_payload()
    component = raw["trial_documents"]["model"]["interior"]["components"]["readout"]
    owner_type = "entry_point_provider.ObservedReadout"
    component.update(type=owner_type, revision=1)
    request = DocumentTrainingRequestSpec.model_validate(raw)
    training = lower_canonical_train(request, registry.training_programs, registries=registry)
    prepared = prepare_managed_training(training, registry)
    initial = prepared.initial_slots["model"]
    update = prepared.kernel_context["studio_graph_update"]
    state = update.reported_update(prepared.initial_slots)
    expected_weights = state["model"].nodes["readout"].layer.weight
    assert not np.array_equal(expected_weights, initial.nodes["readout"].layer.weight)

    graph = GraphSpec.model_validate(training.graph.inline)
    writer = SemanticModelWriter.for_graph(graph, None, registry.components)
    assert writer is not None
    published = write_checkpoint_transaction(
        tmp_path / "checkpoint",
        run_spec=training,
        phase_program=training.worker_execution.effective_phase.phase_program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="external-owner", phase="train_batch", program_step=1,
            completed_barrier="after_train_batch",
        ),
        slots=state,
        semantic_model=writer,
    )
    slot = open_published_model_slot(published.root, published.latest_pointer)
    assert slot is not None
    validate_evaluation_source(slot.graph, registry.components)
    custody = resolve_checkpoint_custody_ref(
        authenticate_published_checkpoint_custody(published.root).parent_ref,
        allowed_root=published.root, components=registry.components,
    )
    np.testing.assert_array_equal(custody.slots["model"].nodes["readout"].layer.weight, expected_weights)

    documents = request.trial_documents
    address = {"document": documents.model_name, "composites": []}
    differences = MODEL_DIFFERENCE_BATCHES.validate_python({"observation": [
        {
            "op": "add_component",
            "address": address | {"element": {"kind": "component", "id": "probe"}},
            "component": {
                "kind": "leaf", "name": "Probe", "type": owner_type, "revision": 1,
                "fields": {
                    "input_size": 2, "output_size": 2, "initialization": "zeros", "bias_init": 0.125,
                },
            },
        },
        {
            "op": "add_wire",
            "address": address | {"element": {"kind": "wire", "id": "measure"}},
            "source": {"owner": "component", "component": "readout", "port": "output"},
            "target": {"owner": "component", "component": "probe", "port": "input"},
        },
    ]})
    _, _, edits = lower_evaluate_attachments(
        documents.model_name, documents.model, None, registry.components,
        tuple(differences["observation"]),
    )
    variant = resolve_evaluation_variant(slot.graph, documents.task_source, edits, components=registry.components)
    pairing = compile_trial_execution(
        documents.model_name, documents.model, variant.task,
        registry.components, registry.expression_functions,
        objective=documents.objective, graph_override=variant.graph,
    )
    assert isinstance(pairing, PreparedTrialExecution), pairing
    resolved_graph = pairing.executable.resolved.graph
    restored = slot.restore(slot.template(registry.components), target=ModelRestoreTarget(
        resolved_graph, pairing.executable.graph, variant.source_overrides,
    ))
    np.testing.assert_array_equal(restored.nodes["readout"].layer.weight, expected_weights)
    (probe_reset,) = (
        field.address.to_text() for field in pairing.execution.reset.fields
        if field.address.element.parent.id == "probe"
    )
    marker = jnp.asarray([0.75, -0.5])
    reset = pairing.execution.reset.prepare(restored, {probe_reset: marker}, key=jax.random.key(5))
    np.testing.assert_array_equal(restored.nodes["probe"].state_view(reset.state), marker)

    layout = bind_raw_state(restored, resolved_graph)
    trial, *_ = pairing.execution.task.condition_ids
    evaluation = RolloutScorer(pairing.execution, pairing.objective, 17, (trial,)).evaluate(
        restored, capture_state=layout.capture,
    )
    probe = semantic_key("model", "state", ("probe",), "last")
    readout = semantic_key("model", "state", ("readout",), "last")
    np.testing.assert_array_equal(evaluation.history[probe][0, 0], jnp.zeros(2))
    np.testing.assert_array_equal(
        evaluation.history[probe][0, 1:], np.full((pairing.execution.task.n_steps, 2), 0.125),
    )
    assert np.any(evaluation.history[readout][0, 1:] != 0)
