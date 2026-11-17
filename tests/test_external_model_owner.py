"""A package-owned model component participates in managed Train and Evaluate."""

import os

import jax
import jax.numpy as jnp
import numpy as np

from feedbax.compiler.model_attachment import lower_evaluate_attachments
from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.model_attachment import MODEL_DIFFERENCE_BATCHES
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.studio_project import StudioProjectCreateFile
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.jobs import JobService
from feedbax.persistence.array_container import decode_array_container
from feedbax.plugins.composition import load_application
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.runtime.raw_state import bind_raw_state
from feedbax.training.evaluation_variant import resolve_evaluation_variant, validate_evaluation_source
from feedbax.training.method_checkpoint import read_method_checkpoint
from feedbax.training.model_checkpoint import ModelRestoreTarget
from feedbax.training.rollout_scoring import RolloutScorer
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.installed_package import install_discovery_fixture
from tests.support.managed_train_fixture import managed_train_payload
from tests.test_local_job_host import host, job_row, restart, wait_until  # noqa: F401


def test_external_owner_trains_checkpoints_and_records_an_evaluate_difference(host, tmp_path, monkeypatch):
    installed_site = install_discovery_fixture(tmp_path / "installed")
    monkeypatch.syspath_prepend(str(installed_site))
    paths = (str(installed_site), os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, paths)))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    restart(host)
    registry = load_application().bundle

    raw = managed_train_payload()
    component = raw["trial_documents"]["model"]["interior"]["components"]["readout"]
    owner_type = "entry_point_provider.ObservedReadout"
    component.update(type=owner_type, revision=1)
    folders = ProjectFolderService(
        tmp_path / "projects", components=registry.components,
        functions=registry.expression_functions, python_tasks=registry.python_tasks,
        training_methods=registry.training_methods,
    )
    snapshot = folders.new().snapshot
    project = snapshot.project_id
    snapshot = folders.create_file(project, StudioProjectCreateFile.model_validate({
        "revision": snapshot.revision,
        "file": {"kind": "step", "id": "train", "step": {"kind": "train", "base": {"inline": raw}}},
    })).snapshot
    reviewed = folders.read_plan(project, revision=snapshot.revision)
    assert not reviewed.plan.diagnostics, reviewed.plan.diagnostics
    (train,) = reviewed.plan.runs
    request = decode_request(
        reviewed.plan.requests[train.request_id], reviewed.plan.documents.__getitem__,
    ).spec
    assert isinstance(request, DocumentTrainingRequestSpec)
    (admission,) = JobService(host, registry).admit_plan(
        folders, reviewed, run_keys=(train.key,), token="external-owner",
    ).items
    assert admission.disposition == "admitted", admission
    row = wait_until(
        lambda: job_row(host, admission.job_id, "succeeded") or job_row(host, admission.job_id, "failed"),
        timeout=24,
    )
    assert row["state"] == "succeeded", row["outcome"]
    published = PublishedTrainingCheckpoint.model_validate(row["outcome"]["completion"]["checkpoint"])
    assert published.completed_updates == request.target_total
    slot = read_method_checkpoint(
        published.checkpoint_root, published.pointer, components=registry.components,
    ).model
    validate_evaluation_source(slot.graph, registry.components)

    documents = request.trial_documents
    initial = compile_trial_execution(
        documents.model_name, documents.model, documents.task_source,
        registry.components, registry.expression_functions,
        objective=documents.objective, initialization_key=model_initialization_key(request.seed),
    )
    assert isinstance(initial, PreparedTrialExecution), initial
    initial_model = initial.executable.graph
    (replica,) = slot.replicas
    tensor = semantic_key("model", "value", ("readout",), "transform")
    expected_weights = decode_array_container(
        slot.provider.get_object(replica.arrays),
        expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
    ).tensors[tensor]
    assert not np.array_equal(expected_weights, initial_model.nodes["readout"].layer.weight)
    restored_source = slot.restore(initial_model)
    np.testing.assert_array_equal(restored_source.nodes["readout"].layer.weight, expected_weights)

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
    restored = slot.restore(initial_model, target=ModelRestoreTarget(
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
