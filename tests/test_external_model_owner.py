"""A package-owned model component participates in managed Train and Evaluate."""

import os

import jax
import jax.numpy as jnp
import numpy as np

from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.jobs import JobService
from feedbax.persistence.array_container import decode_array_container
from feedbax.application.composition import load_application
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.host.method_checkpoint import read_method_checkpoint
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.installed_package import install_discovery_fixture
from tests.support.model_documents import read_changed_model
from tests.support.managed_train_fixture import managed_evaluation_step, managed_train_project
from tests.test_canonical_evaluate_admission import _save_project
from tests.support.host import job_row, token, wait_until, worker_environment


def test_external_owner_trains_checkpoints_and_records_an_evaluate_difference(host, tmp_path, monkeypatch):
    installed_site = install_discovery_fixture(tmp_path / "installed")
    monkeypatch.syspath_prepend(str(installed_site))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    registry = load_application().bundle

    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    sources = raw["steps"]["train"]["base"]["inline"]["trial_documents"]
    model_name = sources["model"]["document"]
    model = raw["documents"][model_name]["payload"]
    task = raw["documents"][sources["task"]["document"]]["payload"]
    task["validation_trials"] = task["validation_trials"][:1]
    component = model["interior"]["components"]["readout"]
    owner_type = "entry_point_provider.ObservedReadout"
    component.update(type=owner_type, revision=1)
    address = {"composites": []}
    raw["steps"]["evaluate"] = managed_evaluation_step("train")
    read_changed_model(raw, "evaluate", [
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
    ], base=model_name, components=registry.components)
    folders = ProjectFolderService(
        tmp_path / "projects", components=registry.components,
        functions=registry.expression_functions, python_tasks=registry.python_tasks,
        training_methods=registry.training_methods,
    )
    project = "installed-owner"
    _save_project(folders, project, Project.model_validate(raw))
    reviewed = folders.read_plan(project, revision=None)
    assert not reviewed.plan.diagnostics, reviewed.plan.diagnostics
    train, evaluate = (
        next(run for run in reviewed.plan.runs if run.key.step == name)
        for name in ("train", "evaluate")
    )
    request = decode_request(
        reviewed.plan.requests[train.request_id], reviewed.plan.documents.__getitem__,
    ).spec
    assert isinstance(request, DocumentTrainingRequestSpec)
    jobs = JobService(host, registry)
    paths = (str(installed_site), os.environ.get("PYTHONPATH", ""))
    worker_environment(jobs, {"PYTHONPATH": os.pathsep.join(filter(None, paths))})

    def run_to_success(key):
        (admission,) = jobs.admit_plan(folders, reviewed, run_keys=(key,), token=token(key.step)).items
        assert admission.disposition == "admitted", admission
        row = wait_until(
            lambda: job_row(host, admission.job_id, "succeeded") or job_row(host, admission.job_id, "failed"),
        )
        assert row["state"] == "succeeded", row["outcome"]
        return row

    trained = run_to_success(train.key)
    published = PublishedTrainingCheckpoint.model_validate(trained["outcome"]["completion"]["checkpoint"])
    assert published.completed_updates == request.target_total
    slot = read_method_checkpoint(
        published.checkpoint_root, published.pointer, components=registry.components,
    ).model

    documents = request.trial_documents
    initial = compile_trial_execution(
        documents.model, documents.task_source,
        registry.components, registry.expression_functions,
        objective=documents.objective, initialization_key=model_initialization_key(request.seed),
    )
    assert isinstance(initial, PreparedTrialExecution), initial
    initial_model = initial.executable.graph
    (replica,) = slot.replicas
    tensors = decode_array_container(
        slot.provider.get_object(replica.arrays),
        expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
    ).tensors
    expected_weights = tensors[semantic_key("model", "value", ("readout",), "transform")]
    expected_bias = tensors[semantic_key("model", "value", ("readout",), "offset")]
    assert not np.array_equal(expected_weights, initial_model.nodes["readout"].layer.weight)
    restored_source = slot.restore(initial_model)
    np.testing.assert_array_equal(restored_source.nodes["readout"].layer.weight, expected_weights)

    (readout_reset,) = (
        field.address.to_text() for field in initial.execution.reset.fields
        if field.address.element.parent.id == "readout"
    )
    marker = jnp.asarray([0.75, -0.5])
    reset = initial.execution.reset.prepare(restored_source, {readout_reset: marker}, key=jax.random.key(5))
    np.testing.assert_array_equal(restored_source.nodes["readout"].state_view(reset.state), marker)

    evaluated = run_to_success(evaluate.key)
    (edge,) = evaluated["edges"]
    assert edge["binding"]["terminal"]["receipt"] == trained["outcome"]["store_success"]
    result = jobs.read_evaluation_result(
        project=reviewed.project_id, store=reviewed.store_path, job_id=evaluated["id"],
    )
    with result.open_states() as states:
        probe = states.data("/v1/model/state/probe/last", (0, 0, 0))
        np.testing.assert_array_equal(probe[0], np.zeros(2))
        np.testing.assert_array_equal(probe[1:], np.full((task["fixed_steps"], 2), 0.125))
        readout = states.data("/v1/model/state/readout/last", (0, 0, 0))
        controller = states.data("/v1/model/state/cell/output", (0, 0, 0))
        np.testing.assert_allclose(
            readout[1:], controller[1:] @ expected_weights.T + expected_bias,
            rtol=1e-6, atol=1e-7,
        )
