"""Plan and shared graph compilation for a model document with nested owners."""

from copy import deepcopy

import jax
import numpy as np
import equinox as eqx

from feedbax.cli import main
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.jobs import JobService
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_train import lower_canonical_train
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.invocation import prepare_training_invocation
from feedbax.training.preparation import ExecutionPreparationResult
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.contracts.studio_training import StudioGraphMethodPayload
from feedbax.contracts.training import TrainingSpec
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from feedbax.web.worker.execution import compile_training_run, prototypes_from_task_bindings
from tests.test_canonical_evaluate_admission import _save_project, authored_evaluation_project
from tests.test_local_job_host import host as host_fixture, job_row, wait_until

host = host_fixture


def _end(component, port):
    return {"owner": "component", "component": component, "port": port}


def _wire(source, target, attachments=()):
    return {"source": source, "target": target, "attachments": list(attachments)}


def _port(id_, name, *, shape=None):
    return {"id": id_, "name": name, "type": {"shape": shape}}


def _model_project() -> dict:
    project, _ = authored_evaluation_project()
    raw = project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    train = raw["steps"]["train"]["base"]["inline"]
    source = train["graph"]
    leaves = {
        key: {"kind": "leaf", "name": key.capitalize(), "type": value["type"], "fields": value["params"]}
        for key, value in source["nodes"].items()
    }
    period = leaves["plant"]["fields"].pop("dt")
    delay = {"kind": "delay", "id": "carry", "delay": {"value": 1, "unit": "st"}}
    network = {"kind": "composite", "name": "Controller", "ports": {
        "inputs": [_port("target", "Target", shape=[2])],
        "outputs": [_port("force", "Force", shape=[2])],
    }, "interior": {"components": {key: leaves[key] for key in ("cell", "readout")}, "wires": {
        "target": _wire({"owner": "inputs", "port": "target"}, _end("cell", "input")),
        "carry": _wire(_end("cell", "hidden"), _end("cell", "hidden"), (delay,)),
        "decode": _wire(_end("cell", "output"), _end("readout", "input")),
        "read": _wire(_end("readout", "output"), {"owner": "outputs", "port": "force"}),
    }}}
    wrapper = {"kind": "composite", "name": "Wrapper", "ports": deepcopy(network["ports"]), "interior": {
        "components": {"controller": network}, "wires": {
            "pass-target": _wire({"owner": "inputs", "port": "target"}, _end("controller", "target")),
            "pass-force": _wire(_end("controller", "force"), {"owner": "outputs", "port": "force"}),
        },
    }}
    model = {
        "step_seconds": period,
        "ports": {"inputs": [_port("task-input", "reach_target", shape=[2])],
                  "outputs": [_port("effector", "effector")]},
        "interior": {"components": {"wrapper": wrapper, "plant": leaves["plant"]}, "wires": {
            "task-wire": _wire({"owner": "inputs", "port": "task-input"}, _end("wrapper", "target")),
            "plant-wire": _wire(_end("wrapper", "force"), _end("plant", "force")),
            "effector-wire": _wire(_end("plant", "effector"), {"owner": "outputs", "port": "effector"}),
        }},
    }
    train["graph"] = {"document": "reach-model"}
    raw["documents"] = {"reach-model": {"schema": MODEL_DOCUMENT_SCHEMA_ID, "payload": model}}
    raw["steps"] = {key: raw["steps"][key] for key in ("train", "baseline")}
    return raw


def _rename_components(model):
    for component in model["interior"]["components"].values():
        component["name"] += " display name"
        if component["kind"] == "composite":
            for ports in component["ports"].values():
                for port in ports:
                    port["name"] += " display name"
            _rename_components(component)


def test_nested_model_plans_with_stable_identity_and_compiles():
    raw = _model_project()
    before = compile_project(Project.model_validate(raw))
    assert not before.diagnostics
    _rename_components(raw["documents"]["reach-model"]["payload"])
    after = compile_project(Project.model_validate(raw))
    assert not after.diagnostics
    assert [run.request_id for run in before.runs] == [run.request_id for run in after.runs]
    assert before.requests == after.requests
    assert before.documents == after.documents

    run = next(run for run in before.runs if run.kind == "train")
    spec = decode_request(before.requests[run.request_id], before.documents.__getitem__).spec
    registry = ComponentRegistry()
    graph = compile_graph(
        GraphDocument(graph=spec.graph), registry,
        input_prototypes=prototypes_from_task_bindings(spec.task_binding_spec),
        initialization_key=jax.random.key(42),
    ).graph
    assert graph.nodes["wrapper"].nodes["controller"].nodes["cell"].cell.weight_ih.shape[1] == 2
    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    prepared = compile_training_run(
        graph_spec=spec.graph,
        training_spec=TrainingSpec(
            optimizer=method.optimizer, loss=spec.objective.loss,
            n_batches=spec.training_config.n_batches, batch_size=1,
        ),
        task_spec=spec.task.model_dump(mode="json", exclude_none=True),
        task_binding_spec=spec.task_binding_spec,
        cfg=spec.training_config, component_registry=registry,
    )
    assert prepared.graph.nodes["wrapper"].nodes["controller"].nodes["cell"].cell.weight_ih.shape[1] == 2

    raw["documents"]["reach-model"]["payload"]["ports"]["inputs"][0]["name"] = "unknown"
    refused = compile_project(Project.model_validate(raw))
    problem = next(item for item in refused.diagnostics if item.model_address is not None)
    assert problem.document == "reach-model"
    assert problem.model_address.element.end.port == "task-input"


def test_three_level_model_train_evaluate_and_rename_current(host, tmp_path, capsys):
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    raw = _model_project()
    project = "nested"
    _save_project(folders, project, Project.model_validate(raw))
    plan = folders.read_plan(project, revision=None)
    assert main(["plan", str(folders.root / project)]) == 0
    assert set(line.split()[1] for line in capsys.readouterr().out.splitlines()) == {
        run.request_id for run in plan.plan.runs
    }
    train = RunKey("default", "train", "default")
    evaluate = RunKey("default", "baseline", "default")
    (train_job,) = jobs.admit_plan(folders, plan, run_keys=(train,), token="nested-train").items
    (evaluation_job,) = jobs.admit_plan(folders, plan, run_keys=(evaluate,), token="nested-evaluate").items
    assert [train_job.disposition, evaluation_job.disposition] == ["admitted", "admitted"], (
        train_job, evaluation_job,
    )

    def finished(job):
        return wait_until(lambda: (
            job_row(host, job.job_id, "succeeded") or job_row(host, job.job_id, "failed")
        ), timeout=25)

    trained = finished(train_job)
    assert trained["state"] == "succeeded", trained.get("outcome")
    checkpoint = PublishedTrainingCheckpoint.model_validate(trained["outcome"]["completion"]["checkpoint"])
    slot = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
    assert slot is not None
    restored = slot.restore()
    assert restored.nodes["wrapper"].nodes["controller"].nodes["cell"].cell.weight_ih.shape[1] == 2
    evaluated = finished(evaluation_job)
    assert evaluated["state"] == "succeeded", evaluated.get("outcome")
    result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluation_job.job_id)
    with result.open_states() as dataset:
        assert dataset.data("/v1/model/state/wrapper/controller/cell/hidden").shape[-2] == 101
        assert dataset.data("/v1/model/carry/wrapper/controller/carry/carry").shape[-2] == 101
        assert np.isfinite(dataset.data("objective/total")).all()

    _rename_components(raw["documents"]["reach-model"]["payload"])
    _save_project(folders, project, Project.model_validate(raw))
    renamed = folders.read_plan(project, revision=None)
    assert [run.request_id for run in renamed.plan.runs] == [run.request_id for run in plan.plan.runs]
    joined = ProjectPlanService(folders, jobs).read(project, revision=renamed.revision)
    assert joined.attempts_by_run[train][0].job_id == train_job.job_id
    assert joined.attempts_by_run[evaluate][0].successful_receipt == result.receipt


def test_same_seed_nested_rename_preserves_values_and_batch_loss():
    raw = _model_project()
    registry = new_application_registry_bundle()

    def trained():
        plan = compile_project(Project.model_validate(raw))
        row = next(run for run in plan.runs if run.kind == "train")
        request = decode_request(plan.requests[row.request_id], plan.documents.__getitem__).spec
        run = lower_canonical_train(request, registry.training_programs)
        prepared = prepare_training_invocation(run, registry, run_id=None, resume=False)
        assert isinstance(prepared, ExecutionPreparationResult)
        update = prepared.kernel_context["studio_graph_update"].reported_update
        state, losses = prepared.initial_slots, []
        for _ in range(2):
            state = update(state)
            losses.append((state["train_loss"], state["loss_terms"]))
        model = state["model"].nodes["wrapper"].nodes["controller"]
        return losses, jax.tree.leaves(eqx.filter(model, eqx.is_array))

    before, values = trained()
    _rename_components(raw["documents"]["reach-model"]["payload"])
    after, renamed = trained()
    assert before == after
    assert len(values) == len(renamed) and values
    for a, b in zip(values, renamed, strict=True):
        np.testing.assert_array_equal(a, b)
