"""An Evaluate step reads a Model and a Task document: its subject's, or one it names."""

from copy import deepcopy

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project_sweep import model_field_path
from feedbax.contracts.project import Project, diff
from feedbax.planner.project_plan import ProjectPlanMemo, compile_project
from feedbax.contracts.project_resolution import ResolutionCode
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import LastMember
from tests.support.managed_train_fixture import managed_evaluation_step, managed_train_project

# Request identities for the shipped point-mass fixture, with Model defaults filled and keyed Task trials.
# These witnesses hold the same changes expressed on the Evaluate step or in its Model and Task documents.
FEE = "train:52d8987cbf9cea9e6706ab37025a8b6d00c1063bd6f73cef8ba46ba0d1f6feec"
DEFAULT = "evaluate:514c6ccf4b7b76dd3af60b5ed619de36ead2e541e865c933a4597778b5388694"
FIELD_CHANGED = "evaluate:86580f29b2edcad563ddcdf7af55bd40ee65f0a175b433cb1062137283822774"
ATTACHMENT_ADDED = "evaluate:4957f6e7585d7c1a725c6ac7e2ce5f84b62f21d9fb750e78402d80859cad7805"
# The replacement Task now pairs target_pos by stable ID without a generated override.
FEWER_CONDITIONS = "evaluate:6bff30ca35c87f7a9c4db2dccc3875a42dc5500005c0c4ead5b6d72d0c5f371d"
SWEPT_PUSHED = (
    ("trained==347a995705cde9b16ee3a665dede47e6594d0c1158f3809255731f622a8fbe30",
     "evaluate:489ba0f6bb0478121871baab6c167d474ac28e35dc70474630ac94b9dece99eb"),
    ("trained==abb0ffe5124ac74b8d431462660c668b987dd969c5bf76fc07a20894a2b46eea",
     "evaluate:337ed9ae58ac63a634efda4cd1a0f5c0307aa8b78636602e6975fc617038ce0e"),
)
NEAREST_SHARED = {
    "fee": "train:f917ec7c972f5c6905bb219c09b774bc7cf919de5cd19b0a9e6654c16ddf20bc",
    "other": FEE,
    "check": "evaluate:18ba9ba2989d3cc138878c9c0d96605512ad2b16fc769681cfd8ff908fd056ae",
}
UNPAIRED_TASK = "Model input 'target_pos' needs a Task signal."

TASK = "simple-reaches-task"
MASS = {"composites": [], "element": {"kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"]}}
PUSH = {"kind": "intervention", "id": "push", "operation": "Scale", "operands": {"factor": {"kind": "value", "value": 0.5}}}


def _data():
    data = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    data["documents"]["model"] = data["documents"].pop("point-mass")
    fee = data["steps"].pop("train")
    fee["base"]["inline"]["seed"] = 0
    fee["base"]["inline"]["trial_documents"]["model"] = {"document": "model"}
    data["steps"] = {"fee": fee, "check": managed_evaluation_step("fee")}
    return data


def _model(data, name="model"):
    return deepcopy(data["documents"][name]["payload"])


def _with_mass(payload, mass):
    payload["interior"]["components"]["plant"]["fields"]["mass"] = mass
    return payload


def _with_push(payload):
    payload["interior"]["wires"]["drive"]["attachments"] = {
        PUSH["id"]: {key: deepcopy(value) for key, value in PUSH.items() if key != "id"},
    }
    return payload


def _based(data, name, content, base="model"):
    data["documents"][name] = {
        "label": "Document",
        "schema": data["documents"][base]["schema"], "base": {"document": base},
        "differences": {
            path: value.model_dump(mode="json") for path, value in diff(_model(data, base), content).items()
        },
    }


def _reads(data, step, role, document):
    data["steps"][step]["base"]["inline"].setdefault("trial_documents", {})[role] = {"document": document}


def _plan(data):
    return compile_project(Project.model_validate(data))


def _ids(data):
    return {run.key.step: run.request_id for run in _plan(data).runs}


def test_a_change_held_in_a_document_gives_the_ids_of_the_change_stored_on_the_step():
    for change, expected in ((lambda payload: _with_mass(payload, 2.0), FIELD_CHANGED), (_with_push, ATTACHMENT_ADDED)):
        data = _data()
        _based(data, "model-2", change(_model(data)))
        _reads(data, "check", "model", "model-2")
        assert _ids(data) == {"fee": FEE, "check": expected}


def test_the_subject_documents_or_an_unchanged_variant_give_the_default_ids():
    data = _data()
    assert _ids(data) == {"fee": FEE, "check": DEFAULT}
    _reads(data, "check", "model", "model")
    assert _ids(data) == {"fee": FEE, "check": DEFAULT}
    _based(data, "model-2", _model(data))
    _reads(data, "check", "model", "model-2")
    assert _ids(data) == {"fee": FEE, "check": DEFAULT}


def test_an_evaluation_names_the_documents_it_reads_and_reads_its_subjects_as_a_follower():
    data = _data()
    plan = _plan(data)
    check = next(row for row in plan.rows if row.key.step == "check")
    assert dict(check.document_ids) == {
        "payload.trial_documents.model": "model", "payload.trial_documents.task": TASK,
    }
    assert {"model", TASK} <= check.document_uses
    assert "check" not in plan.document_users("default", "model")

    _based(data, "model-2", _with_push(_model(data)))
    _reads(data, "check", "model", "model-2")
    plan = _plan(data)
    check = next(row for row in plan.rows if row.key.step == "check")
    assert check.document_ids["payload.trial_documents.model"] == "model-2"
    assert plan.document_users("default", "model-2") == {"check"}
    assert "check" not in plan.document_users("default", TASK)


def test_an_evaluation_keeps_its_subjects_swept_value_under_its_documents_change():
    mass = model_field_path(ModelAddress.model_validate(MASS))

    def swept():
        data = _data()
        data["steps"]["fee"]["sweep"] = {"axes": {"mass": {"path": mass, "values": [1.5, 3.0]}}}
        data["steps"]["check"]["sweep"] = {"axes": {"trained": {"input": "subject"}}}
        return data

    data = swept()
    _based(data, "model-2", _with_push(_model(data)))
    _reads(data, "check", "model", "model-2")
    plan = _plan(data)
    assert tuple(sorted((run.key.row, run.request_id) for run in plan.runs if run.kind == "evaluate")) == SWEPT_PUSHED
    trained = {
        run.request_id: decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        for run in plan.runs if run.kind == "train"
    }
    measured = set()
    for run in (run for run in plan.runs if run.kind == "evaluate"):
        request = decode_request(plan.requests[run.request_id], plan.documents.__getitem__)
        (subject,) = request.inputs["subject"]
        checkpoint = subject.reference
        assert checkpoint.role == "checkpoint" and checkpoint.select == LastMember(kind="last")
        paired = trained[checkpoint.request_id].trial_documents
        measured.add(paired.model.interior.components["plant"].fields["mass"])
        assert request.spec.params["trials"] == list(paired.task.condition_ids)
        variation = request.spec.params["variation"]
        assert all("mass" not in edit["path"] for edit in variation)
        (push,) = [edit for edit in variation if edit["path"].startswith("graph.nodes.")]
        assert push["op"] == "add" and push["value"]["type"] == "feedbax.Scale"
        assert push["value"]["params"] == {"fixed": True, "scale": 0.5}
    assert measured == {1.5, 3.0}


def test_evaluate_ungroup_eligibility_reads_the_selected_axis_model(tmp_path):
    """Catch a selected Model value inheriting the trained Model's Ungroup verdict."""
    from fastapi.testclient import TestClient

    from feedbax.web.app import create_app
    from feedbax.web.services.project_folder import ProjectFolderService
    from tests.test_canonical_evaluate_admission import _save_project

    data = _data()
    data["steps"]["fee"]["label"] = "Training source"
    wrapper = "idle-wrapper"
    data["documents"]["model"]["payload"]["interior"]["components"][wrapper] = {
        "kind": "composite", "label": "Idle wrapper",
        "ports": {"inputs": {"in": {"label": "In", "type": {}}}}, "interior": {},
    }
    _reads(data, "check", "model", "model")
    default = f"trial_documents.model.interior.components.{wrapper}.ports.inputs.in.default"
    data["steps"]["check"]["sweep"] = {"axes": {"model-choice": {"values": {
        "open": {},
        "blocked": {"patches": {default: {"op": "add", "value": {"value": 1.0}}}},
    }}}}
    folders = ProjectFolderService(tmp_path)
    project = "ungroup-axis-value"
    _save_project(folders, project, Project.model_validate(data))
    snapshot = folders.read(project)
    address = {"composites": [], "element": {"kind": "component", "id": wrapper}}

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        def reason(value):
            response = client.post(f"/api/projects/{project}/model/evaluate-model", json={
                "revision": snapshot.revision,
                "run": {"variant": "default", "step": "check", "row": f"model-choice={value}"},
                "scope": {"kind": "axis_value", "axis": "model-choice", "value": value},
            })
            assert response.status_code == 200, response.text
            state = response.json()
            assert state["source"] == {
                "kind": "trained", "reference": {"kind": "step", "key": "fee"}, "name": "Training source",
            }
            assert state["resolved_model"] is not None, state["problems"]
            assert wrapper in state["resolved_model"]["interior"]["components"]
            return next(item["reason"] for item in state["ungroup"] if item["address"] == address)

        assert reason("open") is None
        assert reason("blocked") == "Ungroup would remove an unconnected defaulted input"


def test_a_model_document_that_shares_none_with_the_subjects_is_the_evaluations_problem():
    data = _data()
    data["documents"]["model"]["label"] = "Point mass"
    data["documents"]["unrelated"] = deepcopy(data["documents"]["model"])
    _reads(data, "check", "model", "unrelated")
    plan = _plan(data)
    assert [run.key.step for run in plan.runs] == ["fee"]
    assert [(problem.code, problem.step, problem.document, problem.path) for problem in plan.diagnostics] == [
        (ResolutionCode.BAD_REFERENCE, "check", "unrelated", "payload.trial_documents.model"),
    ]
    assert plan.diagnostics[0].detail == "Shares no document with Point mass."


def test_an_evaluation_reads_its_documents_differences_from_the_nearest_shared_document():
    data = _data()
    data["steps"]["other"] = deepcopy(data["steps"]["fee"])
    _based(data, "model-2", _with_push(_model(data)))
    _reads(data, "check", "model", "model-2")
    _based(data, "model-3", _with_mass(_model(data), 3.0))
    _reads(data, "fee", "model", "model-3")
    assert _ids(data) == NEAREST_SHARED

    _reads(data, "fee", "model", "model")
    assert _ids(data) == {"fee": FEE, "other": FEE, "check": ATTACHMENT_ADDED}


def test_an_incremental_plan_follows_the_documents_an_evaluation_reads():
    data = _data()
    _based(data, "model-2", _with_push(_model(data)))
    _reads(data, "check", "model", "model-2")
    memo = ProjectPlanMemo()
    compile_project(Project.model_validate(data), memo=memo)
    for change in (
        lambda: _with_mass(data["documents"]["model"]["payload"], 3.0),
        lambda: _based(data, "model-3", _with_mass(_model(data), 4.0)) or _reads(data, "fee", "model", "model-3"),
        lambda: data["documents"]["model-2"].update(base={"document": "model-3"}),
    ):
        change()
        project = Project.model_validate(data)
        assert {run.key.step: run.request_id for run in compile_project(project, memo=memo).runs} == {
            run.key.step: run.request_id for run in compile_project(project).runs
        }


def test_a_task_document_replaces_the_subjects_task_unless_it_holds_the_same_content():
    data = _data()
    data["documents"]["task-2"] = deepcopy(data["documents"][TASK])
    _reads(data, "check", "task", "task-2")
    assert _ids(data) == {"fee": FEE, "check": DEFAULT}

    shorter = data["documents"]["task-2"]["payload"]
    shorter["validation_trials"].popitem()
    plan = _plan(data)
    assert {run.key.step: run.request_id for run in plan.runs} == {"fee": FEE, "check": FEWER_CONDITIONS}
    request = decode_request(plan.requests[FEWER_CONDITIONS], plan.documents.__getitem__)
    assert request.inputs["subject"][0].reference.request_id == FEE
    assert request.spec.params["trials"] == list(shorter["validation_trials"])

    shorter["validation_trials"] = deepcopy(data["documents"][TASK]["payload"]["validation_trials"])
    del shorter["signals"]["target_pos"]
    plan = _plan(data)
    assert not plan.diagnostics
    check = next(run for run in plan.runs if run.key.step == "check")
    assert check.document_ids["payload.trial_documents.task"] == "task-2"
    assert [(problem.path, problem.detail) for problem in check.problems] == [("/trial_documents/task", UNPAIRED_TASK)]


def test_a_continuation_reads_a_variant_made_in_an_evaluation():
    data = _data()
    del data["steps"]["check"]
    _based(data, "model-2", _with_push(_model(data)))
    continuation = deepcopy(data["steps"]["fee"])
    continuation["base"]["inline"]["checkpoint_initialization"] = "initialize_from"
    continuation["inputs"] = {"checkpoint": {"step": "fee", "checkpoint": "last"}}
    data["steps"]["cont"] = continuation
    _reads(data, "cont", "model", "model-2")
    plan = _plan(data)
    assert not plan.diagnostics
    run = next(run for run in plan.runs if run.key.step == "cont")
    assert run.inputs["checkpoint"].member.reference.request_id == FEE
    spec = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
    assert [attachment.id for attachment in spec.trial_documents.model.interior.wires["drive"].attachments] == ["push"]
