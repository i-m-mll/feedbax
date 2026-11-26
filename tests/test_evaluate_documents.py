"""An Evaluate step reads a Model and a Task document: its subject's, or one it names."""

from copy import deepcopy

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project, diff
from feedbax.planner.project_plan import ProjectPlanMemo, compile_project
from feedbax.contracts.project_resolution import ResolutionCode
from feedbax.contracts.request_identity import decode_request
from tests.support.managed_train_fixture import managed_evaluation_step, managed_train_project

# Request identities for the shipped point-mass Task, with its signal labels projected by ID.
# These witnesses hold the same changes expressed on the Evaluate step or in its Model and Task documents.
FEE = "train:7ee0aec47e9eb5d0408a91707503e31114c55bf912e95406f206bd654db1bbe4"
DEFAULT = "evaluate:7c6fad3f13cb972b160877409c34fda69843e13141ffdaf788250959b0289647"
FIELD_CHANGED = "evaluate:36403904c6fbf41fda182a0997b17ef9afb0bb92f6034ab11ab1217c60775884"
ATTACHMENT_ADDED = "evaluate:fb014cda6d2da4cb22047330a52f6013a32f812d968fbae93e78d45b5ece424b"
# The replacement Task now pairs target_pos by stable ID without a generated override.
FEWER_CONDITIONS = "evaluate:aef41d53a6e65f832f58a1281f33d810626c951e894423d6faf73053a3ecf563"
SWEPT_PUSHED = (
    ("trained==347a995705cde9b16ee3a665dede47e6594d0c1158f3809255731f622a8fbe30",
     "evaluate:dcf3f064b759cc1bfc8cc90b3f6f76b6af6947a432b708e288456a53967fa04a"),
    ("trained==abb0ffe5124ac74b8d431462660c668b987dd969c5bf76fc07a20894a2b46eea",
     "evaluate:26c6ab856a6094343bbcbb74ed40ef5d75fae9d3316a0ecaf3fd4bac061a3216"),
)
NEAREST_SHARED = {
    "fee": "train:4e5dbd251fb8ae4d207e7a6223b5daec96cdb92517047cbbe3a0b734ed92318b",
    "other": FEE,
    "check": "evaluate:a2dc720dec386e1fd5951caece29c1570d6f7589a5ce512d34fe54676127552c",
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
    mass = ModelAddress.model_validate(MASS).to_text()

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
    trained = {run.request_id for run in plan.runs if run.kind == "train"}
    for run in (run for run in plan.runs if run.kind == "evaluate"):
        spec = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        assert run.inputs["subject"].member.reference.request_id in trained
        assert all("mass" not in edit["path"] for edit in spec.params["variation"])


def test_evaluate_ungroup_eligibility_reads_the_selected_axis_model(tmp_path):
    """Catch a selected Model value inheriting the trained Model's Ungroup verdict."""
    from fastapi.testclient import TestClient

    from feedbax.web.app import create_app
    from feedbax.web.services.project_folder import ProjectFolderService
    from tests.test_canonical_evaluate_admission import _save_project

    data = _data()
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
    shorter["validation_trials"] = shorter["validation_trials"][:-1]
    assert _ids(data) == {"fee": FEE, "check": FEWER_CONDITIONS}

    shorter["validation_trials"] = deepcopy(data["documents"][TASK]["payload"]["validation_trials"])
    shorter["signals"] = [signal for signal in shorter["signals"] if signal["id"] != "target_pos"]
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
