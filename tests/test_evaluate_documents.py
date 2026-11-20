"""An Evaluate step reads a Model and a Task document: its subject's, or one it names."""

from copy import deepcopy

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project, diff
from feedbax.contracts.project_plan import ProjectPlanMemo, compile_project
from feedbax.contracts.project_resolution import ResolutionCode
from feedbax.contracts.request_identity import decode_request
from tests.support.managed_train_fixture import managed_evaluation_step, managed_train_project

# Read at develop 76a6c51d, where the same changes were stored on the Evaluate step itself,
# and a Task choice was a `params.variation` entry with path `task`; the swept and nearest-shared ids at a9b3bb7d.
FEE = "train:40b570d54d78b55b6a06cce52dabd7968d1e0e09c3b92a6a09ab05e0d06f3117"
DEFAULT = "evaluate:9e1f087eb03b07a9106a08054a90b5bce78b5775c23b582ed213998535edc062"
FIELD_CHANGED = "evaluate:af856b63bdc773b5ed9f350993b003223f6b3a6c74c185eb676f4f471530882b"
ATTACHMENT_ADDED = "evaluate:3ad98387a55ff6903ae938428689f61b996d4b33eecb2552d8ec2a3487353caf"
FEWER_CONDITIONS = "evaluate:93633f66c3a47e11b1ea2d5acf5718a9c031b496319876725a53cc8c322594cc"
SWEPT_PUSHED = (
    ("trained==347a995705cde9b16ee3a665dede47e6594d0c1158f3809255731f622a8fbe30",
     "evaluate:449db0e3369f600420a8faf2642b1b661296c5d006b9e4acc0356aac1bc5f1e2"),
    ("trained==abb0ffe5124ac74b8d431462660c668b987dd969c5bf76fc07a20894a2b46eea",
     "evaluate:c9ccc993dbbf9a51b2e92053741ebc7140901744a8bd2f53574786ec53805677"),
)
NEAREST_SHARED = {
    "fee": "train:79900a9a69a9f18592ed334d91afb7cc6e13c8af1e1e22411f4c5cbcde43d1f0",
    "other": FEE,
    "check": "evaluate:59359e9c566cf281ef63419ca6948b365d145982adc65418ab1adc840de50cc3",
}
UNPAIRED_TASK = "Model input 'target_pos' reads signal 'target_pos', which the Task does not declare."

TASK = "simple-reaches-task"
MASS = {"composites": [], "element": {"kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"]}}
PUSH = {"kind": "intervention", "id": "push", "operation": "Scale", "fields": {"factor": 0.5}}


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
    payload["interior"]["wires"]["drive"]["attachments"] = [deepcopy(PUSH)]
    return payload


def _based(data, name, content, base="model"):
    data["documents"][name] = {
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


def test_a_model_document_that_shares_none_with_the_subjects_is_the_evaluations_problem():
    data = _data()
    data["documents"]["unrelated"] = deepcopy(data["documents"]["model"])
    _reads(data, "check", "model", "unrelated")
    plan = _plan(data)
    assert [run.key.step for run in plan.runs] == ["fee"]
    assert [(problem.code, problem.step, problem.document, problem.path) for problem in plan.diagnostics] == [
        (ResolutionCode.BAD_REFERENCE, "check", "unrelated", "payload.trial_documents.model"),
    ]


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
    assert [run.key.step for run in plan.runs] == ["fee"]
    assert [(problem.step, problem.document, problem.path, problem.detail) for problem in plan.diagnostics] == [
        ("check", "task-2", "/trial_documents/task", UNPAIRED_TASK),
    ]


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
