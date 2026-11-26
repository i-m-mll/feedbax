"""Family rules keep authored size fixed and preserve the scientific members they stand for."""

from copy import deepcopy

import pytest

from feedbax.contracts.expression_values import CallValue, DrawValue, LiteralValue
from feedbax.contracts.member_family import FamilyProblem, expand_members
from feedbax.contracts.project import Project
from feedbax.contracts.project_storage import write_project_file
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID, SetEntry
from feedbax.planner.project_plan import compile_project
from tests.test_document_train_pair import authored_documents, document_project


def _shift_entry(last):
    return {
        "label": "Shift {n}", "over": {"n": f"0..{last}"},
        "task_variables": {"amount": {"label": "Amplitude", "type": {"base": "real"}, "value": "n + 1"}},
    }


def test_set_family_size_depends_on_the_rule_digits_and_first_read_stays_small():
    small = SetEntry.model_validate(_shift_entry(9))
    large = SetEntry.model_validate(_shift_entry(999))
    small_bytes, large_bytes = map(write_project_file, (small, large))
    assert len(large_bytes) - len(small_bytes) == 2
    assert SetEntry.model_validate_json(large_bytes) == large
    respelt = SetEntry.model_validate(_shift_entry(999) | {"over": {"n": " 0  ..  999 "}})
    assert write_project_file(respelt) == large_bytes

    [first] = expand_members({"shift": large}, first=True, unique_labels=True)
    assert (first.source_key, first.key, first.value.label) == ("shift", "shift-0", "Shift 0")
    assert first.value.task_variables["amount"].value == CallValue(
        function="add", args=(LiteralValue(value=0), LiteralValue(value=1)),
    )


def test_family_combines_in_order_and_addresses_a_collision_at_the_stored_member():
    family = SetEntry.model_validate({
        **_shift_entry(1),
        "label": "Shift {n} {{toward}} {side}",
        "over": {"n": "-1..0", "side": {"left": {"value": -1, "label": "Left"}, "right": 1}},
        "combination": "zip",
    })
    rows = tuple(expand_members({"shift": family}, unique_labels=True))
    assert [(row.key, row.value.label) for row in rows] == [
        ("shift-m1-left", "Shift -1 {toward} Left"), ("shift-0-right", "Shift 0 {toward} right"),
    ]
    with pytest.raises(FamilyProblem, match="already exists") as error:
        tuple(expand_members({"shift": family, "shift-0-right": SetEntry(label="Other")}))
    assert (error.value.source_key, error.value.path, error.value.coordinates) == (
        "shift", ("over",), {"n": "0", "side": "right"},
    )


def _decoded(raw):
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    assert all(not row.problems for row in plan.rows), plan.rows
    return {
        run.key.step: decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec for run in plan.runs
    }


def test_evaluate_selects_a_validation_family_after_crossing_a_listed_set_family():
    raw = document_project(authored_documents(), evaluation=True)
    task = raw["documents"]["task"]["payload"]
    original = deepcopy(next(iter(task["validation_trials"].values())))
    task["validation_trials"] = {"reach": original | {"label": "Reach {k}", "over": {"k": "0..2"}}}
    raw["documents"]["bank"] = {
        "label": "Loads", "schema": SET_DOCUMENT_SCHEMA_ID,
        "payload": {"entries": {"shift": _shift_entry(1)}},
    }
    raw["steps"]["train"]["set_fields"] = {"case": {"axis": {"set": "bank"}, "sampling": {"kind": "listed"}}}
    params = raw["steps"]["baseline"]["base"]["inline"]["params"]
    params["trial_set"] = {"kind": "conditions", "ids": ["reach"]}
    params["variation"] = [{"op": "replace", "path": "task.epochs.hold.end.length", "value": "6 st"}]
    specs = _decoded(raw)
    expected = [f"reach-{k}--case=shift-{n}" for k in range(3) for n in range(2)]
    assert list(specs["train"].trial_documents.task.condition_ids) == expected
    assert specs["baseline"].params["trials"] == expected

    params["variation"].append({
        "op": "replace", "path": "task.validation_trials.reach.over.k", "value": "0..1",
    })
    narrower = _decoded(raw)
    assert list(narrower["train"].trial_documents.task.condition_ids) == expected
    assert narrower["baseline"].params["trials"] == expected[:4]


def test_set_family_row_retains_its_authored_entry_when_it_only_adds_task_variables():
    raw = document_project(authored_documents())
    raw["documents"]["bank"] = {
        "label": "Loads", "schema": SET_DOCUMENT_SCHEMA_ID,
        "payload": {"entries": {"shift": _shift_entry(1)}},
    }
    raw["steps"]["train"]["sweep"] = {"axes": {"case": {"set": "bank"}}}
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    assert [run.axis_keys["case"] for run in plan.runs] == ["shift-0", "shift-1"]
    assert [run.set_entries for run in plan.runs] == [{"case": "shift"}, {"case": "shift"}]


def test_set_weight_defaults_to_the_family_and_can_name_one_generated_value():
    raw = document_project(authored_documents())
    raw["documents"]["bank"] = {
        "label": "Loads", "schema": SET_DOCUMENT_SCHEMA_ID,
        "payload": {"entries": {"shift": _shift_entry(1)}},
    }
    raw["steps"]["train"]["set_fields"] = {"case": {"axis": {"set": "bank"}, "sampling": {
        "kind": "weighted", "weights": {"shift": 3, "shift-1": 5},
    }}}
    task = _decoded(raw)["train"].trial_documents.task
    selection = next(variable for variable in task.variables if variable.id == "case")
    assert isinstance(selection.value, DrawValue)
    assert selection.value.weights == (1, 3, 5)
    assert selection.set_choices.entries == (None, "shift-0", "shift-1")
