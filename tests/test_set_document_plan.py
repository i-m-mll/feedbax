"""A Set document swept by Train and Evaluate steps plans one run per entry."""

from copy import deepcopy

import pytest

from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import compile_project
from feedbax.contracts.project_sweep import NamedAxis
from feedbax.contracts.set_document import (
    SET_DOCUMENT_SCHEMA_ID, SetDocument, axis_from_set, set_from_axis, sets_offered, step_model,
)
from feedbax.plugins.composition import load_application
from tests.test_document_train_pair import authored_documents, document_project

GAIN = {
    "kind": "leaf", "name": "Gain", "type": "feedbax.Scale", "revision": None, "fields": {"scale": 0.5},
    "port_counts": {}, "attachments": {"sources": {}, "sinks": {}},
}


def _entries():
    return {
        "gain": {"label": "Gain", "differences": {
            "/interior/components/gain": {"value": GAIN},
            "/interior/wires/plant-wire/target": {"value": {"owner": "component", "component": "gain", "port": "input"}},
            "/interior/wires/gain-wire": {"value": {
                "source": {"owner": "component", "component": "gain", "port": "output"},
                "target": {"owner": "component", "component": "plant", "port": "force"}, "attachments": [],
            }},
        }},
        "scaled": {"label": "Scaled force", "differences": {"/interior/wires/plant-wire/attachments": {"value": [
            {"kind": "intervention", "id": "scale", "operation": "Scale", "fields": {"factor": 0.5}},
        ]}}},
        "heavy": {"label": "Heavy", "differences": {"/interior/components/plant/fields/mass": {"value": 2.0}}},
    }


def _project(entries=None):
    raw = document_project(authored_documents(), evaluation=True)
    raw["steps"]["plain"] = deepcopy(raw["steps"]["train"])
    evaluate = raw["steps"]["baseline"]
    evaluate["inputs"]["subject"]["step"] = "plain"
    evaluate["base"]["inline"]["trial_documents"] = {"model": {"document": "reach-model"}}
    raw["documents"]["alterations"] = {
        "schema": SET_DOCUMENT_SCHEMA_ID, "payload": {"entries": entries or _entries()},
    }
    for step in ("train", "baseline"):
        raw["steps"][step]["sweep"] = {"axes": {"alteration": {"set": "alterations"}}}
    return raw


def _plan(raw):
    components = load_application().bundle.components
    return compile_project(Project.model_validate(raw), components=components)


def _requests(plan):
    return {(run.key.step, run.key.row): run.request_id for run in plan.runs}


def test_a_set_plans_one_labelled_run_per_entry_on_train_and_evaluate():
    plan = _plan(_project())
    assert not plan.diagnostics
    labels = {(run.key.step, run.label) for run in plan.runs if run.key.step != "plain"}
    assert labels == {
        (step, f"alteration={label}") for step in ("train", "baseline") for label in ("Gain", "Scaled force", "Heavy")
    }
    before = _requests(plan)
    plain = before[("plain", "default")]
    assert plain not in {before[("baseline", row)] for row in ("alteration=gain", "alteration=heavy")}

    edited = _entries()
    edited["heavy"]["differences"]["/interior/components/plant/fields/mass"]["value"] = 3.0
    after = _requests(_plan(_project(edited)))
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {("train", "alteration=heavy"), ("baseline", "alteration=heavy")}


def test_equal_entries_share_a_request_and_each_is_listed():
    entries = _entries()
    entries["scaled"] = deepcopy(entries["heavy"]) | {"label": "Also heavy"}
    plan = _plan(_project(entries))
    assert not plan.diagnostics
    requests = _requests(plan)
    for step in ("train", "baseline"):
        assert requests[step, "alteration=scaled"] == requests[step, "alteration=heavy"]
        assert requests[step, "alteration=gain"] != requests[step, "alteration=heavy"]


def test_an_inline_axis_and_its_set_plan_the_same_requests():
    raw = _project()
    project = Project.model_validate(raw)
    document = SetDocument.model_validate(raw["documents"]["alterations"]["payload"])
    model = step_model(project, "train")
    assert model is not None and model[0] == "reach-model"
    axis = axis_from_set(document, *model)
    assert set_from_axis(axis, "reach-model") == document
    inline = deepcopy(raw)
    for step in ("train", "baseline"):
        inline["steps"][step]["sweep"] = {"axes": {"alteration": axis.model_dump(mode="json", exclude_unset=True)}}
    assert isinstance(Project.model_validate(inline).steps["train"].sweep.axes["alteration"], NamedAxis)
    assert _requests(_plan(inline)) == _requests(_plan(raw))


@pytest.mark.parametrize("pointer", ["/interior/components/missing/fields/mass", "/interior/wires/plant-wire/gone/x"])
def test_a_set_that_does_not_fit_is_a_diagnostic_on_the_axis(pointer):
    entries = _entries()
    entries["heavy"]["differences"] = {pointer: {"value": 2.0}}
    raw = _project(entries)
    plan = _plan(raw)
    problems = {(problem.step, problem.path) for problem in plan.diagnostics}
    assert problems == {(step, "sweep.axes.alteration.set") for step in ("train", "baseline")}
    assert all("'heavy'" in problem.detail and pointer in problem.detail for problem in plan.diagnostics)
    assert not any(run.key.step in ("train", "baseline") for run in plan.runs)
    raw["documents"]["fits"] = {"schema": SET_DOCUMENT_SCHEMA_ID, "payload": {"entries": {"heavy": _entries()["heavy"]}}}
    assert sets_offered(Project.model_validate(raw), "train") == ("fits",)
