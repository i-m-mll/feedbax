"""A Set document swept by Train and Evaluate steps plans one run per entry."""

from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from feedbax.application.project_writer import dump_text_project
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, parse_text_project_files
from feedbax.planner.project_plan import RunKey, compile_project
from feedbax.contracts.project_sweep import NamedAxis
from feedbax.contracts.set_document import (
    SET_DOCUMENT_SCHEMA_ID, SetDocument, axis_from_set, set_from_axis, sets_offered, step_model,
)
from feedbax.application.composition import load_application
from tests.test_document_train_pair import authored_documents, document_project

GAIN = {
    "kind": "leaf", "label": "Gain", "type": "feedbax.Scale", "revision": None, "fields": {"scale": 0.5},
    "port_counts": {}, "attachments": {"sources": {}, "sinks": {}},
}


def _entries():
    return {
        "gain": {"label": "Gain", "differences": {
            "/interior/components/gain": {"value": GAIN},
            "/interior/wires/plant-wire/target": {"value": {"owner": "component", "component": "gain", "port": "input"}},
            "/interior/wires/gain-wire": {"value": {
                "source": {"owner": "component", "component": "gain", "port": "output"},
                "target": {"owner": "component", "component": "plant", "port": "force"},
            }},
        }},
        "scaled": {"label": "Scaled force", "differences": {"/interior/wires/plant-wire/attachments": {"value": {
            "scale": {"kind": "intervention", "operation": "Scale", "operands": {"factor": {"kind": "value", "value": 0.5}}},
        }}}},
        "heavy": {"label": "Heavy", "differences": {"/interior/components/plant/fields/mass": {"value": 2.0}}},
    }


def _project(entries=None):
    raw = document_project(authored_documents(), evaluation=True)
    raw["steps"]["plain"] = deepcopy(raw["steps"]["train"])
    evaluate = raw["steps"]["baseline"]
    evaluate["inputs"]["subject"]["step"] = "plain"
    evaluate["base"]["inline"]["trial_documents"] = {"model": {"document": "reach-model"}}
    raw["documents"]["alterations"] = {
        "label": "Alterations",
        "schema": SET_DOCUMENT_SCHEMA_ID, "payload": {"entries": entries or _entries()},
    }
    for step in ("train", "baseline"):
        raw["steps"][step]["sweep"] = {"axes": {"alteration": {"set": "alterations"}}}
    return raw


def _plan(raw, *, source_runs=()):
    components = load_application().bundle.components
    return compile_project(Project.model_validate(raw), components=components, source_runs=source_runs)


def _requests(plan):
    return {(run.key.step, run.key.row): run.request_id for run in plan.runs}


def test_a_set_plans_one_labelled_run_per_entry_on_train_and_evaluate():
    plan = _plan(_project())
    assert not plan.diagnostics
    labels = {(run.key.step, run.label) for run in plan.runs if run.key.step != "plain"}
    assert labels == {
        (step, label) for step in ("train", "baseline") for label in ("Gain", "Scaled force", "Heavy")
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


def test_a_saved_set_has_names_for_every_authored_entry():
    with pytest.raises(ValidationError, match="label"):
        SetDocument.model_validate({"entries": {"unlabelled": {}}})

    inline = NamedAxis.model_validate({"values": {"removed": None, "first": {}, "second": {"label": "Heavy"}}})
    saved = set_from_axis(inline, "reach-model")
    assert [entry.label for entry in saved.entries.values()] == ["Value 1", "Heavy"]


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
    selected = RunKey("default", "train", "alteration=heavy")
    plan = _plan(raw, source_runs=(selected,))
    assert selected not in plan.selected_definitions
    problems = {(problem.step, problem.path) for problem in plan.diagnostics}
    assert problems == {(step, "sweep.axes.alteration.set") for step in ("train", "baseline")}
    assert all("'heavy'" in problem.detail and pointer in problem.detail for problem in plan.diagnostics)
    assert not any(run.key.step in ("train", "baseline") for run in plan.runs)
    raw["documents"]["fits"] = {"label": "Fits", "schema": SET_DOCUMENT_SCHEMA_ID, "payload": {"entries": {"heavy": _entries()["heavy"]}}}
    assert sets_offered(Project.model_validate(raw), "train") == ("alterations", "fits")


def test_sets_and_based_models_do_not_change_the_bytes_of_their_base(tmp_path):
    attachment = {"kind": "intervention", "operation": "Scale",
                  "operands": {"factor": {"kind": "value", "value": 0.5}}}
    attachment_path = "/interior/wires/plant-wire/attachments/scale"
    differences = {
        attachment_path: {"value": attachment},
        "/interior/components/plant/fields/mass": {"value": 2},
    }
    raw = _project({"scaled": {"label": "Scaled", "differences": differences}})
    model = raw["documents"]["reach-model"]
    model["payload"]["interior"]["components"]["plant"]["fields"] = {}
    model["payload"]["interior"]["wires"]["plant-wire"]["attachments"] = {}
    configuration = ProjectConfiguration(project="study", store="results")
    base_files = dump_text_project(Project.model_validate({"documents": {"reach-model": model}}), configuration)

    raw["documents"]["derived"] = {
        "schema": model["schema"], "label": "Derived", "base": {"document": "reach-model"},
        "differences": differences,
    }
    train = raw["steps"]["plain"]
    raw["steps"]["derived"] = deepcopy(train)
    raw["steps"]["derived"]["base"]["inline"]["trial_documents"]["model"] = {"document": "derived"}
    raw["documents"]["force-set"] = {
        "label": "Force", "schema": SET_DOCUMENT_SCHEMA_ID,
        "payload": {"entries": {"scaled": {"label": "Scaled", "differences": {
            attachment_path: differences[attachment_path],
        }}}},
    }
    raw["steps"]["trial-set"] = deepcopy(train) | {
        "set_fields": {"alteration": {"axis": {"set": "force-set"}, "sampling": {"kind": "cooccurring"}}},
    }
    train["sweep"] = {"axes": {}}
    raw["sweep"] = None

    files = dump_text_project(Project.model_validate(raw), configuration)
    saved_base = files["documents/reach-model.json"]
    assert saved_base == base_files["documents/reach-model.json"]
    graph = json.loads(saved_base)["payload"]["interior"]
    assert "attachments" not in graph["wires"]["plant-wire"]
    assert "fields" not in graph["components"]["plant"]
    assert "sweep" not in json.loads(files["steps/plain.json"])
    assert not {"sweep", "variants"} & json.loads(files["study.feedbax"]).keys()

    saved = parse_text_project_files(tmp_path, files).project
    after = _plan(saved.model_dump(mode="json", by_alias=True, exclude_unset=True))
    before = _plan(raw)
    for plan in (before, after):
        assert not plan.diagnostics, plan.diagnostics
        assert {run.key.step for run in plan.runs} == {"train", "baseline", "plain", "derived", "trial-set"}, {row.key.step: row.problems for row in plan.rows}
        assert all(run.request_id and run.admission_problem is None for run in plan.runs)
    assert _requests(after) == _requests(before)

    key = RunKey("default", "train", "alteration=scaled")
    selected = compile_project(saved, source_runs=(key,))
    assert _requests(selected) == _requests(before)
    definition = selected.selected_definitions[key]
    model_path = "/payload/trial_documents/model"
    source = definition.sources[model_path + attachment_path + "/operands/factor/value"]
    assert (source.kind, source.name, source.step, source.axis, source.value) == (
        "document", "alterations", "train", "alteration", "scaled",
    )
    assert source.path == (
        "/entries/scaled/differences/~1interior~1wires~1plant-wire~1attachments~1scale"
        "/value/operands/factor/value"
    )
    inherited = definition.sources[model_path + "/interior/wires/plant-wire/source/port"]
    assert (inherited.kind, inherited.name, inherited.path) == (
        "document", "reach-model", "/interior/wires/plant-wire/source/port",
    )
