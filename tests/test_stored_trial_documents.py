"""The stored Task and Objective are keyed, ordered, and lower to their execution values."""

from copy import deepcopy
from typing import get_origin

import pytest
from pydantic import ValidationError

from feedbax.contracts.expression_values import CallValue, LiteralValue, NameValue, UnresolvedValue
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.objective_document import ObjectiveDocument, StoredObjectiveDocument
from feedbax.contracts.stored_members import StoredAddressedMembers, StoredMembers
from feedbax.contracts.task_document import (
    BLANK_STORED_TASK, StoredTaskAdditions, StoredTaskDocument, StoredTaskDraftError, TaskDocument, Until,
)
from feedbax.contracts.time_units import TimeValue
from feedbax.contracts.window import StoredWindow, WindowInterval, WindowMoment
from feedbax.examples import load_shipped_projects
from feedbax.planner.project_plan import compile_project


def _task():
    start = ModelAddress.at_state((), "body", "position").to_authored_text()
    return {
        "id": "root-draw-key",
        "variables": {
            "z": {"label": "z variable", "type": {"base": "real"}, "value": 2, "count": 2},
            "a": {"label": "a variable", "type": {"base": "real"}, "value": "z + 1"},
        },
        "epochs": {
            "end": {"label": "Last part", "end": {"kind": "rest"}},
            "first": {"label": "First part", "end": {"kind": "until", "condition": False, "limit": "20 st"}},
        },
        "signals": {
            "gate": {
                "label": "Gate",
                "type": {"base": "boolean"},
                "intervention": "wire.drive#pulse",
                "absent": True,
                "cells": {
                    "z": {"first_epoch": "end", "last_epoch": "end", "value": True},
                    "a": {"first_epoch": "first", "last_epoch": "first", "value": False},
                },
            },
        },
        "repeats": {
            "repeat-block": {"first_epoch": "end", "last_epoch": "first", "count": 2},
        },
        "links": {start: "z"},
        "validation_trials": {
            "right": {"variables": {"z": 3}},
            "left": {"variables": {"z": 2}},
        },
    }


def test_task_lowers_member_order_and_draw_root_without_using_trial_keys():
    payload = _task()
    task = StoredTaskDocument.model_validate(payload)
    lowered = task.lower()
    assert isinstance(lowered, TaskDocument)
    assert lowered.id == "root-draw-key"
    assert [(value.id, value.name) for value in lowered.variables] == [("z", "z variable"), ("a", "a variable")]
    assert tuple(epoch.id for epoch in lowered.epochs) == ("end", "first")
    assert tuple(cell.id for cell in lowered.signals[0].cells) == ("z", "a")
    assert lowered.repeats[0].id == "repeat-block"
    assert [(link.model_field, link.variable) for link in lowered.links] == [
        (ModelAddress.from_authored_text(address).to_text(), key) for address, key in payload["links"].items()
    ]
    assert lowered.variables[0].value == lowered.variables[0].count == LiteralValue(value=2)
    assert isinstance(lowered.variables[1].value, CallValue)
    assert lowered.variables[1].value.args[0] == NameValue(reference="z")
    assert isinstance(until := lowered.epochs[1].end, Until)
    assert until.limit == TimeValue(value=20, unit="st")
    assert lowered.signals[0].absent == LiteralValue(value=True)
    assert lowered.signals[0].intervention == ModelAddress.from_authored_text("wire.drive#pulse").to_text()

    authored = StoredTaskDocument.from_lowered(lowered, trial_keys=("right", "left"))
    assert authored.model_dump(mode="json", exclude_unset=True) == payload
    generated = StoredTaskDocument.from_lowered(lowered)
    assert tuple(generated.validation_trials) == ("trial-1", "trial-2")
    assert generated.lower().condition_ids == lowered.condition_ids


def test_sampled_task_uses_the_empty_storage_default():
    assert BLANK_STORED_TASK.validation_trials == {}
    payload = _task()
    payload["validation_trials"] = {}
    task = StoredTaskDocument.model_validate(payload).lower()
    assert task.condition_ids == ()
    assert len(task.validation_trials) == 1
    assert StoredTaskDocument.from_lowered(task).validation_trials == {}
    assert StoredTaskDocument.from_lowered(task, trial_keys=()).validation_trials == {}


def test_objective_lowers_keyed_terms_and_keeps_optional_labels():
    payload = {
        "id": "objective",
        "terms": {
            "z": {"measure": 3},
            "a": {"label": "Terminal error", "measure": "sqnorm(outputs.effector.pos)", "enabled": False},
        },
    }
    objective = StoredObjectiveDocument.model_validate(payload).lower()
    assert isinstance(objective, ObjectiveDocument)
    assert [(term.id, term.name) for term in objective.terms] == [("z", None), ("a", "Terminal error")]
    assert objective.terms[0].weight == LiteralValue(value=1)
    measure = objective.terms[1].measure
    assert isinstance(measure, CallValue)
    assert measure.args[0] == NameValue(reference=ModelAddress.from_authored_text("outputs.effector.pos").to_text())
    assert StoredObjectiveDocument.from_lowered(objective).model_dump(mode="json", exclude_unset=True) == payload


@pytest.mark.parametrize("field", ["variables", "epochs", "signals", "validation_trials", "links", "repeats"])
def test_task_rejects_runtime_arrays_in_keyed_collections(field):
    payload = _task()
    payload[field] = list(payload[field].values())
    with pytest.raises(ValidationError, match=field):
        StoredTaskDocument.model_validate(payload)


def test_keyed_members_require_stable_keys_and_exclude_inline_ids():
    payload = _task()
    signal = payload["signals"]["gate"]
    payload["signals"] = {"1starts-with-number": signal}
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        StoredTaskDocument.model_validate(payload)

    payload["signals"] = {"z": signal}
    with pytest.raises(ValidationError, match="keys must be disjoint: z"):
        StoredTaskDocument.model_validate(payload)

    payload["signals"] = {"gate": signal}
    signal["cells"]["z"]["id"] = "z"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        StoredTaskDocument.model_validate(payload)

    with pytest.raises(ValidationError, match="terms"):
        StoredObjectiveDocument.model_validate({"id": "objective", "terms": []})


def test_task_and_set_member_keys_do_not_shadow_expression_words():
    payload = _task()
    variable = payload["variables"]["z"]
    payload["variables"]["time"] = variable
    with pytest.raises(ValidationError, match="reserved in expressions: time"):
        StoredTaskDocument.model_validate(payload)
    with pytest.raises(ValidationError, match="reserved in expressions: outputs"):
        StoredTaskAdditions.model_validate({"task_variables": {"outputs": variable}})


def test_task_additions_project_even_a_map_edited_in_place():
    payload = _task()
    additions = StoredTaskAdditions()
    additions.task_variables.update(StoredTaskDocument.model_validate(payload).variables)
    additions.task_links.update(payload["links"])
    assert StoredTaskAdditions.from_lowered(additions.lower()) == additions

    assert get_origin(StoredTaskAdditions.model_fields["task_variables"].annotation) is StoredMembers
    assert get_origin(StoredTaskAdditions.model_fields["task_links"].annotation) is StoredAddressedMembers
    with pytest.raises(ValidationError, match="authored Model address|canonical"):
        StoredTaskAdditions.model_validate({
            "task_links": {f" {next(iter(payload['links']))}": "z"},
        })


def test_stored_windows_keep_literal_units_and_lower_timed_expressions():
    payload = {"spans": [
        {
            "kind": "interval", "start": {"anchor": {"kind": "trial", "edge": "start"}, "offset": "onset + 2 st"},
            "end": {"offset": "30 ms"},
        },
        {"kind": "moment", "at": {"anchor": {"kind": "epoch", "epoch": "first", "edge": "end"}, "offset": "true"}},
    ]}
    stored = StoredWindow.model_validate(payload)
    assert stored.model_dump(mode="json", exclude_unset=True)["spans"][0]["end"]["offset"] == "30 ms"
    window = stored.lower()
    interval, moment = window.spans
    assert isinstance(interval, WindowInterval) and isinstance(interval.start.offset, CallValue)
    assert interval.start.offset.args[0] == NameValue(reference="onset")
    assert interval.end.offset == TimeValue(value=0.03, unit="s")
    assert isinstance(moment, WindowMoment) and moment.at.offset == UnresolvedValue(text="true")
    assert StoredWindow.from_lowered(window).lower() == window


def test_old_expression_time_and_address_objects_are_refused_in_stored_tasks():
    for field, old in [
        (("variables", "z", "value"), {"kind": "literal", "value": 2}),
        (("epochs", "first", "end", "limit"), {"value": 20, "unit": "st"}),
        (("links",), {ModelAddress.at_state((), "body", "position").to_text(): "z"}),
    ]:
        payload = _task()
        holder = payload
        for key in field[:-1]:
            holder = holder[key]
        holder[field[-1]] = old
        with pytest.raises(ValidationError, match=field[-1]):
            StoredTaskDocument.model_validate(payload)


def test_invalid_until_limit_stays_in_the_saved_task_until_lowering():
    payload = _task()
    payload["epochs"]["first"]["end"]["limit"] = "wait +"
    task = StoredTaskDocument.model_validate(payload)
    assert task.model_dump(mode="json", exclude_unset=True) == payload

    with pytest.raises(StoredTaskDraftError, match="literal time") as error:
        task.lower()
    assert error.value.path == ("epochs", "first", "end", "limit")
    assert error.value.text == "wait +"


def test_label_changes_keep_the_shipped_train_request_identity():
    project = load_shipped_projects()["point-mass-reaching"]
    before = compile_project(project)
    assert not before.diagnostics and len(before.runs) == 3
    assert all(row.admission_problem is None for row in before.runs)
    edited = deepcopy(project)
    payload = edited.documents["simple-reaches-task"].payload
    for family, key in (("variables", "start"), ("epochs", "move"), ("signals", "target_pos")):
        payload[family][key]["label"] += " renamed"

    after = compile_project(edited)
    assert not after.diagnostics
    assert {row.key: row.request_id for row in after.runs} == {row.key: row.request_id for row in before.runs}
