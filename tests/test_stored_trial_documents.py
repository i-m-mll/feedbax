"""The stored Task and Objective are keyed, ordered, and lower to their execution values."""

from typing import get_origin

import pytest
from pydantic import ValidationError

from feedbax.contracts.expression_values import LiteralValue
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.objective_document import ObjectiveDocument, StoredObjectiveDocument
from feedbax.contracts.stored_members import StoredAddressedMembers, StoredMembers
from feedbax.contracts.task_document import StoredTaskAdditions, StoredTaskDocument, TaskDocument


def _task():
    start = ModelAddress.at_state((), "body", "position").to_text()
    return {
        "id": "root-draw-key",
        "variables": {
            "z": {"label": "z variable", "type": {"base": "real"}, "value": {"kind": "literal", "value": 2.0}},
            "a": {"label": "a variable", "type": {"base": "real"}, "value": {"kind": "literal", "value": 3.0}},
        },
        "epochs": {
            "end": {"label": "Last part", "end": {"kind": "rest"}},
            "first": {"label": "First part", "end": {"kind": "rest"}},
        },
        "signals": {
            "gate": {
                "label": "Gate",
                "type": {"base": "boolean"},
                "cells": {
                    "z": {"first_epoch": "end", "last_epoch": "end", "value": {"kind": "literal", "value": True}},
                    "a": {"first_epoch": "first", "last_epoch": "first", "value": {"kind": "literal", "value": False}},
                },
            },
        },
        "repeats": {
            "repeat-block": {"first_epoch": "end", "last_epoch": "first", "count": {"kind": "literal", "value": 2}},
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
    assert [(link.model_field, link.variable) for link in lowered.links] == list(payload["links"].items())
    assert type(lowered.variables[0].value.value) is float

    authored = StoredTaskDocument.from_lowered(lowered, trial_keys=("right", "left"))
    assert authored.model_dump(mode="json", exclude_unset=True) == payload
    generated = StoredTaskDocument.from_lowered(lowered)
    assert tuple(generated.validation_trials) == ("trial-1", "trial-2")
    assert generated.lower().condition_ids == lowered.condition_ids


def test_objective_lowers_keyed_terms_and_keeps_optional_labels():
    payload = {
        "id": "objective",
        "terms": {
            "z": {"measure": {"kind": "literal", "value": 3}},
            "a": {"label": "Terminal error", "measure": {"kind": "literal", "value": 2.0}, "enabled": False},
        },
    }
    objective = StoredObjectiveDocument.model_validate(payload).lower()
    assert isinstance(objective, ObjectiveDocument)
    assert [(term.id, term.name) for term in objective.terms] == [("z", None), ("a", "Terminal error")]
    assert objective.terms[0].weight == LiteralValue(value=1)
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


def test_task_additions_project_even_a_map_edited_in_place():
    payload = _task()
    additions = StoredTaskAdditions()
    additions.task_variables.update(StoredTaskDocument.model_validate(payload).variables)
    additions.task_links.update(payload["links"])
    assert StoredTaskAdditions.from_lowered(additions.lower()) == additions

    assert get_origin(StoredTaskAdditions.model_fields["task_variables"].annotation) is StoredMembers
    assert get_origin(StoredTaskAdditions.model_fields["task_links"].annotation) is StoredAddressedMembers
    with pytest.raises(ValidationError, match="Model address text must be canonical"):
        StoredTaskAdditions.model_validate({
            "task_links": {f" {next(iter(payload['links']))}": "z"},
        })
