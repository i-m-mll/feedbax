from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from feedbax.web.app import create_app
from tests.test_document_task_parity import _model_payload, _pair_model, example
from feedbax.contracts.expression_values import LENGTH, CallValue, NameValue, ValueBase, ValueType
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.objective_document import ObjectiveDocument, ObjectiveTerm
from feedbax.expression_builtins import register_builtins
from feedbax.contracts.expression_registry import FunctionRegistry


@pytest.fixture
def expression_functions():
    registry = FunctionRegistry()
    register_builtins(registry)
    registry.seal()
    return registry


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def authored_preview(kind, expression_functions):
    interface, _ = _pair_model()
    document, _ = example(kind, expression_functions, interface=interface)
    return {
        "model": _model_payload(),
        "task": document.model_dump(mode="json"),
        "input_bindings": {"task-input": "target_pos"},
        "seed": 17,
        "count": 4,
    }


@pytest.mark.parametrize("kind", ["simple", "delayed", "stabilization"])
def test_sample_task_trials_returns_the_authored_trial(client, expression_functions, kind):
    payload = authored_preview(kind, expression_functions)
    first = client.post("/api/execution/task-trials/sample", json=payload)
    second = client.post("/api/execution/task-trials/sample", json=payload)
    assert first.status_code == second.status_code == 200, first.text
    result = first.json()
    assert result == second.json()
    assert result["task"] == kind
    assert result["n_steps"] == 80
    assert len(result["drawn"]["trials"]) == 4
    trial = result["drawn"]["trials"][0]
    assert len(trial["inputs"]["target_pos"]) == result["n_steps"]
    assert list(trial["inits"].values()) == [trial["variables"]["start"]]
    assert trial["timeline"][0]["start"] == 0
    assert trial["timeline"][-1]["end"] == result["n_steps"]


# Catches an aggregate that counts every trial as in an epoch whose end is drawn (or none), and percentiles taken over
# steps where the signal is absent.
def test_aggregate_of_delayed_reaches_shares_the_drawn_epoch(client, expression_functions):
    payload = authored_preview("delayed", expression_functions) | {"count": 64, "seed": 0, "aggregate": True}
    response = client.post("/api/execution/task-trials/sample", json=payload)
    assert response.status_code == 200, response.text
    drawn = response.json()["drawn"]
    assert drawn["kind"] == "aggregate"
    hold = next(epoch["share"] for epoch in drawn["epochs"] if epoch["epoch"] == "hold")
    assert hold[:5] == [1.0] * 5
    assert all(0 < share < 1 for share in hold[6:14])
    assert hold[14:] == [0.0] * (len(hold) - 14)
    target = drawn["signals"]["target_pos"]
    absent = [step for step, share in enumerate(target["presence"]) if share == 0]
    assert absent and all(target["percentiles"][step] is None for step in absent)
    present = target["percentiles"][-1]
    assert len(present) == 2 and all(low <= high for component in present for low, high in zip(component, component[1:]))
    catch = drawn["variables"]["catch"]
    assert (catch["least"], catch["greatest"]) == (False, True)


def test_a_binding_problem_does_not_hide_the_trial(client, expression_functions):
    payload = authored_preview("simple", expression_functions)
    payload["input_bindings"] = {"task-input": "target_vel"}
    response = client.post("/api/execution/task-trials/sample", json=payload)
    assert response.status_code == 200, response.text
    trial = response.json()["drawn"]["trials"][0]
    assert trial["inputs"]["target_pos"]


def test_a_link_to_a_missing_model_field_does_not_hide_the_trial(client, expression_functions):
    payload = authored_preview("delayed", expression_functions)
    missing = ModelAddress.model_validate({
        "composites": [],
        "element": {"kind": "field", "parent": {"kind": "component", "id": "missing"}, "path": ["position"]},
    }).to_text()
    payload["task"]["links"].append({"model_field": missing, "variable": "start"})
    response = client.post("/api/execution/task-trials/sample", json=payload)
    assert response.status_code == 200, response.text
    trial = response.json()["drawn"]["trials"][0]
    assert list(trial["inits"].values()) == [trial["variables"]["start"]]
    assert missing not in trial["inits"]


def test_preview_marks_the_model_period_without_an_http_error(client, expression_functions):
    payload = authored_preview("delayed", expression_functions)
    del payload["model"]["step_seconds"]
    response = client.post("/api/execution/task-trials/sample", json=payload)
    assert response.status_code == 200, response.text
    drawn = response.json()["drawn"]
    assert drawn["kind"] == "problems"
    assert drawn["problems"][0]["path"] == ["model", "step_seconds"]


def test_delayed_reaches_term_applies_where_move_and_target_overlap(client, expression_functions):
    payload = authored_preview("delayed", expression_functions)
    measure = CallValue(function="sqnorm", args=(NameValue(reference="target_pos"),))
    measure_type = ValueType(base=ValueBase.REAL, dimensions=LENGTH.raised(2))
    payload["objective"] = ObjectiveDocument(id="reach", terms=(
        ObjectiveTerm(id="target", measure=measure, measure_type=measure_type, epochs=("move",)),
        ObjectiveTerm(id="absent", measure=measure, measure_type=measure_type, epochs=("hold",)),
    )).model_dump(mode="json")
    response = client.post("/api/execution/task-trials/sample", json=payload)
    assert response.status_code == 200, response.text
    trial = response.json()["drawn"]["trials"][0]
    (move,) = [cue for cue in trial["timeline"] if cue["epoch"] == "move"]
    expected = [present and move["start"] <= step < move["end"]
                for step, present in enumerate(trial["present"]["target_pos"])]
    assert any(expected) and not all(expected)
    np.testing.assert_array_equal(trial["terms"]["target"]["steps"], expected)
    assert trial["terms"]["target"]["covered"]
    assert not any(trial["terms"]["absent"]["steps"])
    assert not trial["terms"]["absent"]["covered"]


def test_nominal_trial_uses_representative_values_and_the_same_task_timeline(client, expression_functions):
    """The nominal trial must keep its values and bars when the sampler controls change."""
    payload = authored_preview("delayed", expression_functions) | {"nominal": True}
    response = client.post("/api/execution/task-trials/sample", json=payload)
    changed_controls = client.post("/api/execution/task-trials/sample", json=payload | {"seed": 2, "count": 10})
    assert response.status_code == changed_controls.status_code == 200, response.text
    assert response.json() == changed_controls.json()
    (trial,) = response.json()["drawn"]["trials"]
    assert trial["id"] == "nominal"
    assert trial["variables"]["start"] == pytest.approx([0.05, 0.075])
    assert trial["variables"]["catch"] is False
    assert [(cue["epoch"], cue["start"], cue["end"]) for cue in trial["timeline"]] == [
        ("hold", 0, 10), ("cue", 10, 24), ("move", 24, 80),
    ]
    assert trial["present"]["target_pos"] == [False] * 10 + [True] * 70


def test_term_preview_distinguishes_compile_problem_from_empty_window(client, expression_functions):
    """A term with a bad measure type must be marked even if its shelf row has no bars."""
    payload = authored_preview("delayed", expression_functions) | {"count": 2}
    measure = CallValue(function="sqnorm", args=(NameValue(reference="target_pos"),))
    payload["objective"] = ObjectiveDocument(id="reach", terms=(
        ObjectiveTerm(id="bad-measure", measure=measure, epochs=("move",)),
        ObjectiveTerm(
            id="valid-without-coverage", measure=measure, epochs=("hold",),
            measure_type=ValueType(base=ValueBase.REAL, dimensions=LENGTH.raised(2)),
        ),
    )).model_dump(mode="json")

    for controls in ({}, {"nominal": True}, {"aggregate": True}):
        response = client.post("/api/execution/task-trials/sample", json=payload | controls)
        assert response.status_code == 200, response.text
        drawn = response.json()["drawn"]
        records = [drawn] if drawn["kind"] == "aggregate" else drawn["trials"]
        for record in records:
            broken = record["terms"]["bad-measure"]
            absent = record["terms"]["valid-without-coverage"]
            assert not any(broken["steps"]) and not any(absent["steps"])
            assert not broken["covered"] and not absent["covered"]
            (problem,) = broken["problems"]
            assert problem["code"] == "unit_mismatch"
            assert problem["path"][:4] == ["objective", "terms", "bad-measure", "measure"]
            assert absent["problems"] == []
