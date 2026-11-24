from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from feedbax.web.app import create_app
from tests.test_document_task_parity import _model_payload, _pair_model, example
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
        "input_overrides": {"reach_target": "target_pos"},
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


def test_sample_task_trials_reports_model_pairing_problem(client, expression_functions):
    payload = authored_preview("simple", expression_functions)
    payload["input_overrides"] = {"reach_target": "target_vel"}
    response = client.post("/api/execution/task-trials/sample", json=payload)
    assert response.status_code == 422
    (problem,) = response.json()["detail"]
    assert problem["path"][:2] == ["model", "inputs"]
    assert problem["code"] == "type_mismatch"
