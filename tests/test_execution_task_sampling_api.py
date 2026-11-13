from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from feedbax.web.app import create_app
from tests.test_document_task_parity import _model_payload, _pair_model, example
from feedbax.expression_builtins import register_builtins
from feedbax.expression_registry import FunctionRegistry


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
        "model_name": "reach-model",
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
    assert result["schema_version"] == "feedbax.execution.sampled_task_trials.v2"
    assert result["task"] == kind
    assert result["n_steps"] == 80
    assert len(result["trials"]) == 4
    trial = result["trials"][0]
    assert len(trial["inputs"]["target_pos"]) == result["n_steps"]
    assert list(trial["inits"].values()) == [trial["variables"]["start"]]
    assert trial["timeline"][0]["start"] == 0
    assert trial["timeline"][-1]["end"] == result["n_steps"]


def test_sample_task_trials_reports_model_pairing_problem(client, expression_functions):
    payload = authored_preview("simple", expression_functions)
    payload["input_overrides"] = {"reach_target": "target_vel"}
    response = client.post("/api/execution/task-trials/sample", json=payload)
    assert response.status_code == 422
    (problem,) = response.json()["detail"]
    assert problem["path"][:2] == ["model", "inputs"]
    assert problem["code"] == "type_mismatch"
