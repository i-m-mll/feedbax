from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from feedbax.web.app import create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_sample_task_trials_is_deterministic_for_simple_reaches(client: TestClient) -> None:
    request = {
        "task_spec": {
            "type": "SimpleReaches",
            "params": {
                "n_steps": 12,
                "workspace": [[-0.5, -0.25], [0.5, 0.25]],
                "eval_reach_length": 0.4,
            },
        },
        "seed": 17,
        "count": 4,
    }

    first = client.post("/api/execution/task-trials/sample", json=request)
    second = client.post("/api/execution/task-trials/sample", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    payload = first.json()
    assert payload["schema_version"] == "feedbax.execution.sampled_task_trials.v1"
    assert payload["task_type"] == "SimpleReaches"
    assert len(payload["trials"]) == 4
    assert payload["trials"][0]["n_steps"] == 12
    assert len(payload["trials"][0]["start"]) == 2
    assert len(payload["trials"][0]["goal"]) == 2


def test_sample_task_trials_accepts_reaching_task_registry_alias(client: TestClient) -> None:
    response = client.post(
        "/api/execution/task-trials/sample",
        json={
            "task_spec": {
                "type": "feedbax.task.ReachingTask",
                "params": {"n_steps": 6},
            },
            "seed": 0,
            "count": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["task_type"] == "SimpleReaches"


def test_sample_task_trials_supports_delayed_center_out_timeline_cues(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/execution/task-trials/sample",
        json={
            "task_spec": {
                "type": "DelayedReaches",
                "params": {
                    "preset": "delayed_center_out",
                    "n_control_stages": 8,
                    "workspace": [[-1.0, -1.0], [1.0, 1.0]],
                    "epoch_len_ranges": [[2, 2]],
                    "p_catch_trial": 0.0,
                },
            },
            "seed": 4,
            "count": 2,
        },
    )

    assert response.status_code == 200
    trial = response.json()["trials"][0]
    assert trial["n_steps"] == 9
    assert {"label": "prep", "step": 0, "kind": "epoch"} in trial["timeline"]
    assert {"label": "go_cue", "step": 2, "kind": "event"} in trial["timeline"]


def test_sample_task_trials_rejects_unsupported_task_type(client: TestClient) -> None:
    response = client.post(
        "/api/execution/task-trials/sample",
        json={"task_spec": {"type": "Stabilization", "params": {}}, "seed": 0, "count": 1},
    )

    assert response.status_code == 422
    assert "not supported" in response.json()["detail"]
