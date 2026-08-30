"""Studio HTTP operations expose, authenticate, and project durable reservations."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from feedbax.web.api import orchestration as orchestration_api
from feedbax.web.orchestration import controller as studio_controller_module
from tests.test_durable_controller import _gcp_plan


def test_launch_endpoint_stops_at_an_inert_named_reservation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(studio_controller_module, "studio_controller", None)
    invocation, plan = _gcp_plan()
    app = FastAPI()
    app.include_router(orchestration_api.router, prefix="/api/orchestration")
    client = TestClient(app)

    response = client.post(
        "/api/orchestration/launch",
        json={
            "invocation": invocation.model_dump(mode="json"),
            "backend_plan": plan.model_dump(mode="json"),
            "project": "inert-project",
            "zone": "northamerica-northeast1-a",
            "machine_type": "n1-standard-4",
            "preemptible": True,
            "worker_port": 8765,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "awaiting_authentication"
    assert payload["intent_id"]
    assert payload["reservation_id"]
    events = studio_controller_module.get_studio_controller().controller.store.read_all()
    assert [event.event_type for event in events][-1] == "effect_reservation_created"
    assert "external_effect_dispatched" not in {event.event_type for event in events}
    status = client.get(
        "/api/orchestration/status",
        params={"intent_id": payload["intent_id"], "refresh": "false"},
    )
    assert status.status_code == 200
    assert status.json()["reservation_id"] == payload["reservation_id"]
    assert status.json()["status"] == "awaiting_authentication"


def test_authentication_rejects_evidence_for_an_unknown_reservation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(studio_controller_module, "studio_controller", None)
    app = FastAPI()
    app.include_router(orchestration_api.router, prefix="/api/orchestration")

    response = TestClient(app).post(
        "/api/orchestration/intents/missing/reservations/missing/authenticate",
        json={
            "operator_identity": "operator",
            "authentication_id": "auth-missing",
            "confirmation_token": "launch-billable-gcp-worker",
            "max_cost_usd": 1.0,
        },
    )

    assert response.status_code == 404
