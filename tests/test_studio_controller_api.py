"""Studio HTTP operations expose, authenticate, and project durable reservations."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from feedbax.orchestration.controller import OrphanHandlingPolicy, ProviderResourceObservation
from feedbax.web.api import orchestration as orchestration_api
from feedbax.web.orchestration import controller as studio_controller_module
from tests.test_durable_controller import _Adapter, _gcp_plan


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
    assert status.json()["orphaned_resources"] == []


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


def test_retry_endpoint_admits_same_effect_without_dispatch(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(studio_controller_module, "studio_controller", None)
    invocation, plan = _gcp_plan()
    studio = studio_controller_module.get_studio_controller()
    intent, reservation = studio.reserve_cloud_launch(
        invocation,
        plan,
        parameters={
            **plan.configuration,
            "instance_name": "feedbax-worker-retry",
            "startup_timeout_seconds": 120,
            "worker_health_timeout_seconds": 120,
            "poll_interval_seconds": 2,
        },
        ttl_seconds=300,
        secrets={},
    )
    studio.controller.authenticate_reservation(
        intent.intent_id,
        reservation.reservation_id,
        operator_identity="operator",
        authentication_id="auth-first",
    )
    adapter = _Adapter(backend_id="gcp")
    attempt = asyncio.run(
        studio.controller.dispatch(intent.intent_id, reservation.reservation_id, adapter)
    )
    studio.controller.observe_attempt_terminal(
        intent.intent_id,
        attempt.attempt_id,
        status="failed",
        exit_classification="worker_exit",
        reservation_id=reservation.reservation_id,
    )
    app = FastAPI()
    app.include_router(orchestration_api.router, prefix="/api/orchestration")

    response = TestClient(app).post(
        f"/api/orchestration/intents/{intent.intent_id}/reservations/"
        f"{reservation.reservation_id}/retry",
        json={"invocation": invocation.model_dump(mode="json")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "awaiting_authentication"
    assert payload["external_effect_key"] == reservation.external_effect_key
    retry_projection = studio.status(intent.intent_id).reservations[payload["reservation_id"]]
    assert retry_projection.status == "inert"
    assert retry_projection.reservation.requires_authentication is True
    events = studio.controller.store.read_all()
    assert [event.event_type for event in events].count("retry_admitted") == 1
    assert [event.event_type for event in events].count("external_effect_dispatched") == 1


def test_status_endpoint_exposes_durable_orphan_handling(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(studio_controller_module, "studio_controller", None)
    invocation, plan = _gcp_plan()
    studio = studio_controller_module.get_studio_controller()
    intent, reservation = studio.reserve_cloud_launch(
        invocation,
        plan,
        parameters={
            **plan.configuration,
            "instance_name": "feedbax-worker-managed",
            "startup_timeout_seconds": 120,
            "worker_health_timeout_seconds": 120,
            "poll_interval_seconds": 2,
        },
        ttl_seconds=300,
        secrets={},
    )
    studio.controller.authenticate_reservation(
        intent.intent_id,
        reservation.reservation_id,
        operator_identity="operator",
        authentication_id="auth-managed",
    )
    adapter = _Adapter(backend_id="gcp")
    asyncio.run(studio.controller.dispatch(intent.intent_id, reservation.reservation_id, adapter))
    adapter.inventory_resources = (
        ProviderResourceObservation(
            provider_resource_handle="feedbax-worker-managed",
            external_effect_key=reservation.external_effect_key,
            status="RUNNING",
        ),
        ProviderResourceObservation(
            provider_resource_handle="feedbax-worker-orphan",
            external_effect_key="feedbax-worker-orphan",
            status="RUNNING",
        ),
    )
    asyncio.run(
        studio.controller.observe_provider_inventory(
            intent.intent_id,
            adapter,
            backend_id="gcp",
            scope={"project": "inert-project", "zone": "northamerica-northeast1-a"},
            reservation_ids=(reservation.reservation_id,),
            policy=OrphanHandlingPolicy(policy_id="test.require-operator"),
        )
    )
    app = FastAPI()
    app.include_router(orchestration_api.router, prefix="/api/orchestration")

    response = TestClient(app).get(
        "/api/orchestration/status",
        params={"intent_id": intent.intent_id, "refresh": "false"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "operator_action_required"
    assert payload["orphaned_resources"] == [
        {
            "backend_id": "gcp",
            "provider_resource_handle": "feedbax-worker-orphan",
            "external_effect_key": "feedbax-worker-orphan",
            "status": "operator_action_required",
            "handling_policy": {
                "schema_id": "feedbax.orchestration.orphan_handling_policy",
                "schema_version": "feedbax.orchestration.orphan_handling_policy.v1",
                "policy_id": "test.require-operator",
                "action": "require-operator",
            },
        }
    ]
