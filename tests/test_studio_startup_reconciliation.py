"""Studio restart recovery against an on-disk ledger and a fake provider."""

from __future__ import annotations

import asyncio
from collections import Counter
from unittest.mock import Mock

from anyio import CancelScope
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from feedbax.orchestration.controller import ProviderResourceObservation
from feedbax.web.api import orchestration as orchestration_api
from feedbax.web.app import create_app
from feedbax.web.orchestration import controller as studio_controller_module
from feedbax.web.orchestration.controller import StudioController
from feedbax.web.services.training_service import training_service
from tests.test_durable_controller import _Adapter, _gcp_plan


_PRIMARY_SCOPE = ("inert-project", "northamerica-northeast1-a")


class _ScopedAdapter(_Adapter):
    def __init__(self):
        super().__init__(backend_id="gcp")
        self.reconciled = []
        self.inventories = Counter()
        self.resources_by_scope = {}

    async def reconcile(self, reservation):
        self.reconciled.append(reservation.external_effect_key)
        return await super().reconcile(reservation)

    async def observe_inventory(self, scope):
        scope_key = scope["project"], scope["zone"]
        self.inventories[scope_key] += 1
        self.inventory_resources = self.resources_by_scope.get(scope_key, ())
        return await super().observe_inventory(scope)


def _studio(path, adapter):
    studio = StudioController(event_path=path)
    studio.gcp = adapter
    studio.adapters["gcp"] = adapter
    return studio


def _reserve(studio, key, *, scope=_PRIMARY_SCOPE):
    project, zone = scope
    invocation, plan = _gcp_plan(project=project, zone=zone)
    intent, reservation = studio.reserve_cloud_launch(
        invocation,
        plan,
        parameters={**plan.configuration, "instance_name": key},
        ttl_seconds=300,
        secrets={},
    )
    return intent.intent_id, reservation.reservation_id


async def _dispatch(studio, key, *, scope=_PRIMARY_SCOPE):
    intent_id, reservation_id = _reserve(studio, key, scope=scope)
    await studio.authenticate_and_dispatch(
        intent_id, reservation_id, operator_identity="operator", authentication_id=f"auth-{key}"
    )
    return intent_id, reservation_id


def test_startup_groups_durable_replay_and_inventory(monkeypatch, tmp_path):
    async def exercise():
        path = tmp_path / "events.jsonl"
        adapter = _ScopedAdapter()
        writer = _studio(path, adapter)
        running_id, _ = await _dispatch(writer, "running")
        missing_id, _ = await _dispatch(writer, "missing")
        historical_scope = ("inert-project", "northamerica-northeast1-b")
        terminal_id, terminal_reservation = await _dispatch(
            writer, "terminal", scope=historical_scope
        )
        terminal_attempt = next(iter(writer.status(terminal_id).attempts.values()))
        writer.controller.observe_attempt_terminal(
            terminal_id,
            terminal_attempt.attempt_id,
            status="succeeded",
            exit_classification="completed",
            reservation_id=terminal_reservation,
        )
        del adapter.resource_by_key["missing"]
        adapter.resources_by_scope[_PRIMARY_SCOPE] = (
            ProviderResourceObservation(
                provider_resource_handle=adapter.resource_by_key["running"],
                external_effect_key="running",
                status="RUNNING",
            ),
            ProviderResourceObservation(
                provider_resource_handle="orphan-instance",
                external_effect_key="orphan-instance",
                status="RUNNING",
            ),
        )
        adapter.resources_by_scope[historical_scope] = (
            ProviderResourceObservation(
                provider_resource_handle="same-name-in-another-zone",
                external_effect_key="running",
                status="RUNNING",
            ),
        )

        reader = _studio(path, adapter)
        store = reader.controller.store
        full_read = Mock(wraps=store._read_unlocked)
        monkeypatch.setattr(store, "_read_unlocked", full_read)
        await reader.reconcile_on_startup()

        assert full_read.call_count == 1
        assert Counter(adapter.reconciled) == {"running": 1, "missing": 1}
        assert adapter.inventories == {
            _PRIMARY_SCOPE: 1,
            historical_scope: 1,
        }
        assert adapter.dispatch_count == 3

        replayed = _studio(path, adapter)
        running = replayed.status(running_id)
        missing = replayed.status(missing_id)
        assert next(iter(running.attempts.values())).status == "running"
        assert next(iter(missing.attempts.values())).status == "failed"
        for projection in (running, missing):
            (orphan,) = projection.orphans.values()
            assert orphan.resource.external_effect_key == "orphan-instance"
            assert orphan.status == "operator_action_required"
        (other_scope_orphan,) = replayed.status(terminal_id).orphans.values()
        assert other_scope_orphan.resource.provider_resource_handle == "same-name-in-another-zone"

    asyncio.run(exercise())


def test_startup_failure_preserves_evidence_and_gates_only_affected_intents(
    caplog, monkeypatch, tmp_path
):
    async def exercise():
        adapter = _ScopedAdapter()
        path = tmp_path / "events.jsonl"
        writer = _studio(path, adapter)
        offline_scope = ("offline", "northamerica-northeast1-a")
        failed_id, failed_reservation = await _dispatch(writer, "failed-probe", scope=offline_scope)
        sibling_id, _ = await _dispatch(writer, "sibling", scope=offline_scope)
        healthy_id, healthy_reservation = await _dispatch(writer, "missing")
        del adapter.resource_by_key["missing"]
        point_lookup = adapter.reconcile
        inventory_lookup = adapter.observe_inventory

        async def reconcile(reservation):
            if reservation.external_effect_key == "failed-probe":
                raise RuntimeError("fake point lookup failure")
            return await point_lookup(reservation)

        async def observe_inventory(scope):
            if scope["project"] == "offline":
                raise RuntimeError("fake inventory failure")
            return await inventory_lookup(scope)

        monkeypatch.setattr(adapter, "reconcile", reconcile)
        monkeypatch.setattr(adapter, "observe_inventory", observe_inventory)
        studio = _studio(path, adapter)
        monkeypatch.setattr(studio_controller_module, "studio_controller", studio)
        await studio.reconcile_on_startup()
        assert any(failed_id in record.getMessage() for record in caplog.records)
        assert any("offline" in record.getMessage() for record in caplog.records)

        replayed = _studio(path, adapter)
        assert next(iter(replayed.status(failed_id).attempts.values())).status == "running"
        assert next(iter(replayed.status(healthy_id).attempts.values())).status == "failed"
        evidence = replayed.controller.store.read_all()
        assert any(
            event.intent_id == sibling_id and event.event_type == "external_effect_reconciled"
            for event in evidence
        )
        assert {
            event.intent_id
            for event in evidence
            if event.event_type == "external_effect_absence_reconciled"
        } == {healthy_id}

        invocation, _ = _gcp_plan()
        app = FastAPI()
        app.include_router(orchestration_api.router, prefix="/api/orchestration")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://studio"
        ) as client:
            failed_retry = (
                f"/api/orchestration/intents/{failed_id}/reservations/{failed_reservation}/retry"
            )
            payload = {"invocation": invocation.model_dump(mode="json")}
            failed = await client.post(failed_retry, json=payload)
            healthy = await client.post(
                f"/api/orchestration/intents/{healthy_id}/reservations/{healthy_reservation}/retry",
                json=payload,
            )
            assert failed.status_code == 503
            assert healthy.status_code == 200

            monkeypatch.setattr(adapter, "reconcile", point_lookup)
            monkeypatch.setattr(adapter, "observe_inventory", inventory_lookup)
            del adapter.resource_by_key["failed-probe"]
            refreshed = await client.get(
                "/api/orchestration/status", params={"intent_id": failed_id}
            )
            assert refreshed.status_code == 200
            recovered = await client.post(failed_retry, json=payload)
            assert recovered.status_code == 200
            assert recovered.json()["status"] == "awaiting_authentication"

        assert adapter.inventories == {_PRIMARY_SCOPE: 1, offline_scope: 1}
        assert adapter.dispatch_count == 3

    asyncio.run(exercise())


def test_lifespan_serves_reads_and_owns_pending_reconciliation(monkeypatch, tmp_path):
    async def exercise():
        adapter = _ScopedAdapter()
        path = tmp_path / "events.jsonl"
        writer = _studio(path, adapter)
        intent_id, _ = await _dispatch(writer, "running")
        inert_id, inert_reservation = _reserve(writer, "inert")
        prefix = "/api/orchestration"
        invocation, _ = _gcp_plan()
        reached_provider = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_reconcile(_reservation):
            reached_provider.set()
            try:
                await asyncio.Future()
            finally:
                with CancelScope(shield=True):
                    await asyncio.sleep(0)
                    cancelled.set()

        monkeypatch.setattr(adapter, "reconcile", blocked_reconcile)
        studio = _studio(path, adapter)
        monkeypatch.setattr(training_service, "rebuild_cache_from_state_docs", lambda: None)

        def terminate_worker():
            assert cancelled.is_set()

        terminate_worker_mock = Mock(side_effect=terminate_worker)
        monkeypatch.setattr(training_service, "_terminate_worker", terminate_worker_mock)
        app = create_app(studio_controller=studio)
        before = path.read_bytes()
        async with asyncio.timeout(5):
            async with app.router.lifespan_context(app):
                await reached_provider.wait()
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://studio"
                ) as client:
                    reservation_url = (
                        f"{prefix}/intents/{inert_id}/reservations/{inert_reservation}"
                    )
                    authenticate = await client.post(
                        f"{reservation_url}/authenticate",
                        json={
                            "operator_identity": "operator",
                            "authentication_id": "auth-inert",
                            "confirmation_token": "launch-billable-gcp-worker",
                            "max_cost_usd": 1.0,
                        },
                    )
                    retry = await client.post(
                        f"{reservation_url}/retry",
                        json={"invocation": invocation.model_dump(mode="json")},
                    )
                    terminate = await client.delete(
                        f"{prefix}/instance", params={"intent_id": intent_id}
                    )
                    assert (
                        authenticate.status_code
                        == retry.status_code
                        == terminate.status_code
                        == 503
                    )
                    assert path.read_bytes() == before
                    assert adapter.dispatch_count == 1
                    status = await client.get(
                        f"{prefix}/status", params={"intent_id": intent_id, "refresh": "false"}
                    )
                    assert status.status_code == 200
                    assert status.json()["intent_id"] == intent_id
                    assert (await client.get("/api/components")).status_code == 200

        terminate_worker_mock.assert_called_once()
        assert path.read_bytes() == before

    asyncio.run(exercise())
