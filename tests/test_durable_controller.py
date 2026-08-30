"""Durable controller reservations, replay, and recovery contracts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from feedbax.orchestration.controller import (
    ControllerEventStore,
    ControllerProtocolError,
    DurableController,
    EffectDispatchAmbiguous,
    EffectObservation,
    EffectReservation,
    OperatorGateError,
    RunIntent,
    controller_event_from_document,
    effect_reservation_from_document,
    run_intent_from_document,
)
from feedbax.orchestration.drivers.local import local_driver_registration
from feedbax.orchestration.drivers.runpod import runpod_driver_registration
from feedbax.orchestration.gcp_backend import GCP_CONTROLLER_CAPABILITIES
from feedbax.orchestration.realization import (
    BackendRealizationRequest,
    ExpectedCost,
    MachineShape,
    OrchestrationBackend,
)
from tests.test_invocation_backend_realization import _sisu_invocation
from feedbax.web.orchestration.controller import GcpEffectAdapter, StudioController


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 29, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


class _Adapter:
    def __init__(self, *, ambiguous: bool = False) -> None:
        self.ambiguous = ambiguous
        self.status = "running"
        self.dispatch_count = 0
        self.reconcile_count = 0
        self.resource_by_key: dict[str, str] = {}

    def bind_secret_material(self, _reservation_id, _secrets) -> None:
        pass

    async def dispatch(self, reservation):
        self.dispatch_count += 1
        handle = self.resource_by_key.setdefault(
            reservation.external_effect_key,
            f"resource-{len(self.resource_by_key) + 1}",
        )
        if self.ambiguous:
            raise EffectDispatchAmbiguous("response dropped after provider accepted create")
        return EffectObservation(
            provider_resource_handle=handle,
            status=(
                "succeeded"
                if reservation.effect_class == "cloud-machine-termination"
                else "running"
            ),
        )

    async def reconcile(self, reservation):
        self.reconcile_count += 1
        handle = self.resource_by_key.get(reservation.external_effect_key)
        if handle is None:
            return None
        return EffectObservation(provider_resource_handle=handle, status=self.status)


def _plan(*, paid: bool):
    invocation = _sisu_invocation()
    if paid:
        registration = runpod_driver_registration()
        variant = "engine-acquired"
        machine = MachineShape(
            accelerator_type="NVIDIA GeForce RTX 4090",
            accelerator_count=1,
            regions=("CA-MTL-1",),
        )
        expected_cost = ExpectedCost(maximum=2.5, basis="inert test ceiling")
        confirmation = "authenticated-effect-reservation"
    else:
        registration = local_driver_registration()
        variant = "local-stop"
        machine = MachineShape()
        expected_cost = None
        confirmation = None
    backend = OrchestrationBackend(
        backend_id=registration.name,
        supported_scientific_capabilities=frozenset({"training"}),
        driver_capabilities=registration.supported_capabilities,
    )
    request = BackendRealizationRequest(
        adapter_id=f"feedbax.orchestration.{variant}",
        adapter_version="1",
        capability_variant=variant,
        code_bundle_id="git:feedbax@controller-test",
        environment_bundle_id="uv-lock:" + "d" * 64,
        command=("feedbax", "execute-training-run-spec", "sisu-tier-a.json"),
        machine=machine,
        timeout_seconds=120,
        retry_classification="same-plan",
        expected_cost=expected_cost,
        billable_confirmation_class=confirmation,
        external_effect_key=f"sisu-{variant}-controller-test",
    )
    return invocation, backend.realize("training", (invocation, request))


def _gcp_plan():
    invocation = _sisu_invocation()
    backend = OrchestrationBackend(
        backend_id="gcp",
        supported_scientific_capabilities=frozenset({"training"}),
        driver_capabilities=GCP_CONTROLLER_CAPABILITIES,
    )
    request = BackendRealizationRequest(
        adapter_id="feedbax.orchestration.gcp-controller",
        adapter_version="1",
        capability_variant="controller-acquired",
        code_bundle_id="git:feedbax@controller-test",
        environment_bundle_id="uv-lock:" + "d" * 64,
        command=("feedbax", "worker"),
        machine=MachineShape(cpu_count=4, memory_gib=15),
        network_requirements=("egress:https",),
        secret_names=("gcp_application_credentials",),
        timeout_seconds=120,
        retry_classification="same-plan",
        expected_cost=ExpectedCost(maximum=1.0, basis="one-hour test ceiling"),
        billable_confirmation_class="authenticated-effect-reservation",
        external_effect_key="gcp-controller-test",
        configuration={
            "project": "inert-project",
            "zone": "northamerica-northeast1-a",
            "machine_type": "n1-standard-4",
            "preemptible": True,
            "worker_port": 8765,
            "install_spec": {
                "schema_version": "feedbax.orchestration.install.v1",
                "source": "git",
                "repository": "https://github.com/mlll-io/feedbax.git",
                "ref": "develop",
                "extras": [],
            },
        },
    )
    return invocation, backend.realize("training", (invocation, request))


def _admit(controller: DurableController, *, paid: bool):
    invocation, plan = _plan(paid=paid)
    intent = RunIntent(
        intent_id="intent-paid" if paid else "intent-local",
        invocation_id=invocation.invocation_id,
        desired_outcome="satisfied",
        operator_gate_policy=("per-effect-authentication" if paid else "none"),
        idempotency_boundary=plan.external_effect_key,
    )
    controller.admit_intent(intent)
    controller.select_backend_plan(intent.intent_id, plan)
    return intent, plan


def test_controller_documents_reject_unsupported_versions() -> None:
    intent = RunIntent(
        intent_id="intent",
        invocation_id="invocation",
        desired_outcome="satisfied",
        idempotency_boundary="effect",
    ).model_dump(mode="json")
    with pytest.raises(ControllerProtocolError, match="no migration"):
        run_intent_from_document(
            {**intent, "schema_version": "feedbax.orchestration.run_intent.v0"}
        )

    for loader, schema_id in (
        (effect_reservation_from_document, "feedbax.orchestration.effect_reservation"),
        (controller_event_from_document, "feedbax.orchestration.controller_event"),
    ):
        with pytest.raises(ControllerProtocolError, match="no migration"):
            loader({"schema_id": schema_id, "schema_version": "unsupported"})


def test_paid_reservation_expires_inert_without_adapter_contact(tmp_path) -> None:
    clock = _Clock()
    controller = DurableController(
        ControllerEventStore(tmp_path / "events.jsonl"),
        producer_id="test-controller",
        clock=clock,
    )
    intent, plan = _admit(controller, paid=True)
    reservation = controller.reserve_effect(
        intent.intent_id,
        plan,
        effect_class="cloud-machine-acquisition",
        normalized_parameters={"machine_type": "RTX 4090"},
        expires_at=clock.now + timedelta(minutes=5),
        reservation_id="R-expire",
    )
    adapter = _Adapter()

    with pytest.raises(OperatorGateError, match="not been authenticated"):
        asyncio.run(controller.dispatch(intent.intent_id, reservation.reservation_id, adapter))
    clock.now += timedelta(minutes=6)
    projection = controller.expire_reservations(intent.intent_id)

    assert projection.reservations["R-expire"].status == "expired"
    assert projection.attempts == {}
    assert adapter.dispatch_count == 0
    assert [event.event_type for event in controller.store.read_all()].count(
        "effect_reservation_expired"
    ) == 1
    controller.expire_reservations(intent.intent_id)
    assert [event.event_type for event in controller.store.read_all()].count(
        "effect_reservation_expired"
    ) == 1


def test_rejected_transition_is_never_appended(tmp_path) -> None:
    clock = _Clock()
    controller = DurableController(
        ControllerEventStore(tmp_path / "events.jsonl"),
        producer_id="test-controller",
        clock=clock,
    )
    intent, plan = _admit(controller, paid=True)
    reservation = controller.reserve_effect(
        intent.intent_id,
        plan,
        effect_class="cloud-machine-acquisition",
        normalized_parameters={"machine_type": "RTX 4090"},
        expires_at=clock.now + timedelta(minutes=5),
    )
    controller.authenticate_reservation(
        intent.intent_id,
        reservation.reservation_id,
        operator_identity="operator",
        authentication_id="auth-one",
    )
    before = controller.store.read_all()

    with pytest.raises(ControllerProtocolError, match="cannot authenticate"):
        controller.authenticate_reservation(
            intent.intent_id,
            reservation.reservation_id,
            operator_identity="operator",
            authentication_id="auth-two",
        )

    assert controller.store.read_all() == before
    assert controller.project(intent.intent_id).reservations[reservation.reservation_id].status == (
        "authenticated"
    )


def test_authenticated_ambiguous_dispatch_recovers_once_after_restart(tmp_path) -> None:
    clock = _Clock()
    path = tmp_path / "events.jsonl"
    controller = DurableController(
        ControllerEventStore(path), producer_id="test-controller", clock=clock
    )
    intent, plan = _admit(controller, paid=True)
    reservation = controller.reserve_effect(
        intent.intent_id,
        plan,
        effect_class="cloud-machine-acquisition",
        normalized_parameters={"machine_type": "RTX 4090"},
        expires_at=clock.now + timedelta(minutes=5),
        reservation_id="R-disconnect",
    )
    controller.authenticate_reservation(
        intent.intent_id,
        reservation.reservation_id,
        operator_identity="operator",
        authentication_id="auth-R-disconnect",
    )
    adapter = _Adapter(ambiguous=True)
    attempt = asyncio.run(
        controller.dispatch(intent.intent_id, reservation.reservation_id, adapter)
    )

    assert attempt.status == "unknown"
    assert adapter.dispatch_count == 1
    restarted = DurableController(
        ControllerEventStore(path), producer_id="test-controller", clock=clock
    )
    adapter.ambiguous = False
    projection = asyncio.run(restarted.reconcile(intent.intent_id, {"runpod": adapter}))

    assert adapter.dispatch_count == 1
    assert adapter.reconcile_count == 1
    assert len(projection.attempts) == 1
    recovered = next(iter(projection.attempts.values()))
    assert recovered.provider_resource_handle == "resource-1"
    assert recovered.status == "running"
    assert projection.reservations["R-disconnect"].status == "reconciled"
    replayed = asyncio.run(restarted.reconcile(intent.intent_id, {"runpod": adapter}))
    assert adapter.reconcile_count == 2
    assert adapter.dispatch_count == 1
    assert replayed == projection


def test_local_effect_uses_same_controller_without_operator_gate(tmp_path) -> None:
    controller = DurableController(
        ControllerEventStore(tmp_path / "events.jsonl"), producer_id="test-controller"
    )
    intent, plan = _admit(controller, paid=False)
    reservation = controller.reserve_effect(
        intent.intent_id,
        plan,
        effect_class="local-execution",
        normalized_parameters={"command": list(plan.command)},
        reservation_id="R-local",
    )
    adapter = _Adapter()

    attempt = asyncio.run(
        controller.dispatch(intent.intent_id, reservation.reservation_id, adapter)
    )

    assert attempt.status == "running"
    assert attempt.provider_resource_handle == "resource-1"
    assert adapter.dispatch_count == 1
    replayed_attempt = asyncio.run(
        controller.dispatch(intent.intent_id, reservation.reservation_id, adapter)
    )
    assert replayed_attempt == attempt
    assert adapter.dispatch_count == 1
    controller.record_output_staged(
        intent.intent_id,
        attempt_id=attempt.attempt_id,
        artifact_refs=("artifact-version:checkpoint",),
    )
    controller.record_publication_committed(
        intent.intent_id,
        attempt_id=attempt.attempt_id,
        artifact_refs=("artifact-version:checkpoint", "artifact-version:manifest"),
        publication_id="publication-local",
    )
    assert controller.inspect_artifacts(intent.intent_id) == (
        "artifact-version:checkpoint",
        "artifact-version:manifest",
    )


def test_reconciliation_records_progressive_observations_until_terminal(tmp_path) -> None:
    controller = DurableController(
        ControllerEventStore(tmp_path / "events.jsonl"), producer_id="test-controller"
    )
    intent, plan = _admit(controller, paid=False)
    reservation = controller.reserve_effect(
        intent.intent_id,
        plan,
        effect_class="local-execution",
        normalized_parameters={"command": list(plan.command)},
        reservation_id="R-progressive",
    )
    adapter = _Adapter()
    asyncio.run(controller.dispatch(intent.intent_id, reservation.reservation_id, adapter))

    asyncio.run(controller.reconcile(intent.intent_id, {"local": adapter}))
    adapter.status = "succeeded"
    terminal = asyncio.run(controller.reconcile(intent.intent_id, {"local": adapter}))
    replayed = asyncio.run(controller.reconcile(intent.intent_id, {"local": adapter}))

    attempt = next(iter(terminal.attempts.values()))
    assert attempt.status == "succeeded"
    assert adapter.reconcile_count == 2
    assert replayed == terminal


def test_studio_gcp_reservation_is_inert_and_recovers_from_events(tmp_path) -> None:
    invocation, plan = _gcp_plan()
    studio = StudioController(event_path=tmp_path / "events.jsonl")
    intent, reservation = studio.reserve_cloud_launch(
        invocation,
        plan,
        parameters={
            "project": "inert-project",
            "zone": "northamerica-northeast1-a",
            "machine_type": "n1-standard-4",
            "preemptible": True,
            "worker_port": 8765,
            "instance_name": "feedbax-worker-inert",
            "install_spec": {
                "schema_version": "feedbax.orchestration.install.v1",
                "source": "git",
                "repository": "https://github.com/mlll-io/feedbax.git",
                "ref": "develop",
                "extras": [],
            },
            "startup_timeout_seconds": 120,
            "worker_health_timeout_seconds": 120,
            "poll_interval_seconds": 2,
        },
        ttl_seconds=300,
        secrets={"worker_auth_token": "not-persisted"},
    )

    assert studio.status(intent.intent_id).reservations[reservation.reservation_id].status == (
        "inert"
    )
    serialized = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "not-persisted" not in serialized
    restarted = StudioController(event_path=tmp_path / "events.jsonl")
    recovered = restarted.status(intent.intent_id)
    assert recovered.intent == intent
    assert recovered.reservations[reservation.reservation_id].reservation.backend_id == "gcp"
    assert recovered.attempts == {}


def test_studio_cancellation_is_a_separately_reserved_cleanup_effect(tmp_path) -> None:
    invocation, plan = _gcp_plan()
    studio = StudioController(event_path=tmp_path / "events.jsonl")
    adapter = _Adapter()
    studio.gcp = adapter
    studio.adapters["gcp"] = adapter
    intent, reservation = studio.reserve_cloud_launch(
        invocation,
        plan,
        parameters={
            **plan.configuration,
            "instance_name": "feedbax-worker-cancel",
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
        authentication_id="auth-launch",
    )
    acquisition = asyncio.run(
        studio.controller.dispatch(intent.intent_id, reservation.reservation_id, adapter)
    )

    asyncio.run(studio.terminate(intent.intent_id))
    projection = studio.status(intent.intent_id)
    cleanup = next(
        item
        for item in projection.reservations.values()
        if item.reservation.effect_class == "cloud-machine-termination"
    )

    assert adapter.dispatch_count == 2
    assert cleanup.status == "observed"
    assert projection.attempts[acquisition.attempt_id].status == "cancelled"
    asyncio.run(studio.terminate(intent.intent_id))
    assert adapter.dispatch_count == 2


def test_gcp_cleanup_reconciliation_treats_absence_as_terminal(monkeypatch) -> None:
    async def missing_instance(*_args):
        raise RuntimeError("instance not found")

    monkeypatch.setattr("feedbax.web.orchestration.controller.get_instance", missing_instance)
    reservation = EffectReservation(
        reservation_id="R-cleanup",
        intent_id="intent-cleanup",
        invocation_id="invocation-cleanup",
        backend_plan_id="plan-cleanup",
        effect_class="cloud-machine-termination",
        external_effect_key="gcp-machine:terminate",
        normalized_parameters={
            "project": "inert-project",
            "zone": "northamerica-northeast1-a",
            "instance_name": "already-gone",
        },
        backend_id="gcp",
        machine=MachineShape(),
        requires_authentication=False,
        created_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
    )

    observation = asyncio.run(GcpEffectAdapter().reconcile(reservation))

    assert observation is not None
    assert observation.status == "succeeded"
    assert observation.provider_resource_handle == "already-gone"
