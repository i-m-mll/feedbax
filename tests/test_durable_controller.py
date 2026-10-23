"""Durable controller reservations, replay, and recovery contracts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from feedbax.orchestration.controller import (
    ControllerEvent,
    ControllerEventStore,
    ControllerConflictError,
    ControllerProtocolError,
    DurableController,
    EffectDispatchAmbiguous,
    EffectObservation,
    EffectReservation,
    OrphanHandlingPolicy,
    OperatorGateError,
    ProviderResourceObservation,
    RunIntent,
    controller_event_from_document,
    effect_reservation_from_document,
    provider_inventory_observation,
    run_intent_from_document,
)
from feedbax.orchestration.stages import StageEngine
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
from feedbax.web.orchestration.gcp import InstanceInfo, InstanceStatus
from feedbax.orchestration.transition_authority import assert_disjoint_transition_authorities


def test_controller_and_stage_engine_own_disjoint_durable_identity_domains() -> None:
    assert DurableController.transition_authority.identity_field == "intent_id"
    assert StageEngine.transition_authority.identity_field == "run_set_id"
    assert_disjoint_transition_authorities(
        DurableController.transition_authority,
        StageEngine.transition_authority,
    )


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 29, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


class _Adapter:
    def __init__(self, *, ambiguous: bool = False, backend_id: str = "local") -> None:
        self.ambiguous = ambiguous
        self.backend_id = backend_id
        self.status = "running"
        self.dispatch_count = 0
        self.reconcile_count = 0
        self.inventory_count = 0
        self.resource_by_key: dict[str, str] = {}
        self.inventory_resources: tuple[ProviderResourceObservation, ...] = ()

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

    async def observe_inventory(self, scope):
        self.inventory_count += 1
        return provider_inventory_observation(
            backend_id=self.backend_id,
            scope=scope,
            observed_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
            resources=self.inventory_resources,
        )


def _plan(*, paid: bool, retry_classification: str = "same-plan"):
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
        retry_classification=retry_classification,
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

    event = ControllerEvent(
        event_id="event",
        intent_id="intent",
        sequence=0,
        event_type="intent_admitted",
        producer_id="test",
        occurred_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        observed_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
    ).model_dump(mode="json")
    migrated = controller_event_from_document(
        {**event, "schema_version": "feedbax.orchestration.controller_event.v1"}
    )
    assert migrated.schema_version == "feedbax.orchestration.controller_event.v2"
    with pytest.raises(ControllerProtocolError, match="v1 does not define"):
        controller_event_from_document(
            {
                **event,
                "schema_version": "feedbax.orchestration.controller_event.v1",
                "event_type": "provider_inventory_observed",
            }
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


def test_effect_key_has_one_initial_reservation_across_intents(tmp_path) -> None:
    controller = DurableController(
        ControllerEventStore(tmp_path / "events.jsonl"), producer_id="test-controller"
    )
    invocation, plan = _plan(paid=False)
    for intent_id in ("intent-one", "intent-two"):
        controller.admit_intent(
            RunIntent(
                intent_id=intent_id,
                invocation_id=invocation.invocation_id,
                desired_outcome="satisfied",
                idempotency_boundary=plan.external_effect_key,
            )
        )
        controller.select_backend_plan(intent_id, plan)
    controller.reserve_effect(
        "intent-one",
        plan,
        effect_class="local-execution",
        normalized_parameters={"command": list(plan.command)},
    )

    with pytest.raises(ControllerConflictError, match="reused with different content"):
        controller.reserve_effect(
            "intent-two",
            plan,
            effect_class="local-execution",
            normalized_parameters={"command": list(plan.command)},
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


def test_retry_admission_enforces_invocation_policy_and_preserves_effect_key(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    controller = DurableController(ControllerEventStore(path), producer_id="test-controller")
    invocation, plan = _plan(paid=False)
    intent = RunIntent(
        intent_id="intent-retry",
        invocation_id=invocation.invocation_id,
        desired_outcome="satisfied",
        idempotency_boundary=plan.external_effect_key,
    )
    controller.admit_intent(intent)
    controller.select_backend_plan(intent.intent_id, plan)
    first = controller.reserve_effect(
        intent.intent_id,
        plan,
        effect_class="local-execution",
        normalized_parameters={"command": list(plan.command)},
        reservation_id="R-first",
    )
    adapter = _Adapter()
    first_attempt = asyncio.run(
        controller.dispatch(intent.intent_id, first.reservation_id, adapter)
    )
    controller.observe_attempt_terminal(
        intent.intent_id,
        first_attempt.attempt_id,
        status="failed",
        exit_classification="worker_exit",
        reservation_id=first.reservation_id,
    )

    retry = controller.admit_retry(intent.intent_id, invocation, first.reservation_id)
    restarted = DurableController(ControllerEventStore(path), producer_id="restarted")
    replayed = restarted.admit_retry(intent.intent_id, invocation, first.reservation_id)

    assert retry == replayed
    assert retry.external_effect_key == first.external_effect_key
    assert restarted.project(intent.intent_id).retry_count == 1
    second_attempt = asyncio.run(
        restarted.dispatch(intent.intent_id, retry.reservation_id, adapter)
    )
    restarted.observe_attempt_terminal(
        intent.intent_id,
        second_attempt.attempt_id,
        status="failed",
        exit_classification="worker_exit",
        reservation_id=retry.reservation_id,
    )
    with pytest.raises(ControllerProtocolError, match="exhausted"):
        restarted.admit_retry(intent.intent_id, invocation, retry.reservation_id)


def test_retry_is_forbidden_by_backend_plan_even_when_invocation_allows_it(tmp_path) -> None:
    controller = DurableController(
        ControllerEventStore(tmp_path / "events.jsonl"), producer_id="test-controller"
    )
    invocation, plan = _plan(paid=False, retry_classification="never")
    intent = RunIntent(
        intent_id="intent-no-retry",
        invocation_id=invocation.invocation_id,
        desired_outcome="satisfied",
        idempotency_boundary=plan.external_effect_key,
    )
    controller.admit_intent(intent)
    controller.select_backend_plan(intent.intent_id, plan)
    reservation = controller.reserve_effect(
        intent.intent_id,
        plan,
        effect_class="local-execution",
        normalized_parameters={"command": list(plan.command)},
    )
    attempt = asyncio.run(
        controller.dispatch(intent.intent_id, reservation.reservation_id, _Adapter())
    )
    controller.observe_attempt_terminal(
        intent.intent_id,
        attempt.attempt_id,
        status="failed",
        exit_classification="worker_exit",
        reservation_id=reservation.reservation_id,
    )

    with pytest.raises(ControllerProtocolError, match="forbids retry"):
        controller.admit_retry(intent.intent_id, invocation, reservation.reservation_id)


def test_complete_inventory_proves_ambiguous_absence_before_same_key_retry(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    controller = DurableController(ControllerEventStore(path), producer_id="test-controller")
    invocation, plan = _plan(paid=False)
    intent = RunIntent(
        intent_id="intent-ambiguous-retry",
        invocation_id=invocation.invocation_id,
        desired_outcome="satisfied",
        idempotency_boundary=plan.external_effect_key,
    )
    controller.admit_intent(intent)
    controller.select_backend_plan(intent.intent_id, plan)
    reservation = controller.reserve_effect(
        intent.intent_id,
        plan,
        effect_class="local-execution",
        normalized_parameters={"command": list(plan.command)},
        reservation_id="R-ambiguous",
    )
    adapter = _Adapter(ambiguous=True)
    unknown = asyncio.run(
        controller.dispatch(intent.intent_id, reservation.reservation_id, adapter)
    )
    with pytest.raises(ControllerProtocolError, match="reconcile ambiguity"):
        controller.admit_retry(intent.intent_id, invocation, reservation.reservation_id)

    adapter.resource_by_key.clear()
    asyncio.run(
        controller.observe_provider_inventory(
            intent.intent_id,
            adapter,
            backend_id="local",
            scope={"runtime": "test-worker"},
            reservation_ids=(reservation.reservation_id,),
            policy=OrphanHandlingPolicy(policy_id="test.require-operator"),
        )
    )
    assert controller.project(intent.intent_id).attempts[unknown.attempt_id].status == "failed"

    retry = controller.admit_retry(intent.intent_id, invocation, reservation.reservation_id)
    adapter.ambiguous = False
    recovered = asyncio.run(controller.dispatch(intent.intent_id, retry.reservation_id, adapter))

    assert retry.external_effect_key == reservation.external_effect_key
    assert recovered.provider_resource_handle == "resource-1"
    assert adapter.dispatch_count == 2
    assert len(adapter.resource_by_key) == 1


def test_provider_inventory_detects_and_handles_orphan_replay_safely(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    controller = DurableController(ControllerEventStore(path), producer_id="test-controller")
    invocation, plan = _plan(paid=False)
    intent = RunIntent(
        intent_id="intent-orphan",
        invocation_id=invocation.invocation_id,
        desired_outcome="satisfied",
        idempotency_boundary=plan.external_effect_key,
    )
    controller.admit_intent(intent)
    controller.select_backend_plan(intent.intent_id, plan)
    reservation = controller.reserve_effect(
        intent.intent_id,
        plan,
        effect_class="local-execution",
        external_effect_key="managed-effect",
        normalized_parameters={"command": list(plan.command)},
    )
    adapter = _Adapter()
    asyncio.run(controller.dispatch(intent.intent_id, reservation.reservation_id, adapter))
    adapter.inventory_resources = (
        ProviderResourceObservation(
            provider_resource_handle="resource-managed",
            external_effect_key="managed-effect",
            status="RUNNING",
        ),
        ProviderResourceObservation(
            provider_resource_handle="resource-orphan",
            external_effect_key="orphan-effect",
            status="RUNNING",
        ),
    )
    policy = OrphanHandlingPolicy(policy_id="test.require-operator")

    projection = asyncio.run(
        controller.observe_provider_inventory(
            intent.intent_id,
            adapter,
            backend_id="local",
            scope={"runtime": "test-worker"},
            reservation_ids=(reservation.reservation_id,),
            policy=policy,
        )
    )
    before = controller.store.read_all()
    replayed = asyncio.run(
        controller.observe_provider_inventory(
            intent.intent_id,
            adapter,
            backend_id="local",
            scope={"runtime": "test-worker"},
            reservation_ids=(reservation.reservation_id,),
            policy=policy,
        )
    )
    restarted = DurableController(ControllerEventStore(path), producer_id="restarted")

    assert len(projection.orphans) == 1
    orphan = next(iter(projection.orphans.values()))
    assert orphan.resource.provider_resource_handle == "resource-orphan"
    assert orphan.status == "operator_action_required"
    assert orphan.handling_policy == policy
    assert replayed == projection
    assert controller.store.read_all() == before
    assert restarted.project(intent.intent_id) == projection


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


def test_gcp_inventory_observation_is_complete_versioned_and_canonical(monkeypatch) -> None:
    async def inventory(*_args):
        return [
            InstanceInfo(name="feedbax-worker-z", status=InstanceStatus.RUNNING),
            InstanceInfo(name="feedbax-worker-a", status=InstanceStatus.STOPPING),
        ]

    monkeypatch.setattr("feedbax.web.orchestration.controller.list_instances", inventory)

    observation = asyncio.run(
        GcpEffectAdapter().observe_inventory(
            {"project": "inert-project", "zone": "northamerica-northeast1-a"}
        )
    )

    assert observation.schema_version == "feedbax.orchestration.provider_inventory_observation.v1"
    assert observation.complete is True
    assert [item.external_effect_key for item in observation.resources] == [
        "feedbax-worker-a",
        "feedbax-worker-z",
    ]
