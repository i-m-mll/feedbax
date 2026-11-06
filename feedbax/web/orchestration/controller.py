"""Studio adapters for the durable orchestration controller."""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import anyio

from feedbax.execution.records import Invocation
from feedbax.orchestration.controller import (
    ControllerEventStore,
    ControllerProjection,
    DurableController,
    EffectObservation,
    EffectReservation,
    OrphanHandlingPolicy,
    ProviderInventoryObservation,
    ProviderResourceObservation,
    RunIntent,
    provider_inventory_observation,
    utc_now,
)
from feedbax.orchestration.realization import BackendPlan, ExpectedCost
from feedbax.web.orchestration.gcp import (
    InstanceInfo,
    InstanceStatus,
    delete_instance,
    get_instance,
    list_instances,
)
from feedbax.web.worker.transport import WorkerConfigurationError


logger = logging.getLogger(__name__)
_RECOVERY_CONCURRENCY = 4
_PROVIDER_READ_TIMEOUT_SECONDS = 30
_ORPHAN_POLICY = OrphanHandlingPolicy(policy_id="feedbax.orchestration.orphan.require_operator")
_GcpScope = tuple[str, str]
_ScopeMembers = dict[str, list[EffectReservation]]


class StudioRecoveryUnavailable(RuntimeError):
    """Provider recovery is pending or failed for an orchestration operation."""


def _controller_path() -> Path:
    configured = os.environ.get("FEEDBAX_CONTROLLER_EVENT_LOG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "feedbax" / "controller" / "events.jsonl"


class GcpEffectAdapter:
    """GCP provider calls behind exact durable reservations."""

    def __init__(self, training_service: Any | None = None) -> None:
        self.training_service = training_service
        self._secret_material: dict[str, dict[str, str]] = {}

    def bind_secret_material(self, reservation_id: str, secrets: Mapping[str, str]) -> None:
        self._secret_material[reservation_id] = dict(secrets)

    async def dispatch(self, reservation: EffectReservation) -> EffectObservation:
        parameters = reservation.normalized_parameters
        if reservation.effect_class == "cloud-machine-termination":
            await delete_instance(
                str(parameters["project"]),
                str(parameters["zone"]),
                str(parameters["instance_name"]),
            )
            if self.training_service is not None:
                self.training_service._terminate_worker()
            return EffectObservation(
                provider_resource_handle=str(parameters["instance_name"]),
                status="succeeded",
                observations=({"provider_status": "TERMINATED"},),
            )
        if reservation.effect_class != "cloud-machine-acquisition":
            raise ValueError(f"GCP adapter does not implement {reservation.effect_class!r}")
        raise WorkerConfigurationError(
            "GCP Studio worker acquisition is unavailable because the legacy adapter "
            "cannot establish a verifiable HTTPS worker origin"
        )

    async def reconcile(self, reservation: EffectReservation) -> EffectObservation | None:
        parameters = reservation.normalized_parameters
        try:
            with anyio.fail_after(_PROVIDER_READ_TIMEOUT_SECONDS):
                info = await get_instance(
                    str(parameters["project"]),
                    str(parameters["zone"]),
                    str(parameters["instance_name"]),
                )
        except RuntimeError as exc:
            if "not found" in str(exc).lower():
                if reservation.effect_class == "cloud-machine-termination":
                    return EffectObservation(
                        provider_resource_handle=str(parameters["instance_name"]),
                        status="succeeded",
                        observations=({"provider_status": "TERMINATED"},),
                    )
                return None
            raise
        if reservation.effect_class == "cloud-machine-termination":
            terminal = info.status in {InstanceStatus.TERMINATED, InstanceStatus.PREEMPTED}
            return EffectObservation(
                provider_resource_handle=info.name,
                status="succeeded" if terminal else "running",
                observations=({"provider_status": info.status.value},),
            )
        return self._observation(info, None, reservation.expected_cost)

    async def observe_inventory(self, scope: Mapping[str, Any]) -> ProviderInventoryObservation:
        if set(scope) != {"project", "zone"}:
            raise ValueError("GCP inventory scope requires exactly project and zone")
        with anyio.fail_after(_PROVIDER_READ_TIMEOUT_SECONDS):
            instances = await list_instances(str(scope["project"]), str(scope["zone"]))
        resources = tuple(
            ProviderResourceObservation(
                provider_resource_handle=info.name,
                external_effect_key=info.name,
                status=info.status.value,
                attributes={
                    "internal_ip": info.internal_ip,
                    "external_ip": info.external_ip,
                    "zone": info.zone,
                    "machine_type": info.machine_type,
                },
            )
            for info in instances
        )
        return provider_inventory_observation(
            backend_id="gcp",
            scope=scope,
            observed_at=utc_now(),
            resources=resources,
        )

    @staticmethod
    def _observation(
        info: InstanceInfo,
        worker_url: str | None,
        expected_cost: ExpectedCost | None,
    ) -> EffectObservation:
        status = "running"
        if info.status in {InstanceStatus.TERMINATED, InstanceStatus.PREEMPTED}:
            status = "failed"
        return EffectObservation(
            provider_resource_handle=info.name,
            worker_identity=worker_url,
            status=status,
            observed_cost=expected_cost,
            observations=(
                {
                    "provider_status": info.status.value,
                    "internal_ip": info.internal_ip,
                    "external_ip": info.external_ip,
                    "zone": info.zone,
                    "machine_type": info.machine_type,
                },
            ),
        )


class StudioController:
    """Studio's sole launch, status, refresh, cancellation, and inspection authority."""

    def __init__(
        self,
        *,
        event_path: Path | str | None = None,
        training_service: Any | None = None,
    ) -> None:
        self.controller = DurableController(
            ControllerEventStore(event_path or _controller_path()),
            producer_id="feedbax-studio-controller",
        )
        self.gcp = GcpEffectAdapter(training_service)
        self.adapters = {"gcp": self.gcp}
        self._discovering = False
        self._discovery_failed = False
        self._pending: Counter[str] = Counter()
        self._failed: set[str] = set()

    def reserve_cloud_launch(
        self,
        invocation: Invocation,
        backend_plan: BackendPlan,
        *,
        parameters: Mapping[str, Any],
        ttl_seconds: int,
        secrets: Mapping[str, str],
    ) -> tuple[RunIntent, EffectReservation]:
        if backend_plan.invocation_id != invocation.invocation_id:
            raise ValueError("BackendPlan does not realize the submitted Invocation")
        if backend_plan.backend_id != "gcp":
            raise ValueError("the GCP Studio endpoint requires a GCP BackendPlan")
        for field_name in (
            "project",
            "zone",
            "machine_type",
            "preemptible",
            "worker_port",
            "install_spec",
        ):
            if backend_plan.configuration.get(field_name) != parameters.get(field_name):
                raise ValueError(
                    f"GCP launch parameter {field_name!r} differs from the exact BackendPlan"
                )
        intent = RunIntent(
            intent_id=f"studio-{os.urandom(12).hex()}",
            invocation_id=invocation.invocation_id,
            desired_outcome="remote-worker-available",
            operator_gate_policy="per-effect-authentication",
            idempotency_boundary=backend_plan.external_effect_key,
        )
        self.controller.admit_intent(intent)
        self.controller.select_backend_plan(intent.intent_id, backend_plan)
        reservation = self.controller.reserve_effect(
            intent.intent_id,
            backend_plan,
            effect_class="cloud-machine-acquisition",
            normalized_parameters=parameters,
            expires_at=utc_now() + timedelta(seconds=ttl_seconds),
            external_effect_key=str(parameters["instance_name"]),
        )
        self.gcp.bind_secret_material(reservation.reservation_id, secrets)
        return intent, reservation

    async def authenticate_and_dispatch(
        self,
        intent_id: str,
        reservation_id: str,
        *,
        operator_identity: str,
        authentication_id: str,
    ) -> None:
        self.authenticate_reservation(
            intent_id,
            reservation_id,
            operator_identity=operator_identity,
            authentication_id=authentication_id,
        )
        await self.controller.dispatch(intent_id, reservation_id, self.gcp)

    def authenticate_reservation(
        self,
        intent_id: str,
        reservation_id: str,
        *,
        operator_identity: str,
        authentication_id: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.require_recovered(intent_id)
        self.controller.authenticate_reservation(
            intent_id,
            reservation_id,
            operator_identity=operator_identity,
            authentication_id=authentication_id,
            evidence=evidence,
        )

    def status(self, intent_id: str) -> ControllerProjection:
        if self._discovering or self._pending[intent_id] or intent_id in self._failed:
            return self.controller.project(intent_id)
        return self.controller.expire_reservations(intent_id)

    async def refresh(self, intent_id: str) -> ControllerProjection:
        if self._discovering or self._pending[intent_id]:
            return self.controller.project(intent_id)
        self.require_recovered(intent_id, allow_failed=True)
        failures: set[str] = set()
        self._pending[intent_id] += 1
        try:
            scopes = await anyio.to_thread.run_sync(self._read_scope_members)
            for scope, members in scopes.items():
                if intent_id in members:
                    failures.update(
                        await self._recover_scope(scope, members, only_intent=intent_id)
                    )
        finally:
            del self._pending[intent_id]
        if failures:
            self._failed.add(intent_id)
            self.require_recovered(intent_id)
        self._failed.discard(intent_id)
        return self.status(intent_id)

    def _read_scope_members(self) -> dict[_GcpScope | None, _ScopeMembers]:
        scopes: dict[_GcpScope | None, _ScopeMembers] = {}
        for intent_id, projection in self.controller.project_all().items():
            for reserved in projection.reservations.values():
                reservation = reserved.reservation
                parameters = reservation.normalized_parameters
                scope = None
                if reservation.backend_id == "gcp" and {"project", "zone"} <= parameters.keys():
                    scope = (str(parameters["project"]), str(parameters["zone"]))
                scopes.setdefault(scope, {}).setdefault(intent_id, []).append(reservation)
        return scopes

    async def _recover_scope(
        self,
        scope: _GcpScope | None,
        members: _ScopeMembers,
        *,
        only_intent: str | None = None,
    ) -> set[str]:
        selected = members if only_intent is None else {only_intent: members[only_intent]}
        failed: set[str] = set()
        for intent_id, reservations in selected.items():
            try:
                await self.controller.reconcile(
                    intent_id,
                    self.adapters,
                    reservation_ids=tuple(
                        reservation.reservation_id for reservation in reservations
                    ),
                )
            except Exception:
                failed.add(intent_id)
                logger.exception("Failed to reconcile Studio intent %s", intent_id)
        if scope is None:
            return failed
        acquisitions = {
            intent_id: sorted(
                reservation.reservation_id
                for reservation in reservations
                if reservation.effect_class == "cloud-machine-acquisition"
            )
            for intent_id, reservations in members.items()
        }
        targets = {
            intent_id: acquisitions[intent_id]
            for intent_id in selected
            if acquisitions.get(intent_id)
        }
        if targets:
            project, zone = scope
            try:
                provider_scope = {"project": project, "zone": zone}
                observation = await self.gcp.observe_inventory(provider_scope)
                await anyio.to_thread.run_sync(
                    partial(
                        self.controller.record_provider_scope,
                        observation,
                        backend_id="gcp",
                        scope=provider_scope,
                        reservations_by_intent=targets,
                        scope_members=acquisitions,
                        policy=_ORPHAN_POLICY,
                    )
                )
            except Exception:
                failed.update(targets)
                logger.exception("Failed to reconcile Studio inventory in %s/%s", project, zone)
        return failed

    def require_recovered(self, intent_id: str, *, allow_failed: bool = False) -> None:
        if self._discovering or self._pending[intent_id]:
            raise StudioRecoveryUnavailable("Studio orchestration recovery is pending")
        if self._discovery_failed or (intent_id in self._failed and not allow_failed):
            raise StudioRecoveryUnavailable("Studio orchestration recovery failed")

    def admit_retry(
        self,
        intent_id: str,
        invocation: Invocation,
        previous_reservation_id: str,
        *,
        ttl_seconds: int,
    ) -> EffectReservation:
        self.require_recovered(intent_id)
        return self.controller.admit_retry(
            intent_id,
            invocation,
            previous_reservation_id,
            expires_at=utc_now() + timedelta(seconds=ttl_seconds),
        )

    @asynccontextmanager
    async def startup_recovery(self):
        self._discovering = True
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(self.reconcile_on_startup)
            try:
                yield
            finally:
                tasks.cancel_scope.cancel()

    async def reconcile_on_startup(self) -> None:
        self._discovering = True
        self._discovery_failed = False
        self._failed.clear()
        try:
            scopes = await anyio.to_thread.run_sync(self._read_scope_members)
            self._pending = Counter(
                intent_id for members in scopes.values() for intent_id in members
            )
        except Exception:
            self._discovery_failed = True
            logger.exception("Failed to replay the Studio orchestration ledger")
            return
        finally:
            self._discovering = False

        remaining = iter(scopes.items())

        async def recover_scopes() -> None:
            for scope, members in remaining:
                self._failed.update(await self._recover_scope(scope, members))
                self._pending.subtract(members.keys())

        try:
            async with anyio.create_task_group() as tasks:
                for _ in range(min(len(scopes), _RECOVERY_CONCURRENCY)):
                    tasks.start_soon(recover_scopes)
        finally:
            self._pending.clear()

    async def terminate(self, intent_id: str) -> None:
        self.require_recovered(intent_id, allow_failed=True)
        projection = self.controller.project(intent_id)
        cleanup = {
            item.status
            for item in projection.reservations.values()
            if item.reservation.effect_class == "cloud-machine-termination"
        }
        if "dispatched" in cleanup:
            raise StudioRecoveryUnavailable("Studio termination must be reconciled before retry")
        if cleanup.intersection({"observed", "reconciled"}):
            return
        source = next(
            (
                item
                for item in reversed(tuple(projection.reservations.values()))
                if item.reservation.effect_class == "cloud-machine-acquisition"
                and item.status in {"observed", "reconciled"}
                and item.provider_resource_handle is not None
            ),
            None,
        )
        if source is None:
            self.require_recovered(intent_id)
            return
        source_attempt = next(
            (
                attempt
                for attempt in projection.attempts.values()
                if attempt.reservation_id == source.reservation.reservation_id
            ),
            None,
        )
        self.controller.request_cancellation(intent_id, reason="Studio operator requested stop")
        plan = projection.backend_plan
        assert plan is not None
        parameters = source.reservation.normalized_parameters
        cleanup = self.controller.reserve_effect(
            intent_id,
            plan,
            effect_class="cloud-machine-termination",
            external_effect_key=f"{source.reservation.external_effect_key}:terminate",
            normalized_parameters={
                "project": parameters["project"],
                "zone": parameters["zone"],
                "instance_name": parameters["instance_name"],
            },
        )
        cleanup_attempt = await self.controller.dispatch(
            intent_id, cleanup.reservation_id, self.gcp
        )
        if (
            cleanup_attempt.status == "succeeded"
            and source_attempt is not None
            and (source_attempt.status not in {"succeeded", "failed", "cancelled"})
        ):
            self.controller.observe_attempt_terminal(
                intent_id,
                source_attempt.attempt_id,
                status="cancelled",
                exit_classification="operator_cancelled_after_cleanup",
                reservation_id=source.reservation.reservation_id,
            )

    def latest_intent_id(self) -> str | None:
        return self.controller.latest_intent_id()

    def inspect_artifacts(self, intent_id: str) -> tuple[str, ...]:
        return self.controller.inspect_artifacts(intent_id)


studio_controller: StudioController | None = None


def get_studio_controller(training_service: Any | None = None) -> StudioController:
    global studio_controller
    if studio_controller is None:
        studio_controller = StudioController(training_service=training_service)
    elif training_service is not None:
        studio_controller.gcp.training_service = training_service
    return studio_controller


__all__ = ["GcpEffectAdapter", "StudioController", "get_studio_controller"]
