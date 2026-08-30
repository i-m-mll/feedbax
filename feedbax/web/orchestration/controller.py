"""Studio adapters for the durable orchestration controller."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

from feedbax.execution.records import Invocation
from feedbax.orchestration.controller import (
    ControllerEventStore,
    ControllerProjection,
    DurableController,
    EffectObservation,
    EffectReservation,
    RunIntent,
    utc_now,
)
from feedbax.orchestration.realization import BackendPlan, ExpectedCost
from feedbax.web.orchestration.gcp import (
    InstanceConfig,
    InstanceInfo,
    InstanceStatus,
    create_instance,
    delete_instance,
    get_instance,
)
from feedbax.web.orchestration.startup_script import FeedbaxInstallSpec
from feedbax.web.worker.client import wait_for_health


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
        config = self._config(reservation)
        instance_name = str(parameters["instance_name"])
        info = await create_instance(config, instance_name)
        deadline = asyncio.get_running_loop().time() + float(parameters["startup_timeout_seconds"])
        while info.status != InstanceStatus.RUNNING:
            if info.status in {
                InstanceStatus.TERMINATED,
                InstanceStatus.STOPPING,
                InstanceStatus.PREEMPTED,
            }:
                raise RuntimeError(f"GCP instance entered {info.status.value}")
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("timed out waiting for GCP instance readiness")
            await asyncio.sleep(float(parameters["poll_interval_seconds"]))
            info = await get_instance(config.project, config.zone, instance_name)
        worker_url = self._worker_url(info, config.worker_port)
        secrets = self._secret_material.get(reservation.reservation_id, {})
        auth_token = secrets.get("worker_auth_token")
        await wait_for_health(
            worker_url,
            timeout=float(parameters["worker_health_timeout_seconds"]),
            interval=float(parameters["poll_interval_seconds"]),
            auth_token=auth_token,
        )
        if self.training_service is not None:
            self.training_service.connect_remote(worker_url, auth_token)
        return self._observation(info, worker_url, reservation.expected_cost)

    async def reconcile(self, reservation: EffectReservation) -> EffectObservation | None:
        parameters = reservation.normalized_parameters
        try:
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
        worker_url = None
        if info.status == InstanceStatus.RUNNING and info.external_ip is not None:
            worker_url = self._worker_url(info, int(parameters["worker_port"]))
        return self._observation(info, worker_url, reservation.expected_cost)

    def _config(self, reservation: EffectReservation) -> InstanceConfig:
        parameters = reservation.normalized_parameters
        secrets = self._secret_material.get(reservation.reservation_id, {})
        return InstanceConfig(
            project=str(parameters["project"]),
            zone=str(parameters["zone"]),
            machine_type=str(parameters["machine_type"]),
            preemptible=bool(parameters["preemptible"]),
            worker_port=int(parameters["worker_port"]),
            auth_token=secrets.get("worker_auth_token"),
            ts_auth_key=secrets.get("tailscale_auth_key"),
            install_spec=FeedbaxInstallSpec.from_mapping(dict(parameters["install_spec"])),
        )

    @staticmethod
    def _worker_url(info: InstanceInfo, worker_port: int) -> str:
        if info.external_ip is None:
            raise RuntimeError("running GCP instance has no external IP")
        return f"http://{info.external_ip}:{worker_port}"

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
        self.controller.authenticate_reservation(
            intent_id,
            reservation_id,
            operator_identity=operator_identity,
            authentication_id=authentication_id,
        )
        await self.controller.dispatch(intent_id, reservation_id, self.gcp)

    def status(self, intent_id: str) -> ControllerProjection:
        return self.controller.expire_reservations(intent_id)

    async def refresh(self, intent_id: str) -> ControllerProjection:
        return await self.controller.reconcile(intent_id, self.adapters)

    async def reconcile_on_startup(self) -> None:
        intent_ids = tuple(
            dict.fromkeys(event.intent_id for event in self.controller.store.read_all())
        )
        for intent_id in intent_ids:
            await self.refresh(intent_id)

    async def terminate(self, intent_id: str) -> None:
        projection = self.controller.project(intent_id)
        if any(
            item.reservation.effect_class == "cloud-machine-termination"
            and item.status in {"observed", "reconciled"}
            for item in projection.reservations.values()
        ):
            return
        source = next(
            (
                item
                for item in reversed(tuple(projection.reservations.values()))
                if item.reservation.effect_class == "cloud-machine-acquisition"
                and item.provider_resource_handle is not None
            ),
            None,
        )
        if source is None:
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
        plan_document = next(
            event.payload["backend_plan"]
            for event in self.controller.store.read_all()
            if event.intent_id == intent_id and event.event_type == "backend_plan_selected"
        )
        plan = BackendPlan.model_validate(plan_document)
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
        events = self.controller.store.read_all()
        return events[-1].intent_id if events else None

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
