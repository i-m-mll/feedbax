"""Durable intent-to-effect controller protocol.

The append-only event stream in this module is the authority for controller
state.  Projections are rebuilt from that stream and adapters are invoked only
after the corresponding effect reservation is durable.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from feedbax.contracts.manifest import StrictModel, canonical_json_bytes
from feedbax.orchestration.realization import (
    Attempt,
    BackendPlan,
    ExpectedCost,
    MachineShape,
    backend_plan_from_document,
)


RUN_INTENT_SCHEMA_ID = "feedbax.orchestration.run_intent"
RUN_INTENT_SCHEMA_VERSION = "feedbax.orchestration.run_intent.v1"
EFFECT_RESERVATION_SCHEMA_ID = "feedbax.orchestration.effect_reservation"
EFFECT_RESERVATION_SCHEMA_VERSION = "feedbax.orchestration.effect_reservation.v1"
CONTROLLER_EVENT_SCHEMA_ID = "feedbax.orchestration.controller_event"
CONTROLLER_EVENT_SCHEMA_VERSION = "feedbax.orchestration.controller_event.v1"
CONTROLLER_PROJECTION_SCHEMA_ID = "feedbax.orchestration.controller_projection"
CONTROLLER_PROJECTION_SCHEMA_VERSION = "feedbax.orchestration.controller_projection.v1"

ControllerEventType = Literal[
    "intent_admitted",
    "intent_rejected",
    "intent_cancelled",
    "intent_superseded",
    "backend_plan_selected",
    "backend_plan_rejected",
    "effect_reservation_created",
    "effect_reservation_authenticated",
    "effect_reservation_expired",
    "effect_reservation_cancelled",
    "effect_reservation_invalidated",
    "external_effect_dispatched",
    "external_effect_observed",
    "external_effect_reconciled",
    "external_effect_abandoned",
    "attempt_started",
    "attempt_heartbeat_observed",
    "attempt_terminal_observed",
    "attempt_state_unknown",
    "output_staged",
    "publication_committed",
    "publication_failed",
    "invocation_satisfied",
    "retry_admitted",
    "invocation_permanently_failed",
    "operator_action_required",
]
ReservationStatus = Literal[
    "inert",
    "authenticated",
    "dispatched",
    "observed",
    "reconciled",
    "expired",
    "cancelled",
    "invalidated",
    "abandoned",
]
AttemptStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "unknown"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ControllerProtocolError(ValueError):
    """A durable controller document or transition violates the protocol."""


class ControllerConflictError(ControllerProtocolError):
    """A duplicate identity carries different durable content."""


class OperatorGateError(ControllerProtocolError):
    """A requested effect has not satisfied its durable operator gate."""


class EffectDispatchAmbiguous(RuntimeError):
    """An adapter may have caused an effect but did not return an observation."""


class RunIntent(StrictModel):
    """Provider-neutral request for the controller to satisfy one invocation."""

    schema_id: Literal["feedbax.orchestration.run_intent"] = RUN_INTENT_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.run_intent.v1"] = RUN_INTENT_SCHEMA_VERSION
    intent_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    workflow_plan_id: str | None = None
    desired_outcome: str = Field(min_length=1)
    operator_gate_policy: Literal["none", "per-effect-authentication"] = "none"
    cancellation_policy: Literal["best-effort", "require-cleanup"] = "require-cleanup"
    idempotency_boundary: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)


class EffectReservation(StrictModel):
    """Inert authorization boundary for one exact possible external effect."""

    schema_id: Literal["feedbax.orchestration.effect_reservation"] = EFFECT_RESERVATION_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.effect_reservation.v1"] = (
        EFFECT_RESERVATION_SCHEMA_VERSION
    )
    reservation_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    backend_plan_id: str = Field(min_length=1)
    effect_class: str = Field(min_length=1)
    external_effect_key: str = Field(min_length=1)
    normalized_parameters: dict[str, Any]
    backend_id: str = Field(min_length=1)
    machine: MachineShape
    expected_cost: ExpectedCost | None = None
    requires_authentication: bool
    created_at: datetime
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_gate(self) -> "EffectReservation":
        if self.requires_authentication and self.expires_at is None:
            raise ValueError("authenticated effect reservations require an expiry")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("effect reservation expiry must follow creation")
        try:
            canonical_json_bytes(self.normalized_parameters)
        except Exception as exc:
            raise ValueError("effect parameters must have a canonical JSON encoding") from exc
        return self


class ControllerEvent(StrictModel):
    """One replay-safe event in an intent's monotonic controller stream."""

    schema_id: Literal["feedbax.orchestration.controller_event"] = CONTROLLER_EVENT_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.controller_event.v1"] = (
        CONTROLLER_EVENT_SCHEMA_VERSION
    )
    event_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    event_type: ControllerEventType
    producer_id: str = Field(min_length=1)
    occurred_at: datetime
    observed_at: datetime
    invocation_id: str | None = None
    backend_plan_id: str | None = None
    reservation_id: str | None = None
    attempt_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ReservationProjection(StrictModel):
    reservation: EffectReservation
    status: ReservationStatus = "inert"
    authenticated_by: str | None = None
    authentication_id: str | None = None
    authenticated_at: datetime | None = None
    dispatch_count: int = 0
    provider_resource_handle: str | None = None
    observed_cost: ExpectedCost | None = None


class ControllerProjection(StrictModel):
    schema_id: Literal["feedbax.orchestration.controller_projection"] = (
        CONTROLLER_PROJECTION_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.controller_projection.v1"] = (
        CONTROLLER_PROJECTION_SCHEMA_VERSION
    )
    intent: RunIntent | None = None
    status: str = "absent"
    backend_plan_id: str | None = None
    reservations: dict[str, ReservationProjection] = Field(default_factory=dict)
    attempts: dict[str, Attempt] = Field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    diagnostics: tuple[dict[str, Any], ...] = ()
    last_sequence: int = -1


class EffectObservation(StrictModel):
    """Adapter observation returned after dispatch or reconciliation."""

    provider_resource_handle: str = Field(min_length=1)
    worker_identity: str | None = None
    status: AttemptStatus = "pending"
    observed_cost: ExpectedCost | None = None
    observations: tuple[dict[str, Any], ...] = ()


class EffectAdapter(Protocol):
    """Provider adapter used only by the durable controller."""

    async def dispatch(self, reservation: EffectReservation) -> EffectObservation: ...

    async def reconcile(self, reservation: EffectReservation) -> EffectObservation | None: ...


class ControllerEventStore:
    """Fsync'd JSONL authority with process and filesystem locking."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._process_lock = threading.RLock()

    def read_all(self) -> list[ControllerEvent]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._process_lock, self.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            return self._read_unlocked()

    def append(
        self,
        *,
        event_type: ControllerEventType,
        intent_id: str,
        producer_id: str,
        occurred_at: datetime,
        observed_at: datetime,
        invocation_id: str | None = None,
        backend_plan_id: str | None = None,
        reservation_id: str | None = None,
        attempt_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        validator: Callable[[Sequence[ControllerEvent], ControllerEvent], None] | None = None,
    ) -> ControllerEvent:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._process_lock, self.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            events = self._read_unlocked()
            sequence = 1 + max(
                (event.sequence for event in events if event.intent_id == intent_id),
                default=-1,
            )
            content = {
                "intent_id": intent_id,
                "event_type": event_type,
                "producer_id": producer_id,
                "occurred_at": occurred_at,
                "invocation_id": invocation_id,
                "backend_plan_id": backend_plan_id,
                "reservation_id": reservation_id,
                "attempt_id": attempt_id,
                "payload": dict(payload or {}),
            }
            event_id = idempotency_key or hashlib.sha256(canonical_json_bytes(content)).hexdigest()
            duplicate = next((event for event in events if event.event_id == event_id), None)
            if duplicate is not None:
                comparable = {
                    "intent_id": duplicate.intent_id,
                    "event_type": duplicate.event_type,
                    "producer_id": duplicate.producer_id,
                    "invocation_id": duplicate.invocation_id,
                    "backend_plan_id": duplicate.backend_plan_id,
                    "reservation_id": duplicate.reservation_id,
                    "attempt_id": duplicate.attempt_id,
                    "payload": duplicate.payload,
                }
                requested = {key: value for key, value in content.items() if key != "occurred_at"}
                if requested != comparable:
                    raise ControllerConflictError(
                        f"controller event id {event_id!r} was reused with different content"
                    )
                return duplicate
            event = ControllerEvent(
                event_id=event_id,
                sequence=sequence,
                observed_at=observed_at,
                **content,
            )
            if validator is not None:
                validator(events, event)
            line = json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return event

    def _read_unlocked(self) -> list[ControllerEvent]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        events: list[ControllerEvent] = []
        sequences: dict[str, int] = {}
        identities: dict[str, ControllerEvent] = {}
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = controller_event_from_document(json.loads(line))
            except Exception as exc:
                raise ControllerProtocolError(
                    f"invalid controller event at {self.path}:{line_number}"
                ) from exc
            expected = sequences.get(event.intent_id, -1) + 1
            if event.sequence != expected:
                raise ControllerProtocolError(
                    f"controller stream {event.intent_id!r} expected sequence {expected}, "
                    f"got {event.sequence}"
                )
            previous = identities.get(event.event_id)
            if previous is not None and previous != event:
                raise ControllerConflictError(
                    f"controller event id {event.event_id!r} has conflicting records"
                )
            identities[event.event_id] = event
            sequences[event.intent_id] = event.sequence
            events.append(event)
        return events


def run_intent_from_document(document: Any) -> RunIntent:
    return _load_versioned(
        RunIntent,
        document,
        schema_id=RUN_INTENT_SCHEMA_ID,
        schema_version=RUN_INTENT_SCHEMA_VERSION,
        label="RunIntent",
    )


def effect_reservation_from_document(document: Any) -> EffectReservation:
    return _load_versioned(
        EffectReservation,
        document,
        schema_id=EFFECT_RESERVATION_SCHEMA_ID,
        schema_version=EFFECT_RESERVATION_SCHEMA_VERSION,
        label="EffectReservation",
    )


def controller_event_from_document(document: Any) -> ControllerEvent:
    return _load_versioned(
        ControllerEvent,
        document,
        schema_id=CONTROLLER_EVENT_SCHEMA_ID,
        schema_version=CONTROLLER_EVENT_SCHEMA_VERSION,
        label="ControllerEvent",
    )


def _load_versioned(
    model: type[RunIntent] | type[EffectReservation] | type[ControllerEvent],
    document: Any,
    *,
    schema_id: str,
    schema_version: str,
    label: str,
) -> Any:
    if not isinstance(document, Mapping):
        raise ControllerProtocolError(f"{label} document must be a mapping")
    if document.get("schema_id") != schema_id:
        raise ControllerProtocolError(
            f"unsupported {label} schema_id {document.get('schema_id')!r}; expected {schema_id!r}"
        )
    if document.get("schema_version") != schema_version:
        raise ControllerProtocolError(
            f"unsupported {label} schema_version {document.get('schema_version')!r}; "
            "no migration is registered"
        )
    return model.model_validate(document)


def project_controller_events(
    events: Sequence[ControllerEvent],
    *,
    intent_id: str,
) -> ControllerProjection:
    """Deterministically reconstruct one intent from its append-only stream."""

    projection = ControllerProjection()
    expected = 0
    for event in (item for item in events if item.intent_id == intent_id):
        if event.sequence != expected:
            raise ControllerProtocolError(
                f"controller stream {intent_id!r} expected sequence {expected}, got {event.sequence}"
            )
        projection = _apply_event(projection, event)
        expected += 1
    return projection


def _apply_event(
    projection: ControllerProjection,
    event: ControllerEvent,
) -> ControllerProjection:
    """The single authoritative lifecycle transition validator and reducer."""

    event_type = event.event_type
    update: dict[str, Any] = {"last_sequence": event.sequence}
    if event_type == "intent_admitted":
        if projection.intent is not None:
            _invalid(event, "intent already exists")
        intent = run_intent_from_document(event.payload["intent"])
        if intent.intent_id != event.intent_id:
            _invalid(event, "intent payload identity does not match stream")
        return projection.model_copy(update={**update, "intent": intent, "status": "admitted"})
    if projection.intent is None:
        _invalid(event, "intent has not been admitted")
    if event_type in {"intent_rejected", "intent_cancelled", "intent_superseded"}:
        if projection.status in {"satisfied", "permanently_failed"}:
            _invalid(event, f"terminal intent cannot transition through {event_type}")
        return projection.model_copy(
            update={**update, "status": event_type.removeprefix("intent_")}
        )
    if event_type == "backend_plan_selected":
        plan = backend_plan_from_document(event.payload["backend_plan"])
        if plan.backend_plan_id != event.backend_plan_id:
            _invalid(event, "backend plan payload identity does not match event")
        if plan.invocation_id != projection.intent.invocation_id:
            _invalid(event, "backend plan realizes a different invocation")
        if projection.backend_plan_id not in (None, event.backend_plan_id):
            _invalid(event, "intent already selected a different backend plan")
        return projection.model_copy(
            update={**update, "backend_plan_id": event.backend_plan_id, "status": "planned"}
        )
    if event_type == "backend_plan_rejected":
        return projection.model_copy(update={**update, "status": "backend_rejected"})
    if event_type == "effect_reservation_created":
        if projection.backend_plan_id is None:
            _invalid(event, "backend plan has not been selected")
        reservation = effect_reservation_from_document(event.payload["reservation"])
        if reservation.reservation_id != event.reservation_id:
            _invalid(event, "reservation payload identity does not match event")
        if reservation.backend_plan_id != projection.backend_plan_id:
            _invalid(event, "reservation does not bind the selected backend plan")
        if reservation.reservation_id in projection.reservations:
            _invalid(event, "reservation already exists")
        reservations = dict(projection.reservations)
        reservations[reservation.reservation_id] = ReservationProjection(reservation=reservation)
        return projection.model_copy(
            update={**update, "reservations": reservations, "status": "reserved"}
        )
    if event.reservation_id is not None:
        reservation_projection = projection.reservations.get(event.reservation_id)
    else:
        reservation_projection = None
    if event_type.startswith("effect_reservation_") or event_type.startswith("external_effect_"):
        if reservation_projection is None:
            _invalid(event, "event references an unknown reservation")
        reservations = dict(projection.reservations)
        current = reservation_projection
        assert current is not None
        if event_type == "effect_reservation_authenticated":
            if current.status != "inert":
                _invalid(event, f"cannot authenticate reservation from {current.status}")
            current = current.model_copy(
                update={
                    "status": "authenticated",
                    "authenticated_by": event.payload.get("operator_identity"),
                    "authentication_id": event.payload.get("authentication_id"),
                    "authenticated_at": event.occurred_at,
                }
            )
        elif event_type in {
            "effect_reservation_expired",
            "effect_reservation_cancelled",
            "effect_reservation_invalidated",
        }:
            if current.status not in {"inert", "authenticated"}:
                _invalid(event, f"cannot close reservation from {current.status}")
            current = current.model_copy(
                update={"status": event_type.removeprefix("effect_reservation_")}
            )
        elif event_type == "external_effect_dispatched":
            if current.status not in {"inert", "authenticated"}:
                _invalid(event, f"cannot dispatch reservation from {current.status}")
            if current.reservation.requires_authentication and current.status != "authenticated":
                _invalid(event, "reservation requires authentication")
            if current.dispatch_count != 0:
                _invalid(event, "reservation may be dispatched only once")
            current = current.model_copy(update={"status": "dispatched", "dispatch_count": 1})
        elif event_type in {"external_effect_observed", "external_effect_reconciled"}:
            if current.status not in {"dispatched", "observed", "reconciled"}:
                _invalid(event, f"cannot observe reservation from {current.status}")
            current = current.model_copy(
                update={
                    "status": "observed" if event_type.endswith("observed") else "reconciled",
                    "provider_resource_handle": event.payload.get("provider_resource_handle"),
                    "observed_cost": (
                        ExpectedCost.model_validate(event.payload["observed_cost"])
                        if event.payload.get("observed_cost") is not None
                        else current.observed_cost
                    ),
                }
            )
        elif event_type == "external_effect_abandoned":
            if current.status not in {"dispatched", "observed", "reconciled"}:
                _invalid(event, f"cannot abandon reservation from {current.status}")
            current = current.model_copy(update={"status": "abandoned"})
        reservations[event.reservation_id] = current
        attempts = dict(projection.attempts)
        if (
            event.attempt_id is not None
            and event.attempt_id in attempts
            and event_type
            in {
                "external_effect_observed",
                "external_effect_reconciled",
            }
        ):
            attempt = attempts[event.attempt_id]
            observed_status = event.payload.get("status", attempt.status)
            attempt_update: dict[str, Any] = {
                "provider_resource_handle": event.payload.get("provider_resource_handle"),
                "worker_identity": event.payload.get("worker_identity"),
                "observations": tuple(event.payload.get("observations", attempt.observations)),
                "event_refs": (*attempt.event_refs, event.event_id),
            }
            if observed_status not in {"succeeded", "failed", "cancelled"}:
                attempt_update["status"] = observed_status
                if observed_status == "running" and attempt.started_at is None:
                    attempt_update["started_at"] = event.occurred_at
            attempts[event.attempt_id] = attempt.model_copy(update=attempt_update)
        return projection.model_copy(
            update={
                **update,
                "reservations": reservations,
                "attempts": attempts,
                "status": current.status,
            }
        )
    if event_type in {
        "attempt_started",
        "attempt_heartbeat_observed",
        "attempt_terminal_observed",
        "attempt_state_unknown",
    }:
        attempts = dict(projection.attempts)
        existing = attempts.get(event.attempt_id or "")
        if event_type == "attempt_started":
            if existing is not None:
                _invalid(event, "attempt already exists")
            attempt = Attempt.model_validate(event.payload["attempt"])
            attempt = attempt.model_copy(
                update={"event_refs": (*attempt.event_refs, event.event_id)}
            )
        elif existing is None:
            _invalid(event, "event references an unknown attempt")
        elif event_type == "attempt_heartbeat_observed":
            if existing.status != "running":
                _invalid(event, "heartbeats require a running attempt")
            attempt = existing.model_copy(
                update={
                    "observations": (*existing.observations, dict(event.payload)),
                    "event_refs": (*existing.event_refs, event.event_id),
                }
            )
        elif event_type == "attempt_state_unknown":
            if existing.status in {"succeeded", "failed", "cancelled"}:
                _invalid(event, "terminal attempt cannot become unknown")
            attempt = existing.model_copy(
                update={
                    "status": "unknown",
                    "event_refs": (*existing.event_refs, event.event_id),
                }
            )
        else:
            if existing.status in {"succeeded", "failed", "cancelled"}:
                _invalid(event, "attempt already has a terminal outcome")
            attempt = existing.model_copy(
                update={
                    "status": event.payload["status"],
                    "terminal_at": event.occurred_at,
                    "exit_classification": event.payload["exit_classification"],
                    "observations": (*existing.observations, dict(event.payload)),
                    "event_refs": (*existing.event_refs, event.event_id),
                }
            )
        attempts[attempt.attempt_id] = attempt
        return projection.model_copy(
            update={**update, "attempts": attempts, "status": attempt.status}
        )
    if event_type in {"output_staged", "publication_committed"}:
        if event.attempt_id is None or event.attempt_id not in projection.attempts:
            _invalid(event, "event references an unknown attempt")
        refs = tuple(
            dict.fromkeys((*projection.artifact_refs, *event.payload.get("artifact_refs", ())))
        )
        attempts = dict(projection.attempts)
        attempt = attempts[event.attempt_id]
        attempt_update: dict[str, Any] = {
            "event_refs": (*attempt.event_refs, event.event_id),
        }
        if event_type == "publication_committed":
            attempt_update["publication_refs"] = (
                *attempt.publication_refs,
                event.payload["publication_id"],
            )
        attempts[event.attempt_id] = attempt.model_copy(update=attempt_update)
        status = "published" if event_type == "publication_committed" else "outputs_staged"
        return projection.model_copy(
            update={
                **update,
                "artifact_refs": refs,
                "attempts": attempts,
                "status": status,
            }
        )
    if event_type == "publication_failed":
        return projection.model_copy(update={**update, "status": "publication_failed"})
    if event_type == "invocation_satisfied":
        return projection.model_copy(update={**update, "status": "satisfied"})
    if event_type == "retry_admitted":
        return projection.model_copy(update={**update, "status": "retry_admitted"})
    if event_type == "invocation_permanently_failed":
        return projection.model_copy(update={**update, "status": "permanently_failed"})
    if event_type == "operator_action_required":
        diagnostics = (*projection.diagnostics, dict(event.payload))
        return projection.model_copy(
            update={**update, "diagnostics": diagnostics, "status": "operator_action_required"}
        )
    raise AssertionError(f"unhandled controller event type {event_type!r}")


def _invalid(event: ControllerEvent, reason: str) -> None:
    raise ControllerProtocolError(
        f"invalid {event.event_type} transition at sequence {event.sequence}: {reason}"
    )


class DurableController:
    """One controller authority over durable intent, effects, and attempts."""

    def __init__(
        self,
        store: ControllerEventStore,
        *,
        producer_id: str,
        clock: Callable[[], datetime] = utc_now,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.store = store
        self.producer_id = producer_id
        self.clock = clock
        self.id_factory = id_factory

    def project(self, intent_id: str) -> ControllerProjection:
        return project_controller_events(self.store.read_all(), intent_id=intent_id)

    def admit_intent(self, intent: RunIntent) -> ControllerProjection:
        self._record(
            "intent_admitted",
            intent.intent_id,
            invocation_id=intent.invocation_id,
            payload={"intent": intent.model_dump(mode="json")},
            idempotency_key=f"intent:{intent.intent_id}:admitted",
        )
        return self.project(intent.intent_id)

    def select_backend_plan(self, intent_id: str, plan: BackendPlan) -> ControllerProjection:
        projection = self.project(intent_id)
        if projection.intent is None or projection.intent.invocation_id != plan.invocation_id:
            raise ControllerProtocolError("backend plan does not realize the admitted invocation")
        self._record(
            "backend_plan_selected",
            intent_id,
            invocation_id=plan.invocation_id,
            backend_plan_id=plan.backend_plan_id,
            payload={"backend_plan": plan.model_dump(mode="json")},
            idempotency_key=f"intent:{intent_id}:backend:{plan.backend_plan_id}",
        )
        return self.project(intent_id)

    def reserve_effect(
        self,
        intent_id: str,
        plan: BackendPlan,
        *,
        effect_class: str,
        normalized_parameters: Mapping[str, Any],
        expires_at: datetime | None = None,
        reservation_id: str | None = None,
        external_effect_key: str | None = None,
    ) -> EffectReservation:
        projection = self.project(intent_id)
        if projection.backend_plan_id != plan.backend_plan_id:
            raise ControllerProtocolError("effect reservation requires the selected backend plan")
        now = self.clock()
        requires_authentication = (
            projection.intent is not None
            and projection.intent.operator_gate_policy == "per-effect-authentication"
            and effect_class in {"cloud-machine-acquisition", "billable-publication"}
        ) or (plan.expected_cost is not None and effect_class == "cloud-machine-acquisition")
        reservation = EffectReservation(
            reservation_id=reservation_id or self.id_factory(),
            intent_id=intent_id,
            invocation_id=plan.invocation_id,
            backend_plan_id=plan.backend_plan_id,
            effect_class=effect_class,
            external_effect_key=external_effect_key or plan.external_effect_key,
            normalized_parameters=dict(normalized_parameters),
            backend_id=plan.backend_id,
            machine=plan.machine,
            expected_cost=plan.expected_cost,
            requires_authentication=requires_authentication,
            created_at=now,
            expires_at=expires_at,
        )
        self._record(
            "effect_reservation_created",
            intent_id,
            invocation_id=plan.invocation_id,
            backend_plan_id=plan.backend_plan_id,
            reservation_id=reservation.reservation_id,
            payload={"reservation": reservation.model_dump(mode="json")},
            idempotency_key=f"reservation:{reservation.reservation_id}:created",
        )
        return reservation

    def authenticate_reservation(
        self,
        intent_id: str,
        reservation_id: str,
        *,
        operator_identity: str,
        authentication_id: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> ReservationProjection:
        if not operator_identity.strip() or not authentication_id.strip():
            raise OperatorGateError("reservation authentication requires durable operator evidence")
        projection = self._expire_due(intent_id)
        reservation = _reservation(projection, reservation_id)
        if reservation.status == "expired":
            raise OperatorGateError("effect reservation has expired")
        self._record(
            "effect_reservation_authenticated",
            intent_id,
            invocation_id=reservation.reservation.invocation_id,
            backend_plan_id=reservation.reservation.backend_plan_id,
            reservation_id=reservation_id,
            payload={
                "operator_identity": operator_identity,
                "authentication_id": authentication_id,
                "evidence": dict(evidence or {}),
            },
            idempotency_key=f"reservation:{reservation_id}:auth:{authentication_id}",
        )
        return _reservation(self.project(intent_id), reservation_id)

    def expire_reservations(self, intent_id: str) -> ControllerProjection:
        return self._expire_due(intent_id)

    async def dispatch(
        self,
        intent_id: str,
        reservation_id: str,
        adapter: EffectAdapter,
    ) -> Attempt:
        projection = self._expire_due(intent_id)
        reserved = _reservation(projection, reservation_id)
        if reserved.status == "expired":
            raise OperatorGateError("effect reservation has expired")
        existing = next(
            (
                attempt
                for attempt in projection.attempts.values()
                if attempt.reservation_id == reservation_id
            ),
            None,
        )
        if reserved.status in {"dispatched", "observed", "reconciled"}:
            if existing is None:
                raise ControllerProtocolError(
                    "dispatched reservation has no Attempt; reconcile it before retrying"
                )
            return existing
        if reserved.status not in {"inert", "authenticated"}:
            raise ControllerProtocolError(
                f"cannot dispatch effect reservation from {reserved.status}"
            )
        if reserved.reservation.requires_authentication and reserved.status != "authenticated":
            raise OperatorGateError("effect reservation has not been authenticated")
        attempt_id = "attempt-" + hashlib.sha256(reservation_id.encode()).hexdigest()[:24]
        self._record(
            "external_effect_dispatched",
            intent_id,
            invocation_id=reserved.reservation.invocation_id,
            backend_plan_id=reserved.reservation.backend_plan_id,
            reservation_id=reservation_id,
            attempt_id=attempt_id,
            payload={"external_effect_key": reserved.reservation.external_effect_key},
            idempotency_key=f"reservation:{reservation_id}:dispatched",
        )
        pending = Attempt(
            attempt_id=attempt_id,
            invocation_id=reserved.reservation.invocation_id,
            backend_plan_id=reserved.reservation.backend_plan_id,
            reservation_id=reservation_id,
            status="pending",
        )
        self._record(
            "attempt_started",
            intent_id,
            invocation_id=pending.invocation_id,
            backend_plan_id=pending.backend_plan_id,
            reservation_id=reservation_id,
            attempt_id=attempt_id,
            payload={"attempt": pending.model_dump(mode="json")},
            idempotency_key=f"attempt:{attempt_id}:started",
        )
        try:
            observation = await adapter.dispatch(reserved.reservation)
        except EffectDispatchAmbiguous as exc:
            self._record(
                "attempt_state_unknown",
                intent_id,
                reservation_id=reservation_id,
                attempt_id=attempt_id,
                payload={"reason": str(exc)},
                idempotency_key=f"attempt:{attempt_id}:unknown",
            )
            return self.project(intent_id).attempts[attempt_id]
        except Exception as exc:
            self._record(
                "attempt_terminal_observed",
                intent_id,
                reservation_id=reservation_id,
                attempt_id=attempt_id,
                payload={
                    "status": "failed",
                    "exit_classification": type(exc).__name__,
                    "error": str(exc),
                },
                idempotency_key=f"attempt:{attempt_id}:failed",
            )
            return self.project(intent_id).attempts[attempt_id]
        self._record_observation(
            intent_id, reservation_id, attempt_id, observation, reconciled=False
        )
        return self.project(intent_id).attempts[attempt_id]

    async def reconcile(
        self,
        intent_id: str,
        adapters: Mapping[str, EffectAdapter],
    ) -> ControllerProjection:
        """Recover from records first, then query only unresolved reserved effects."""

        projection = self._expire_due(intent_id)
        for reservation_id, reserved in tuple(projection.reservations.items()):
            if reserved.status not in {"dispatched", "observed", "reconciled"}:
                continue
            terminal_attempt = next(
                (
                    attempt
                    for attempt in projection.attempts.values()
                    if attempt.reservation_id == reservation_id
                    and attempt.status in {"succeeded", "failed", "cancelled"}
                ),
                None,
            )
            if terminal_attempt is not None:
                continue
            adapter = adapters.get(reserved.reservation.backend_id)
            if adapter is None:
                self._record(
                    "operator_action_required",
                    intent_id,
                    reservation_id=reservation_id,
                    payload={
                        "code": "backend_adapter_unavailable",
                        "backend_id": reserved.reservation.backend_id,
                    },
                    idempotency_key=f"reservation:{reservation_id}:adapter-unavailable",
                )
                continue
            observation = await adapter.reconcile(reserved.reservation)
            if observation is None:
                self._record(
                    "operator_action_required",
                    intent_id,
                    reservation_id=reservation_id,
                    payload={
                        "code": "effect_state_unresolved",
                        "external_effect_key": reserved.reservation.external_effect_key,
                    },
                    idempotency_key=f"reservation:{reservation_id}:unresolved",
                )
                continue
            attempt_id = next(
                (
                    attempt.attempt_id
                    for attempt in projection.attempts.values()
                    if attempt.reservation_id == reservation_id
                ),
                self.id_factory(),
            )
            if attempt_id not in projection.attempts:
                pending = Attempt(
                    attempt_id=attempt_id,
                    invocation_id=reserved.reservation.invocation_id,
                    backend_plan_id=reserved.reservation.backend_plan_id,
                    reservation_id=reservation_id,
                    status="pending",
                )
                self._record(
                    "attempt_started",
                    intent_id,
                    reservation_id=reservation_id,
                    attempt_id=attempt_id,
                    payload={"attempt": pending.model_dump(mode="json")},
                    idempotency_key=f"attempt:{attempt_id}:started",
                )
            self._record_observation(
                intent_id, reservation_id, attempt_id, observation, reconciled=True
            )
            projection = self.project(intent_id)
        return self.project(intent_id)

    def cancel_reservation(self, intent_id: str, reservation_id: str, *, reason: str) -> None:
        reserved = _reservation(self.project(intent_id), reservation_id)
        if reserved.status not in {"inert", "authenticated"}:
            raise ControllerProtocolError(
                "a dispatched external effect requires a separately reserved cleanup effect"
            )
        self._record(
            "effect_reservation_cancelled",
            intent_id,
            reservation_id=reservation_id,
            payload={"reason": reason},
            idempotency_key=f"reservation:{reservation_id}:cancelled",
        )

    def request_cancellation(self, intent_id: str, *, reason: str) -> ControllerProjection:
        self._record(
            "intent_cancelled",
            intent_id,
            payload={"reason": reason},
            idempotency_key=f"intent:{intent_id}:cancelled",
        )
        return self.project(intent_id)

    def observe_attempt_terminal(
        self,
        intent_id: str,
        attempt_id: str,
        *,
        status: Literal["succeeded", "failed", "cancelled"],
        exit_classification: str,
        reservation_id: str | None = None,
    ) -> Attempt:
        self._record(
            "attempt_terminal_observed",
            intent_id,
            reservation_id=reservation_id,
            attempt_id=attempt_id,
            payload={"status": status, "exit_classification": exit_classification},
            idempotency_key=f"attempt:{attempt_id}:terminal",
        )
        return self.project(intent_id).attempts[attempt_id]

    def inspect_artifacts(self, intent_id: str) -> tuple[str, ...]:
        return self.project(intent_id).artifact_refs

    def record_output_staged(
        self,
        intent_id: str,
        *,
        attempt_id: str,
        artifact_refs: Sequence[str],
    ) -> ControllerProjection:
        if not artifact_refs or any(not ref.strip() for ref in artifact_refs):
            raise ControllerProtocolError("staged output requires non-empty artifact references")
        self._record(
            "output_staged",
            intent_id,
            attempt_id=attempt_id,
            payload={"artifact_refs": list(artifact_refs)},
            idempotency_key=f"attempt:{attempt_id}:outputs:{_refs_identity(artifact_refs)}",
        )
        return self.project(intent_id)

    def record_publication_committed(
        self,
        intent_id: str,
        *,
        attempt_id: str,
        artifact_refs: Sequence[str],
        publication_id: str,
    ) -> ControllerProjection:
        if not publication_id.strip():
            raise ControllerProtocolError("publication identity must be non-empty")
        self._record(
            "publication_committed",
            intent_id,
            attempt_id=attempt_id,
            payload={
                "artifact_refs": list(artifact_refs),
                "publication_id": publication_id,
            },
            idempotency_key=f"publication:{publication_id}:committed",
        )
        return self.project(intent_id)

    def _expire_due(self, intent_id: str) -> ControllerProjection:
        projection = self.project(intent_id)
        now = self.clock()
        for reservation_id, reserved in tuple(projection.reservations.items()):
            expires_at = reserved.reservation.expires_at
            if (
                reserved.status in {"inert", "authenticated"}
                and expires_at is not None
                and now >= expires_at
            ):
                self._record(
                    "effect_reservation_expired",
                    intent_id,
                    reservation_id=reservation_id,
                    payload={},
                    idempotency_key=f"reservation:{reservation_id}:expired",
                )
        return self.project(intent_id)

    def _record_observation(
        self,
        intent_id: str,
        reservation_id: str,
        attempt_id: str,
        observation: EffectObservation,
        *,
        reconciled: bool,
    ) -> None:
        payload = observation.model_dump(mode="json")
        observation_identity = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        self._record(
            "external_effect_observed",
            intent_id,
            reservation_id=reservation_id,
            attempt_id=attempt_id,
            payload=payload,
            idempotency_key=f"reservation:{reservation_id}:observed:{observation_identity}",
        )
        if reconciled:
            self._record(
                "external_effect_reconciled",
                intent_id,
                reservation_id=reservation_id,
                attempt_id=attempt_id,
                payload=payload,
                idempotency_key=f"reservation:{reservation_id}:reconciled:{observation_identity}",
            )
        if observation.status in {"succeeded", "failed", "cancelled"}:
            self.observe_attempt_terminal(
                intent_id,
                attempt_id,
                status=observation.status,
                exit_classification=observation.status,
                reservation_id=reservation_id,
            )

    def _record(
        self,
        event_type: ControllerEventType,
        intent_id: str,
        *,
        invocation_id: str | None = None,
        backend_plan_id: str | None = None,
        reservation_id: str | None = None,
        attempt_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ControllerEvent:
        now = self.clock()
        event = self.store.append(
            event_type=event_type,
            intent_id=intent_id,
            producer_id=self.producer_id,
            occurred_at=now,
            observed_at=now,
            invocation_id=invocation_id,
            backend_plan_id=backend_plan_id,
            reservation_id=reservation_id,
            attempt_id=attempt_id,
            payload=payload,
            idempotency_key=idempotency_key,
            validator=lambda events, candidate: project_controller_events(
                (*events, candidate), intent_id=intent_id
            ),
        )
        return event


def _reservation(
    projection: ControllerProjection,
    reservation_id: str,
) -> ReservationProjection:
    try:
        return projection.reservations[reservation_id]
    except KeyError as exc:
        raise ControllerProtocolError(f"unknown effect reservation {reservation_id!r}") from exc


def _refs_identity(refs: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(refs))).hexdigest()


__all__ = [
    "CONTROLLER_EVENT_SCHEMA_ID",
    "CONTROLLER_EVENT_SCHEMA_VERSION",
    "CONTROLLER_PROJECTION_SCHEMA_ID",
    "CONTROLLER_PROJECTION_SCHEMA_VERSION",
    "EFFECT_RESERVATION_SCHEMA_ID",
    "EFFECT_RESERVATION_SCHEMA_VERSION",
    "RUN_INTENT_SCHEMA_ID",
    "RUN_INTENT_SCHEMA_VERSION",
    "ControllerConflictError",
    "ControllerEvent",
    "ControllerEventStore",
    "ControllerProjection",
    "ControllerProtocolError",
    "DurableController",
    "EffectAdapter",
    "EffectDispatchAmbiguous",
    "EffectObservation",
    "EffectReservation",
    "OperatorGateError",
    "ReservationProjection",
    "RunIntent",
    "controller_event_from_document",
    "effect_reservation_from_document",
    "project_controller_events",
    "run_intent_from_document",
]
