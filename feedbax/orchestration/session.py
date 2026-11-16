"""Durable venue acquisition, attachment and owned-resource teardown."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import ConfigDict, Field, SecretStr

from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.execution_target import (
    LOCAL_TARGET_ID, CloseWhenIdle as CloseWhenIdle, KeepOpen as KeepOpen, OpeningObservation, OpeningStep,
    SessionLifecycle, SessionTarget, TargetId, TargetRealization,
)
from feedbax.host.client import HostClient
from feedbax.host.io import private_directory
from feedbax.host.protocol import JsonObject, SessionAccess


class SessionError(RuntimeError):
    pass


class ProviderUnreachable(SessionError):
    """The configured provider did not answer its control-plane request."""


class SessionOpeningFailed(SessionError):
    """An opening stopped and requires reconciling its owned resources."""


class SessionOpeningUnconfirmed(SessionOpeningFailed):
    """The create may have succeeded without a confirmed resource identity."""


class SessionOpeningRejected(SessionError):
    """The driver rejected the request before it could acquire a venue."""


class SessionModel(StrictModel):
    model_config = ConfigDict(frozen=True)


type Nonempty = Annotated[str, Field(min_length=1)]
type Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type Positive = Annotated[int, Field(gt=0, strict=True)]
type Nonnegative = Annotated[int, Field(ge=0, strict=True)]


class SessionRequest(SessionModel):
    id: UUID
    target: Nonempty
    resources: JsonObject
    environment: Digest
    lifecycle: SessionLifecycle
    maximum_microusd: Nonnegative
    profile: TargetId | None = Field(default=None, exclude_if=lambda profile: profile is None)

    @property
    def placement(self) -> SessionTarget:
        return SessionTarget(
            kind="session", profile=self.profile or self.target, driver=self.target, session_id=self.id,
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            canonical_json_v2_bytes(self.model_dump(mode="json"))
        ).hexdigest()


class SessionAuthorization(SessionModel):
    request_digest: Digest
    operator: Nonempty
    method: Nonempty
    evidence: Nonempty
    expires_at_ms: Positive
    ceiling_microusd: Nonnegative


class SessionLaunch(SessionModel):
    request: SessionRequest
    authorization: SessionAuthorization
    adoption_sha256: Digest
    acquired_at_ms: Positive
    hourly_microusd: Nonnegative

    @property
    def intent(self) -> str:
        return f"feedbax-session-{self.request.digest}"

    @property
    def ends_at_ms(self) -> int:
        expiry = self.authorization.expires_at_ms
        if not self.hourly_microusd:
            return expiry
        budget_ms = self.request.maximum_microusd * 3_600_000 // self.hourly_microusd
        return min(expiry, self.acquired_at_ms + budget_ms)


class ProviderResource(SessionModel):
    id: Nonempty
    intent: str | None
    kind: Literal["venue", "volume", "secret"]


class ProviderMachineObservation(SessionModel):
    state: Literal["starting", "on", "off"]
    observed_at_ms: Positive
    billed_microusd: Nonnegative | None = None
    billing_started_at_ms: Positive | None = None


class SessionRealization(SessionModel):
    launch: SessionLaunch
    environment: Digest
    probe: JsonObject


class CloseEvidence(SessionModel):
    checked_at_ms: Positive
    owned: tuple[ProviderResource, ...]
    remaining: tuple[ProviderResource, ...]
    note: str | None = Field(default=None, exclude_if=lambda value: value is None)


class SessionRecord(SessionModel):
    launch: SessionLaunch
    phase: Literal["opening", "open", "closing", "closed"] = "opening"
    venue: ProviderResource | None = None
    realization: SessionRealization | None = None
    epoch: Nonnegative = 0
    close_evidence: CloseEvidence | None = None
    opening_observations: tuple[OpeningObservation, ...] = Field(default=(), exclude_if=lambda values: not values)
    opening_failure: str | None = Field(default=None, exclude_if=lambda value: value is None)
    unconfirmed_create: bool = Field(default=False, exclude_if=lambda value: not value)

    @property
    def placement(self) -> SessionTarget:
        realized = self.realization
        if realized is None or realized.launch != self.launch or realized.environment != self.launch.request.environment:
            raise SessionError("The venue has no matching realized target.")
        facts = realized.probe.get("placement")
        if facts is not None and not isinstance(facts, dict):
            raise SessionError("The venue reported invalid placement evidence.")
        target = self.launch.request.placement
        return SessionTarget(
            kind=target.kind, profile=target.profile, driver=target.driver, session_id=target.session_id,
            realization=TargetRealization(environment=realized.environment, placement=facts),
        )


class SessionProvider(Protocol):
    """A provider for one target. The launch intent is the acquisition idempotency key.

    Acquisition must set the intent on creation and reconcile by that key even when
    the original reply is lost. A used intent cannot create a second venue, even
    after removal. An open venue enforces ends_at_ms independently of the
    controller and Host. Opening is reconciled by the acquisition intent.
    Realization and close can be
    retried. Remove must compare the resource identity and intent before acting,
    and treat an already absent resource as success. Adoption reads realization
    evidence from venue custody.
    """

    target: str
    supports_idle_close: bool

    @property
    def enforces_cutoff(self) -> bool: ...

    def hourly_microusd(self, request: SessionRequest) -> int: ...

    def inventory(self) -> Sequence[ProviderResource]: ...

    def machine_observation(self, resource: ProviderResource) -> ProviderMachineObservation: ...

    def check_opening_deadline(self, launch: SessionLaunch) -> None:
        """Check an opening bound owned by the provider before querying inventory."""
        ...

    def acquire(self, launch: SessionLaunch) -> ProviderResource: ...

    def realize(
        self, resource: ProviderResource, launch: SessionLaunch, *,
        adoption_token: str, controller_token: str, opening: Callable[[OpeningStep], None],
    ) -> SessionRealization: ...

    def realization(self, resource: ProviderResource) -> SessionRealization: ...

    def client(self, resource: ProviderResource, access: SessionAccess | None) -> HostClient: ...

    def remove(self, resource: ProviderResource) -> None: ...

    def claim_opening_close(
        self, resource: ProviderResource | None, launch: SessionLaunch,
    ) -> tuple[ProviderResource, ...] | None:
        """Prevent initialization and return known resources, or leave closure unclaimed."""
        ...


def session_capability_problem(provider: SessionProvider, lifecycle: SessionLifecycle) -> str | None:
    if isinstance(lifecycle, CloseWhenIdle) and not provider.supports_idle_close:
        return "The provider cannot enforce close-when-idle on the venue."
    if not provider.enforces_cutoff:
        return "The running venue cannot independently enforce the authorized cutoff."
    return None


class _ControllerRecord(SessionModel):
    session: SessionRecord
    token: SecretStr


def _capability_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class SessionDriver:
    """Own one target's venues. Jobs go through the attached, ordinary Host."""

    def __init__(
        self, root: Path, provider: SessionProvider, *,
        verify_authorization: Callable[[SessionAuthorization], None],
        clock_ms: Callable[[], int] = _now_ms,
    ):
        self.root = private_directory(root.resolve())
        self.provider = provider
        self._verify_authorization = verify_authorization
        self._clock_ms = clock_ms
        with self.connection() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS sessions(
                    id TEXT PRIMARY KEY, state TEXT NOT NULL, token TEXT NOT NULL
                )"""
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.root / "sessions.sqlite3")) as db, db:
            yield db

    @contextmanager
    def _exclusive(self, session_id: UUID, *, approval: bool = False) -> Iterator[None]:
        name = f"{session_id}.approval.lock" if approval else f"{session_id}.lock"
        with (self.root / name).open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _read(self, session_id: UUID) -> _ControllerRecord | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT state, token FROM sessions WHERE id=?", (str(session_id),)
            ).fetchone()
        if row is None:
            return None
        state, token = row
        return _ControllerRecord(session=SessionRecord.model_validate_json(state), token=token)

    def _require(self, session_id: UUID) -> _ControllerRecord:
        control = self._read(session_id)
        if control is None:
            raise SessionError(f"Unknown session {session_id}.")
        return control

    def record(self, session_id: UUID) -> SessionRecord:
        return self._require(session_id).session

    def find_record(self, session_id: UUID) -> SessionRecord | None:
        control = self._read(session_id)
        return control.session if control is not None else None

    def records(self) -> tuple[SessionRecord, ...]:
        with self.connection() as db:
            return tuple(SessionRecord.model_validate_json(state) for (state,) in db.execute(
                "SELECT state FROM sessions ORDER BY rowid",
            ))

    def _write(self, state: SessionRecord, token: str) -> None:
        with self.connection() as db:
            db.execute(
                """INSERT INTO sessions VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET state=excluded.state, token=excluded.token""",
                (str(state.launch.request.id), state.model_dump_json(), token),
            )

    def _validate(
        self, request: SessionRequest, authorization: SessionAuthorization, *, require_current: bool = True,
    ) -> None:
        if request.target != self.provider.target:
            raise SessionError(f"Session target {request.target!r} does not match the provider.")
        if request.profile == LOCAL_TARGET_ID:
            raise SessionError("A venue session cannot use the This Mac target.")
        if authorization.request_digest != request.digest:
            raise SessionError("Authorization is bound to a different session request.")
        if require_current and authorization.expires_at_ms <= self._clock_ms():
            raise SessionError("Session authorization has expired.")
        if authorization.ceiling_microusd < request.maximum_microusd:
            raise SessionError("The requested maximum exceeds the authorized cost ceiling.")
        self._verify_authorization(authorization)

    def inventory(self) -> tuple[ProviderResource, ...]:
        return tuple(self.provider.inventory())

    def _find_venue(self, intent: str, inventory: Sequence[ProviderResource]) -> ProviderResource | None:
        matches = [item for item in inventory if item.kind == "venue" and item.intent == intent]
        if len(matches) > 1:
            raise SessionError(f"Provider reports multiple venues for acquisition intent {intent}.")
        return matches[0] if matches else None

    @staticmethod
    def _check_realization(launch: SessionLaunch, realized: SessionRealization) -> None:
        if realized.launch != launch or realized.environment != launch.request.environment or not realized.probe:
            raise SessionError("Venue realization does not match the authorized session environment.")

    @staticmethod
    def _check_owner(launch: SessionLaunch, adoption_token: str) -> None:
        if not hmac.compare_digest(launch.adoption_sha256, _capability_digest(adoption_token)):
            raise SessionError("The adoption capability does not match the session owner.")

    def open(
        self, request: SessionRequest, authorization: SessionAuthorization, *, adoption_token: str,
        expected_hourly_microusd: int | None = None,
    ) -> SessionRecord:
        request = request.model_copy(deep=True)
        self._validate(request, authorization, require_current=False)
        if len(adoption_token) < 32:
            raise SessionError("Session adoption requires a private capability of at least 32 characters.")
        if problem := session_capability_problem(self.provider, request.lifecycle):
            raise SessionError(problem)
        with self._exclusive(request.id):
            control = self._read(request.id)
            if control is None:
                if authorization.expires_at_ms <= self._clock_ms():
                    raise SessionOpeningRejected("Approval expired before the machine opened.")
                if self._find_venue(f"feedbax-session-{request.digest}", self.inventory()) is not None:
                    raise SessionError("This session already has a venue. Adopt it explicitly.")
                hourly_microusd = self.provider.hourly_microusd(request)
                if expected_hourly_microusd is not None and hourly_microusd != expected_hourly_microusd:
                    raise SessionOpeningRejected("The machine rate changed. Review a new quote.")
                launch = SessionLaunch(
                    request=request, authorization=authorization, acquired_at_ms=self._clock_ms(),
                    adoption_sha256=_capability_digest(adoption_token),
                    hourly_microusd=hourly_microusd,
                )
                state = SessionRecord(launch=launch)
                token = secrets.token_urlsafe()
                self._write(state, token)
            else:
                state, secret = control.session, control.token
                token = secret.get_secret_value()
                if state.launch.request != request or state.launch.authorization != authorization:
                    raise SessionError("Session ID already belongs to a different request or authorization.")
                if expected_hourly_microusd is not None and state.launch.hourly_microusd != expected_hourly_microusd:
                    raise SessionError("The machine rate differs from the recorded quote.")
                self._check_owner(state.launch, adoption_token)
                if state.phase != "opening":
                    if authorization.expires_at_ms <= self._clock_ms():
                        raise SessionError("Session authorization has expired.")
                    return state

            launch = state.launch

            def report_opening(step: OpeningStep) -> None:
                nonlocal state
                last = state.opening_observations[-1] if state.opening_observations else None
                if last is None or last.step != step:
                    events = (*state.opening_observations, OpeningObservation(step=step, observed_at_ms=self._clock_ms()))
                    state = state.model_copy(update={"opening_observations": events})
                    self._write(state, token)

            try:
                if state.opening_failure is not None:
                    raise SessionOpeningFailed(state.opening_failure)
                if state.venue is None:
                    report_opening("acquiring")
                    self.provider.check_opening_deadline(launch)
                if self._clock_ms() >= launch.ends_at_ms:
                    raise SessionOpeningFailed("The machine reached its time or cost limit while opening.")
                resource = self._find_venue(launch.intent, self.inventory())
                if state.venue is None:
                    self.provider.check_opening_deadline(launch)
                if state.venue is not None and resource != state.venue:
                    raise SessionOpeningFailed("The machine disappeared or changed identity while opening.")
                if resource is None:
                    resource = self.provider.acquire(launch)
                if resource.intent != launch.intent or resource.kind != "venue":
                    raise SessionError("The acquired venue does not carry this session's intent.")
                state = state.model_copy(update={"venue": resource})
                self._write(state, token)
                report_opening("connecting")
                realized = self.provider.realize(
                    resource, launch, adoption_token=adoption_token, controller_token=token, opening=report_opening,
                )
            except SessionOpeningFailed as error:
                state = state.model_copy(update={"opening_failure": str(error)})
                if isinstance(error, SessionOpeningUnconfirmed):
                    state = state.model_copy(update={"phase": "closing", "unconfirmed_create": True})
                self._write(state, token)
                try:
                    self._close(state, token)
                except Exception as cleanup:
                    raise SessionError(f"{error} Owned resource closure is unresolved: {cleanup}") from error
                raise
            self._check_realization(launch, realized)
            state = state.model_copy(update={"phase": "open", "realization": realized})
            self._write(state, token)
            return state

    @staticmethod
    def _access(record: _ControllerRecord) -> SessionAccess:
        state = record.session
        if state.phase != "open" or state.venue is None:
            raise SessionError(f"Session {state.launch.request.id} has no open Host.")
        return SessionAccess(str(state.launch.request.id), state.epoch, record.token.get_secret_value())

    def session_access(self, session_id: UUID) -> SessionAccess:
        return self._access(self._require(session_id))

    def attach(self, session_id: UUID, *, access: SessionAccess | None = None) -> HostClient:
        record = self._require(session_id)
        current = self._access(record)
        if access is not None and access != current:
            raise SessionError("The local controller fence changed while attaching to the session.")
        assert record.session.venue is not None
        return self.provider.client(record.session.venue, current)

    def adopt(
        self, request: SessionRequest, authorization: SessionAuthorization, *, adoption_token: str,
    ) -> SessionRecord:
        self._validate(request, authorization)
        with self._exclusive(request.id):
            existing = self._read(request.id)
            if existing is not None and existing.session.opening_failure is not None:
                raise SessionError("This failed opening can only reconcile and close its resources.")
            resource = self._find_venue(f"feedbax-session-{request.digest}", self.inventory())
            if resource is None:
                raise SessionError("No venue carries the authorized session's intent.")
            realized = self.provider.realization(resource)
            launch = realized.launch
            if launch.request != request or launch.authorization != authorization:
                raise SessionError("Venue custody identifies a different request or authorization.")
            self._check_realization(launch, realized)
            self._check_owner(launch, adoption_token)
            if launch.ends_at_ms <= self._clock_ms():
                raise SessionError("Session has exhausted its authorized time or cost ceiling.")
            local = self._read(request.id)
            if local is not None and local.session.phase in ("closing", "closed"):
                raise SessionError("A closing or closed session cannot be adopted.")
            host = self.provider.client(resource, None)
            identity = str(request.id)
            operator = authorization.operator
            epoch = host.session_epoch(identity, operator, adoption_token)
            token = secrets.token_urlsafe()
            adopted = host.adopt_session(identity, operator, epoch, adoption_token, token)
            state = SessionRecord(launch=launch, phase="open", venue=resource, realization=realized, epoch=adopted)
            self._write(state, token)
            return state

    def recover_close(
        self, request: SessionRequest, authorization: SessionAuthorization, *, adoption_token: str,
    ) -> CloseEvidence:
        """Recover a prior venue close claim after losing the controller journal."""
        self._validate(request, authorization, require_current=False)
        with self._exclusive(request.id):
            local = self._read(request.id)
            if local is not None:
                if local.session.launch.request != request or local.session.launch.authorization != authorization:
                    raise SessionError("The local session belongs to a different request or authorization.")
                self._check_owner(local.session.launch, adoption_token)
                if local.session.phase in ("closing", "closed"):
                    return self._close(local.session, local.token.get_secret_value())
            resource = self._find_venue(f"feedbax-session-{request.digest}", self.inventory())
            if resource is None:
                raise SessionError("No venue remains from which to recover close authority.")
            realized = self.provider.realization(resource)
            launch = realized.launch
            if launch.request != request or launch.authorization != authorization:
                raise SessionError("Venue custody identifies a different request or authorization.")
            self._check_realization(launch, realized)
            self._check_owner(launch, adoption_token)
            host = self.provider.client(resource, None)
            claim = host.recover_session_close(str(request.id), authorization.operator, adoption_token)
            if claim["session_id"] != str(request.id) or claim["operator"] != authorization.operator:
                raise SessionError("Recovered close authority does not identify this session.")
            state = SessionRecord(
                launch=launch, phase="closing", venue=resource, realization=realized, epoch=claim["epoch"],
            )
            return self._close(state, "")

    def close(self, session_id: UUID) -> CloseEvidence:
        with self._exclusive(session_id):
            control = self._require(session_id)
            return self._close(control.session, control.token.get_secret_value())

    def reconcile_unconfirmed_creates(self) -> None:
        """Check expired acquisition intents for late resources on controller startup."""
        records = [
            record for record in self.records()
            if record.unconfirmed_create and record.launch.ends_at_ms <= self._clock_ms()
        ]
        if not records:
            return
        intents = {resource.intent for resource in self.inventory()}
        for record in records:
            if record.launch.intent not in intents and record.phase == "closed":
                continue
            session_id = record.launch.request.id
            with self._exclusive(session_id):
                control = self._require(session_id)
                closing = control.session.model_copy(update={"phase": "closing"})
                self._write(closing, control.token.get_secret_value())
                self._close(closing, control.token.get_secret_value())

    def _close(self, state: SessionRecord, token: str) -> CloseEvidence:
        if state.phase == "closed":
            assert state.close_evidence is not None
            return state.close_evidence
        session_id = state.launch.request.id
        resources = self.inventory()
        owned = tuple(item for item in resources if item.intent == state.launch.intent)
        previous = state.close_evidence.owned if state.close_evidence is not None else ()
        current_venue = self._find_venue(state.launch.intent, owned)
        recorded = state.venue or current_venue
        venues = (recorded,) if recorded is not None else ()
        known = tuple(dict.fromkeys((*previous, *venues, *owned)))
        known_ids = {(item.kind, item.id) for item in known}
        claimed: tuple[ProviderResource, ...] = ()
        if any(
            (item.kind, item.id) in known_ids and item.intent != state.launch.intent for item in resources
        ) or (current_venue is not None and current_venue != recorded):
            raise SessionError(f"Provider custody changed for session {session_id}.")
        end_reached = self._clock_ms() >= state.launch.ends_at_ms
        recheck_unconfirmed = state.unconfirmed_create and (current_venue is not None or end_reached and recorded is None)
        if state.phase != "closing" or recheck_unconfirmed:
            unopened = (
                self.provider.claim_opening_close(current_venue, state.launch)
                if (state.phase == "opening" or state.unconfirmed_create)
                and (current_venue is not None or recorded is None)
                else None
            )
            if unopened is not None:
                claimed = unopened
            elif current_venue is not None:
                access = SessionAccess(str(session_id), state.epoch, token)
                claim = self.provider.client(current_venue, access).claim_session_close()
                if (
                    claim["session_id"] != str(session_id) or claim["epoch"] != state.epoch
                    or claim["operator"] != state.launch.authorization.operator
                ):
                    raise SessionError("The venue close claim does not identify this controller.")
            elif recorded is None:
                raise SessionError("Opening has no reconciled venue identity or definitive create rejection.")
        if any(item.intent != state.launch.intent for item in claimed):
            raise SessionError("Opening close claim includes a resource with a different intent.")
        known = tuple(dict.fromkeys((*known, *claimed)))
        known_ids.update((item.kind, item.id) for item in claimed)
        pending = state.close_evidence.remaining if state.close_evidence is not None else ()
        removable = tuple(dict.fromkeys((*pending, *owned, *claimed)))
        evidence = CloseEvidence(checked_at_ms=self._clock_ms(), owned=known, remaining=removable)
        state = state.model_copy(update={"phase": "closing", "venue": recorded, "close_evidence": evidence})
        self._write(state, token)
        waiting_for_end = state.unconfirmed_create and not end_reached
        for item in sorted(removable, key=lambda item: item.kind != "venue"):
            if not waiting_for_end or item.kind == "venue":
                self.provider.remove(item)
        remaining = tuple(
            item for item in self.inventory()
            if item.intent == state.launch.intent or (item.kind, item.id) in known_ids
        )
        phase = "closing" if remaining or waiting_for_end else "closed"
        note = (
            "Create unresolved. Exact-name listing was empty through the approved end time."
            if state.unconfirmed_create and phase == "closed" and recorded is None else None
        )
        evidence = CloseEvidence(checked_at_ms=self._clock_ms(), owned=known, remaining=remaining, note=note)
        self._write(state.model_copy(update={"phase": phase, "close_evidence": evidence}), token)
        if waiting_for_end:
            raise SessionError("Unconfirmed RunPod create remains in cleanup through the approved end time.")
        if remaining:
            raise SessionError(f"Owned resources remain for session {session_id}: {[r.id for r in remaining]}")
        return evidence
