"""Run-set state document persistence and advisory locking."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from feedbax.contracts.manifest import StrictModel
from feedbax.orchestration.repo_realization import RepoRealizationPlan


RUN_SET_STATE_SCHEMA_ID = "feedbax.orchestration.run_set_state"
RUN_SET_STATE_SCHEMA_VERSION_V1 = "feedbax.orchestration.run_set_state.v1"
RUN_SET_STATE_SCHEMA_VERSION_V2 = "feedbax.orchestration.run_set_state.v2"
RUN_SET_STATE_SCHEMA_VERSION_V3 = "feedbax.orchestration.run_set_state.v3"
RUN_SET_STATE_SCHEMA_VERSION = "feedbax.orchestration.run_set_state.v4"
ROW_STATUSES = ("pending", "launched", "ready", "running", "completed", "failed", "stopped")
STAGE_STATUSES = ("pending", "running", "completed", "failed", "skipped")

RowStatusName = Literal["pending", "launched", "ready", "running", "completed", "failed", "stopped"]
StageStatusName = Literal["pending", "running", "completed", "failed", "skipped"]
CheckStatusName = Literal["pass", "fail"]
AcquisitionIntentState = Literal[
    "intended",
    "acquired",
    "failed-unacquired",
    "ambiguous",
    "resolved-torn-down",
    "ambiguous-unresolved",
]


def utc_now() -> datetime:
    """Return timezone-aware UTC now with stable second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class StateLockError(RuntimeError):
    """Raised when a run-set state lock cannot be acquired."""


class PreflightCheckEntry(StrictModel):
    """One named preflight check result."""

    name: str
    status: CheckStatusName
    detail: str | None = None
    observed: Any = None


class StageState(StrictModel):
    """Durable state for one orchestration stage."""

    status: StageStatusName = "pending"
    attempts: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    checks: list[PreflightCheckEntry] = Field(default_factory=list)


class RowState(StrictModel):
    """Durable state for one row."""

    status: RowStatusName = "pending"
    event_seq_high_water_mark: int = -1
    pid: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_event_type: str | None = None
    event_discrepancies: list[dict[str, Any]] = Field(default_factory=list)
    collected_outputs: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class AcquisitionIntent(StrictModel):
    """One durable engine-owned record for exactly one provider create invocation."""

    intent_id: str
    datacenter_candidate: str | None = None
    config_identity: str
    intended_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    state: AcquisitionIntentState = "intended"
    pod_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    teardown_evidence: list[dict[str, Any]] = Field(default_factory=list)


class RunSetState(StrictModel):
    """Atomic JSON state document for one run set."""

    schema_id: Literal["feedbax.orchestration.run_set_state"] = RUN_SET_STATE_SCHEMA_ID
    schema_version: Literal[
        "feedbax.orchestration.run_set_state.v4"
    ] = RUN_SET_STATE_SCHEMA_VERSION
    run_set_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    current_stage: str | None = None
    stages: dict[str, StageState] = Field(default_factory=dict)
    rows: dict[str, RowState] = Field(default_factory=dict)
    provision_record: dict[str, Any] | None = None
    provisioning_attempts: list[dict[str, Any]] = Field(default_factory=list)
    acquisition_intents: list[AcquisitionIntent] = Field(default_factory=list)
    provisioning_stop_reason: str | None = None
    environment_fingerprint: str | None = None
    repo_realization_plan: RepoRealizationPlan | None = None
    budget_counters: dict[str, Any] = Field(default_factory=dict)
    certificate_ref: str | None = None
    registration_payload: dict[str, Any] | None = None
    abort_reason: str | None = None

    def stage(self, stage_id: str) -> StageState:
        """Return the state for ``stage_id``, defaulting to pending."""
        return self.stages.get(stage_id, StageState())

    def with_stage(self, stage_id: str, stage: StageState) -> "RunSetState":
        """Return a copy with one stage state replaced."""
        stages = dict(self.stages)
        stages[stage_id] = stage
        return self.model_copy(
            update={"stages": stages, "current_stage": stage_id, "updated_at": utc_now()}
        )

    def with_row(self, row_id: str, row: RowState) -> "RunSetState":
        """Return a copy with one row state replaced."""
        rows = dict(self.rows)
        rows[row_id] = row
        return self.model_copy(update={"rows": rows, "updated_at": utc_now()})


class RunSetStateStore:
    """Read/write helper for one run-set state document."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def load(self) -> RunSetState:
        """Load the current state document."""
        return RunSetState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, state: RunSetState, *, crash_before_replace: bool = False) -> Path:
        """Atomically write ``state`` using temp-file plus ``os.replace``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = state.model_copy(update={"updated_at": utc_now()}).model_dump(mode="json")
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if crash_before_replace:
                return tmp_path
            os.replace(tmp_path, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return self.path
        finally:
            if crash_before_replace:
                pass
            elif tmp_path.exists():
                tmp_path.unlink()

    def initialize(self, state: RunSetState) -> RunSetState:
        """Write initial state if absent; otherwise return existing state."""
        if self.path.exists():
            return self.load()
        self.save(state)
        return state

    @contextmanager
    def lock(self, *, break_stale: bool = False) -> Iterator[None]:
        """Acquire an advisory PID lock around state mutation."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "acquired_at": time.time(),
            "state_path": str(self.path),
        }
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError as exc:
                existing = _read_lock(self.lock_path)
                pid = existing.get("pid")
                if isinstance(pid, int) and not _pid_alive(pid) and break_stale:
                    self.lock_path.unlink(missing_ok=True)
                    continue
                state = "stale" if isinstance(pid, int) and not _pid_alive(pid) else "active"
                raise StateLockError(
                    f"run-set state lock is {state}: {self.lock_path} pid={pid!r}"
                ) from exc
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, sort_keys=True)
                    handle.write("\n")
                break
        try:
            yield
        finally:
            try:
                current = _read_lock(self.lock_path)
                if current.get("pid") == os.getpid():
                    self.lock_path.unlink(missing_ok=True)
            except OSError:
                pass


def _read_lock(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
