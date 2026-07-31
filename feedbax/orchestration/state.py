"""Run-set state document persistence and advisory locking."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
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
RUN_SET_STATE_SCHEMA_VERSION_V4 = "feedbax.orchestration.run_set_state.v4"
RUN_SET_STATE_SCHEMA_VERSION = "feedbax.orchestration.run_set_state.v5"
EMERGENCY_RUN_SET_RECORD_SCHEMA_ID = "feedbax.orchestration.emergency_run_set_record"
EMERGENCY_RUN_SET_RECORD_SCHEMA_VERSION = (
    "feedbax.orchestration.emergency_run_set_record.v1"
)
REGISTRATION_HISTORY_SCHEMA_ID = "feedbax.orchestration.registration_history"
REGISTRATION_HISTORY_SCHEMA_VERSION = "feedbax.orchestration.registration_history.v1"
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
PreservationState = Literal[
    "preserve-required",
    "preserved",
    "release-authorized",
    "unknown",
]

DEFAULT_CONTROL_RESERVE_BYTES = 1024 * 1024
DEFAULT_EMERGENCY_RESERVE_BYTES = 64 * 1024
DEFAULT_STATE_UPDATE_BYTES = 1024 * 1024
MAX_CONTROL_RESERVE_BYTES = 64 * 1024 * 1024
MAX_EMERGENCY_RECORD_BYTES = 32 * 1024


def utc_now() -> datetime:
    """Return timezone-aware UTC now with stable second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class StateLockError(RuntimeError):
    """Raised when a run-set state lock cannot be acquired."""


class ControlFilesystemPreflightError(RuntimeError):
    """Raised when the control filesystem cannot support durable state updates."""


class PreflightCheckEntry(StrictModel):
    """One named preflight check result."""

    name: str
    status: CheckStatusName
    detail: str | None = None
    observed: Any = None


DEPENDENCY_SKIP_OUTCOME = "skipped-due-to-dependency"


def dependency_skip_observed(*dependencies: str) -> dict[str, Any]:
    """Build the canonical `observed` payload for a dependency-skipped preflight check.

    Single-sources the sentinel shape so every writer of a dependency-skip result -
    whether it fills a `PreflightCheckEntry.observed` field directly or a per-row
    entry nested inside one - agrees on the same `outcome`/`dependencies` spelling
    that `_is_dependency_skipped_preflight_check`-style readers inspect.
    """
    ordered_dependencies = list(dict.fromkeys(dependencies))
    return {
        "outcome": DEPENDENCY_SKIP_OUTCOME,
        "dependencies": ordered_dependencies,
    }


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


class RegistrationHistoryEntry(StrictModel):
    """One immutable registration outcome superseded by explicit recertification."""

    registration_payload: dict[str, Any]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    certificate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_certificate_ref: str = Field(min_length=1)


class RegistrationHistory(StrictModel):
    """Versioned fail-to-pass registration history for one run set."""

    schema_id: Literal[
        "feedbax.orchestration.registration_history"
    ] = REGISTRATION_HISTORY_SCHEMA_ID
    schema_version: Literal[
        "feedbax.orchestration.registration_history.v1"
    ] = REGISTRATION_HISTORY_SCHEMA_VERSION
    run_set_id: str = Field(min_length=1)
    entries: list[RegistrationHistoryEntry] = Field(min_length=1, max_length=1)


class EmergencyProviderIdentity(StrictModel):
    """Minimal provider identity needed to recover a preserved resource."""

    provider: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    endpoint: str | None = None


class EmergencyRunSetRecord(StrictModel):
    """Bounded recovery record independent of the primary run-set state document."""

    schema_id: Literal[
        "feedbax.orchestration.emergency_run_set_record"
    ] = EMERGENCY_RUN_SET_RECORD_SCHEMA_ID
    schema_version: Literal[
        "feedbax.orchestration.emergency_run_set_record.v1"
    ] = EMERGENCY_RUN_SET_RECORD_SCHEMA_VERSION
    run_set_id: str = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=utc_now)
    provider_identity: EmergencyProviderIdentity
    preservation_state: PreservationState
    lease_state: str = Field(min_length=1)
    custody_complete: bool
    spend_boundary: str = Field(min_length=1)
    primary_failure: str = Field(min_length=1)
    next_recovery_action: str = Field(min_length=1)


class ControlFilesystemPreflight(StrictModel):
    """Observed capacity and reserved bytes for one control-state filesystem."""

    filesystem_path: str
    writable: Literal[True] = True
    observed_free_bytes: int = Field(ge=0)
    required_free_bytes: int = Field(ge=0)
    state_update_bytes: int = Field(ge=1, le=MAX_CONTROL_RESERVE_BYTES)
    control_reserve_bytes: int = Field(ge=0, le=MAX_CONTROL_RESERVE_BYTES)
    emergency_reserve_bytes: int = Field(
        ge=MAX_EMERGENCY_RECORD_BYTES,
        le=MAX_CONTROL_RESERVE_BYTES,
    )


class RunSetState(StrictModel):
    """Atomic JSON state document for one run set."""

    schema_id: Literal["feedbax.orchestration.run_set_state"] = RUN_SET_STATE_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.run_set_state.v5"] = RUN_SET_STATE_SCHEMA_VERSION
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
        self.control_reserve_path = self.path.with_suffix(self.path.suffix + ".reserve")
        self.emergency_path = self.path.with_suffix(self.path.suffix + ".emergency.json")
        self.emergency_reserve_path = self.path.with_suffix(
            self.path.suffix + ".emergency.reserve"
        )

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

    def preflight_and_reserve(
        self,
        *,
        control_reserve_bytes: int = DEFAULT_CONTROL_RESERVE_BYTES,
        emergency_reserve_bytes: int = DEFAULT_EMERGENCY_RESERVE_BYTES,
        state_update_bytes: int = DEFAULT_STATE_UPDATE_BYTES,
    ) -> ControlFilesystemPreflight:
        """Verify control storage and durably reserve bounded recovery capacity.

        The control reserve is intentionally not consumed by normal state writes. A later
        lifecycle integration may release it for bounded collection metadata. The independent
        emergency reserve is consumed only by :meth:`save_emergency`.
        """
        _validate_reserve_sizes(
            control_reserve_bytes,
            emergency_reserve_bytes,
            state_update_bytes,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _probe_directory_writable(self.path.parent, self.path.name)

        requested_reserves = (
            (self.control_reserve_path, control_reserve_bytes),
            (self.emergency_reserve_path, emergency_reserve_bytes),
        )
        reserve_allocation_bytes = sum(
            size
            for path, size in requested_reserves
            if not path.exists() or path.stat().st_size != size
        )
        required_free_bytes = reserve_allocation_bytes + state_update_bytes
        observed_free_bytes = shutil.disk_usage(self.path.parent).free
        if observed_free_bytes < required_free_bytes:
            raise ControlFilesystemPreflightError(
                "control filesystem capacity preflight failed: "
                f"required_free_bytes={required_free_bytes} "
                f"observed_free_bytes={observed_free_bytes} path={self.path.parent}"
            )

        for reserve_path, reserve_size in requested_reserves:
            if not reserve_path.exists() or reserve_path.stat().st_size != reserve_size:
                _reserve_file(reserve_path, reserve_size)
        return ControlFilesystemPreflight(
            filesystem_path=str(self.path.parent),
            observed_free_bytes=observed_free_bytes,
            required_free_bytes=required_free_bytes,
            state_update_bytes=state_update_bytes,
            control_reserve_bytes=control_reserve_bytes,
            emergency_reserve_bytes=emergency_reserve_bytes,
        )

    def release_control_reserve(self) -> None:
        """Release the bounded non-emergency reserve for later lifecycle-owned metadata."""
        self.control_reserve_path.unlink(missing_ok=True)
        _fsync_directory(self.path.parent)

    def load_emergency(self) -> EmergencyRunSetRecord:
        """Load and strictly validate the current emergency recovery record."""
        return EmergencyRunSetRecord.model_validate_json(
            self.emergency_path.read_text(encoding="utf-8")
        )

    def save_emergency(self, record: EmergencyRunSetRecord) -> Path:
        """Publish a recovery record using only the preallocated reserve inode.

        A successful publish atomically renames the reserve over ``emergency_path`` and
        therefore consumes the reserve name. Call :meth:`preflight_and_reserve` to
        replenish it before a later update. If replenishment or replacement fails, an
        already-published emergency record remains readable.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_copy(update={"recorded_at": utc_now()}).model_dump(mode="json")
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(encoded) > MAX_EMERGENCY_RECORD_BYTES:
            raise ValueError(
                "emergency run-set record exceeds bounded channel: "
                f"bytes={len(encoded)} maximum={MAX_EMERGENCY_RECORD_BYTES}"
            )
        if not self.emergency_reserve_path.exists():
            raise ControlFilesystemPreflightError(
                "emergency channel has no reserved inode; call preflight_and_reserve "
                "before the first record and before each replacement"
            )

        _publish_reserved_emergency(
            reserve_path=self.emergency_reserve_path,
            destination_path=self.emergency_path,
            payload=encoded,
        )
        return self.emergency_path

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


def _validate_reserve_sizes(
    control_bytes: int,
    emergency_bytes: int,
    state_update_bytes: int,
) -> None:
    if not 0 <= control_bytes <= MAX_CONTROL_RESERVE_BYTES:
        raise ValueError(
            f"control reserve must be between 0 and {MAX_CONTROL_RESERVE_BYTES} bytes"
        )
    if not MAX_EMERGENCY_RECORD_BYTES <= emergency_bytes <= MAX_CONTROL_RESERVE_BYTES:
        raise ValueError(
            "emergency reserve must cover the maximum emergency record and remain bounded: "
            f"minimum={MAX_EMERGENCY_RECORD_BYTES} maximum={MAX_CONTROL_RESERVE_BYTES}"
        )
    if not 1 <= state_update_bytes <= MAX_CONTROL_RESERVE_BYTES:
        raise ValueError(
            "state update preflight must be positive and bounded: "
            f"minimum=1 maximum={MAX_CONTROL_RESERVE_BYTES}"
        )


def _probe_directory_writable(directory: Path, state_name: str) -> None:
    try:
        fd, probe_name = tempfile.mkstemp(prefix=f".{state_name}.", suffix=".probe", dir=directory)
        probe_path = Path(probe_name)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
            probe_path.unlink(missing_ok=True)
        _fsync_directory(directory)
    except OSError as exc:
        raise ControlFilesystemPreflightError(
            f"control filesystem writability preflight failed: path={directory}: {exc}"
        ) from exc


def _reserve_file(path: Path, size: int) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        if size:
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(fd, 0, size)
            else:
                chunk = bytes(min(size, 1024 * 1024))
                remaining = size
                while remaining:
                    written = os.write(fd, chunk[:remaining])
                    remaining -= written
        os.fsync(fd)
        _require_reserved_capacity(os.fstat(fd), size)
        os.close(fd)
        fd = -1
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        tmp_path.unlink(missing_ok=True)


def _publish_reserved_emergency(
    *,
    reserve_path: Path,
    destination_path: Path,
    payload: bytes,
) -> None:
    """Write a preallocated inode and atomically expose it without fresh allocation."""
    try:
        descriptor = os.open(reserve_path, os.O_RDWR)
    except FileNotFoundError as exc:
        raise ControlFilesystemPreflightError(
            "emergency reserve was consumed concurrently; an existing emergency record, "
            "if any, remains authoritative"
        ) from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        descriptor_stat = os.fstat(descriptor)
        try:
            reserve_stat = reserve_path.stat()
        except FileNotFoundError as exc:
            raise ControlFilesystemPreflightError(
                "emergency reserve was consumed concurrently; the published emergency "
                "record remains authoritative"
            ) from exc
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            reserve_stat.st_dev,
            reserve_stat.st_ino,
        ):
            raise ControlFilesystemPreflightError(
                "emergency reserve changed concurrently; refusing to overwrite an "
                "unverified inode"
            )
        _require_reserved_capacity(descriptor_stat, MAX_EMERGENCY_RECORD_BYTES)

        _pwrite_all(descriptor, payload)
        os.ftruncate(descriptor, len(payload))
        os.fsync(descriptor)
        os.replace(reserve_path, destination_path)
        _fsync_directory(destination_path.parent)
    finally:
        os.close(descriptor)


def _pwrite_all(descriptor: int, payload: bytes) -> None:
    """Overwrite already-allocated bytes without changing the file offset."""
    view = memoryview(payload)
    offset = 0
    while view:
        written = os.pwrite(descriptor, view, offset)
        if written <= 0:
            raise OSError("failed to write emergency record into reserved inode")
        offset += written
        view = view[written:]


def _require_reserved_capacity(stat: os.stat_result, required_bytes: int) -> None:
    """Reject short or sparse files that cannot prove the requested capacity was reserved."""
    allocated_blocks = getattr(stat, "st_blocks", None)
    allocated_bytes = allocated_blocks * 512 if isinstance(allocated_blocks, int) else None
    if stat.st_size < required_bytes or (
        allocated_bytes is not None and allocated_bytes < required_bytes
    ):
        raise ControlFilesystemPreflightError(
            "reserve file is not fully allocated: "
            f"size_bytes={stat.st_size} allocated_bytes={allocated_bytes!r} "
            f"required_bytes={required_bytes}; rerun preflight_and_reserve"
        )


def _fsync_directory(directory: Path) -> None:
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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
