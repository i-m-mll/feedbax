"""Focused tests for bounded run-set control-state persistence."""

from __future__ import annotations

import errno
import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from feedbax.orchestration import state as orchestration_state
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.orchestration.state import (
    EMERGENCY_RUN_SET_RECORD_SCHEMA_ID,
    EMERGENCY_RUN_SET_RECORD_SCHEMA_VERSION,
    MAX_CONTROL_RESERVE_BYTES,
    MAX_EMERGENCY_RECORD_BYTES,
    ControlFilesystemPreflightError,
    EmergencyProviderIdentity,
    EmergencyRunSetRecord,
    RunSetState,
    RunSetStateStore,
)


def _emergency_record() -> EmergencyRunSetRecord:
    return EmergencyRunSetRecord(
        run_set_id="run-set",
        provider_identity=EmergencyProviderIdentity(
            provider="runpod",
            resource_id="pod-123",
            endpoint="ssh://example.invalid",
        ),
        preservation_state="preserve-required",
        lease_state="active until 2026-08-01T00:00:00Z",
        custody_complete=False,
        spend_boundary="stop no later than 2026-08-01T00:00:00Z",
        primary_failure="OSError: [Errno 28] No space left on device",
        next_recovery_action="collect remote outputs before authorizing teardown",
    )


def test_control_filesystem_preflight_reserves_bounded_capacity(tmp_path: Path) -> None:
    store = RunSetStateStore(tmp_path / "state.json")

    evidence = store.preflight_and_reserve(
        control_reserve_bytes=8192,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )

    assert evidence.writable is True
    assert evidence.control_reserve_bytes == 8192
    assert evidence.state_update_bytes == 4096
    assert store.control_reserve_path.stat().st_size == 8192
    assert store.emergency_reserve_path.stat().st_size == MAX_EMERGENCY_RECORD_BYTES
    with pytest.raises(ValueError, match="remain bounded"):
        store.preflight_and_reserve(
            control_reserve_bytes=0,
            emergency_reserve_bytes=MAX_CONTROL_RESERVE_BYTES + 1,
        )


def test_control_filesystem_preflight_fails_before_reserving_insufficient_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    monkeypatch.setattr(
        "feedbax.orchestration.state.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )

    with pytest.raises(ControlFilesystemPreflightError, match="capacity preflight failed"):
        store.preflight_and_reserve(
            control_reserve_bytes=8192,
            emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
            state_update_bytes=4096,
        )

    assert not store.control_reserve_path.exists()
    assert not store.emergency_reserve_path.exists()


def test_control_filesystem_preflight_fails_closed_on_unwritable_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")

    def fail_probe(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr("feedbax.orchestration.state.tempfile.mkstemp", fail_probe)
    with pytest.raises(ControlFilesystemPreflightError, match="writability preflight failed"):
        store.preflight_and_reserve()


def test_primary_enospc_leaves_old_state_and_emergency_fallback_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.save(RunSetState(run_set_id="old"))
    store.preflight_and_reserve(
        control_reserve_bytes=8192,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    original_pwrite = os.pwrite
    emergency_writes = 0

    def fail_primary_write(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    def fail_fresh_allocation(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError(errno.ENOSPC, "No space left on device")

    def recording_pwrite(descriptor: int, payload: bytes, offset: int) -> int:
        nonlocal emergency_writes
        emergency_writes += 1
        return original_pwrite(descriptor, payload, offset)

    monkeypatch.setattr("feedbax.orchestration.state.os.pwrite", recording_pwrite)
    monkeypatch.setattr("feedbax.orchestration.state.json.dump", fail_primary_write)
    with pytest.raises(OSError, match="No space left on device"):
        store.save(RunSetState(run_set_id="monitor-failure"))
    monkeypatch.setattr(
        "feedbax.orchestration.state.tempfile.mkstemp",
        fail_fresh_allocation,
    )
    with pytest.raises(OSError, match="No space left on device"):
        store.save(RunSetState(run_set_id="abort-failure"))

    emergency_path = store.save_emergency(_emergency_record())
    loaded_emergency = store.load_emergency()

    assert store.load().run_set_id == "old"
    assert emergency_path == store.emergency_path
    assert not store.emergency_reserve_path.exists()
    assert store.control_reserve_path.exists()
    assert emergency_writes >= 1
    assert not list(tmp_path.glob(".state.json.*.tmp"))
    assert loaded_emergency == _emergency_record().model_copy(
        update={"recorded_at": loaded_emergency.recorded_at}
    )


def test_emergency_record_is_atomic_bounded_and_rejects_unknown_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    fsynced_modes: list[int] = []
    original_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        fsynced_modes.append(os.fstat(fd).st_mode)
        original_fsync(fd)

    monkeypatch.setattr("feedbax.orchestration.state.os.fsync", recording_fsync)
    store.save_emergency(_emergency_record())
    assert not list(tmp_path.glob(".state.json.emergency.json.*.tmp"))
    assert any(mode & 0o170000 == 0o100000 for mode in fsynced_modes)
    assert any(mode & 0o170000 == 0o040000 for mode in fsynced_modes)

    oversized = _emergency_record().model_copy(
        update={"primary_failure": "x" * MAX_EMERGENCY_RECORD_BYTES}
    )
    with pytest.raises(ValueError, match="exceeds bounded channel"):
        store.save_emergency(oversized)

    stale_payload = _emergency_record().model_dump(mode="json")
    stale_payload["schema_version"] = f"{EMERGENCY_RUN_SET_RECORD_SCHEMA_ID}.v0"
    store.emergency_path.write_text(json.dumps(stale_payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="EmergencyRunSetRecord"):
        store.load_emergency()

    family = default_spec_registry.resolve("EmergencyRunSetRecord")
    assert family.identity == EMERGENCY_RUN_SET_RECORD_SCHEMA_ID
    assert family.current_version == EMERGENCY_RUN_SET_RECORD_SCHEMA_VERSION
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate("EmergencyRunSetRecord", stale_payload)


def test_emergency_record_replace_failure_preserves_previous_durable_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    original = _emergency_record()
    store.save_emergency(original)
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    previous = store.load_emergency()

    original_replace = os.replace

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("feedbax.orchestration.state.os.replace", fail_replace)
    with pytest.raises(OSError, match="No space left on device"):
        store.save_emergency(
            original.model_copy(update={"next_recovery_action": "new recovery action"})
        )

    assert store.load_emergency() == previous
    assert store.emergency_reserve_path.exists()
    with pytest.raises(ControlFilesystemPreflightError, match="not fully allocated"):
        store.save_emergency(
            original.model_copy(update={"next_recovery_action": "retry without replenish"})
        )
    assert not list(tmp_path.glob(".state.json.emergency.json.*.tmp"))

    monkeypatch.setattr("feedbax.orchestration.state.os.replace", original_replace)
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    replacement = original.model_copy(
        update={"next_recovery_action": "replacement after replenishment"}
    )
    store.save_emergency(replacement)
    loaded = store.load_emergency()
    assert loaded.next_recovery_action == "replacement after replenishment"


def test_emergency_save_requires_reserved_capacity(tmp_path: Path) -> None:
    store = RunSetStateStore(tmp_path / "state.json")

    with pytest.raises(ControlFilesystemPreflightError, match="lock is absent"):
        store.save_emergency(_emergency_record())

    assert not store.emergency_path.exists()

    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    store.emergency_reserve_path.unlink()
    descriptor = os.open(store.emergency_reserve_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(descriptor, MAX_EMERGENCY_RECORD_BYTES)
    finally:
        os.close(descriptor)
    with pytest.raises(ControlFilesystemPreflightError, match="not fully allocated"):
        store.save_emergency(_emergency_record())
    assert not store.emergency_path.exists()


def test_concurrent_first_emergency_publish_keeps_one_complete_record(tmp_path: Path) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    records = [
        _emergency_record().model_copy(update={"next_recovery_action": f"recovery-{index}"})
        for index in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.save_emergency, record) for record in records]
        outcomes: list[Path | Exception] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except Exception as exc:
                outcomes.append(exc)

    assert sum(isinstance(outcome, Path) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, ControlFilesystemPreflightError) for outcome in outcomes) == 1
    assert store.load_emergency().next_recovery_action in {"recovery-0", "recovery-1"}


def test_concurrent_publish_and_replenish_rename_the_json_bearing_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    original_pwrite_all = orchestration_state._pwrite_all
    original_flock = fcntl.flock
    payload_written = Event()
    allow_publish = Event()
    replenisher_waiting = Event()
    written_inode: list[int] = []
    failures: list[BaseException] = []

    def blocking_pwrite_all(descriptor: int, payload: bytes) -> None:
        original_pwrite_all(descriptor, payload)
        written_inode.append(os.fstat(descriptor).st_ino)
        payload_written.set()
        assert allow_publish.wait(timeout=5)

    def recording_flock(descriptor: int, operation: int) -> None:
        if current_thread().name == "replenisher":
            replenisher_waiting.set()
        original_flock(descriptor, operation)

    monkeypatch.setattr(
        "feedbax.orchestration.state._pwrite_all",
        blocking_pwrite_all,
    )
    monkeypatch.setattr("feedbax.orchestration.state.fcntl.flock", recording_flock)

    def publish() -> None:
        try:
            store.save_emergency(_emergency_record())
        except BaseException as exc:
            failures.append(exc)

    def replenish() -> None:
        try:
            store.preflight_and_reserve(
                control_reserve_bytes=0,
                emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
                state_update_bytes=4096,
            )
        except BaseException as exc:
            failures.append(exc)

    publisher = Thread(target=publish, name="publisher")
    publisher.start()
    assert payload_written.wait(timeout=5)
    replenisher = Thread(target=replenish, name="replenisher")
    replenisher.start()
    assert replenisher_waiting.wait(timeout=5)
    assert replenisher.is_alive()
    allow_publish.set()
    publisher.join(timeout=5)
    replenisher.join(timeout=5)

    assert not publisher.is_alive()
    assert not replenisher.is_alive()
    assert failures == []
    assert store.emergency_path.stat().st_ino == written_inode[0]
    assert store.emergency_reserve_path.stat().st_ino != written_inode[0]
    assert store.emergency_reserve_path.stat().st_size == MAX_EMERGENCY_RECORD_BYTES
    assert store.load_emergency().next_recovery_action == (
        "collect remote outputs before authorizing teardown"
    )
