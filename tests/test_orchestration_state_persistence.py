"""Focused tests for bounded run-set control-state persistence."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

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
    original_dump = json.dump

    def fail_primary_dump(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("feedbax.orchestration.state.json.dump", fail_primary_dump)
    with pytest.raises(OSError, match="No space left on device"):
        store.save(RunSetState(run_set_id="monitor-failure"))
    with pytest.raises(OSError, match="No space left on device"):
        store.save(RunSetState(run_set_id="abort-failure"))
    monkeypatch.setattr("feedbax.orchestration.state.json.dump", original_dump)

    emergency_path = store.save_emergency(_emergency_record())
    loaded_emergency = store.load_emergency()

    assert store.load().run_set_id == "old"
    assert emergency_path == store.emergency_path
    assert not store.emergency_reserve_path.exists()
    assert store.control_reserve_path.exists()
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

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("feedbax.orchestration.state.os.replace", fail_replace)
    with pytest.raises(OSError, match="No space left on device"):
        store.save_emergency(
            original.model_copy(update={"next_recovery_action": "new recovery action"})
        )

    assert store.load_emergency() == previous
    assert not list(tmp_path.glob(".state.json.emergency.json.*.tmp"))
