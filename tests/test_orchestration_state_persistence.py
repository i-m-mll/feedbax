"""Focused tests for bounded run-set control-state persistence."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import stat
import subprocess
import sys
import time
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
    STATE_LOCK_PROTOCOL,
    ControlFilesystemPreflightError,
    EmergencyProviderIdentity,
    EmergencyRunSetRecord,
    RunSetState,
    RunSetStateStore,
    StateLockError,
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


def _spawn_lock_process(
    state_path: Path,
    ready_path: Path,
    action: str,
) -> subprocess.Popen[str]:
    script = """
import sys
from pathlib import Path

from feedbax.orchestration.state import RunSetStateStore

store = RunSetStateStore(Path(sys.argv[1]))
try:
    with store.lock(break_stale=True):
        Path(sys.argv[2]).write_text("ready", encoding="utf-8")
        if sys.argv[3] == "wait":
            sys.stdin.readline()
        elif sys.argv[3] == "exception":
            raise RuntimeError("release lock through context-manager exception")
except RuntimeError:
    pass
"""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(state_path), str(ready_path), action],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_path(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            _stdout, stderr = process.communicate()
            pytest.fail(f"lock subprocess exited before readiness: {stderr}")
        time.sleep(0.01)
    process.kill()
    process.wait(timeout=5)
    pytest.fail("lock subprocess did not become ready before the deadline")


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


@pytest.mark.parametrize("substitution", ["symlink", "hardlink"])
def test_emergency_reserve_substitution_preserves_target(
    tmp_path: Path,
    substitution: str,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    target = tmp_path / "do-not-modify"
    original = b"owner-controlled bytes"
    target.write_bytes(original)
    store.emergency_reserve_path.unlink()
    if substitution == "symlink":
        store.emergency_reserve_path.symlink_to(target)
    else:
        os.link(target, store.emergency_reserve_path)

    with pytest.raises(ControlFilesystemPreflightError):
        store.save_emergency(_emergency_record())

    assert target.read_bytes() == original


@pytest.mark.parametrize("reserve_name", ["control_reserve_path", "emergency_reserve_path"])
@pytest.mark.parametrize("substitution", ["symlink", "hardlink"])
def test_preflight_replaces_substituted_reserve_without_modifying_target(
    tmp_path: Path,
    reserve_name: str,
    substitution: str,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=4096,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    reserve_path = getattr(store, reserve_name)
    target = tmp_path / "do-not-modify"
    original = b"owner-controlled bytes"
    target.write_bytes(original)
    reserve_path.unlink()
    if substitution == "symlink":
        reserve_path.symlink_to(target)
    else:
        os.link(target, reserve_path)

    store.preflight_and_reserve(
        control_reserve_bytes=4096,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )

    assert target.read_bytes() == original
    reserve_stat = reserve_path.stat()
    assert reserve_stat.st_nlink == 1
    expected_size = 4096 if reserve_name == "control_reserve_path" else MAX_EMERGENCY_RECORD_BYTES
    assert reserve_stat.st_size == expected_size


def test_first_lock_acquisition_creates_and_reuses_stable_inode(tmp_path: Path) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    assert not store.lock_path.exists()

    with store.lock():
        original_inode = store.lock_path.stat().st_ino
        payload = json.loads(store.lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()

    with store.lock():
        assert store.lock_path.stat().st_ino == original_inode

    assert store.lock_path.stat().st_ino == original_inode


@pytest.mark.parametrize("substitution", ["symlink", "hardlink"])
def test_state_lock_substitution_preserves_target(
    tmp_path: Path,
    substitution: str,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    target = tmp_path / "do-not-modify"
    original = b"owner-controlled bytes"
    target.write_bytes(original)
    if substitution == "symlink":
        store.lock_path.symlink_to(target)
    else:
        os.link(target, store.lock_path)

    with pytest.raises(StateLockError):
        with store.lock(break_stale=True):
            pass

    assert target.read_bytes() == original


def test_simultaneous_stale_lock_breakers_have_one_owner(tmp_path: Path) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.lock_path.write_text(json.dumps({"pid": 999999999}), encoding="utf-8")
    start = Event()
    acquired = Event()
    contender_rejected = Event()
    allow_release = Event()
    outcomes: list[str] = []

    def contend() -> None:
        assert start.wait(timeout=5)
        try:
            with store.lock(break_stale=True):
                outcomes.append("acquired")
                acquired.set()
                assert allow_release.wait(timeout=5)
        except StateLockError:
            outcomes.append("rejected")
            contender_rejected.set()

    contenders = [Thread(target=contend) for _ in range(2)]
    for contender in contenders:
        contender.start()
    start.set()
    assert acquired.wait(timeout=5)
    assert contender_rejected.wait(timeout=5)
    allow_release.set()
    for contender in contenders:
        contender.join(timeout=5)

    assert outcomes.count("acquired") == 1
    assert outcomes.count("rejected") == 1
    assert all(not contender.is_alive() for contender in contenders)
    with store.lock(break_stale=True):
        pass


@pytest.mark.parametrize("reserve_name", ["control_reserve_path", "emergency_reserve_path"])
def test_fifo_reserve_substitution_is_bounded_and_replaced(
    tmp_path: Path,
    reserve_name: str,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=4096,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    reserve_path = getattr(store, reserve_name)
    reserve_path.unlink()
    os.mkfifo(reserve_path)
    script = """
import sys
from pathlib import Path

from feedbax.orchestration.state import MAX_EMERGENCY_RECORD_BYTES, RunSetStateStore

RunSetStateStore(Path(sys.argv[1])).preflight_and_reserve(
    control_reserve_bytes=4096,
    emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
    state_update_bytes=4096,
)
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(store.path)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert stat.S_ISREG(reserve_path.stat().st_mode)


@pytest.mark.parametrize("substitution", ["symlink", "hardlink"])
def test_emergency_lock_link_substitution_preserves_target(
    tmp_path: Path,
    substitution: str,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    store.emergency_lock_path.unlink()
    target = tmp_path / "do-not-modify"
    original = b"owner-controlled bytes"
    target.write_bytes(original)
    if substitution == "symlink":
        store.emergency_lock_path.symlink_to(target)
    else:
        os.link(target, store.emergency_lock_path)

    with pytest.raises(ControlFilesystemPreflightError):
        store.preflight_and_reserve(
            control_reserve_bytes=0,
            emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
            state_update_bytes=4096,
        )

    assert target.read_bytes() == original


def test_emergency_lock_fifo_substitution_fails_within_subprocess_deadline(
    tmp_path: Path,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    store.emergency_lock_path.unlink()
    os.mkfifo(store.emergency_lock_path)
    script = """
import sys
from pathlib import Path

from feedbax.orchestration.state import (
    MAX_EMERGENCY_RECORD_BYTES,
    ControlFilesystemPreflightError,
    RunSetStateStore,
)

try:
    RunSetStateStore(Path(sys.argv[1])).preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
except ControlFilesystemPreflightError:
    pass
else:
    raise AssertionError("FIFO emergency lock was accepted")
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(store.path)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert stat.S_ISFIFO(store.emergency_lock_path.stat().st_mode)


def test_final_rename_substitution_cannot_report_false_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    diverted = tmp_path / "verified-json-inode"
    attacker_payload = b"attacker replacement bytes"
    original_replace = os.replace

    def substitute_at_publication(source: object, destination: object) -> None:
        if Path(source) == store.emergency_reserve_path:
            original_replace(source, diverted)
            store.emergency_reserve_path.write_bytes(attacker_payload)
        original_replace(source, destination)

    monkeypatch.setattr("feedbax.orchestration.state.os.replace", substitute_at_publication)

    with pytest.raises(ControlFilesystemPreflightError, match="published emergency record"):
        store.save_emergency(_emergency_record())

    assert store.emergency_path.read_bytes() == attacker_payload
    EmergencyRunSetRecord.model_validate_json(diverted.read_text(encoding="utf-8"))


def test_reserve_allocation_rename_substitution_cannot_report_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    diverted = tmp_path / "verified-reserve-inode"
    attacker_payload = b"attacker replacement bytes"
    original_replace = os.replace

    def substitute_at_publication(source: object, destination: object) -> None:
        if Path(destination) == store.control_reserve_path:
            original_replace(source, diverted)
            Path(source).write_bytes(attacker_payload)
        original_replace(source, destination)

    monkeypatch.setattr("feedbax.orchestration.state.os.replace", substitute_at_publication)

    with pytest.raises(ControlFilesystemPreflightError, match="published reserve file"):
        store.preflight_and_reserve(
            control_reserve_bytes=4096,
            emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
            state_update_bytes=4096,
        )

    assert store.control_reserve_path.read_bytes() == attacker_payload
    assert diverted.stat().st_size == 4096


def test_emergency_lock_substitution_before_return_cannot_report_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.preflight_and_reserve(
        control_reserve_bytes=0,
        emergency_reserve_bytes=MAX_EMERGENCY_RECORD_BYTES,
        state_update_bytes=4096,
    )
    displaced_lock = tmp_path / "displaced-emergency-lock"
    original_publish = orchestration_state._publish_reserved_emergency

    def publish_then_substitute_lock(**kwargs: object) -> None:
        original_publish(**kwargs)
        os.replace(store.emergency_lock_path, displaced_lock)
        store.emergency_lock_path.write_bytes(b"replacement lock")

    monkeypatch.setattr(
        "feedbax.orchestration.state._publish_reserved_emergency",
        publish_then_substitute_lock,
    )

    with pytest.raises(ControlFilesystemPreflightError, match="emergency channel lock"):
        store.save_emergency(_emergency_record())

    assert store.load_emergency().run_set_id == "run-set"


@pytest.mark.parametrize("error_number", [errno.ENOTSUP, errno.ENOLCK])
def test_state_lock_translates_unsupported_flock_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")

    def fail_flock(_descriptor: int, _operation: int) -> None:
        raise OSError(error_number, os.strerror(error_number))

    monkeypatch.setattr("feedbax.orchestration.state.fcntl.flock", fail_flock)
    with pytest.raises(StateLockError, match="unable to acquire") as exc_info:
        with store.lock():
            pass

    assert exc_info.value.errno == error_number


@pytest.mark.parametrize("error_number", [errno.ENOTSUP, errno.ENOLCK])
def test_emergency_lock_translates_unsupported_flock_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")

    def fail_flock(_descriptor: int, _operation: int) -> None:
        raise OSError(error_number, os.strerror(error_number))

    monkeypatch.setattr("feedbax.orchestration.state.fcntl.flock", fail_flock)
    with pytest.raises(ControlFilesystemPreflightError, match="unable to acquire") as exc_info:
        store.preflight_and_reserve()

    assert exc_info.value.errno == error_number


@pytest.mark.parametrize("operation", ["state-lock", "emergency-preflight"])
def test_missing_secure_open_capability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    monkeypatch.delattr(os, "O_NONBLOCK")

    expected_error = StateLockError if operation == "state-lock" else ControlFilesystemPreflightError
    with pytest.raises(expected_error) as exc_info:
        if operation == "state-lock":
            with store.lock():
                pass
        else:
            store.preflight_and_reserve()

    assert exc_info.value.errno == errno.ENOTSUP


def test_real_subprocess_lock_contention(tmp_path: Path) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    ready = tmp_path / "holder-ready"
    process = _spawn_lock_process(store.path, ready, "wait")
    try:
        _wait_for_path(ready, process)
        with pytest.raises(StateLockError, match="active"):
            with store.lock(break_stale=True):
                pass
        assert process.stdin is not None
        process.stdin.write("release\n")
        process.stdin.flush()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    with store.lock(break_stale=True):
        pass


@pytest.mark.parametrize("release_mode", ["crash", "exception"])
def test_real_subprocess_owner_release(
    tmp_path: Path,
    release_mode: str,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    ready = tmp_path / f"{release_mode}-ready"
    action = "wait" if release_mode == "crash" else "exception"
    process = _spawn_lock_process(store.path, ready, action)
    _wait_for_path(ready, process)
    if release_mode == "crash":
        process.kill()
    assert process.wait(timeout=5) == (0 if release_mode == "exception" else -9)

    with store.lock(break_stale=True):
        pass


@pytest.mark.parametrize("legacy_bytes", [b"", b"not-json", b"[]", b"{}", b'{"pid": null}'])
@pytest.mark.parametrize("break_stale", [False, True])
def test_unverifiable_legacy_locks_require_offline_cleanup(
    tmp_path: Path,
    legacy_bytes: bytes,
    break_stale: bool,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.lock_path.write_bytes(legacy_bytes)

    with pytest.raises(StateLockError, match="remove or migrate the lock offline"):
        with store.lock(break_stale=break_stale):
            pass

    assert store.lock_path.read_bytes() == legacy_bytes


def test_empty_legacy_lock_cannot_be_claimed_before_writer_publishes_pid(
    tmp_path: Path,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    inode_created = Event()
    allow_pid_publication = Event()
    legacy_owner_entered = Event()

    def legacy_writer() -> None:
        descriptor = os.open(store.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            inode_created.set()
            assert allow_pid_publication.wait(timeout=5)
            os.write(descriptor, json.dumps({"pid": os.getpid()}).encode("utf-8"))
            os.fsync(descriptor)
            legacy_owner_entered.set()
        finally:
            os.close(descriptor)

    writer = Thread(target=legacy_writer)
    writer.start()
    assert inode_created.wait(timeout=5)
    new_owner_entered = False
    try:
        with pytest.raises(StateLockError, match="remove or migrate the lock offline"):
            with store.lock(break_stale=True):
                new_owner_entered = True
    finally:
        allow_pid_publication.set()
        writer.join(timeout=5)

    assert not writer.is_alive()
    assert legacy_owner_entered.is_set()
    assert new_owner_entered is False
    assert json.loads(store.lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_legacy_lock_payloads_preserve_active_and_stale_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    store.lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    with pytest.raises(StateLockError, match="active"):
        with store.lock(break_stale=True):
            pass

    store.lock_path.write_text(json.dumps({"pid": 424242}), encoding="utf-8")
    monkeypatch.setattr("feedbax.orchestration.state._pid_alive", lambda _pid: False)
    with pytest.raises(StateLockError, match="stale"):
        with store.lock():
            pass
    with store.lock(break_stale=True):
        pass


def test_stable_lock_protocol_migration_is_explicitly_one_way(tmp_path: Path) -> None:
    store = RunSetStateStore(tmp_path / "state.json")

    with store.lock():
        pass

    payload = json.loads(store.lock_path.read_text(encoding="utf-8"))
    assert payload["protocol"] == STATE_LOCK_PROTOCOL
    assert payload["pid"] == os.getpid()
    assert store.lock_path.exists()


def test_displaced_state_lock_owner_fails_after_replacement_owner_enters(
    tmp_path: Path,
) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    displaced_lock = tmp_path / "displaced-state-lock"
    replacement_owner_entered = False

    with pytest.raises(StateLockError, match="pathname changed"):
        with store.lock():
            original_inode = store.lock_path.stat().st_ino
            os.replace(store.lock_path, displaced_lock)
            store.lock_path.touch()
            with store.lock():
                replacement_owner_entered = True

    assert replacement_owner_entered is True
    assert displaced_lock.stat().st_ino == original_inode
    assert store.lock_path.stat().st_ino != original_inode
