from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path

import pytest

from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.orchestration.events import (
    RUN_EVENT_SCHEMA_ID,
    RUN_EVENT_SCHEMA_VERSION,
    BatchLineFormatError,
    RunEvent,
    RunEventEmitter,
    RunEventProtocolError,
    RunEventReader,
    format_batch_line,
)


class FailingJsonlHandle:
    def __init__(self, *, fail_writes: int | None) -> None:
        self.fail_writes = fail_writes
        self.lines: list[str] = []
        self.closed = False

    def write(self, line: str) -> int:
        if self.fail_writes is None or self.fail_writes > 0:
            if self.fail_writes is not None:
                self.fail_writes -= 1
            raise OSError("simulated write failure")
        self.lines.append(line)
        return len(line)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _install_failing_handle(emitter: RunEventEmitter, handle: FailingJsonlHandle) -> None:
    assert emitter._handle is not None
    emitter._handle.close()
    emitter._handle = handle


def test_run_event_envelope_round_trips_and_preserves_payload_unknowns(tmp_path: Path) -> None:
    path = tmp_path / "events" / "row-1.events.jsonl"
    emitter = RunEventEmitter(run_set_id="set-1", row_id="row-1", path=path)
    try:
        event = emitter.emit("progress", {"batch": 1, "custom": {"kept": True}})
    finally:
        emitter.close()

    assert event.schema_id == RUN_EVENT_SCHEMA_ID
    assert event.schema_version == RUN_EVENT_SCHEMA_VERSION
    assert event.payload["custom"] == {"kept": True}

    [read_event] = RunEventReader(path).read_all()
    assert read_event == event


def test_run_event_rejects_unknown_schema_and_extra_fields() -> None:
    base = {
        "schema_id": RUN_EVENT_SCHEMA_ID,
        "schema_version": RUN_EVENT_SCHEMA_VERSION,
        "run_set_id": "set-1",
        "row_id": "row-1",
        "seq": 0,
        "emitted_at_ms": 1,
        "type": "progress",
        "payload": {},
    }

    with pytest.raises(RunEventProtocolError, match="schema_version"):
        RunEvent.model_validate({**base, "schema_version": "feedbax.run_event.v1"})
    with pytest.raises(Exception, match="Extra inputs"):
        RunEvent.model_validate({**base, "unexpected": True})


def test_run_event_registry_declares_reject_policy() -> None:
    family = default_spec_registry.resolve("RunEvent")

    assert family.identity == RUN_EVENT_SCHEMA_ID
    assert family.current_version == RUN_EVENT_SCHEMA_VERSION
    assert family.policy is not None
    assert family.policy.stance == "reject"

    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate("RunEvent", {"schema_version": "feedbax.run_event.v1"})


def test_reader_rejects_non_monotonic_seq(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = RunEvent(
        run_set_id="set-1",
        row_id="row-1",
        seq=1,
        emitted_at_ms=1,
        type="progress",
    )
    second = first.model_copy(update={"seq": 1, "emitted_at_ms": 2})
    path.write_text(
        first.model_dump_json() + "\n" + second.model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RunEventProtocolError, match="strictly monotonic"):
        RunEventReader(path).read_all()


def test_emitter_cadence_terminal_atomicity_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    warnings: list[str] = []
    emitter = RunEventEmitter(run_set_id="set-1", row_id="row-1", path=path, warn=warnings.append)
    try:
        emitted = [
            emitter.emit_progress({"phase": "train", "batch": batch, "total_batches": 100}, batch=batch, total_batches=100)
            for batch in range(1, 12)
        ]
        terminal = emitter.emit_terminal("complete", {"status": "completed"})
        ignored = emitter.emit_terminal("failed", {"status": "failed"})
    finally:
        emitter.close()

    assert [event.seq for event in emitted if event is not None] == [0, 1]
    assert terminal is not None and terminal.type == "complete"
    assert ignored is None
    assert warnings == ["RunEvent terminal event already emitted for row 'row-1'"]

    lines = path.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line)["schema_id"] == RUN_EVENT_SCHEMA_ID for line in lines)
    assert all(line.endswith("}") for line in lines)
    assert [event.seq for event in RunEventReader(path).read_all(from_seq=1)] == [1, 2]


def test_emitter_heartbeat_and_io_errors_do_not_propagate(tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeat.jsonl"
    emitter = RunEventEmitter(
        run_set_id="set-1",
        row_id="row-1",
        path=heartbeat_path,
        heartbeat_seconds=0.01,
    )
    try:
        deadline = time.monotonic() + 1.0
        events = []
        while time.monotonic() < deadline:
            events = RunEventReader(heartbeat_path).read_all()
            if any(event.type == "heartbeat" for event in events):
                break
            time.sleep(0.01)
    finally:
        emitter.close()

    assert any(event.type == "heartbeat" for event in events)

    warnings: list[str] = []
    blocked_dir = tmp_path / "not-a-dir"
    blocked_dir.write_text("file blocks directory creation", encoding="utf-8")
    broken = RunEventEmitter(
        run_set_id="set-1",
        row_id="row-2",
        events_dir=blocked_dir,
        warn=warnings.append,
    )
    broken.emit("progress", {"batch": 1})
    broken.close()

    assert broken.diagnostics
    assert warnings


def test_emitter_retries_transient_jsonl_failures_in_order(tmp_path: Path) -> None:
    warnings: list[str] = []
    handle = FailingJsonlHandle(fail_writes=4)
    emitter = RunEventEmitter(
        run_set_id="set-1",
        row_id="row-1",
        path=tmp_path / "events.jsonl",
        heartbeat_seconds=0.01,
        retry_initial_seconds=0.001,
        warn=warnings.append,
    )
    _install_failing_handle(emitter, handle)

    try:
        for seq in range(3):
            emitter.emit("progress", {"batch": seq})
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(handle.lines) < 3:
            time.sleep(0.01)
    finally:
        emitter.close()

    assert [json.loads(line)["seq"] for line in handle.lines[:3]] == [0, 1, 2]
    assert emitter.stats["dropped_events"] == 0
    assert warnings


def test_emitter_drops_after_bounded_retries_and_keeps_heartbeat_alive(
    tmp_path: Path,
) -> None:
    warnings: list[str] = []
    handle = FailingJsonlHandle(fail_writes=None)
    emitter = RunEventEmitter(
        run_set_id="set-1",
        row_id="row-1",
        path=tmp_path / "events.jsonl",
        heartbeat_seconds=0.01,
        max_write_attempts=3,
        retry_initial_seconds=0.001,
        warn=warnings.append,
    )
    _install_failing_handle(emitter, handle)

    try:
        emitter.emit("progress", {"batch": 1})
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and emitter.stats["dropped_events"] == 0:
            time.sleep(0.01)
    finally:
        emitter.close()

    assert emitter.stats["io_failures"] >= 3
    assert emitter.stats["dropped_events"] > 0
    assert any("dropped JSONL event" in warning for warning in warnings)


def test_emitter_heartbeat_fires_after_retry_storm(tmp_path: Path) -> None:
    handle = FailingJsonlHandle(fail_writes=2)
    emitter = RunEventEmitter(
        run_set_id="set-1",
        row_id="row-1",
        path=tmp_path / "events.jsonl",
        heartbeat_seconds=0.01,
        retry_initial_seconds=0.001,
    )
    _install_failing_handle(emitter, handle)

    try:
        emitter.emit("progress", {"batch": 1})
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            event_types = [json.loads(line)["type"] for line in handle.lines]
            if "heartbeat" in event_types:
                break
            time.sleep(0.01)
    finally:
        emitter.close()

    assert "heartbeat" in [json.loads(line)["type"] for line in handle.lines]


def test_reader_reconciles_terminal_events_and_sentinels(tmp_path: Path) -> None:
    def write_events(name: str, event_type: str | None) -> Path:
        path = tmp_path / f"{name}.jsonl"
        if event_type is not None:
            path.write_text(
                RunEvent(
                    run_set_id="set",
                    row_id=name,
                    seq=0,
                    emitted_at_ms=1,
                    type=event_type,
                ).model_dump_json()
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text("", encoding="utf-8")
        return path

    done = tmp_path / "row.done"
    failed = tmp_path / "row.failed"

    done.write_text("", encoding="utf-8")
    agree = RunEventReader(write_events("agree", "complete")).reconcile_sentinels(
        done_sentinel=done,
        failed_sentinel=failed,
    )
    assert agree.status == "completed"
    assert agree.discrepancies == ()

    sentinel_only = RunEventReader(write_events("sentinel-only", "progress")).reconcile_sentinels(
        done_sentinel=done,
        failed_sentinel=failed,
    )
    assert sentinel_only.status == "completed"
    assert sentinel_only.synthesized_event is not None
    assert sentinel_only.discrepancies[0]["code"] == "sentinel_without_terminal_event"

    done.unlink()
    event_only = RunEventReader(write_events("event-only", "failed")).reconcile_sentinels(
        done_sentinel=done,
        failed_sentinel=failed,
    )
    assert event_only.status == "failed"
    assert event_only.discrepancies[0]["code"] == "terminal_event_without_sentinel"

    done.write_text("", encoding="utf-8")
    disagree = RunEventReader(write_events("disagree", "failed")).reconcile_sentinels(
        done_sentinel=done,
        failed_sentinel=failed,
    )
    assert disagree.status == "error"
    assert disagree.discrepancies[0]["code"] == "terminal_sentinel_disagree"


def test_batch_line_rendering_contract() -> None:
    event = RunEvent(
        run_set_id="set-1",
        row_id="row-1",
        seq=0,
        emitted_at_ms=1,
        type="progress",
        payload={
            "phase": "train",
            "batch": 12,
            "total_batches": 300,
            "loss": 0.012345,
            "extras": {"lambda": "0.5"},
            "elapsed_seconds": 4.56,
        },
    )

    assert format_batch_line(event) == (
        "BATCH phase=train batch=12/300 loss=0.01235 lambda=0.5 elapsed=4.6s"
    )
    with pytest.raises(BatchLineFormatError, match="missing required"):
        format_batch_line({"phase": "train"})


def test_batch_line_sink_receives_progress_lines(tmp_path: Path) -> None:
    sink = StringIO()
    emitter = RunEventEmitter(
        run_set_id="set-1",
        row_id="row-1",
        path=tmp_path / "events.jsonl",
        render_batch_lines=True,
        batch_line_sink=sink,
        heartbeat_seconds=None,
    )
    try:
        emitter.emit("progress", {"phase": "train", "batch": 1, "total_batches": 1})
    finally:
        emitter.close()

    assert sink.getvalue() == "BATCH phase=train batch=1/1\n"
