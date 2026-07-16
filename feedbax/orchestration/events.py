"""Versioned run-event envelopes and JSONL transport helpers."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.metric_values import NumericBooleanJsonValue
from feedbax.contracts.worker import MaterializedSlotAxisBinding, ProgressCoordinate


RUN_EVENT_SCHEMA_ID = "feedbax.run_event"
RUN_EVENT_SCHEMA_VERSION = "feedbax.run_event.v2"
RUN_EVENT_TERMINAL_TYPES = frozenset({"complete", "failed"})
RUN_EVENT_EXTENSION_TYPES = frozenset({"log", "trajectory"})
RUN_EVENT_CORE_TYPES = frozenset(
    {
        "ready",
        "progress",
        "heartbeat",
        "checkpoint_written",
        "phase_changed",
        "complete",
        "failed",
    }
)
MAPPED_METRIC_VALUE_SCHEMA_ID = "feedbax.manifest.mapped_metric_value"
MAPPED_METRIC_VALUE_SCHEMA_VERSION = "feedbax.manifest.mapped_metric_value.v1"
STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_ID = (
    "feedbax.manifest.structured_mapped_metric_value"
)
STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_VERSION = (
    "feedbax.manifest.structured_mapped_metric_value.v1"
)


class MappedMetricValue(StrictModel):
    """Lossless JSON envelope for one metric retaining declared mapped axes."""

    schema_id: Literal["feedbax.manifest.mapped_metric_value"] = (
        MAPPED_METRIC_VALUE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.manifest.mapped_metric_value.v1"] = (
        MAPPED_METRIC_VALUE_SCHEMA_VERSION
    )
    value: Any
    shape: tuple[int, ...]
    dtype: str
    axes: tuple[MaterializedSlotAxisBinding, ...]


def _validate_nonempty_named_nodes(value: dict[str, NumericBooleanJsonValue]):
    if not value:
        raise ValueError("structured mapped metric has an empty named node")
    for item in value.values():
        if isinstance(item, dict):
            _validate_nonempty_named_nodes(item)
    return value


class StructuredMappedMetricValue(StrictModel):
    """Named numeric/boolean metric leaves retaining one replica axis."""

    schema_id: Literal["feedbax.manifest.structured_mapped_metric_value"] = (
        STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.manifest.structured_mapped_metric_value.v1"] = (
        STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_VERSION
    )
    value: dict[str, NumericBooleanJsonValue]
    axes: tuple[MaterializedSlotAxisBinding, ...]
    _validate_value = field_validator("value")(_validate_nonempty_named_nodes)


def _normalize_structured_mapped_value(
    value: Mapping[Any, Any],
    *,
    metric_name: str,
    replica_count: int,
    path: str = "value",
) -> dict[str, Any]:
    if not value:
        raise RunEventProtocolError(f"mapped metric {metric_name!r} has empty named node {path}")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise RunEventProtocolError(
                f"mapped metric {metric_name!r} requires a string name at {path}"
            )
        item_path = f"{path}.{key}"
        if isinstance(item, Mapping):
            normalized[key] = _normalize_structured_mapped_value(
                item,
                metric_name=metric_name,
                replica_count=replica_count,
                path=item_path,
            )
            continue
        try:
            array = np.asarray(item)
        except (TypeError, ValueError) as exc:
            raise RunEventProtocolError(
                f"mapped metric {metric_name!r} leaf {item_path} is not an array"
            ) from exc
        if array.ndim < 1:
            raise RunEventProtocolError(
                f"mapped metric {metric_name!r} leaf {item_path} lacks a replica dimension"
            )
        if array.shape[0] != replica_count:
            raise RunEventProtocolError(
                f"mapped metric {metric_name!r} leaf {item_path} has leading size "
                f"{array.shape[0]}; expected {replica_count}"
            )
        if array.dtype.kind not in "biuf":
            raise RunEventProtocolError(
                f"mapped metric {metric_name!r} leaf {item_path} has unsupported dtype "
                f"{array.dtype}"
            )
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise RunEventProtocolError(
                f"mapped metric {metric_name!r} leaf {item_path} must be finite"
            )
        normalized[key] = array.tolist()
    return normalized


def normalize_serialized_metrics(
    coordinate: ProgressCoordinate,
    named_metrics: Mapping[str, Any],
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strip mapped metrics from a coordinate and envelope them in their carrier."""
    mapped = {
        name: bindings
        for name, bindings in slot_axis_bindings.items()
        if bindings and bindings[0].mode == "mapped" and name in named_metrics
    }
    if not mapped:
        return coordinate.model_dump(mode="json", exclude_none=True), dict(named_metrics)
    coordinate_metrics = {
        name: value for name, value in coordinate.metrics.items() if name not in mapped
    }
    coordinate_payload = coordinate.model_copy(
        update={"metrics": coordinate_metrics}
    ).model_dump(mode="json", exclude_none=True)
    normalized = dict(named_metrics)
    for name, axes in mapped.items():
        if isinstance(named_metrics[name], Mapping):
            if (
                len(axes) != 1
                or axes[0].role != "replicate"
                or axes[0].level != 0
                or axes[0].array_axis != 0
            ):
                raise RunEventProtocolError(
                    f"mapped metric {name!r} requires one leading replica axis"
                )
            value = _normalize_structured_mapped_value(
                named_metrics[name],
                metric_name=name,
                replica_count=axes[0].size,
            )
            normalized[name] = StructuredMappedMetricValue(
                value=value,
                axes=axes,
            ).model_dump(mode="json")
            continue
        try:
            array = np.asarray(named_metrics[name])
            value = array.tolist()
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RunEventProtocolError(
                f"mapped metric {name!r} must have a finite JSON representation"
            ) from exc
        normalized[name] = MappedMetricValue(
            value=value,
            shape=array.shape,
            dtype=str(array.dtype),
            axes=axes,
        ).model_dump(mode="json")
    return coordinate_payload, normalized


class RunEventProtocolError(ValueError):
    """Raised when a run-event stream violates the transport contract."""


class BatchLineFormatError(ValueError):
    """Raised when a progress event cannot be rendered as an RLRMP BATCH line."""


class RunEvent(BaseModel):
    """One versioned run event in the canonical JSONL envelope."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str = RUN_EVENT_SCHEMA_ID
    schema_version: str = RUN_EVENT_SCHEMA_VERSION
    run_set_id: str
    row_id: str
    seq: int = Field(ge=0)
    emitted_at_ms: int = Field(ge=0)
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> "RunEvent":
        event = super().model_validate(obj, *args, **kwargs)
        if event.schema_id != RUN_EVENT_SCHEMA_ID:
            raise RunEventProtocolError(
                "Unsupported RunEvent schema_id: "
                f"{event.schema_id!r}; expected {RUN_EVENT_SCHEMA_ID!r}"
            )
        if event.schema_version != RUN_EVENT_SCHEMA_VERSION:
            raise RunEventProtocolError(
                "Unsupported RunEvent schema_version: "
                f"{event.schema_version!r}; expected {RUN_EVENT_SCHEMA_VERSION!r}"
            )
        return event


@dataclass(frozen=True)
class ReconciledRunStatus:
    """Row status derived from a run-event log and completion sentinels."""

    status: str
    terminal_event: RunEvent | None = None
    sentinel_status: str | None = None
    discrepancies: tuple[dict[str, Any], ...] = ()
    synthesized_event: RunEvent | None = None


@dataclass
class _PendingJsonlLine:
    line: str
    attempts: int
    next_retry_at: float


class RunEventEmitter:
    """Emit run events to canonical JSONL and optional human BATCH lines.

    JSONL writes use one ``write()`` call per complete line under a process-local
    lock. I/O failures are converted to diagnostics and warnings so training
    loops do not fail because event persistence is unavailable.
    """

    def __init__(
        self,
        *,
        run_set_id: str,
        row_id: str,
        path: Path | str | None = None,
        events_dir: Path | str | None = None,
        render_batch_lines: bool = False,
        batch_line_sink: TextIO | None = None,
        heartbeat_seconds: float | None = 60.0,
        max_write_attempts: int = 5,
        retry_initial_seconds: float = 0.1,
        retry_max_seconds: float = 3.0,
        now_ms: Callable[[], int] | None = None,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        if path is not None and events_dir is not None:
            raise ValueError("RunEventEmitter accepts either path or events_dir, not both")
        self.run_set_id = run_set_id
        self.row_id = row_id
        self.path = (
            Path(path)
            if path is not None
            else (Path(events_dir) / f"{row_id}.events.jsonl" if events_dir is not None else None)
        )
        self.render_batch_lines = render_batch_lines
        self.batch_line_sink = batch_line_sink if batch_line_sink is not None else sys.stdout
        self.heartbeat_seconds = heartbeat_seconds
        if max_write_attempts < 1:
            raise ValueError("max_write_attempts must be at least 1")
        self.max_write_attempts = max_write_attempts
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._warn = warn or (lambda message: warnings.warn(message, RuntimeWarning, stacklevel=2))
        self._seq = 0
        self._terminal_emitted = False
        self._closed = False
        self._handle: TextIO | None = None
        self._lock = threading.RLock()
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._last_write_monotonic = time.monotonic()
        self._retry_queue: deque[_PendingJsonlLine] = deque()
        self._retry_wakeup = threading.Event()
        self._io_failures = 0
        self._dropped_events = 0
        self.diagnostics: list[dict[str, Any]] = []

        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self.path.open("a", encoding="utf-8", buffering=1)
            except OSError as exc:
                self._record_io_failure("open", exc)
        if self._handle is not None and heartbeat_seconds is not None and heartbeat_seconds > 0:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"feedbax-run-event-heartbeat-{row_id}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    @classmethod
    def from_env(
        cls,
        *,
        render_batch_lines: bool = False,
        batch_line_sink: TextIO | None = None,
        heartbeat_seconds: float | None = 60.0,
    ) -> "RunEventEmitter | None":
        """Build an emitter from ``FEEDBAX_RUN_*`` variables when present."""
        run_set_id = os.environ.get("FEEDBAX_RUN_SET_ID")
        row_id = os.environ.get("FEEDBAX_ROW_ID")
        events_dir = os.environ.get("FEEDBAX_RUN_EVENTS_DIR")
        if not run_set_id or not row_id:
            if render_batch_lines:
                return cls(
                    run_set_id=run_set_id or "ad-hoc",
                    row_id=row_id or "ad-hoc",
                    render_batch_lines=True,
                    batch_line_sink=batch_line_sink,
                    heartbeat_seconds=None,
                )
            return None
        return cls(
            run_set_id=run_set_id,
            row_id=row_id,
            events_dir=events_dir if events_dir else None,
            render_batch_lines=render_batch_lines,
            batch_line_sink=batch_line_sink,
            heartbeat_seconds=heartbeat_seconds if events_dir else None,
        )

    def __enter__(self) -> "RunEventEmitter":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop heartbeat emission and close the JSONL handle."""
        self._closed = True
        self._stop_heartbeat.set()
        self._retry_wakeup.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            self._drain_retry_queue(force=True)
            handle = self._handle
            self._handle = None
            if handle is not None:
                try:
                    handle.close()
                except OSError as exc:
                    self._record_io_failure("close", exc)

    @property
    def stats(self) -> dict[str, int]:
        """Return emitter transport counters for final diagnostics."""
        return {"io_failures": self._io_failures, "dropped_events": self._dropped_events}

    def should_emit_progress(self, *, batch: int, total_batches: int | None) -> bool:
        """Return whether a progress event should be emitted at the default cadence."""
        if total_batches is not None and total_batches <= 50:
            return True
        return batch == 1 or batch % 10 == 0 or (
            total_batches is not None and batch >= total_batches
        )

    def emit_progress(
        self,
        payload: Mapping[str, Any],
        *,
        batch: int,
        total_batches: int | None,
        force: bool = False,
    ) -> RunEvent | None:
        """Emit a progress event if cadence allows it."""
        if not force and not self.should_emit_progress(batch=batch, total_batches=total_batches):
            return None
        return self.emit("progress", payload)

    def emit_terminal(self, event_type: str, payload: Mapping[str, Any]) -> RunEvent | None:
        """Emit exactly one terminal event."""
        if event_type not in RUN_EVENT_TERMINAL_TYPES:
            raise ValueError(f"event_type must be terminal; got {event_type!r}")
        with self._lock:
            if self._terminal_emitted:
                self._record_warning(
                    "terminal_already_emitted",
                    f"RunEvent terminal event already emitted for row {self.row_id!r}",
                )
                return None
            self._terminal_emitted = True
        return self.emit(event_type, payload)

    def emit(self, event_type: str, payload: Mapping[str, Any] | None = None) -> RunEvent:
        """Emit one event and return the validated envelope."""
        with self._lock:
            event = RunEvent(
                run_set_id=self.run_set_id,
                row_id=self.row_id,
                seq=self._seq,
                emitted_at_ms=self._now_ms(),
                type=event_type,
                payload=dict(payload or {}),
            )
            self._seq += 1
            if event.type == "progress" and self.render_batch_lines:
                self._write_batch_line(event)
            self._write_jsonl_event(event)
            return event

    def _write_jsonl_event(self, event: RunEvent) -> None:
        handle = self._handle
        if handle is None:
            return
        line = event.model_dump_json(exclude_none=True) + "\n"
        if self._retry_queue:
            self._retry_queue.append(
                _PendingJsonlLine(line=line, attempts=0, next_retry_at=time.monotonic())
            )
            self._retry_wakeup.set()
            return
        if not self._write_jsonl_line(line):
            self._enqueue_retry(line, attempts=1)

    def _write_jsonl_line(self, line: str) -> bool:
        handle = self._handle
        if handle is None:
            return True
        try:
            handle.write(line)
            handle.flush()
            self._last_write_monotonic = time.monotonic()
            return True
        except OSError as exc:
            self._record_io_failure("write", exc)
            return False

    def _enqueue_retry(self, line: str, *, attempts: int) -> None:
        if attempts >= self.max_write_attempts:
            self._drop_jsonl_line(line, attempts=attempts)
            return
        delay = min(
            self.retry_initial_seconds * (2 ** max(attempts - 1, 0)),
            self.retry_max_seconds,
        )
        self._retry_queue.append(
            _PendingJsonlLine(
                line=line,
                attempts=attempts,
                next_retry_at=time.monotonic() + delay,
            )
        )
        self._retry_wakeup.set()

    def _drain_retry_queue(self, *, force: bool = False) -> None:
        while self._retry_queue and self._handle is not None:
            pending = self._retry_queue[0]
            now = time.monotonic()
            if not force and pending.next_retry_at > now:
                return
            if self._write_jsonl_line(pending.line):
                self._retry_queue.popleft()
                continue
            attempts = pending.attempts + 1
            if attempts >= self.max_write_attempts:
                self._retry_queue.popleft()
                self._drop_jsonl_line(pending.line, attempts=attempts)
            else:
                delay = min(
                    self.retry_initial_seconds * (2 ** max(attempts - 1, 0)),
                    self.retry_max_seconds,
                )
                pending.attempts = attempts
                pending.next_retry_at = time.monotonic() + delay
            if not force:
                return

    def _drop_jsonl_line(self, line: str, *, attempts: int) -> None:
        self._dropped_events += 1
        self._record_warning(
            "io_write_dropped",
            "RunEventEmitter dropped JSONL event after "
            f"{attempts} write attempts for {self.path}",
        )

    def _write_batch_line(self, event: RunEvent) -> None:
        try:
            self.batch_line_sink.write(format_batch_line(event) + "\n")
            self.batch_line_sink.flush()
        except (OSError, BatchLineFormatError) as exc:
            self._record_io_failure("batch_line", exc)

    def _heartbeat_loop(self) -> None:
        assert self.heartbeat_seconds is not None
        while not self._closed:
            emit_heartbeat = False
            with self._lock:
                self._drain_retry_queue()
                now = time.monotonic()
                retry_delay = (
                    max(0.0, self._retry_queue[0].next_retry_at - now)
                    if self._retry_queue
                    else None
                )
                heartbeat_delay = max(
                    0.0,
                    self.heartbeat_seconds - (now - self._last_write_monotonic),
                )
                if heartbeat_delay == 0:
                    emit_heartbeat = True
                    timeout = retry_delay if retry_delay is not None else self.heartbeat_seconds
                elif retry_delay is None:
                    timeout = heartbeat_delay
                else:
                    timeout = min(heartbeat_delay, retry_delay)
            if emit_heartbeat:
                self.emit("heartbeat", {"idle_seconds": self.heartbeat_seconds})
                continue
            if self._stop_heartbeat.is_set():
                return
            self._retry_wakeup.wait(timeout)
            self._retry_wakeup.clear()

    def _record_io_failure(self, operation: str, exc: BaseException) -> None:
        self._io_failures += 1
        self._record_warning(
            f"io_{operation}_failed",
            f"RunEventEmitter {operation} failed for {self.path}: {exc}",
        )

    def _record_warning(self, code: str, message: str) -> None:
        diagnostic = {"severity": "warning", "code": code, "message": message}
        self.diagnostics.append(diagnostic)
        self._warn(message)


class RunEventReader:
    """Read and follow canonical run-event JSONL streams."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def read_all(self, *, from_seq: int | None = None) -> list[RunEvent]:
        """Read and validate all complete events, optionally replaying from ``seq``."""
        events: list[RunEvent] = []
        previous_seq: int | None = None
        if not self.path.exists():
            return events
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                event = self._event_from_line(line, line_number=line_number)
                if previous_seq is not None and event.seq <= previous_seq:
                    raise RunEventProtocolError(
                        "RunEvent seq must be strictly monotonic: "
                        f"path={self.path}, line={line_number}, previous={previous_seq}, "
                        f"current={event.seq}"
                    )
                previous_seq = event.seq
                if from_seq is None or event.seq >= from_seq:
                    events.append(event)
        return events

    def follow(
        self,
        *,
        from_seq: int | None = None,
        poll_interval: float = 0.5,
        stop_when_terminal: bool = False,
    ) -> Iterator[RunEvent]:
        """Yield events as they appear, replaying from ``from_seq`` first."""
        yielded_through: int | None = None
        while True:
            events = self.read_all(from_seq=from_seq if yielded_through is None else yielded_through + 1)
            for event in events:
                yielded_through = event.seq
                yield event
                if stop_when_terminal and event.type in RUN_EVENT_TERMINAL_TYPES:
                    return
            time.sleep(poll_interval)

    def reconcile_sentinels(
        self,
        *,
        done_sentinel: Path | str | None = None,
        failed_sentinel: Path | str | None = None,
    ) -> ReconciledRunStatus:
        """Reconcile terminal events with ``.done``/``.failed`` sentinels."""
        events = self.read_all()
        return reconcile_run_events(
            events,
            done_sentinel=Path(done_sentinel) if done_sentinel is not None else None,
            failed_sentinel=Path(failed_sentinel) if failed_sentinel is not None else None,
        )

    def _event_from_line(self, line: str, *, line_number: int) -> RunEvent:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunEventProtocolError(
                f"Invalid RunEvent JSONL at {self.path}:{line_number}: {exc}"
            ) from exc
        try:
            return RunEvent.model_validate(payload)
        except Exception as exc:
            raise RunEventProtocolError(
                f"Invalid RunEvent envelope at {self.path}:{line_number}: {exc}"
            ) from exc


def reconcile_run_events(
    events: list[RunEvent],
    *,
    done_sentinel: Path | None = None,
    failed_sentinel: Path | None = None,
) -> ReconciledRunStatus:
    """Reconcile a row event log with completion sentinels."""
    terminal_event = next((event for event in reversed(events) if event.type in RUN_EVENT_TERMINAL_TYPES), None)
    sentinel_status = _sentinel_status(done_sentinel=done_sentinel, failed_sentinel=failed_sentinel)
    event_status = _event_status(terminal_event)
    discrepancies: list[dict[str, Any]] = []

    if event_status is not None and sentinel_status is not None:
        if event_status == sentinel_status:
            return ReconciledRunStatus(
                status=event_status,
                terminal_event=terminal_event,
                sentinel_status=sentinel_status,
            )
        discrepancies.append(
            {
                "code": "terminal_sentinel_disagree",
                "event_status": event_status,
                "sentinel_status": sentinel_status,
            }
        )
        return ReconciledRunStatus(
            status="error",
            terminal_event=terminal_event,
            sentinel_status=sentinel_status,
            discrepancies=tuple(discrepancies),
        )

    if event_status is not None:
        discrepancies.append(
            {
                "code": "terminal_event_without_sentinel",
                "event_status": event_status,
            }
        )
        return ReconciledRunStatus(
            status=event_status,
            terminal_event=terminal_event,
            discrepancies=tuple(discrepancies),
        )

    if sentinel_status is not None:
        discrepancies.append(
            {
                "code": "sentinel_without_terminal_event",
                "sentinel_status": sentinel_status,
            }
        )
        synthesized = _synthesized_terminal_event(events, sentinel_status)
        return ReconciledRunStatus(
            status=sentinel_status,
            sentinel_status=sentinel_status,
            discrepancies=tuple(discrepancies),
            synthesized_event=synthesized,
        )

    return ReconciledRunStatus(status="running")


def format_batch_line(event: RunEvent | Mapping[str, Any]) -> str:
    """Render an RLRMP-compatible ``BATCH`` line from a progress event."""
    payload = event.payload if isinstance(event, RunEvent) else dict(event)
    if isinstance(event, RunEvent) and event.type != "progress":
        raise BatchLineFormatError(f"BATCH lines require progress events; got {event.type!r}")
    phase = _required_payload_value(payload, "phase")
    batch = _required_payload_value(payload, "batch")
    total = _required_payload_value(payload, "total_batches")
    parts = [f"BATCH phase={phase}", f"batch={batch}/{total}"]
    if "loss" in payload and payload["loss"] is not None:
        parts.append(f"loss={float(payload['loss']):.4g}")
    for key, value in _batch_line_extras(payload).items():
        parts.append(f"{key}={value}")
    if "elapsed_seconds" in payload and payload["elapsed_seconds"] is not None:
        parts.append(f"elapsed={float(payload['elapsed_seconds']):.1f}s")
    return " ".join(parts)


def _required_payload_value(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None:
        raise BatchLineFormatError(f"progress payload missing required {key!r}")
    return value


def _batch_line_extras(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    extras = payload.get("extras")
    if isinstance(extras, Mapping):
        return extras
    reserved = {
        "phase",
        "batch",
        "total_batches",
        "loss",
        "elapsed_seconds",
        "coordinate",
        "metrics",
        "status",
    }
    return {
        key: value
        for key, value in payload.items()
        if key not in reserved and isinstance(value, (str, int, float, bool))
    }


def _sentinel_status(*, done_sentinel: Path | None, failed_sentinel: Path | None) -> str | None:
    done = bool(done_sentinel and done_sentinel.exists())
    failed = bool(failed_sentinel and failed_sentinel.exists())
    if done and failed:
        return "error"
    if done:
        return "completed"
    if failed:
        return "failed"
    return None


def _event_status(event: RunEvent | None) -> str | None:
    if event is None:
        return None
    if event.type == "complete":
        return "completed"
    if event.type == "failed":
        return "failed"
    return None


def _synthesized_terminal_event(events: list[RunEvent], status: str) -> RunEvent | None:
    if not events:
        return None
    last = events[-1]
    event_type = "complete" if status == "completed" else "failed"
    return RunEvent(
        run_set_id=last.run_set_id,
        row_id=last.row_id,
        seq=last.seq + 1,
        emitted_at_ms=int(time.time() * 1000),
        type=event_type,
        payload={"status": status, "synthetic": True},
    )


def run_event_from_legacy_worker_event(
    event: Mapping[str, Any],
    *,
    run_set_id: str,
    row_id: str,
    seq: int,
    emitted_at_ms: int | None = None,
) -> RunEvent:
    """Wrap an existing Studio worker event in the run-event envelope."""
    legacy_type = str(event.get("type", "log"))
    event_type = _legacy_worker_type_to_run_event_type(legacy_type)
    payload: MutableMapping[str, Any] = dict(event)
    payload["legacy_type"] = legacy_type
    payload.pop("schema_id", None)
    payload.pop("schema_version", None)
    payload.pop("seq", None)
    payload.pop("emitted_at_ms", None)
    return RunEvent(
        run_set_id=run_set_id,
        row_id=row_id,
        seq=seq,
        emitted_at_ms=emitted_at_ms if emitted_at_ms is not None else int(time.time() * 1000),
        type=event_type,
        payload=dict(payload),
    )


def legacy_worker_event_from_run_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a worker run-event envelope back to the legacy Studio event shape."""
    run_event = RunEvent.model_validate(event)
    payload = dict(run_event.payload)
    legacy_type = payload.pop("legacy_type", None) or _run_event_type_to_legacy_worker_type(
        run_event.type
    )
    payload["type"] = legacy_type
    payload["seq"] = run_event.seq
    payload.setdefault("job_id", run_event.row_id)
    return payload


def _legacy_worker_type_to_run_event_type(legacy_type: str) -> str:
    return {
        "training_progress": "progress",
        "training_complete": "complete",
        "training_error": "failed",
        "training_log": "log",
        "training_trajectory": "trajectory",
    }.get(legacy_type, "log")


def _run_event_type_to_legacy_worker_type(event_type: str) -> str:
    return {
        "progress": "training_progress",
        "complete": "training_complete",
        "failed": "training_error",
        "log": "training_log",
        "trajectory": "training_trajectory",
    }.get(event_type, "training_log")
