from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

import feedbax.web.api.training as training_api
import feedbax.web.services.training_service as training_service_module
import feedbax.web.worker.client as worker_client
import feedbax.web.worker.app as worker_app
from feedbax.contracts.studio_api import (
    STUDIO_API_TRANSPORT_SCHEMA_ID,
    STUDIO_API_TRANSPORT_SCHEMA_VERSION,
)
from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
    StudioTrainingAssemblySpec,
)
from feedbax.orchestration.events import RUN_EVENT_SCHEMA_ID, RunEvent
from feedbax.orchestration.assembly import assemble_run_bundle
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.state import RowState, RunSetState, RunSetStateStore
from feedbax.web.services.training_service import TrainingService
from feedbax.web.services.worker_driver import _worker_start_body, load_worker_execution_payload
from feedbax.web.worker.app import WorkerStatus


def test_studio_training_assembly_spec_governs_worker_payload() -> None:
    spec = StudioTrainingAssemblySpec(
        total_batches=3,
        training_config={"learning_rate": 0.01},
        graph_spec={"schema_id": "feedbax.graph", "schema_version": "feedbax.graph.v1"},
    )

    payload = spec.worker_payload()

    assert payload["schema_version"] == STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION
    assert payload["total_batches"] == 3
    assert payload["snapshot_interval"] == 100
    assert payload["training_config"] == {"learning_rate": 0.01}
    assert "job_id" not in payload
    assert "run_set_id" not in payload


def test_studio_training_assembly_spec_rejects_launch_identity_fields() -> None:
    try:
        StudioTrainingAssemblySpec(
            total_batches=3,
            job_id="caller-job",
            run_set_id="caller-set",
        )
    except ValueError as exc:
        assert "extra" in str(exc).lower()
    else:
        raise AssertionError("Studio authored request accepted orchestrator-owned identities")


def test_worker_driver_resolves_registered_typed_execution_payload(tmp_path) -> None:
    spec = StudioTrainingAssemblySpec(total_batches=7, snapshot_interval=11)
    payload_path = tmp_path / "studio-training.json"
    payload_bytes = spec.model_dump_json(exclude_none=True).encode()
    payload_path.write_bytes(payload_bytes)
    payload_ref = SimpleNamespace(
        schema_id=spec.schema_id,
        schema_version=spec.schema_version,
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
        uri=str(payload_path),
    )
    bundle = SimpleNamespace(run_set_id="set-typed")
    row = SimpleNamespace(
        row_id="job-typed",
        execution=SimpleNamespace(payload=payload_ref),
        metadata={"worker_start": {"total_batches": 999}},
    )

    assert _worker_start_body(bundle, row) == {
        "schema_id": spec.schema_id,
        "schema_version": spec.schema_version,
        "total_batches": 7,
        "snapshot_interval": 11,
        "job_id": "job-typed",
        "run_set_id": "set-typed",
    }


def test_training_service_builds_governed_request_and_compiled_worker_payload(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orchestration"))
    service = TrainingService()
    request, context, registry = service._build_worker_assembly_request(
        worker_start={"total_batches": 5}
    )

    bundle = assemble_run_bundle(
        request,
        run_set_id="2026-07-12-a1b2c3d4",
        context=context,
        registry=registry,
    )

    row = bundle.rows[0]
    assert row.row_id == "2026-07-12-a1b2c3d4-studio"
    assert "metadata" not in row.model_dump(mode="json")
    assert row.launch.metadata == {}
    assert load_worker_execution_payload(row)["total_batches"] == 5
    assert row.execution.immutable_inputs == []


def _wait_for_worker_status(
    client: TestClient,
    job_id: str,
    status: WorkerStatus,
    *,
    timeout: float = 2.0,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}/status")
        if response.status_code == 200 and response.json()["status"] == status.value:
            return response
        time.sleep(0.01)
    return client.get(f"/jobs/{job_id}/status")


def test_worker_routes_keep_terminal_state_for_distinct_job_ids(monkeypatch) -> None:
    def fake_run_training(job: worker_app._Job) -> None:
        loss = float(job.total_batches)
        with job._state_lock:
            job.batch = job.total_batches
            job.last_loss = loss
            job.manifest_payload = {"kind": "TrainingRunManifest", "job_id": job.job_id}
        worker_app._mark_job_terminal(job, WorkerStatus.COMPLETED)
        worker_app._emit(
            job,
            {
                "type": "training_complete",
                "job_id": job.job_id,
                "batch": job.total_batches,
                "loss": loss,
            },
        )
        job.event_queue.put(None)

    monkeypatch.setattr(worker_app, "_run_training", fake_run_training)

    client = TestClient(worker_app.create_app())
    first = client.post(
        "/start",
        json={"job_id": "job-a", "run_set_id": "set-a", "total_batches": 1},
    ).json()["job_id"]
    first_status = _wait_for_worker_status(client, first, WorkerStatus.COMPLETED)
    second = client.post(
        "/start",
        json={"job_id": "job-b", "run_set_id": "set-b", "total_batches": 2},
    ).json()["job_id"]
    second_status = _wait_for_worker_status(client, second, WorkerStatus.COMPLETED)

    first_manifest = client.get(f"/jobs/{first}/manifest")
    second_manifest = client.get(f"/jobs/{second}/manifest")

    assert first_status.status_code == 200
    assert second_status.status_code == 200
    assert first == "job-a"
    assert second == "job-b"
    assert first_status.json()["job_id"] == first
    assert second_status.json()["job_id"] == second
    assert first_status.json()["last_loss"] == 1.0
    assert second_status.json()["last_loss"] == 2.0
    assert first_manifest.json()["job_id"] == first
    assert second_manifest.json()["job_id"] == second
    assert client.get("/jobs/missing/status").status_code == 404


def test_worker_start_requires_external_identity() -> None:
    client = TestClient(worker_app.create_app())

    missing_job = client.post("/start", json={"run_set_id": "set-a", "total_batches": 1})
    assert missing_job.status_code == 400
    assert client.post("/start", json={"job_id": "job-a", "total_batches": 1}).status_code == 400


def test_worker_lifespan_publishes_bootstrap_state_before_routes() -> None:
    app = worker_app.create_app()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert app.state.bootstrap_state.bundle.components.get("Gain") is not None


def test_worker_rejects_start_while_job_running(monkeypatch) -> None:
    release = threading.Event()
    entered = threading.Event()

    def fake_run_training(job: worker_app._Job) -> None:
        entered.set()
        assert release.wait(timeout=2)
        with job._state_lock:
            job.batch = job.total_batches
            job.last_loss = 1.0
        worker_app._mark_job_terminal(job, WorkerStatus.COMPLETED)
        worker_app._emit(
            job,
            {
                "type": "training_complete",
                "job_id": job.job_id,
                "batch": job.total_batches,
                "loss": 1.0,
            },
        )
        job.event_queue.put(None)

    monkeypatch.setattr(worker_app, "_run_training", fake_run_training)

    client = TestClient(worker_app.create_app())
    first = client.post(
        "/start",
        json={"job_id": "job-running", "run_set_id": "set-running", "total_batches": 1},
    ).json()["job_id"]
    assert entered.wait(timeout=2)

    conflict = client.post(
        "/start",
        json={"job_id": "job-conflict", "run_set_id": "set-conflict", "total_batches": 1},
    )
    assert conflict.status_code == 409
    assert "already running job" in conflict.json()["detail"]

    release.set()
    assert _wait_for_worker_status(client, first, WorkerStatus.COMPLETED).status_code == 200

    second = client.post(
        "/start",
        json={"job_id": "job-second", "run_set_id": "set-second", "total_batches": 1},
    )
    assert second.status_code == 200


def test_worker_evicts_oldest_terminal_jobs(monkeypatch) -> None:
    def fake_run_training(job: worker_app._Job) -> None:
        with job._state_lock:
            job.batch = job.total_batches
            job.last_loss = float(job.total_batches)
            job.manifest_payload = {"kind": "TrainingRunManifest", "job_id": job.job_id}
        worker_app._mark_job_terminal(job, WorkerStatus.COMPLETED)
        worker_app._emit(
            job,
            {
                "type": "training_complete",
                "job_id": job.job_id,
                "batch": job.total_batches,
                "loss": float(job.total_batches),
            },
        )
        job.event_queue.put(None)

    monkeypatch.setattr(worker_app, "_TERMINAL_JOB_RETENTION_MAX", 2)
    monkeypatch.setattr(worker_app, "_run_training", fake_run_training)

    client = TestClient(worker_app.create_app())
    first = client.post(
        "/start",
        json={"job_id": "job-first", "run_set_id": "set-first", "total_batches": 1},
    ).json()["job_id"]
    assert _wait_for_worker_status(client, first, WorkerStatus.COMPLETED).status_code == 200
    second = client.post(
        "/start",
        json={"job_id": "job-second", "run_set_id": "set-second", "total_batches": 2},
    ).json()["job_id"]
    assert _wait_for_worker_status(client, second, WorkerStatus.COMPLETED).status_code == 200
    third = client.post(
        "/start",
        json={"job_id": "job-third", "run_set_id": "set-third", "total_batches": 3},
    ).json()["job_id"]
    assert _wait_for_worker_status(client, third, WorkerStatus.COMPLETED).status_code == 200

    assert client.get(f"/jobs/{first}/status").status_code == 404
    assert client.get(f"/jobs/{second}/status").status_code == 200
    assert client.get(f"/jobs/{third}/manifest").json()["job_id"] == third


def test_training_service_starts_state_backed_worker_run(monkeypatch, tmp_path) -> None:
    starts: list[dict[str, Any]] = []

    class FakeWorkerDriver:
        def __init__(self, *, base_url: str, auth_token: str | None = None) -> None:
            assert base_url == "http://worker"
            assert auth_token is None

        def provision(self, bundle, state) -> Mapping[str, Any]:
            del state
            for dirname in ("events", "sentinels", "rows", "collected", "inputs"):
                (bundle.run_set_dir / dirname).mkdir(parents=True, exist_ok=True)
            return {"driver": "fake-worker"}

        def realize_env(self, bundle, state) -> str:
            del bundle, state
            return "fake-env"

        def stage_inputs(self, bundle, state) -> Mapping[str, Any]:
            del bundle, state
            return {}

        def launch_row(self, bundle, row, state) -> Mapping[str, Any]:
            del state
            body = load_worker_execution_payload(row)
            body["job_id"] = row.row_id
            body["run_set_id"] = bundle.run_set_id
            starts.append(body)
            events_dir = bundle.run_set_dir / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            event = RunEvent(
                run_set_id=bundle.run_set_id,
                row_id=row.row_id,
                seq=0,
                emitted_at_ms=1783430000000,
                type="complete",
                payload={
                    "legacy_type": "training_complete",
                    "job_id": row.row_id,
                    "batch": body["total_batches"],
                    "total_batches": body["total_batches"],
                    "loss": 0.125,
                },
            )
            (events_dir / f"{row.row_id}.events.jsonl").write_text(
                event.model_dump_json(exclude_none=True) + "\n",
                encoding="utf-8",
            )
            sentinels = bundle.run_set_dir / "sentinels"
            sentinels.mkdir(parents=True, exist_ok=True)
            (sentinels / f"{row.row_id}.done").write_text("done\n", encoding="utf-8")
            return {"row_id": row.row_id}

        def probe(self, bundle, row, state) -> DriverRowProbe:
            del bundle, row, state
            return DriverRowProbe(status="completed")

        def stop_row(self, bundle, row, state) -> Mapping[str, Any]:
            del bundle, state
            return {"row_id": row.row_id, "status": "stopped"}

        def collect(self, bundle, row, state) -> Mapping[str, str]:
            del state
            event_path = bundle.run_set_dir / "events" / f"{row.row_id}.events.jsonl"
            return {event_path.name: str(event_path)}

        def teardown(self, bundle, state) -> Mapping[str, Any]:
            del bundle, state
            return {}

    async def run() -> None:
        monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orch"))
        monkeypatch.setattr(training_service_module, "WorkerHttpDriver", FakeWorkerDriver)

        service = TrainingService()
        service.connect_remote("http://worker")

        job_id = await service.start_training(3)
        deadline = time.monotonic() + 2.0
        status = None
        while time.monotonic() < deadline:
            status = await service.get_status(job_id)
            if status and status["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        assert status is not None
        assert status["run_set_id"] == starts[0]["run_set_id"]
        assert status["last_loss"] == 0.125
        assert starts == [
            {
                "schema_id": "feedbax.spec.studio.training_assembly",
                "schema_version": "feedbax.spec.studio.training_assembly.v1",
                "job_id": job_id,
                "run_set_id": status["run_set_id"],
                "total_batches": 3,
                "snapshot_interval": 100,
            }
        ]
        assert service.list_live_training_runs()[0]["id"] == job_id

    asyncio.run(run())


def test_training_service_reads_legacy_v2_terminal_status_without_mutating(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    bundle = SimpleNamespace(run_set_id="set-terminal", run_set_dir=tmp_path / "set-terminal")
    bundle.run_set_dir.mkdir(parents=True)
    (bundle.run_set_dir / "bundle.json").write_text(
        json.dumps(
            {
                "schema_id": "feedbax.orchestration.run_bundle",
                "schema_version": "feedbax.orchestration.run_bundle.v2",
                "run_set_id": bundle.run_set_id,
                "rows": [
                    {
                        "row_id": "job-terminal",
                        "metadata": {"worker_start": {"total_batches": 1}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    events_dir = bundle.run_set_dir / "events"
    sentinels = bundle.run_set_dir / "sentinels"
    events_dir.mkdir()
    sentinels.mkdir()
    event = RunEvent(
        run_set_id=bundle.run_set_id,
        row_id="job-terminal",
        seq=0,
        emitted_at_ms=1783430000000,
        type="complete",
        payload={"legacy_type": "training_complete", "batch": 1, "loss": 0.5},
    )
    (events_dir / "job-terminal.events.jsonl").write_text(
        event.model_dump_json(exclude_none=True) + "\n",
        encoding="utf-8",
    )
    (sentinels / "job-terminal.done").write_text("done\n", encoding="utf-8")
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    store.save(
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={"job-terminal": RowState(status="running")},
        )
    )

    service = TrainingService()
    status = service._status_from_state("job-terminal")
    assert status is not None
    assert status["status"] == "running"
    assert status["total_batches"] == 1
    assert status["last_loss"] == 0.5
    assert store.load().rows["job-terminal"].status == "running"


def test_training_service_reads_legacy_v2_orphan_status_without_mutating(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    bundle = SimpleNamespace(run_set_id="set-orphan", run_set_dir=tmp_path / "set-orphan")
    bundle.run_set_dir.mkdir(parents=True)
    (bundle.run_set_dir / "bundle.json").write_text(
        json.dumps(
            {
                "schema_id": "feedbax.orchestration.run_bundle",
                "schema_version": "feedbax.orchestration.run_bundle.v2",
                "run_set_id": bundle.run_set_id,
                "rows": [
                    {
                        "row_id": "job-orphan",
                        "metadata": {"worker_start": {"total_batches": 1}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (bundle.run_set_dir / "events").mkdir()
    (bundle.run_set_dir / "sentinels").mkdir()
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    store.save(
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={"job-orphan": RowState(status="running")},
        )
    )

    service = TrainingService()
    status = service._status_from_state("job-orphan")
    assert status is not None
    assert status["status"] == "running"
    assert status["total_batches"] == 1
    assert store.load().rows["job-orphan"].status == "running"


def test_training_service_preserves_worker_seq_in_ws_envelope(monkeypatch) -> None:
    async def fake_stream_events(base_url: str, job_id: str, **kwargs: Any):
        assert base_url == "http://worker"
        yield {
            "type": "training_progress",
            "job_id": job_id,
            "seq": 7,
            "batch": 12,
            "total_batches": 20,
            "loss": 0.25,
        }

    async def run() -> None:
        monkeypatch.setattr(
            training_service_module.worker_client,
            "stream_events",
            fake_stream_events,
        )
        service = TrainingService()
        service.connect_remote("http://worker")
        [event] = [event async for event in service.stream_progress("job-ws")]
        assert event.raw["seq"] == 7
        assert event.raw["worker_seq"] == 7
        assert event.raw["schema_version"] == STUDIO_API_TRANSPORT_SCHEMA_VERSION

    asyncio.run(run())


def test_worker_emit_buffers_run_event_envelopes() -> None:
    job = worker_app._Job(
        job_id="job-events",
        run_set_id="run-set-events",
        total_batches=3,
        event_queue=worker_app.queue.Queue(),
        stop_event=threading.Event(),
    )

    worker_app._emit(
        job,
        {
            "type": "training_progress",
            "job_id": job.job_id,
            "batch": 1,
            "total_batches": 3,
            "loss": 0.5,
        },
    )
    worker_app._emit(
        job,
        {
            "type": "training_complete",
            "job_id": job.job_id,
            "batch": 3,
            "loss": 0.1,
        },
    )

    first = job.event_queue.get_nowait()
    second = job.event_queue.get_nowait()

    assert first["schema_id"] == RUN_EVENT_SCHEMA_ID
    assert first["run_set_id"] == "run-set-events"
    assert first["row_id"] == "job-events"
    assert first["type"] == "progress"
    assert first["payload"]["legacy_type"] == "training_progress"
    assert first["payload"]["batch"] == 1
    assert second["type"] == "complete"
    assert [seq for seq, _event in job.event_buffer] == [0, 1]


def test_training_service_unwraps_run_event_worker_stream(monkeypatch) -> None:
    async def fake_stream_events(base_url: str, job_id: str, **kwargs: Any):
        assert base_url == "http://worker"
        yield RunEvent(
            run_set_id=job_id,
            row_id=job_id,
            seq=9,
            emitted_at_ms=1783430000000,
            type="progress",
            payload={
                "legacy_type": "training_progress",
                "job_id": job_id,
                "batch": 2,
                "total_batches": 5,
                "loss": 0.25,
            },
        ).model_dump(mode="json")

    async def run() -> None:
        monkeypatch.setattr(
            training_service_module.worker_client,
            "stream_events",
            fake_stream_events,
        )

        service = TrainingService()
        service.connect_remote("http://worker")
        [event] = [event async for event in service.stream_progress("job-run-event")]

        assert event.raw["type"] == "training_progress"
        assert event.raw["worker_seq"] == 9
        assert event.raw["seq"] == 9
        assert event.raw["batch"] == 2
        assert event.raw["schema_version"] == STUDIO_API_TRANSPORT_SCHEMA_VERSION

    asyncio.run(run())


def test_training_service_preserves_error_diagnostics(monkeypatch) -> None:
    async def fake_stream_events(base_url: str, job_id: str, **kwargs: Any):
        assert base_url == "http://worker"
        yield {
            "type": "training_error",
            "job_id": job_id,
            "seq": 4,
            "batch": 0,
            "error": "Invalid graph_spec for graph execution",
            "diagnostics": [
                {
                    "severity": "error",
                    "code": "graph.missing_subgraph",
                    "message": "Network node 'network' has no subgraph",
                    "node_ids": ["network"],
                }
            ],
        }

    async def run() -> None:
        monkeypatch.setattr(
            training_service_module.worker_client,
            "stream_events",
            fake_stream_events,
        )

        service = TrainingService()
        service.connect_remote("http://worker")
        [event] = [event async for event in service.stream_progress("job-diagnostics")]

        assert event.raw["type"] == "training_error"
        assert event.raw["worker_seq"] == 4
        assert event.raw["seq"] == 4
        assert event.raw["diagnostics"][0]["code"] == "graph.missing_subgraph"
        assert event.raw["diagnostics"][0]["node_ids"] == ["network"]

    asyncio.run(run())


def test_training_service_surfaces_reconnect_resync_marker(monkeypatch) -> None:
    async def fake_stream_events(base_url: str, job_id: str, **kwargs: Any):
        yield {
            "type": "training_resync",
            "job_id": job_id,
            "expected_worker_seq": 5,
            "observed_worker_seq": 8,
            "missed_events": 3,
            "reason": "gap",
            "message": "Training stream resumed after reconnect with 3 missed event(s).",
        }

    async def run() -> None:
        monkeypatch.setattr(
            training_service_module.worker_client,
            "stream_events",
            fake_stream_events,
        )

        service = TrainingService()
        service.connect_remote("http://worker")
        [event] = [event async for event in service.stream_progress("job-gap")]

        assert event.raw == {
            "type": "training_resync",
            "job_id": "job-gap",
            "schema_id": STUDIO_API_TRANSPORT_SCHEMA_ID,
            "schema_version": STUDIO_API_TRANSPORT_SCHEMA_VERSION,
            "seq": 0,
            "emitted_at_ms": event.raw["emitted_at_ms"],
            "expected_worker_seq": 5,
            "observed_worker_seq": 8,
            "missed_events": 3,
            "reason": "gap",
            "message": "Training stream resumed after reconnect with 3 missed event(s).",
        }

    asyncio.run(run())


def test_worker_client_emits_gap_marker_after_reconnect(monkeypatch) -> None:
    stream_calls: list[dict] = []
    streams = [
        (
            ['data: {"type": "training_progress", "job_id": "job-gap", "seq": 0}\n'],
            worker_client.httpx.ReadError("dropped"),
        ),
        (
            ['data: {"type": "training_complete", "job_id": "job-gap", "seq": 3, "batch": 1}\n'],
            None,
        ),
    ]

    class FakeResponse:
        def __init__(self, lines: list[str], error: Exception | None) -> None:
            self._lines = lines
            self._error = error

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            for line in self._lines:
                yield line
            if self._error is not None:
                raise self._error

    class FakeStream:
        def __init__(self, response: FakeResponse) -> None:
            self._response = response

        async def __aenter__(self) -> FakeResponse:
            return self._response

        async def __aexit__(self, *args: object) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def stream(
            self,
            method: str,
            url: str,
            *,
            params: dict,
            headers: dict,
        ) -> FakeStream:
            stream_calls.append(params)
            lines, error = streams.pop(0)
            return FakeStream(FakeResponse(lines, error))

    async def run() -> list[dict]:
        monkeypatch.setattr(worker_client.httpx, "AsyncClient", FakeClient)
        monkeypatch.setattr(worker_client, "_RECONNECT_DELAY", 0)
        return [event async for event in worker_client.stream_events("http://worker", "job-gap")]

    events = asyncio.run(run())

    assert stream_calls == [{}, {"from_seq": 1}]
    assert events == [
        {"type": "training_progress", "job_id": "job-gap", "seq": 0},
        {
            "type": "training_resync",
            "job_id": "job-gap",
            "expected_worker_seq": 1,
            "observed_worker_seq": 3,
            "worker_seq": 0,
            "missed_events": 2,
            "reason": "gap",
            "message": "Training stream resumed after reconnect with 2 missed event(s).",
        },
        {"type": "training_complete", "job_id": "job-gap", "seq": 3, "batch": 1},
    ]


def test_training_api_routes_pass_path_job_id(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class StubTrainingService:
        async def get_status(self, job_id: str) -> dict:
            calls.append(("status", job_id))
            return {"status": "completed", "job_id": job_id}

        async def stop_training(self, job_id: str) -> None:
            calls.append(("stop", job_id))

    app = FastAPI()
    app.include_router(training_api.router, prefix="/api/training")
    monkeypatch.setattr(training_api, "training_service", StubTrainingService())

    client = TestClient(app)

    assert client.get("/api/training/job-a").json()["data"]["status"]["job_id"] == "job-a"
    delete_payload = client.delete("/api/training/job-b").json()
    assert delete_payload["data"]["success"] is True
    assert delete_payload["data"]["schema_id"] == STUDIO_API_TRANSPORT_SCHEMA_ID
    assert delete_payload["data"]["schema_version"] == STUDIO_API_TRANSPORT_SCHEMA_VERSION
    assert calls == [("status", "job-a"), ("stop", "job-b")]


def test_ensure_worker_serializes_concurrent_spawns(monkeypatch) -> None:
    popen_calls: list[list[str]] = []
    popen_kwargs: list[dict[str, Any]] = []

    class FakeProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    def fake_popen(args: list[str], **kwargs: Any) -> FakeProcess:
        popen_calls.append(args)
        popen_kwargs.append(kwargs)
        return FakeProcess()

    async def fake_wait_for_health(*args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)

    async def run() -> tuple[str, str]:
        service = TrainingService()
        return await asyncio.gather(service._ensure_worker(), service._ensure_worker())

    monkeypatch.setattr(training_service_module, "_find_free_port", lambda: 54321)
    monkeypatch.setattr(training_service_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        training_service_module.worker_client,
        "wait_for_health",
        fake_wait_for_health,
    )

    first_url, second_url = asyncio.run(run())

    assert first_url == "http://127.0.0.1:54321"
    assert second_url == first_url
    assert len(popen_calls) == 1
    assert popen_kwargs[0]["stderr"] is subprocess.PIPE


def test_worker_module_help_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "feedbax.web.worker", "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Feedbax headless training worker" in proc.stdout
