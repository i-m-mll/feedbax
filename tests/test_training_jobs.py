from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

import feedbax.web.api.training as training_api
import feedbax.web.services.training_service as training_service_module
import feedbax.web.worker.app as worker_app
from feedbax.web.services.training_service import TrainingService
from feedbax.web.worker.app import WorkerStatus


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
    first = client.post("/start", json={"total_batches": 1}).json()["job_id"]
    first_status = _wait_for_worker_status(client, first, WorkerStatus.COMPLETED)
    second = client.post("/start", json={"total_batches": 2}).json()["job_id"]
    second_status = _wait_for_worker_status(client, second, WorkerStatus.COMPLETED)

    first_manifest = client.get(f"/jobs/{first}/manifest")
    second_manifest = client.get(f"/jobs/{second}/manifest")

    assert first_status.status_code == 200
    assert second_status.status_code == 200
    assert first_status.json()["job_id"] == first
    assert second_status.json()["job_id"] == second
    assert first_status.json()["last_loss"] == 1.0
    assert second_status.json()["last_loss"] == 2.0
    assert first_manifest.json()["job_id"] == first
    assert second_manifest.json()["job_id"] == second
    assert client.get("/jobs/missing/status").status_code == 404


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
    first = client.post("/start", json={"total_batches": 1}).json()["job_id"]
    assert entered.wait(timeout=2)

    conflict = client.post("/start", json={"total_batches": 1})
    assert conflict.status_code == 409
    assert "already running job" in conflict.json()["detail"]

    release.set()
    assert _wait_for_worker_status(client, first, WorkerStatus.COMPLETED).status_code == 200

    second = client.post("/start", json={"total_batches": 1})
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
    first = client.post("/start", json={"total_batches": 1}).json()["job_id"]
    assert _wait_for_worker_status(client, first, WorkerStatus.COMPLETED).status_code == 200
    second = client.post("/start", json={"total_batches": 2}).json()["job_id"]
    assert _wait_for_worker_status(client, second, WorkerStatus.COMPLETED).status_code == 200
    third = client.post("/start", json={"total_batches": 3}).json()["job_id"]
    assert _wait_for_worker_status(client, third, WorkerStatus.COMPLETED).status_code == 200

    assert client.get(f"/jobs/{first}/status").status_code == 404
    assert client.get(f"/jobs/{second}/status").status_code == 200
    assert client.get(f"/jobs/{third}/manifest").json()["job_id"] == third


def test_training_service_passes_job_id_to_worker_client(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    started = iter(["job-a", "job-b"])

    async def fake_start_job(base_url: str, total_batches: int, **kwargs: Any) -> str:
        assert base_url == "http://worker"
        return next(started)

    async def fake_get_status(base_url: str, job_id: str, **kwargs: Any) -> dict:
        calls.append(("status", job_id))
        return {
            "status": "completed",
            "batch": 1,
            "total_batches": 1,
            "last_loss": 10.0 if job_id == "job-a" else 20.0,
            "job_id": job_id,
        }

    async def fake_stop_job(base_url: str, job_id: str, **kwargs: Any) -> None:
        calls.append(("stop", job_id))

    async def fake_get_checkpoint(base_url: str, job_id: str, **kwargs: Any) -> dict:
        calls.append(("checkpoint", job_id))
        return {"batch": 1, "loss": 3.0, "weights_available": False}

    async def fake_get_manifest(base_url: str, job_id: str, **kwargs: Any) -> dict:
        calls.append(("manifest", job_id))
        return {"job_id": job_id}

    async def fake_download_checkpoint(
        base_url: str,
        job_id: str,
        dest_path: str,
        **kwargs: Any,
    ) -> None:
        calls.append(("download", job_id))

    async def run() -> None:
        monkeypatch.setattr(training_service_module.worker_client, "start_job", fake_start_job)
        monkeypatch.setattr(training_service_module.worker_client, "get_status", fake_get_status)
        monkeypatch.setattr(training_service_module.worker_client, "stop_job", fake_stop_job)
        monkeypatch.setattr(
            training_service_module.worker_client,
            "get_checkpoint",
            fake_get_checkpoint,
        )
        monkeypatch.setattr(
            training_service_module.worker_client,
            "get_manifest",
            fake_get_manifest,
        )
        monkeypatch.setattr(
            training_service_module.worker_client,
            "download_checkpoint",
            fake_download_checkpoint,
        )

        service = TrainingService()
        service.connect_remote("http://worker")

        assert await service.start_training(1) == "job-a"
        assert await service.start_training(1) == "job-b"
        assert (await service.get_status("job-a"))["last_loss"] == 10.0
        assert (await service.get_status("job-b"))["last_loss"] == 20.0
        await service.stop_training("job-b")
        assert (await service.latest_checkpoint("job-a"))["job_id"] == "job-a"
        assert (await service.latest_manifest("job-b"))["job_id"] == "job-b"
        await service.download_checkpoint("job-a", str(tmp_path / "checkpoint.eqx"))

    asyncio.run(run())

    assert calls == [
        ("status", "job-a"),
        ("status", "job-b"),
        ("stop", "job-b"),
        ("checkpoint", "job-a"),
        ("manifest", "job-b"),
        ("download", "job-a"),
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
    assert client.delete("/api/training/job-b").json() == {"data": {"success": True}}
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
