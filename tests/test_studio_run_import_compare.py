from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from feedbax.contracts.manifest import (
    TrainingRunManifest,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.manifest_packet import export_manifest_packet
from feedbax.web.app import create_app


def _write_training_run(
    root: Path,
    run_id: str,
    *,
    learning_rate: float,
    loss: float,
) -> TrainingRunManifest:
    manifest = TrainingRunManifest(
        id=run_id,
        status="completed",
        training_spec=spec_payload(
            "TrainingRunSpec",
            {
                "optimizer": {"params": {"learning_rate": learning_rate}},
                "batch_size": 32,
                "n_warmup_batches": 4,
            },
        ),
        summary_metrics={
            "final_validation_loss": loss,
            "unused_metric": 999,
        },
        metadata={
            "name": run_id,
            "studio": {"axis_coordinates": {"learning_rate": learning_rate}},
        },
    )
    write_manifest(manifest, root=root)
    return manifest


def test_training_compare_route_returns_only_requested_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    _write_training_run(tmp_path, "run:a", learning_rate=0.001, loss=0.2)
    _write_training_run(tmp_path, "run:b", learning_rate=0.0003, loss=0.1)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/runs/training/compare",
            json={
                "run_ids": ["run:a", "run:b"],
                "param_fields": ["learning_rate"],
                "metric_fields": ["final_validation_loss"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == [
        {
            "id": "run:a",
            "params": {"learning_rate": 0.001},
            "metrics": {"final_validation_loss": 0.2},
        },
        {
            "id": "run:b",
            "params": {"learning_rate": 0.0003},
            "metrics": {"final_validation_loss": 0.1},
        },
    ]
    assert "unused_metric" not in payload["rows"][0]["metrics"]


def test_manifest_packet_import_route_indexes_typed_run_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "target"
    source = tmp_path / "source"
    packet = tmp_path / "packet"
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(target))
    _write_training_run(
        source,
        "feedbax-training-run:packet",
        learning_rate=0.001,
        loss=0.2,
    )
    export_manifest_packet(["feedbax-training-run:packet"], root=source, dest=packet)

    with TestClient(create_app()) as client:
        response = client.post("/api/runs/import/packet", json={"path": str(packet)})
        list_response = client.get("/api/runs/training")

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_manifest_ids"] == ["feedbax-training-run:packet"]
    assert payload["training_runs"][0]["id"] == "feedbax-training-run:packet"
    assert list_response.status_code == 200
    assert [row["id"] for row in list_response.json()] == ["feedbax-training-run:packet"]


def test_runs_dir_import_copies_manifest_with_import_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "target"
    source = tmp_path / "source"
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(target))
    _write_training_run(
        source,
        "feedbax-training-run:runs-dir",
        learning_rate=0.001,
        loss=0.2,
    )

    with TestClient(create_app()) as client:
        response = client.post("/api/runs/import/runs-dir", json={"path": str(source)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_manifest_ids"] == ["feedbax-training-run:runs-dir"]
    imported = load_manifest(
        target / "manifests" / "training_runs" / "feedbax-training-run_runs-dir.json"
    )
    assert imported.metadata["imported_from"]["source"] == "runs_dir"
    assert imported.metadata["studio"]["axis_coordinates"]["learning_rate"] == 0.001
