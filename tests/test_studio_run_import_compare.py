from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feedbax.analysis.bundles import iter_candidate_manifests
from feedbax.contracts.manifest import (
    TrainingRunManifest,
    canonical_manifest_path,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.manifest_packet import export_manifest_packet
from feedbax.persistence.manifest_index import find_manifest_paths_by_id
from feedbax.web.app import create_app


def _write_training_run(
    root: Path,
    run_id: str,
    *,
    learning_rate: float,
    loss: float,
    index: bool = True,
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
    write_manifest(manifest, root=root, index=index)
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


@pytest.mark.parametrize("index_state", ["missing", "partial", "stale"])
def test_selection_uses_manifest_files(tmp_path: Path, monkeypatch, index_state: str) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    _write_training_run(tmp_path, "run:unindexed", learning_rate=0.01, loss=0.4, index=False)
    if index_state != "missing":
        _write_training_run(tmp_path, "run:indexed", learning_rate=0.01, loss=0.3)
    if index_state == "stale":
        removed = _write_training_run(tmp_path, "run:removed", learning_rate=0.01, loss=0.2)
        canonical_manifest_path(removed.kind, removed.id, root=tmp_path).unlink()
        current = _write_training_run(tmp_path, "run:changed", learning_rate=0.01, loss=0.1)
        write_manifest(
            current.model_copy(update={"status": "cancelled"}), root=tmp_path, index=False
        )

    expected = {"run:unindexed"} if index_state == "missing" else {"run:unindexed", "run:indexed"}
    with TestClient(create_app()) as client:
        preview = client.post(
            "/api/runs/selection/preview",
            json={
                "selection_spec": {
                    "mode": "query",
                    "manifest_kind": "TrainingRunManifest",
                    "query": {"manifest_kind": "TrainingRunManifest", "statuses": ["completed"]},
                },
            },
        )
        exact = client.get("/api/runs/training/run:unindexed/manifest")

    assert preview.status_code == 200
    assert {ref["id"] for ref in preview.json()["parent_refs"]} == expected
    assert exact.status_code == 200
    assert exact.json()["id"] == "run:unindexed"
    assert {
        manifest.id
        for manifest in iter_candidate_manifests(tmp_path, manifest_kind="TrainingRunManifest")
        if manifest.status == "completed"
    } == expected


def test_runs_dir_import_reads_only_selected_target_receipts(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(target))
    for index in range(15):
        _write_training_run(target, f"run:unrelated-{index}", learning_rate=0.01, loss=0.4)
    _write_training_run(target, "run:existing", learning_rate=0.01, loss=0.2, index=False)
    _write_training_run(source, "run:existing", learning_rate=0.01, loss=0.9)
    _write_training_run(source, "run:new", learning_rate=0.01, loss=0.3)

    target_receipts_read = set()
    read_bytes = Path.read_bytes

    def track_receipt_read(path: Path) -> bytes:
        if path.suffix == ".json" and path.is_relative_to(target / "manifests"):
            target_receipts_read.add(path.stem)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", track_receipt_read)
    with TestClient(create_app()) as client:
        response = client.post("/api/runs/import/runs-dir", json={"path": str(source)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_manifest_ids"] == ["run:new"]
    assert payload["skipped_manifest_ids"] == ["run:existing"]
    assert {
        run["id"]: run["metrics"]["final_validation_loss"] for run in payload["training_runs"]
    } == {
        "run:new": 0.3,
        "run:existing": 0.2,
    }
    assert target_receipts_read <= {"run_existing", "run_new"}
    assert find_manifest_paths_by_id("run:existing", root=target) == [
        canonical_manifest_path("TrainingRunManifest", "run:existing", root=target)
    ]
