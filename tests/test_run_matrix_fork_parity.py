from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TrainingRunMatrixSpec,
)
from feedbax.training.run_matrix import (
    ForkParityError,
    MaterializedMatrixRow,
    MaterializedRunMatrix,
    fork_matrix_checkpoints,
)

from tests.test_run_matrix_materialization import _matrix, _training_run_payload
from feedbax.training.run_matrix import materialize_run_matrix


def _write_latest(root: Path, *, transaction_id: str, digest: str) -> None:
    tx_dir = root / "transactions" / transaction_id
    tx_dir.mkdir(parents=True)
    manifest = {
        "transaction_id": transaction_id,
        "completed_training_batches": 5,
        "content_integrity_digest": {
            "slots": [{"slot": "model", "slot_root_sha256": digest}],
        },
    }
    (tx_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "latest.json").write_text(
        json.dumps({"manifest_relative_path": f"transactions/{transaction_id}/manifest.json"}),
        encoding="utf-8",
    )


def test_fork_matrix_checkpoints_skip_fork_writes_parity_table(tmp_path: Path) -> None:
    spec = _matrix(_training_run_payload())
    materialized = materialize_run_matrix(spec, repo_root=tmp_path)
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_latest(source, transaction_id="tx-source", digest="same")
    _write_latest(target, transaction_id="tx-target", digest="same")

    table = fork_matrix_checkpoints(
        spec,
        MaterializedRunMatrix(
            matrix_spec_sha256=materialized.matrix_spec_sha256,
            run_set_id=materialized.run_set_id,
            base_payload=materialized.base_payload,
            rows=[materialized.rows[0]],
            run_set_manifest=materialized.run_set_manifest,
        ),
        source_checkpoint_root=source,
        target_checkpoint_roots={"lr_hi": target},
        parity_output_path=tmp_path / "parity.json",
        skip_fork=True,
    )

    assert table["schema_version"] == "feedbax.run_matrix_fork_parity.v1"
    assert table["ok"] is True
    assert table["matrix_spec_sha256"] == materialized.matrix_spec_sha256
    assert any(row["kind"] == "slot_parity" and row["ok"] for row in table["rows"])
    assert any(row["kind"] == "lr_continuation" for row in table["rows"])


def test_fork_matrix_checkpoints_reports_mismatched_slot(tmp_path: Path) -> None:
    spec = _matrix(_training_run_payload())
    materialized = materialize_run_matrix(spec, repo_root=tmp_path)
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_latest(source, transaction_id="tx-source", digest="source")
    _write_latest(target, transaction_id="tx-target", digest="target")

    with pytest.raises(ForkParityError, match="row=lr_hi slot=model"):
        fork_matrix_checkpoints(
            spec,
            MaterializedRunMatrix(
                matrix_spec_sha256=materialized.matrix_spec_sha256,
                run_set_id=materialized.run_set_id,
                base_payload=materialized.base_payload,
                rows=[materialized.rows[0]],
                run_set_manifest=materialized.run_set_manifest,
            ),
            source_checkpoint_root=source,
            target_checkpoint_roots={"lr_hi": target},
            parity_output_path=tmp_path / "parity.json",
            skip_fork=True,
        )
