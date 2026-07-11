from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from feedbax.contracts.checkpoints import (
    BatchHistory,
    CheckpointContinuationRequest,
    CheckpointForkBarrierMapping,
)
from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.training.checkpoint_custody import (
    load_latest_checkpoint,
    write_checkpoint_transaction,
)
from feedbax.training.run_matrix import (
    ForkParityError,
    MaterializedRunMatrix,
    fork_matrix_checkpoints,
    main,
)

from tests.test_run_matrix_materialization import _matrix, _training_run_payload
from tests.test_checkpoint_custody import _coordinate, _minimax_slots, _run_spec
from feedbax.training.run_matrix import materialize_run_matrix
from feedbax.contracts.manifest import canonical_json_bytes


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


def _write_topology_latest(
    root: Path,
    *,
    transaction_id: str,
    digests: dict[str, str],
    blob_digests: dict[str, str],
    provenance: list[dict[str, object]] | None = None,
) -> None:
    tx_dir = root / "transactions" / transaction_id
    tx_dir.mkdir(parents=True)
    manifest: dict[str, object] = {
        "transaction_id": transaction_id,
        "completed_training_batches": 5,
        "slots": [{"slot": slot, "sha256": blob_digests[slot]} for slot in sorted(blob_digests)],
        "content_integrity_digest": {
            "slots": [
                {"slot": slot, "slot_root_sha256": digest}
                for slot, digest in sorted(digests.items())
            ],
        },
    }
    if provenance is not None:
        manifest["fork_provenance"] = {"slots": provenance}
    (tx_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "latest.json").write_text(
        json.dumps({"manifest_relative_path": f"transactions/{transaction_id}/manifest.json"}),
        encoding="utf-8",
    )


def _target_transform_provenance(
    slot: str,
    *,
    source_sha256: str | None,
    target_sha256: str,
    target_only_declaration: dict[str, str] | None = None,
) -> dict[str, object]:
    transform_metadata: dict[str, object] = {
        "stage": "target_post",
        "stages": [
            {
                "stage": "target_post",
                "identity": "tests.target_topology.v1",
                "parameters": {"mode": "adaptive"},
                "metadata": {},
            }
        ],
    }
    if target_only_declaration is not None:
        transform_metadata["target_only_declaration"] = target_only_declaration
    return {
        "slot": slot,
        "source_sha256": source_sha256,
        "target_sha256": target_sha256,
        "transfer_mode": "serialized",
        "transform": {
            "slot": slot,
            "identity": "tests.target_topology.v1",
            "parameters": {"mode": "adaptive"},
            "metadata": transform_metadata,
        },
    }


def _topology_parity_inputs(
    tmp_path: Path,
) -> tuple[TrainingRunMatrixSpec, MaterializedRunMatrix, Path, Path, dict[str, str]]:
    spec = _matrix(_training_run_payload())
    materialized = materialize_run_matrix(spec, repo_root=tmp_path)
    one_row = MaterializedRunMatrix(
        matrix_spec_sha256=materialized.matrix_spec_sha256,
        run_set_id=materialized.run_set_id,
        base_payload=materialized.base_payload,
        rows=[materialized.rows[0]],
        run_set_manifest=materialized.run_set_manifest,
    )
    source = tmp_path / "source"
    target = tmp_path / "target"
    declaration = {"identity": "tests.target_only.v1"}
    _write_topology_latest(
        source,
        transaction_id="tx-source",
        digests={"model": "source-content"},
        blob_digests={"model": "source-blob"},
    )
    _write_topology_latest(
        target,
        transaction_id="tx-target",
        digests={"model": "transformed-content", "adaptive_state": "new-content"},
        blob_digests={"model": "target-model-blob", "adaptive_state": "target-new-blob"},
        provenance=[
            _target_transform_provenance(
                "model",
                source_sha256="source-blob",
                target_sha256="target-model-blob",
            ),
            _target_transform_provenance(
                "adaptive_state",
                source_sha256=None,
                target_sha256="target-new-blob",
                target_only_declaration=declaration,
            ),
        ],
    )
    return spec, one_row, source, target, declaration


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


def test_fork_path_leaves_typed_schedule_payload_byte_identical(tmp_path: Path) -> None:
    payload = _training_run_payload()
    payload["method_payload"]["payload"]["optimizer"]["lr_schedule"] = {
        "kind": "warmup_cosine",
        "learning_rate_0": 0.01,
        "total_steps": 1_000,
        "constant_lr_iterations": 500,
        "origin": {"kind": "segment_start"},
    }
    spec = _matrix(payload)
    materialized = materialize_run_matrix(spec, repo_root=tmp_path)
    row = materialized.rows[0]
    before = canonical_json_bytes(row.payload["method_payload"]["payload"]["optimizer"]["lr_schedule"])
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_latest(source, transaction_id="tx-source", digest="same")
    _write_latest(target, transaction_id="tx-target", digest="same")

    fork_matrix_checkpoints(
        spec,
        MaterializedRunMatrix(
            matrix_spec_sha256=materialized.matrix_spec_sha256,
            run_set_id=materialized.run_set_id,
            base_payload=materialized.base_payload,
            rows=[row],
            run_set_manifest=materialized.run_set_manifest,
        ),
        source_checkpoint_root=source,
        target_checkpoint_roots={"lr_hi": target},
        parity_output_path=tmp_path / "parity.json",
        skip_fork=True,
    )

    after = canonical_json_bytes(row.payload["method_payload"]["payload"]["optimizer"]["lr_schedule"])
    assert after == before


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


def test_fork_matrix_parity_accepts_declared_topology_change(tmp_path: Path) -> None:
    spec, materialized, source, target, declaration = _topology_parity_inputs(tmp_path)

    table = fork_matrix_checkpoints(
        spec,
        materialized,
        source_checkpoint_root=source,
        target_checkpoint_roots={"lr_hi": target},
        parity_output_path=tmp_path / "parity.json",
        row_target_transform_metadata={
            "lr_hi": {
                "identity": "tests.target_topology.v1",
                "parameters": {"mode": "adaptive"},
            }
        },
        row_target_transformed_slots={"lr_hi": ["model"]},
        row_target_only_slots={"lr_hi": {"adaptive_state": declaration}},
        skip_fork=True,
    )

    assert table["ok"] is True
    assert {
        (row["kind"], row["slot"], row["ok"])
        for row in table["rows"]
        if row["kind"] != "lr_continuation"
    } == {
        ("slot_parity", "model", True),
        ("target_only_provenance", "adaptive_state", True),
    }


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_comparable", "missing comparable digest"),
        ("undeclared_extra", "expected_topology"),
        ("preserved_drift", "row=lr_hi slot=model"),
        ("missing_target_only_provenance", "missing fork provenance"),
        ("incorrect_target_only_provenance", "target-only declaration mismatch"),
    ],
)
def test_fork_matrix_parity_rejects_topology_contract_drift(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    spec, materialized, source, target, declaration = _topology_parity_inputs(tmp_path)
    target_manifest_path = next((target / "transactions").glob("*/manifest.json"))
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    transformed_slots: list[str] = ["model"]
    if mutation == "missing_comparable":
        target_manifest["slots"] = [
            slot for slot in target_manifest["slots"] if slot["slot"] != "model"
        ]
        target_manifest["content_integrity_digest"]["slots"] = [
            slot
            for slot in target_manifest["content_integrity_digest"]["slots"]
            if slot["slot"] != "model"
        ]
    elif mutation == "undeclared_extra":
        target_manifest["slots"].append({"slot": "rogue", "sha256": "rogue-blob"})
        target_manifest["content_integrity_digest"]["slots"].append(
            {"slot": "rogue", "slot_root_sha256": "rogue-content"}
        )
    elif mutation == "preserved_drift":
        transformed_slots = []
    elif mutation == "missing_target_only_provenance":
        target_manifest["fork_provenance"]["slots"] = [
            slot
            for slot in target_manifest["fork_provenance"]["slots"]
            if slot["slot"] != "adaptive_state"
        ]
    elif mutation == "incorrect_target_only_provenance":
        target_only = next(
            slot
            for slot in target_manifest["fork_provenance"]["slots"]
            if slot["slot"] == "adaptive_state"
        )
        target_only["transform"]["metadata"]["target_only_declaration"] = {"identity": "wrong"}
    target_manifest_path.write_text(json.dumps(target_manifest), encoding="utf-8")

    with pytest.raises(ForkParityError, match=error):
        fork_matrix_checkpoints(
            spec,
            materialized,
            source_checkpoint_root=source,
            target_checkpoint_roots={"lr_hi": target},
            parity_output_path=tmp_path / "parity.json",
            row_target_transform_metadata={
                "lr_hi": {
                    "identity": "tests.target_topology.v1",
                    "parameters": {"mode": "adaptive"},
                }
            },
            row_target_transformed_slots={"lr_hi": transformed_slots},
            row_target_only_slots={"lr_hi": {"adaptive_state": declaration}},
            skip_fork=True,
        )


def test_matrix_fork_maps_explicit_distinct_barrier_and_reloads_target(
    tmp_path: Path,
) -> None:
    """A continuation can cross barriers only through a declared mapping."""
    continuation = CheckpointContinuationRequest(
        source_completed_batches=12000,
        additional_batches=200,
    )
    source_spec = _run_spec(minimax=True).model_copy(
        update={
            "checkpoint_progress": _run_spec(minimax=True).checkpoint_progress.model_copy(
                update={"continuation": continuation}
            )
        }
    )
    target_program = source_spec.worker_execution.method_contract.phase_program.model_copy(
        deep=True
    )
    target_barrier = "after_adaptive_epsilon_train_chunk"
    target_program.checkpoint_barriers[0].name = target_barrier
    target_program.checkpoint_barriers[0].resume_coordinate.completed_barrier = target_barrier
    target_program.phases[0].checkpoint_barrier = target_barrier
    target_program.transitions[0].barrier = target_barrier
    target_method_contract = source_spec.worker_execution.method_contract.model_copy(
        update={"phase_program": target_program}
    )
    target_effective_phase = source_spec.worker_execution.effective_phase.model_copy(
        update={"phase_program": target_program}
    )
    target_spec = source_spec.model_copy(
        update={
            "worker_execution": source_spec.worker_execution.model_copy(
                update={
                    "method_contract": target_method_contract,
                    "effective_phase": target_effective_phase,
                }
            )
        }
    )
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "name": "distinct barrier continuation row",
            "base": {
                "kind": "inline",
                "inline": target_spec.model_dump(mode="json", exclude_none=True),
            },
            "fork": {
                "source_run_id": "feedbax-training-run:source",
                "lr_continuation": "continue",
                "parity": "skip",
            },
            "rows": [{"row_id": "continuation", "overrides": []}],
        }
    )
    materialized = materialize_run_matrix(matrix, repo_root=tmp_path)
    source_slots = _minimax_slots()
    source_slots["controller"] = BatchHistory(
        jnp.arange(5 * 12000, dtype=jnp.float32).reshape(5, 12000)
    )
    write_checkpoint_transaction(
        tmp_path / "source",
        run_spec=source_spec,
        phase_program=source_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=12000),
        slots=source_slots,
        completed_training_batches=12000,
    )
    target_slots = _minimax_slots()
    target_slots["controller"] = BatchHistory(
        jnp.full((5, 200), -1.0, dtype=jnp.float32)
    )
    target_coordinate = ProgressCoordinate(
        run_id="run-1",
        phase="warmup",
        program_step=12000,
        completed_barrier=target_barrier,
    )
    barrier_mapping = CheckpointForkBarrierMapping(
        source_barrier="after_warmup",
        target_barrier=target_barrier,
        target_coordinate=target_coordinate,
        coordinate_mapping={
            "identity": "tests.after_warmup_to_adaptive_epsilon_chunk.v1",
            "parameters": {"program_step": "preserve"},
        },
    )

    fork_matrix_checkpoints(
        matrix,
        materialized,
        source_checkpoint_root=tmp_path / "source",
        target_checkpoint_roots={"continuation": tmp_path / "target"},
        target_slot_templates={"continuation": target_slots},
        row_barrier_mappings={"continuation": barrier_mapping},
        parity_output_path=tmp_path / "parity.json",
    )

    resumed = load_latest_checkpoint(
        tmp_path / "target",
        expected_run_spec=materialized.rows[0].spec,
        expected_phase_program=(
            materialized.rows[0].spec.worker_execution.method_contract.phase_program
        ),
        expected_slots=target_slots,
    )
    assert resumed.manifest.barrier == target_barrier
    assert resumed.manifest.completed_coordinate == target_coordinate
    assert resumed.manifest.fork_provenance is not None
    assert resumed.manifest.fork_provenance.barrier_mapping == barrier_mapping
    assert resumed.manifest.segment_lineage.start_batch == 12000
    assert resumed.manifest.segment_lineage.segment_batch_count == 200
    assert jnp.all(resumed.slots["controller"].value == -1.0)


def test_fork_cli_materializes_targets_and_writes_parity_table(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(_matrix(_training_run_payload()).model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
    )
    source = tmp_path / "source"
    high = tmp_path / "high"
    low = tmp_path / "low"
    _write_latest(source, transaction_id="tx-source", digest="same")
    _write_latest(high, transaction_id="tx-high", digest="same")
    _write_latest(low, transaction_id="tx-low", digest="same")
    parity_path = tmp_path / "parity.json"

    exit_code = main(
        [
            "fork",
            str(matrix_path),
            "--repo-root",
            str(tmp_path),
            "--source-checkpoint-root",
            str(source),
            "--target",
            f"lr_hi={high}",
            "--target",
            f"lr_lo={low}",
            "--parity-output",
            str(parity_path),
            "--skip-fork",
        ]
    )

    assert exit_code == 0
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    assert parity["schema_version"] == "feedbax.run_matrix_fork_parity.v1"
    assert parity["ok"] is True
    assert {row["row_id"] for row in parity["rows"]} == {"lr_hi", "lr_lo"}


def test_fork_cli_rejects_malformed_target(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(_matrix(_training_run_payload()).model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "fork",
                str(matrix_path),
                "--source-checkpoint-root",
                str(tmp_path / "source"),
                "--target",
                "missing-separator",
                "--parity-output",
                str(tmp_path / "parity.json"),
                "--skip-fork",
            ]
        )
