from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from feedbax.contracts.checkpoints import (
    BatchIndexedCheckpointLeafSpec,
    CheckpointContinuationRequest,
)
from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.worker import CheckpointSlotSpec
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


def test_matrix_fork_forwards_declared_continuation_to_custody_extension(
    tmp_path: Path,
) -> None:
    continuation = CheckpointContinuationRequest(
        source_completed_batches=12000,
        additional_batches=200,
        batch_indexed_leaves=[
            BatchIndexedCheckpointLeafSpec(slot="controller", tree_path="/")
        ],
    )
    source_spec = _run_spec(minimax=True).model_copy(deep=True)
    source_spec = source_spec.model_copy(
        update={
            "checkpoint_progress": source_spec.checkpoint_progress.model_copy(
                update={"continuation": continuation}
            )
        }
    )
    target_program = source_spec.worker_execution.method_contract.phase_program.model_copy(
        deep=True
    )
    target_program.checkpoint_barriers[0].slots.append(
        CheckpointSlotSpec(slot="target_diagnostics")
    )
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
            "name": "continuation row",
            "base": {"inline": target_spec.model_dump(mode="json", exclude_none=True)},
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
    source_slots["controller"] = jnp.arange(5 * 12000, dtype=jnp.float32).reshape(
        5, 12000
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
    raw_target_slots = _minimax_slots()
    raw_target_slots["controller"] = jnp.full((5, 12200), -1.0, dtype=jnp.float32)
    target_slots = _minimax_slots()
    target_slots["controller"] = {
        "history": jnp.full((5, 12200), -1.0, dtype=jnp.float32),
        "target_topology": jnp.array([7], dtype=jnp.int32),
    }
    target_slots["target_diagnostics"] = jnp.zeros((2,), dtype=jnp.float32)

    def make_target_topology(slots):
        transformed = dict(slots)
        transformed["controller"] = {
            "history": transformed["controller"],
            "target_topology": jnp.array([7], dtype=jnp.int32),
        }
        transformed["target_diagnostics"] = jnp.zeros((2,), dtype=jnp.float32)
        return transformed

    table = fork_matrix_checkpoints(
        matrix,
        materialized,
        source_checkpoint_root=tmp_path / "source",
        target_checkpoint_roots={"continuation": tmp_path / "target"},
        target_slot_templates={"continuation": target_slots},
        row_continuation_slot_templates={"continuation": raw_target_slots},
        row_target_slot_transforms={"continuation": make_target_topology},
        row_target_transform_metadata={
            "continuation": {
                "identity": "tests.make_target_topology.v1",
                "parameters": {"target_topology": 7},
            }
        },
        row_target_transformed_slots={"continuation": ["controller"]},
        row_target_only_slots={
            "continuation": {"target_diagnostics": {"role": "diagnostic"}}
        },
        parity_output_path=tmp_path / "parity.json",
    )

    assert table["ok"] is False
    resumed = load_latest_checkpoint(
        tmp_path / "target",
        expected_run_spec=materialized.rows[0].spec,
        expected_phase_program=(
            materialized.rows[0].spec.worker_execution.method_contract.phase_program
        ),
        expected_slots=target_slots,
        continuation_request=continuation,
    )
    controller = resumed.slots["controller"]
    assert controller["history"].shape == (5, 12200)
    assert jnp.array_equal(
        controller["history"][..., :12000], source_slots["controller"]
    )
    assert jnp.all(controller["history"][..., 12000:] == -1.0)
    assert jnp.array_equal(controller["target_topology"], jnp.array([7], dtype=jnp.int32))
    assert jnp.array_equal(resumed.slots["target_diagnostics"], jnp.zeros((2,)))
    provenance = {
        slot.slot: slot for slot in resumed.manifest.fork_provenance.slots
    }
    assert provenance["controller"].transform is not None
    assert [
        stage["stage"] for stage in provenance["controller"].transform.metadata["stages"]
    ] == ["continuation_extension", "target_post"]
    assert provenance["target_diagnostics"].source_sha256 is None
    assert provenance["target_diagnostics"].transform is not None
    assert provenance["target_diagnostics"].transform.metadata[
        "target_only_declaration"
    ] == {"role": "diagnostic"}


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
