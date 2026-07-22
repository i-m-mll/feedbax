from __future__ import annotations

import pytest
from pydantic import ValidationError

from feedbax.contracts.remote_smoke import (
    RemoteSmokeEvidence,
    RemoteSmokeRowEvidence,
)
from feedbax.contracts.run_matrix import TrainingRowProvenance
from feedbax.orchestration.bundle import ExecutionIdentityEnvelope
from feedbax.training.diagnostics import NativeExecutionProducerContext


def _derived_context(run_id: str) -> NativeExecutionProducerContext:
    provenance = TrainingRowProvenance(
        row_id="row-a",
        row_index=0,
        planned_run_id=run_id,
        authored_payload_hash="a" * 64,
        lowered_execution_payload_hash="b" * 64,
        axis_coordinates={},
    )
    execution = ExecutionIdentityEnvelope.model_construct(row_provenance=provenance)
    return NativeExecutionProducerContext.model_construct(
        execution=execution,
        environment_fingerprint="environment",
    )


def _passed_row(**updates: object) -> RemoteSmokeRowEvidence:
    values = {
        "row_id": "row-a",
        "status": "passed",
        "planned_run_id": "planned",
        "derived_run_id": "planned--smoke",
        "derived_producer_context": _derived_context("planned--smoke"),
        "scratch_namespace": "/remote/run/smoke/row-a",
        "start_completed_batches": 4,
        "end_completed_batches": 6,
        "update_budget": 2,
        "payload_binding_status": "verified",
        "executor_result_sha256": "c" * 64,
        "protected_paths_before": {"inputs": "d" * 64},
        "protected_paths_after": {"inputs": "d" * 64},
        "cleanup_status": "removed",
        "deadline_seconds": 1800,
    }
    values.update(updates)
    return RemoteSmokeRowEvidence.model_validate(values)


def test_remote_smoke_evidence_records_typed_passed_row() -> None:
    evidence = RemoteSmokeEvidence(
        run_set_id="run-set",
        bundle_sha256="e" * 64,
        rows=(_passed_row(),),
    )

    assert evidence.rows[0].end_completed_batches == 6
    assert evidence.rows[0].derived_run_id != evidence.rows[0].planned_run_id
    assert (
        evidence.rows[0].derived_producer_context.execution.row_provenance.planned_run_id
        == evidence.rows[0].derived_run_id
    )


@pytest.mark.parametrize("budget", [True, False, 0, -1])
def test_remote_smoke_evidence_rejects_invalid_budget(budget: object) -> None:
    with pytest.raises(ValidationError, match="update_budget"):
        _passed_row(update_budget=budget)


def test_remote_smoke_evidence_rejects_identity_or_protected_path_drift() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        _passed_row(derived_run_id="planned")
    with pytest.raises(ValidationError, match="must match derived_run_id"):
        _passed_row(derived_run_id="other--smoke")
    with pytest.raises(ValidationError, match="protected path contents changed"):
        _passed_row(protected_paths_after={"inputs": "f" * 64})


def test_remote_smoke_opt_out_is_durable_and_cannot_claim_execution() -> None:
    row = RemoteSmokeRowEvidence(
        row_id="row-a",
        status="opted-out",
        update_budget=2,
        payload_binding_status="not-run",
        cleanup_status="not-created",
        deadline_seconds=1800,
        opt_out_reason="operator disabled remote smoke",
    )
    assert row.opt_out_reason == "operator disabled remote smoke"

    with pytest.raises(ValidationError, match="cannot claim runtime results"):
        RemoteSmokeRowEvidence.model_validate(
            {**row.model_dump(mode="json"), "planned_run_id": "planned"}
        )


def test_remote_smoke_evidence_rejects_duplicate_rows() -> None:
    row = _passed_row()
    with pytest.raises(ValidationError, match="duplicate row_id"):
        RemoteSmokeEvidence(
            run_set_id="run-set",
            bundle_sha256="e" * 64,
            rows=(row, row),
        )
