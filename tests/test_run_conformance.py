from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from feedbax.orchestration import AuthorizedBatchStop, RowConformanceRuntimeInputs
from feedbax.contracts.manifest import TrainingRunManifest, spec_payload
from feedbax.contracts.migrations import default_spec_registry, migrate_structured_spec_payload
from feedbax.contracts.training import LrScheduleSpec, OptimizerSpec
from feedbax.contracts.resolved_snapshot_decoder import SNAPSHOT_SCHEMA_ID, SNAPSHOT_SCHEMA_VERSION
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.spec_storage import (
    TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_ID,
    TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
    TrainingRunExecutionCapsule,
    build_resolved_semantics_snapshot,
    training_run_execution_hash,
    training_run_intent_hash,
    training_spec_canonical_bytes,
)
from feedbax.orchestration.bundle import (
    AuthoredIntentRef,
    ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    ImmutableInputIdentity,
    ResolvedSnapshotRef,
    SchemaArtifactRef,
)
from feedbax.orchestration.conformance import (
    REALIZED_DEPLOYMENT_RECORD_SCHEMA_ID,
    REALIZED_DEPLOYMENT_RECORD_SCHEMA_VERSION,
    RUN_CONFORMANCE_SCHEMA_ID,
    RUN_CONFORMANCE_SCHEMA_VERSION,
    RUN_CONFORMANCE_SCHEMA_VERSION_V1,
    CheckEntry,
    CheckRegistry,
    ConformanceRowArtifacts,
    RunConformanceCertificate,
    RealizedDeploymentRecord,
    assert_certificate_allows_completed_registration,
    build_core_check_registry,
    check_checkpoint_cadence,
    check_completed_batches,
    check_events_terminal,
    check_execution_identity,
    check_lr_trace,
    check_manifest_valid,
    check_realized_deployment,
    missing_input_check,
    pass_check,
    run_conformance_checks,
    write_conformance_certificate,
)
from feedbax.orchestration.state import RowState
from feedbax.plugins.discovery import load_conformance_check_plugins


pytestmark = pytest.mark.feedbax_contract


GENERATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _row(**updates: object) -> ConformanceRowArtifacts:
    base = {
        "row_id": "row-a",
        "bundle_row_spec": {
            "n_batches": 10,
            "checkpoint_interval": 5,
            "seeds": {"controller": 123},
        },
        "training_diagnostics": {
            "completed_batches": 10,
            "checkpoint_coordinates": [5, 10],
            "seeds": {"controller": 123},
        },
        "manifest_payload": {
            "kind": "TrainingRunManifest",
            "metadata": {
                "environment_fingerprint": {"python": "3.12", "jax": "0.test"},
                "seeds": {"controller": 123},
            },
            "training_spec": {"inline": {"method_payload": {"payload": {}}}},
            "summary_metrics": {"completed_batches": 10},
        },
        "recorded_environment_fingerprint": {"python": "3.12", "jax": "0.test"},
        "preflight_normalized_payload": {"method_payload": {"payload": {}}},
        "deployment_policy": {
            "driver": "local",
            "venue": "local",
            "resources": {"gpu_id": None, "regions": []},
        },
        "realized_deployment_evidence": {
            "driver": "local",
            "venue": "local",
            "provider": "local",
            "gpu_model": None,
            "gpu_count": None,
            "region": None,
            "immutable_image_id": None,
            "environment_fingerprint": '{"runtime":"local"}',
            "provisioned_at": GENERATED_AT.isoformat(),
            "billing_started_at": None,
            "row_started_at": GENERATED_AT.isoformat(),
            "row_completed_at": GENERATED_AT.isoformat(),
            "observed_at": GENERATED_AT.isoformat(),
            "wall_time_seconds": 0.0,
            "hourly_rate": 0.0,
            "accrued_cost": 0.0,
            "currency": "USD",
            "cost_basis": "local-not-billable",
            "observation_basis": {"environment": "fixture", "timing": "fixture", "cost": "fixture"},
            "unavailable": {
                "gpu_model": "not applicable locally",
                "gpu_count": "not applicable locally",
                "region": "not applicable locally",
                "immutable_image_id": "not applicable locally",
                "billing_started_at": "not billable locally",
            },
        },
    }
    base.update(updates)
    return ConformanceRowArtifacts(**base)


def _identity_fixture(
    tmp_path: Path,
    *,
    authored: dict[str, object] | None = None,
    inputs: list[dict[str, object]] | None = None,
):
    if authored is None:
        authored = TrainingRunMatrixSpec(
            name="fixture",
            base={"kind": "inline", "inline": {"training": "fixture"}},
            rows=[{"row_id": "row-a"}],
        ).model_dump(mode="json", exclude_none=True)
    snapshot = build_resolved_semantics_snapshot({"training": "resolved"})
    canonical_inputs = list(inputs or [])
    intent_hash = training_run_intent_hash(authored)
    execution_hash = training_run_execution_hash(snapshot["root_hash"], canonical_inputs)
    capsule = TrainingRunExecutionCapsule(
        materializer_commit="fixture",
        relevant_schema_versions={"fixture": "v1"},
        dependency_lock_digest="1" * 64,
        input_data_identities=canonical_inputs,
        intent_hash=intent_hash,
        resolved_root_hash=snapshot["root_hash"],
        execution_hash=execution_hash,
    ).model_dump(mode="json", exclude_none=True)

    def store(name: str, payload: dict[str, object]) -> tuple[Path, str]:
        data = training_spec_canonical_bytes(payload)
        path = tmp_path / f"{name}.json"
        path.write_bytes(data)
        return path, hashlib.sha256(data).hexdigest()

    authored_path, authored_sha = store("authored", authored)
    payload_path, payload_sha = store("payload", authored)
    snapshot_path, snapshot_sha = store("snapshot", snapshot)
    capsule_path, capsule_sha = store("capsule", capsule)
    envelope = ExecutionIdentityEnvelope(
        payload=SchemaArtifactRef(
            schema_id=TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            schema_version=TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            artifact_id="artifact://fixture/payload",
            sha256=payload_sha,
            uri=str(payload_path),
        ),
        authored_intent=AuthoredIntentRef(
            schema_id=TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            schema_version=TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            artifact_id="artifact://fixture/authored",
            sha256=authored_sha,
            uri=str(authored_path),
            intent_hash=intent_hash,
        ),
        resolved_snapshot=ResolvedSnapshotRef(
            schema_id=SNAPSHOT_SCHEMA_ID,
            schema_version=SNAPSHOT_SCHEMA_VERSION,
            artifact_id="artifact://fixture/snapshot",
            sha256=snapshot_sha,
            uri=str(snapshot_path),
            root_hash=snapshot["root_hash"],
        ),
        execution_capsule=ExecutionCapsuleRef(
            schema_id=TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_ID,
            schema_version=TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
            artifact_id="artifact://fixture/capsule",
            sha256=capsule_sha,
            uri=str(capsule_path),
            execution_hash=execution_hash,
        ),
        immutable_inputs=[ImmutableInputIdentity.model_validate(item) for item in canonical_inputs],
    )
    manifest = TrainingRunManifest(
        id="feedbax-training-run:row-a",
        intent_hash=intent_hash,
        resolved_semantics_root_hash=snapshot["root_hash"],
        execution_hash=execution_hash,
        input_data_identities=canonical_inputs,
    ).model_dump(mode="json", exclude_none=True)
    return envelope, manifest


def test_certificate_assembly_rules_and_deterministic_write(tmp_path: Path) -> None:
    registry = CheckRegistry(
        {
            "z_pass": lambda row: pass_check("z_pass"),
            "a_missing": lambda row: missing_input_check("a_missing", "required"),
            "m_skip": lambda row: CheckEntry(
                check_id="m_skip",
                status="skipped",
                detail="not applicable for fixture",
            ),
        }
    )

    certificate = write_conformance_certificate(
        run_set_dir=tmp_path,
        run_set_id="run-set-a",
        rows=[_row(row_id="row-b"), _row(row_id="row-a")],
        registry=registry,
        generated_at=GENERATED_AT,
    )

    assert certificate.overall == "fail"
    assert list(certificate.rows) == ["row-a", "row-b"]
    assert [check.check_id for check in certificate.rows["row-a"].checks] == [
        "a_missing",
        "m_skip",
        "realized_deployment",
        "z_pass",
    ]
    skipped = next(
        check for check in certificate.rows["row-a"].checks if check.check_id == "m_skip"
    )
    assert skipped.status == "fail"
    assert "did not produce a verdict" in str(skipped.detail)
    written = (tmp_path / "conformance.json").read_text(encoding="utf-8")
    assert (
        written == json.dumps(certificate.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="skipped conformance checks require a detail"):
        CheckEntry(check_id="bad_skip", status="skipped")


def test_empty_registry_cannot_certify() -> None:
    with pytest.raises(ValueError, match="at least one registered"):
        run_conformance_checks(
            run_set_id="run-set-a",
            rows=[_row()],
            registry=CheckRegistry(),
            generated_at=GENERATED_AT,
        )


def test_realized_deployment_cannot_be_declared_inapplicable() -> None:
    with pytest.raises(ValueError, match="cannot be declared inapplicable"):
        run_conformance_checks(
            run_set_id="run-set-a",
            rows=[_row()],
            registry=build_core_check_registry(),
            declared_inapplicable={"realized_deployment": "requested bypass"},
            generated_at=GENERATED_AT,
        )


def test_schema_round_trip_and_registry_identity() -> None:
    payload = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[_row()],
        registry=CheckRegistry({"ok": lambda row: pass_check("ok")}),
        generated_at=GENERATED_AT,
    ).model_dump(mode="json")

    certificate = RunConformanceCertificate.model_validate(payload)
    assert certificate.model_dump(mode="json")["schema_version"] == RUN_CONFORMANCE_SCHEMA_VERSION

    family = default_spec_registry.resolve("RunConformanceCertificate")
    assert family.identity == RUN_CONFORMANCE_SCHEMA_ID
    assert family.current_version == RUN_CONFORMANCE_SCHEMA_VERSION
    migrated = migrate_structured_spec_payload("RunConformanceCertificate", payload)
    assert migrated.payload == payload

    v1 = dict(payload, schema_version=RUN_CONFORMANCE_SCHEMA_VERSION_V1)
    with pytest.raises(ValueError, match="no registered migration"):
        migrate_structured_spec_payload("RunConformanceCertificate", v1)

    record_payload = dict(payload["rows"]["row-a"]["realized_deployment"])
    record = RealizedDeploymentRecord.model_validate(record_payload)
    assert record.schema_id == REALIZED_DEPLOYMENT_RECORD_SCHEMA_ID
    assert record.schema_version == REALIZED_DEPLOYMENT_RECORD_SCHEMA_VERSION
    family = default_spec_registry.resolve("RealizedDeploymentRecord")
    assert family.identity == REALIZED_DEPLOYMENT_RECORD_SCHEMA_ID
    assert family.current_version == REALIZED_DEPLOYMENT_RECORD_SCHEMA_VERSION
    migrated = migrate_structured_spec_payload("RealizedDeploymentRecord", record_payload)
    assert migrated.payload == record_payload
    with pytest.raises(ValueError, match="no registered migration"):
        migrate_structured_spec_payload(
            "RealizedDeploymentRecord",
            dict(record_payload, schema_version=f"{REALIZED_DEPLOYMENT_RECORD_SCHEMA_ID}.v0"),
        )


def test_certificate_schema_rejects_missing_realized_deployment_proof() -> None:
    with pytest.raises(ValueError, match="exactly one realized_deployment check"):
        RunConformanceCertificate.model_validate(
            {
                "schema_id": RUN_CONFORMANCE_SCHEMA_ID,
                "schema_version": RUN_CONFORMANCE_SCHEMA_VERSION,
                "run_set_id": "run-set-a",
                "generated_at": GENERATED_AT.isoformat(),
                "overall": "pass",
                "rows": {"row-a": {"checks": [{"check_id": "ok", "status": "pass"}]}},
            }
        )

    with pytest.raises(ValueError, match="at least 1 item"):
        RunConformanceCertificate.model_validate(
            {
                "schema_id": RUN_CONFORMANCE_SCHEMA_ID,
                "schema_version": RUN_CONFORMANCE_SCHEMA_VERSION,
                "run_set_id": "run-set-a",
                "generated_at": GENERATED_AT.isoformat(),
                "overall": "pass",
                "rows": {},
            }
        )


@pytest.mark.parametrize("tamper_surface", ["typed", "check"])
def test_certificate_schema_binds_raw_typed_and_check_realized_proof(
    tamper_surface: str,
) -> None:
    payload = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[_row()],
        registry=CheckRegistry({"ok": lambda row: pass_check("ok")}),
        generated_at=GENERATED_AT,
    ).model_dump(mode="json")
    row = payload["rows"]["row-a"]
    if tamper_surface == "typed":
        row["realized_deployment"]["cost_basis"] = "tampered"
        message = "does not match its raw evidence"
    else:
        realized_check = next(
            check for check in row["checks"] if check["check_id"] == "realized_deployment"
        )
        realized_check["observed"]["cost_basis"] = "tampered"
        message = "does not bind the typed record"

    with pytest.raises(ValueError, match=message):
        RunConformanceCertificate.model_validate(payload)


def test_realized_deployment_remote_fails_closed_and_preserves_raw_evidence() -> None:
    raw = dict(_row().realized_deployment_evidence or {})
    fingerprint = json.dumps(
        {
            "image_id": "runpod/image@sha256:" + "a" * 64,
            "runtime": {"device_kind": "RTX 5090", "device_count": 1},
        },
        sort_keys=True,
    )
    raw.update(
        {
            "driver": "runpod",
            "venue": "remote",
            "provider": "runpod",
            "gpu_model": "RTX 5090",
            "gpu_count": 1,
            "region": None,
            "immutable_image_id": "runpod/image@sha256:" + "a" * 64,
            "environment_fingerprint": fingerprint,
            "billing_started_at": GENERATED_AT.isoformat(),
            "hourly_rate": 1.0,
            "accrued_cost": 0.0,
            "cost_basis": "billing-start-to-certify-observation",
        }
    )
    raw["unavailable"] = {"region": "pod response lacked region"}
    row = _row(
        deployment_policy={
            "driver": "runpod",
            "venue": "remote",
            "resources": {"gpu_id": "RTX 5090", "regions": ["CA-MTL-1"]},
        },
        realized_deployment_evidence=raw,
    )

    check = check_realized_deployment(row)
    certificate = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[row],
        registry=CheckRegistry({"realized_deployment": check_realized_deployment}),
        generated_at=GENERATED_AT,
    )

    assert check.status == "fail"
    assert "region" in str(check.detail)
    assert certificate.rows["row-a"].realized_deployment is None
    assert certificate.rows["row-a"].realized_deployment_evidence == raw
    assert RealizedDeploymentRecord.model_validate(raw).region is None


def _valid_remote_realized_evidence() -> dict[str, object]:
    image_id = "runpod/image@sha256:" + "c" * 64
    provisioned = GENERATED_AT - timedelta(hours=3)
    started = GENERATED_AT - timedelta(hours=2)
    completed = GENERATED_AT - timedelta(hours=1)
    return {
        "schema_id": REALIZED_DEPLOYMENT_RECORD_SCHEMA_ID,
        "schema_version": REALIZED_DEPLOYMENT_RECORD_SCHEMA_VERSION,
        "driver": "runpod",
        "venue": "remote",
        "provider": "runpod",
        "gpu_model": "RTX 5090",
        "gpu_count": 1,
        "region": "CA-MTL-1",
        "immutable_image_id": image_id,
        "environment_fingerprint": json.dumps(
            {
                "image_id": image_id,
                "runtime": {"device_kind": "RTX 5090", "device_count": 1},
            },
            sort_keys=True,
        ),
        "provisioned_at": provisioned.isoformat(),
        "billing_started_at": provisioned.isoformat(),
        "row_started_at": started.isoformat(),
        "row_completed_at": completed.isoformat(),
        "observed_at": GENERATED_AT.isoformat(),
        "wall_time_seconds": 3600.0,
        "hourly_rate": 2.0,
        "accrued_cost": 6.0,
        "currency": "USD",
        "cost_basis": "billing-start-to-certify-observation",
        "observation_basis": {"provider": "fixture", "timing": "fixture", "cost": "fixture"},
        "provider_observations": {
            "hourly_rate_raw": "2.0",
            "immutable_image_id_raw": image_id,
        },
        "unavailable": {},
    }


@pytest.mark.parametrize(
    "image_id",
    [
        "runpod/image@sha256:" + "a" * 63,
        "runpod/image@sha256:" + "A" * 64,
        "runpod/image@sha256:" + "g" * 64,
        "@sha256:" + "a" * 64,
    ],
)
def test_realized_deployment_rejects_incomplete_immutable_image_identity(
    image_id: str,
) -> None:
    raw = _valid_remote_realized_evidence()
    raw["immutable_image_id"] = image_id

    with pytest.raises(ValueError, match="complete lowercase OCI digest"):
        RealizedDeploymentRecord.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("billing_started_at", "2026-01-02T00:04:05", "timezone-aware"),
        ("row_completed_at", (GENERATED_AT + timedelta(seconds=1)).isoformat(), "chronology"),
        ("wall_time_seconds", 1.0, "wall_time_seconds"),
        ("accrued_cost", 1.0, "accrued_cost"),
        ("hourly_rate", float("inf"), "finite number"),
    ],
)
def test_realized_deployment_recomputes_timing_and_cost_and_rejects_non_finite(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = _valid_remote_realized_evidence()
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        RealizedDeploymentRecord.model_validate(raw)


def test_certificate_generated_at_must_equal_realized_observation() -> None:
    raw = _valid_remote_realized_evidence()
    row = _row(
        deployment_policy={
            "driver": "runpod",
            "venue": "remote",
            "resources": {"gpu_id": "RTX 5090", "regions": ["CA-MTL-1"]},
        },
        realized_deployment_evidence=raw,
    )

    with pytest.raises(ValueError, match="observed_at must equal generated_at"):
        run_conformance_checks(
            run_set_id="run-set-a",
            rows=[row],
            registry=CheckRegistry({"ok": lambda item: pass_check("ok")}),
            generated_at=GENERATED_AT + timedelta(seconds=1),
        )


def test_core_checks_pass_fail_and_missing_inputs() -> None:
    assert check_completed_batches(_row()).status == "pass"
    assert (
        check_completed_batches(_row(training_diagnostics={"completed_batches": 9})).status
        == "fail"
    )
    missing = check_completed_batches(_row(bundle_row_spec={}))
    assert missing.status == "fail"
    assert "missing required input" in str(missing.detail)

    registry = build_core_check_registry()
    certificate = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[_row(event_log=None)],
        registry=registry,
        generated_at=GENERATED_AT,
    )

    checks = {check.check_id: check for check in certificate.rows["row-a"].checks}
    assert checks["completed_batches"].status == "pass"
    assert checks["checkpoint_cadence"].status == "pass"
    assert checks["environment_fingerprint"].status == "pass"
    assert checks["seeds"].status == "pass"
    assert checks["events_terminal"].status == "fail"
    assert "did not produce a verdict" in str(checks["events_terminal"].detail)


def _authorized_stopped_row(**updates: object) -> ConformanceRowArtifacts:
    base = {
        "bundle_row_spec": {"n_batches": 100, "checkpoint_interval": 50},
        "training_diagnostics": {
            "manifest_id": "feedbax-training-run:stopped",
            "terminal_status": "cancelled",
            "completed_batches": 50,
            "checkpoint_coordinates": [50],
            "checkpoint_transactions": [
                {
                    "transaction_id": "tx-stopped",
                    "completed_batches": 50,
                    "cumulative_completed_batches": 50,
                }
            ],
        },
        "manifest_payload": {
            "kind": "TrainingRunManifest",
            "id": "feedbax-training-run:stopped",
            "status": "cancelled",
            "completed_batches": 50,
        },
        "row_state": RowState(
            status="stopped",
            completed_at=GENERATED_AT,
            error="operator-stop-after-checkpoint",
        ),
        "runtime_inputs": RowConformanceRuntimeInputs(
            authorized_batch_stop=AuthorizedBatchStop(stop_after_batches=50)
        ),
    }
    base.update(updates)
    return _row(**base)


def test_completed_batches_accepts_only_a_fully_attested_authorized_stop() -> None:
    result = check_completed_batches(_authorized_stopped_row())

    assert result.status == "pass"
    assert result.expected == {
        "authored_batches": 100,
        "authorized_stop_after_batches": 50,
        "terminal_status": "cancelled",
        "row_status": "stopped",
        "row_error": "operator-stop-after-checkpoint",
        "final_checkpoint_batches": 50,
    }
    assert result.observed["diagnostics_completed_batches"] == 50
    assert result.observed["checkpoint_coordinates"] == [50]
    assert result.observed["checkpoint_completed_batches"] == [50]


@pytest.mark.parametrize(
    ("updates", "detail"),
    [
        (
            {
                "runtime_inputs": RowConformanceRuntimeInputs(
                    authorized_batch_stop=AuthorizedBatchStop(stop_after_batches=40)
                )
            },
            "diagnostics completed batches do not match the authorized stop",
        ),
        (
            {
                "manifest_payload": {
                    "kind": "TrainingRunManifest",
                    "id": "feedbax-training-run:stopped",
                    "status": "completed",
                    "completed_batches": 50,
                }
            },
            "training manifest does not report cancelled status",
        ),
        (
            {
                "manifest_payload": {
                    "kind": "TrainingRunManifest",
                    "id": "feedbax-training-run:stopped",
                    "status": "cancelled",
                    "completed_batches": 49,
                }
            },
            "training manifest completed batches do not match the authorized stop",
        ),
        (
            {
                "manifest_payload": {
                    "kind": "TrainingRunManifest",
                    "id": "feedbax-training-run:stopped",
                    "status": "cancelled",
                }
            },
            "training manifest completed batch count is missing",
        ),
        (
            {"row_state": RowState(status="completed", completed_at=GENERATED_AT)},
            "orchestration row state is not stopped",
        ),
        (
            {
                "training_diagnostics": {
                    "manifest_id": "feedbax-training-run:stopped",
                    "terminal_status": "cancelled",
                    "completed_batches": 50,
                    "checkpoint_coordinates": [40],
                    "checkpoint_transactions": [
                        {
                            "transaction_id": "tx-short",
                            "completed_batches": 40,
                            "cumulative_completed_batches": 40,
                        }
                    ],
                }
            },
            "final checkpoint coordinate does not match the authorized stop",
        ),
    ],
)
def test_completed_batches_rejects_inconsistent_authorized_stops(
    updates: dict[str, object],
    detail: str,
) -> None:
    result = check_completed_batches(_authorized_stopped_row(**updates))

    assert result.status == "fail"
    assert detail in str(result.detail)


def test_completed_batches_does_not_relax_unplanned_or_full_budget_rows() -> None:
    unplanned = check_completed_batches(
        _authorized_stopped_row(runtime_inputs=RowConformanceRuntimeInputs())
    )
    completed = check_completed_batches(
        _authorized_stopped_row(
            training_diagnostics={"completed_batches": 100},
            manifest_payload={"status": "completed"},
            row_state=RowState(status="completed", completed_at=GENERATED_AT),
        )
    )

    assert unplanned.status == "fail"
    assert unplanned.expected == 100
    assert unplanned.observed == 50
    assert completed.status == "pass"
    assert completed.expected == 100
    assert completed.observed == 100


def test_checkpoint_cadence_uses_segment_length_for_continuation() -> None:
    row = _row(
        bundle_row_spec={"expected_batches": 200, "checkpoint_interval": 100},
        training_diagnostics={
            "completed_batches": 12_200,
            "segment_completed_batches": 200,
            "checkpoint_coordinates": [100, 200],
        },
    )

    cadence = check_checkpoint_cadence(row)

    assert check_completed_batches(row).status == "pass"
    assert cadence.status == "pass"
    assert cadence.expected == {
        "coordinate_interval": 100,
        "coordinates": [100, 200],
    }
    assert cadence.observed == {"coordinates": [100, 200], "realized_batches": 200}


def test_checkpoint_cadence_without_segment_length_is_unchanged() -> None:
    passing = check_checkpoint_cadence(_row())
    failing = check_checkpoint_cadence(
        _row(training_diagnostics={"completed_batches": 10, "checkpoint_coordinates": [5]})
    )

    assert passing.status == "pass"
    assert passing.expected == {"coordinate_interval": 5, "coordinates": [5, 10]}
    assert passing.observed == {"coordinates": [5, 10], "realized_batches": 10}
    assert failing.status == "fail"
    assert failing.expected == {"coordinate_interval": 5, "coordinates": [5, 10]}
    assert failing.observed == {"coordinates": [5], "realized_batches": 10}
    assert failing.detail == "cadence length read from training_diagnostics.completed_batches"


def test_checkpoint_cadence_rejects_wrong_segment_coordinates() -> None:
    result = check_checkpoint_cadence(
        _row(
            bundle_row_spec={"expected_batches": 12_200, "checkpoint_interval": 100},
            training_diagnostics={
                "completed_batches": 12_200,
                "segment_completed_batches": 200,
                "checkpoint_coordinates": [100],
            },
        )
    )

    assert result.status == "fail"
    assert result.expected == {"coordinate_interval": 100, "coordinates": [100, 200]}
    assert result.observed == {"coordinates": [100], "realized_batches": 200}
    assert result.detail == (
        "cadence length read from training_diagnostics.segment_completed_batches"
    )


def test_execution_identity_explicit_empty_inputs_passes(tmp_path: Path) -> None:
    envelope, manifest = _identity_fixture(tmp_path)

    result = check_execution_identity(_row(execution=envelope, manifest_payload=manifest))

    assert result.status == "pass"
    assert result.expected == result.observed
    assert result.expected["input_data_identities"] == []
    assert "execution_identity" in dict(build_core_check_registry().items())


def test_execution_identity_hashes_authenticated_matrix_without_normalizing(
    tmp_path: Path,
) -> None:
    authored = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "fixture",
        "base": {"kind": "inline", "inline": {"training": "fixture"}},
        "rows": [{"row_id": "row-a"}],
    }
    normalized = TrainingRunMatrixSpec.model_validate(authored).model_dump(
        mode="json", exclude_none=True
    )
    assert training_run_intent_hash(authored) != training_run_intent_hash(normalized)
    envelope, manifest = _identity_fixture(tmp_path, authored=authored)

    result = check_execution_identity(_row(execution=envelope, manifest_payload=manifest))

    assert result.status == "pass"
    assert result.expected == result.observed


def test_execution_identity_requires_envelope_and_raw_manifest_fields(tmp_path: Path) -> None:
    assert check_execution_identity(_row()).status == "fail"
    envelope, manifest = _identity_fixture(tmp_path)
    manifest.pop("intent_hash")

    result = check_execution_identity(_row(execution=envelope, manifest_payload=manifest))

    assert result.status == "fail"
    assert "manifest.intent_hash" in str(result.detail)


def test_execution_identity_reports_authored_intent_mismatch(tmp_path: Path) -> None:
    envelope, manifest = _identity_fixture(tmp_path)
    manifest["intent_hash"] = "2" * 64

    result = check_execution_identity(_row(execution=envelope, manifest_payload=manifest))

    assert result.status == "fail"
    assert "intent_hash" in str(result.detail)


@pytest.mark.parametrize("drift", ["root", "inputs"])
def test_execution_identity_reports_semantic_or_input_drift(tmp_path: Path, drift: str) -> None:
    input_identity = {
        "role": "dataset",
        "kind": "artifact",
        "identifier": "dataset-v1",
        "digest": {"algorithm": "sha256", "value": "3" * 64},
    }
    envelope, manifest = _identity_fixture(tmp_path, inputs=[input_identity])
    if drift == "root":
        manifest["resolved_semantics_root_hash"] = "4" * 64
    else:
        changed = dict(input_identity)
        changed["identifier"] = "dataset-v2"
        manifest["input_data_identities"] = [changed]
    manifest["execution_hash"] = training_run_execution_hash(
        manifest["resolved_semantics_root_hash"], manifest["input_data_identities"]
    )

    result = check_execution_identity(_row(execution=envelope, manifest_payload=manifest))

    assert result.status == "fail"
    expected_field = "resolved_semantics_root_hash" if drift == "root" else "input_data_identities"
    assert expected_field in str(result.detail)


def test_execution_identity_rejects_execution_hash_inconsistency(tmp_path: Path) -> None:
    envelope, manifest = _identity_fixture(tmp_path)
    manifest["execution_hash"] = "5" * 64

    result = check_execution_identity(_row(execution=envelope, manifest_payload=manifest))

    assert result.status == "fail"
    assert "execution_hash" in str(result.detail)


@pytest.mark.parametrize("failure", ["bytes", "schema"])
def test_execution_identity_rejects_artifact_digest_or_schema_failure(
    tmp_path: Path, failure: str
) -> None:
    envelope, manifest = _identity_fixture(tmp_path)
    if failure == "bytes":
        Path(envelope.payload.uri).write_text("{}", encoding="utf-8")
    else:
        payload = json.loads(Path(envelope.payload.uri).read_text(encoding="utf-8"))
        payload["schema_version"] = "feedbax.spec.training_run_matrix.v0"
        data = training_spec_canonical_bytes(payload)
        Path(envelope.payload.uri).write_bytes(data)
        envelope = envelope.model_copy(
            update={
                "payload": envelope.payload.model_copy(
                    update={"sha256": hashlib.sha256(data).hexdigest()}
                )
            }
        )

    result = check_execution_identity(_row(execution=envelope, manifest_payload=manifest))

    assert result.status == "fail"
    expected_detail = "sha256 mismatch" if failure == "bytes" else "schema_version mismatch"
    assert expected_detail in str(result.detail)


def test_manifest_valid_loads_manifest_and_compares_preflight_payload(tmp_path: Path) -> None:
    training_payload = {"method_payload": {"payload": {"optimizer": {"type": "adamw"}}}, "metadata": {"optional": None}}
    manifest = TrainingRunManifest(
        id="feedbax-training-run:row-a",
        training_spec=spec_payload("TrainingRunSpec", training_payload),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json", exclude_none=True), sort_keys=True),
        encoding="utf-8",
    )
    registry = CheckRegistry({"manifest_valid": check_manifest_valid})

    certificate = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[
            _row(
                manifest_path=manifest_path,
                manifest_payload=None,
                preflight_normalized_payload=training_payload,
            )
        ],
        registry=registry,
        generated_at=GENERATED_AT,
    )

    assert certificate.rows["row-a"].checks[0].status == "pass"


@pytest.mark.parametrize("expected_status", ["complete", "completed", "done"])
def test_events_terminal_complete_event_matches_success_aliases(
    expected_status: str,
) -> None:
    result = check_events_terminal(
        _row(
            event_log=[
                {"type": "started"},
                {"type": "complete", "payload": {"status": "completed"}},
            ],
            bundle_row_spec={"expected_terminal_status": expected_status},
        )
    )

    assert result.status == "pass"


def test_events_terminal_legacy_skip_and_terminal_validation() -> None:
    assert check_events_terminal(_row(event_log=None)).status == "skipped"
    passed = check_events_terminal(
        _row(
            event_log=[
                {"type": "started"},
                {"type": "complete", "payload": {"status": "completed"}},
            ],
            bundle_row_spec={"expected_terminal_status": "completed"},
        )
    )
    failed = check_events_terminal(
        _row(
            event_log=[
                {"type": "complete", "payload": {"status": "completed"}},
                {"type": "failed", "payload": {"status": "failed"}},
            ],
        )
    )

    assert passed.status == "pass"
    assert failed.status == "fail"


@pytest.mark.parametrize(
    ("event_status", "expected_status"),
    [
        ("complete", "failed"),
        ("failed", "complete"),
    ],
)
def test_events_terminal_fails_success_failure_disagreement(
    event_status: str,
    expected_status: str,
) -> None:
    result = check_events_terminal(
        _row(
            event_log=[
                {
                    "type": "complete" if event_status == "complete" else "failed",
                    "payload": {"status": event_status},
                }
            ],
            bundle_row_spec={"expected_terminal_status": expected_status},
        )
    )

    assert result.status == "fail"


def test_events_terminal_cancelled_uses_payload_status_and_agrees_with_run_outputs() -> None:
    result = check_events_terminal(
        _row(
            event_log=[{"type": "complete", "payload": {"status": "cancelled"}}],
            row_status="stopped",
            bundle_row_spec={"expected_terminal_status": "cancelled"},
            training_diagnostics={"terminal_status": "cancelled"},
        )
    )

    assert result.status == "pass"
    assert result.observed["carrier_type"] == "complete"
    assert result.observed["terminal_status"] == "cancelled"


def test_events_terminal_failed_uses_failed_carrier_and_payload_status() -> None:
    result = check_events_terminal(
        _row(
            event_log=[{"type": "failed", "payload": {"status": "failed"}}],
            row_status="failed",
            training_diagnostics={"terminal_status": "failed"},
        )
    )

    assert result.status == "pass"
    assert result.observed["carrier_type"] == "failed"
    assert result.observed["terminal_status"] == "failed"


@pytest.mark.parametrize(
    ("event", "detail"),
    [
        ({"type": "complete", "payload": {}}, "payload.status"),
        ({"type": "complete", "payload": {"status": "done"}}, "payload.status"),
        (
            {"type": "complete", "payload": {"status": "failed"}},
            "carrier type disagrees",
        ),
    ],
)
def test_events_terminal_rejects_malformed_terminal_events(
    event: dict[str, object],
    detail: str,
) -> None:
    result = check_events_terminal(_row(event_log=[event]))

    assert result.status == "fail"
    assert detail in str(result.detail)


def test_events_terminal_rejects_illegal_cancelled_carrier() -> None:
    result = check_events_terminal(
        _row(event_log=[{"type": "cancelled", "payload": {"status": "cancelled"}}])
    )

    assert result.status == "fail"
    assert result.observed["terminal_count"] == 0


def test_events_terminal_rejects_disagreement_between_outputs() -> None:
    result = check_events_terminal(
        _row(
            event_log=[{"type": "complete", "payload": {"status": "cancelled"}}],
            row_status="stopped",
            training_diagnostics={"terminal_status": "completed"},
        )
    )

    assert result.status == "fail"
    assert "statuses disagree" in str(result.detail)


@pytest.mark.parametrize(
    ("row_status", "event_type", "event_status"),
    [
        ("pending", "complete", "completed"),
        ("launched", "complete", "cancelled"),
        ("ready", "failed", "failed"),
        ("running", "complete", "completed"),
    ],
)
def test_events_terminal_rejects_nonterminal_durable_row_status(
    row_status: str,
    event_type: str,
    event_status: str,
) -> None:
    result = check_events_terminal(
        _row(
            event_log=[{"type": event_type, "payload": {"status": event_status}}],
            row_status=row_status,
            training_diagnostics={"terminal_status": event_status},
        )
    )

    assert result.status == "fail"
    assert "statuses disagree" in str(result.detail)


def test_lr_trace_uses_optimizer_builder_resume_context_and_rejects_flat_terminal_lr() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=10,
        constant_lr_iterations=4,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.2,
    )
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=schedule,
    ).model_dump(mode="json")

    passing_row = _row(
        bundle_row_spec={
            "optimizer": optimizer,
            "resume_context": {
                "schedule_origin_step": 100,
                "current_step": 100,
                "optimizer_count_at_current_step": 12_000,
            },
        },
        training_diagnostics={
            "lr_trace": {
                100: 0.01,
                104: 0.1,
                110: 0.02,
            }
        },
    )
    failing_row = _row(
        bundle_row_spec=passing_row.bundle_row_spec,
        training_diagnostics={
            "lr_trace": {
                100: 0.02,
                104: 0.02,
                110: 0.02,
            }
        },
    )

    assert check_lr_trace(passing_row).status == "pass"
    failed = check_lr_trace(failing_row)
    assert failed.status == "fail"
    assert failed.observed[100] == pytest.approx(0.02)


def test_lr_trace_conforms_each_declared_mapped_coordinate_and_rejects_missing() -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=0.1,
            total_steps=10,
            constant_lr_iterations=4,
            warmup_init_fraction=0.1,
            cosine_annealing_alpha=0.2,
        ),
    ).model_dump(mode="json")
    bundle = {
        "optimizer": optimizer,
        "resume_context": {
            "schedule_origin_step": 100,
            "current_step": 100,
            "optimizer_count_at_current_step": 12_000,
        },
        "worker_execution": {
            "mapping_levels": [{"axis": "ensemble"}],
            "method_contract": {"axes": [{"name": "ensemble", "size": 2}]},
        },
    }
    trace = [
        {
            "step": step,
            "learning_rate": value,
            "axis_coordinates": [{"axis": "ensemble", "index": index}],
        }
        for index in range(2)
        for step, value in ((100, 0.01), (104, 0.1), (110, 0.02))
    ]
    passing = check_lr_trace(
        _row(bundle_row_spec=bundle, training_diagnostics={"lr_trace": trace})
    )
    assert passing.status == "pass"

    missing = check_lr_trace(
        _row(bundle_row_spec=bundle, training_diagnostics={"lr_trace": trace[:3]})
    )
    assert missing.status == "fail"
    assert "coordinate coverage mismatch" in str(missing.detail)


def test_lr_trace_discovers_controller_optimizer_and_compares_realized_samples() -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=0.1,
            total_steps=10,
            constant_lr_iterations=4,
            warmup_init_fraction=0.1,
            cosine_annealing_alpha=0.2,
        ),
    ).model_dump(mode="json")
    bundle_row_spec = {
        "method_payload": {"payload": {"controller_optimizer": optimizer}},
        "resume_context": {
            "schedule_origin_step": 100,
            "current_step": 100,
            "optimizer_count_at_current_step": 12_000,
        },
    }
    passing = check_lr_trace(
        _row(
            bundle_row_spec=bundle_row_spec,
            training_diagnostics={"lr_trace": {100: 0.01, 104: 0.1, 110: 0.02}},
        )
    )
    mismatched = check_lr_trace(
        _row(
            bundle_row_spec=bundle_row_spec,
            training_diagnostics={"lr_trace": {100: 0.01, 104: 0.05, 110: 0.02}},
        )
    )

    assert passing.status == "pass"
    assert passing.expected == pytest.approx({100: 0.01, 104: 0.1, 110: 0.02})
    assert passing.observed == pytest.approx({100: 0.01, 104: 0.1, 110: 0.02})
    assert mismatched.status == "fail"
    assert mismatched.expected[104] == pytest.approx(0.1)
    assert mismatched.observed[104] == pytest.approx(0.05)


def test_lr_trace_uses_native_method_training_optimizer_for_constant_schedule() -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(kind="constant", learning_rate_0=3e-5),
    ).model_dump(mode="json")
    training_spec = {
        "method_payload": {"payload": {"training": {"optimizer": optimizer}}},
        "worker_execution": {
            "mapping_levels": [{"axis": "replica"}],
            "method_contract": {"axes": [{"name": "replica", "size": 2}]},
        },
    }
    trace = [
        {
            "step": step,
            "learning_rate": 3e-5,
            "axis_coordinates": [{"axis": "replica", "index": replica}],
        }
        for replica in range(2)
        for step in (500, 1_000, 1_500)
    ]

    result = check_lr_trace(
        _row(
            bundle_row_spec=training_spec,
            manifest_payload={"training_spec": {"inline": training_spec}},
            training_diagnostics={"lr_trace": trace},
        )
    )

    assert result.status == "pass"
    assert result.expected["(('replica', 0),)"][500] == pytest.approx(3e-5)


def test_lr_trace_normalizes_segment_local_continuation_steps() -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=3e-3,
            constant_lr_iterations=1_000,
            total_steps=3_500,
            cosine_annealing_alpha=0.01,
            warmup_init_fraction=0.01,
        ),
    ).model_dump(mode="json")
    result = check_lr_trace(
        _row(
            bundle_row_spec={
                "optimizer": optimizer,
                "resume_context": {
                    "schedule_origin_step": 12_000,
                    "current_step": 12_000,
                    "optimizer_count_at_current_step": 0,
                },
            },
            training_diagnostics={
                "segment_completed_batches": 4_500,
                "cumulative_completed_batches": 16_500,
                "lr_trace": {
                    500: 0.001512030023150146,
                    2_500: 0.0010578848887234926,
                    4_500: 0.000029999999242136255,
                },
            },
        )
    )

    assert result.status == "pass"
    assert result.observed == pytest.approx(
        {
            12_499: 0.001512030023150146,
            14_499: 0.0010578848887234926,
            16_499: 0.000029999999242136255,
        }
    )


def test_lr_trace_preserves_cumulative_continuation_steps() -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(kind="constant", learning_rate_0=3e-5),
    ).model_dump(mode="json")
    result = check_lr_trace(
        _row(
            bundle_row_spec={
                "optimizer": optimizer,
                "resume_context": {
                    "schedule_origin_step": 12_000,
                    "current_step": 12_000,
                    "optimizer_count_at_current_step": 12_000,
                },
            },
            training_diagnostics={
                "segment_completed_batches": 4_500,
                "cumulative_completed_batches": 16_500,
                "lr_trace": {12_500: 3e-5, 14_500: 3e-5, 16_500: 3e-5},
            },
        )
    )

    assert result.status == "pass"
    assert result.observed == pytest.approx({12_500: 3e-5, 14_500: 3e-5, 16_500: 3e-5})


@pytest.mark.parametrize(
    ("segment_completed", "cumulative_completed", "steps", "detail"),
    [
        (4_500, 16_500, (500, 13_000, 4_500), "mixed or out-of-range"),
        (4_500, 16_500, (0, 500, 4_500), "mixed or out-of-range"),
        (200, 300, (150, 175, 200), "ambiguous"),
    ],
)
def test_lr_trace_rejects_mixed_or_ambiguous_continuation_frames(
    segment_completed: int,
    cumulative_completed: int,
    steps: tuple[int, int, int],
    detail: str,
) -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(kind="constant", learning_rate_0=3e-5),
    ).model_dump(mode="json")
    result = check_lr_trace(
        _row(
            bundle_row_spec={"optimizer": optimizer},
            training_diagnostics={
                "segment_completed_batches": segment_completed,
                "cumulative_completed_batches": cumulative_completed,
                "lr_trace": {step: 3e-5 for step in steps},
            },
        )
    )

    assert result.status == "fail"
    assert detail in str(result.detail)


def test_lr_trace_resolves_legacy_program_steps_from_immutable_event_coordinates() -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(kind="constant", learning_rate_0=3e-5),
    ).model_dump(mode="json")
    program_to_batches = {5_000: 17_000, 7_000: 19_000, 9_000: 21_000}
    events = [
        {
            "type": "progress",
            "payload": {
                "coordinate": {
                    "program_step": program_step,
                    "completed_batches": completed_batches,
                }
            },
        }
        for program_step, completed_batches in program_to_batches.items()
    ]
    result = check_lr_trace(
        _row(
            bundle_row_spec={
                "optimizer": optimizer,
                "resume_context": {
                    "schedule_origin_step": 12_000,
                    "current_step": 16_500,
                    "optimizer_count_at_current_step": 16_500,
                },
            },
            training_diagnostics={
                "segment_completed_batches": 4_500,
                "cumulative_completed_batches": 21_000,
                "lr_trace": {step: 3e-5 for step in program_to_batches},
            },
            event_log=events,
        )
    )

    assert result.status == "pass"
    assert result.observed == pytest.approx({
        17_000: 3e-5,
        19_000: 3e-5,
        21_000: 3e-5,
    })


def test_lr_trace_legacy_program_steps_require_complete_unambiguous_event_evidence() -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(kind="constant", learning_rate_0=3e-5),
    ).model_dump(mode="json")
    diagnostics = {
        "segment_completed_batches": 4_500,
        "cumulative_completed_batches": 21_000,
        "lr_trace": {5_000: 3e-5, 7_000: 3e-5, 9_000: 3e-5},
    }
    row = _row(
        bundle_row_spec={"optimizer": optimizer},
        training_diagnostics=diagnostics,
        event_log=[
            {
                "payload": {
                    "coordinate": {
                        "program_step": 5_000,
                        "completed_batches": 17_000,
                    }
                }
            }
        ],
    )

    missing = check_lr_trace(row)
    assert missing.status == "fail"
    assert "lacks immutable completed-batch evidence" in str(missing.detail)

    conflicting = check_lr_trace(
        _row(
            bundle_row_spec={"optimizer": optimizer},
            training_diagnostics=diagnostics,
            event_log=[
                {
                    "payload": {
                        "coordinate": {
                            "program_step": step,
                            "completed_batches": completed_batches,
                        }
                    }
                }
                for step, completed_batches in (
                    (5_000, 17_000),
                    (5_000, 17_001),
                    (7_000, 19_000),
                    (9_000, 21_000),
                )
            ],
        )
    )
    assert conflicting.status == "fail"
    assert "run events disagree" in str(conflicting.detail)


def test_lr_trace_rejects_conflicting_governed_optimizer_authorities() -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"learning_rate": 3e-5},
    ).model_dump(mode="json")
    conflicting = OptimizerSpec(
        type="adamw",
        params={"learning_rate": 4e-5},
    ).model_dump(mode="json")

    result = check_lr_trace(
        _row(
            bundle_row_spec={
                "optimizer": optimizer,
                "method_payload": {"payload": {"training": {"optimizer": conflicting}}},
            },
            training_diagnostics={"lr_trace": {500: 3e-5, 1_000: 3e-5, 1_500: 3e-5}},
        )
    )

    assert result.status == "fail"
    assert result.expected == "one unambiguous governed optimizer spec"
    assert result.observed == "ValueError"
    assert "ambiguous governed optimizer specs" in str(result.detail)


@pytest.mark.parametrize(
    "optimizer_location",
    [
        "optimizer",
        "optimizer_spec",
        "training.optimizer",
        "training_spec.method_payload.payload.optimizer",
        "manifest.training_spec.method_payload.payload.optimizer",
    ],
)
def test_lr_trace_legacy_optimizer_locations_remain_supported(optimizer_location: str) -> None:
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=0.1,
            total_steps=10,
            constant_lr_iterations=4,
            warmup_init_fraction=0.1,
            cosine_annealing_alpha=0.2,
        ),
    ).model_dump(mode="json")
    resume_context = {
        "schedule_origin_step": 100,
        "current_step": 100,
        "optimizer_count_at_current_step": 12_000,
    }
    bundle_row_spec: dict[str, object] = {"resume_context": resume_context}
    manifest_payload = None
    if optimizer_location == "optimizer":
        bundle_row_spec["optimizer"] = optimizer
    elif optimizer_location == "optimizer_spec":
        bundle_row_spec["optimizer_spec"] = optimizer
    elif optimizer_location == "training.optimizer":
        bundle_row_spec["training"] = {"optimizer": optimizer}
    elif optimizer_location == "training_spec.method_payload.payload.optimizer":
        bundle_row_spec["training_spec"] = {"method_payload": {"payload": {"optimizer": optimizer}}}
    else:
        manifest_payload = {
            "training_spec": {"inline": {"method_payload": {"payload": {"optimizer": optimizer}}}}
        }

    result = check_lr_trace(
        _row(
            bundle_row_spec=bundle_row_spec,
            manifest_payload=manifest_payload,
            training_diagnostics={"lr_trace": {100: 0.01, 104: 0.1, 110: 0.02}},
        )
    )

    assert result.status == "pass"


def test_lr_trace_without_optimizer_fails_closed_as_missing_input() -> None:
    result = check_lr_trace(
        _row(
            bundle_row_spec={
                "resume_context": {
                    "schedule_origin_step": 100,
                    "current_step": 100,
                    "optimizer_count_at_current_step": 12_000,
                }
            },
            training_diagnostics={"lr_trace": {100: 0.01, 104: 0.1, 110: 0.02}},
        )
    )

    assert result.status == "fail"
    assert result.expected == {"required": ["bundle_row_spec optimizer spec"]}
    assert result.observed is None
    assert result.detail == "missing required input: bundle_row_spec optimizer spec"


def test_plugin_check_discovery_and_failure_propagation() -> None:
    def plugin_check(row: ConformanceRowArtifacts) -> CheckEntry:
        return CheckEntry(
            check_id="project_check",
            status="fail",
            expected="project invariant",
            observed="broken",
        )

    class FakeEntryPoint:
        name = "fixture"

        def load(self) -> object:
            return SimpleNamespace(
                feedbax_conformance_checks=lambda: [("project_check", plugin_check)]
            )

    registry = CheckRegistry()
    load_conformance_check_plugins(registry=registry, entry_points=[FakeEntryPoint()])
    certificate = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[_row()],
        registry=registry,
        generated_at=GENERATED_AT,
    )

    assert certificate.overall == "fail"
    assert certificate.rows["row-a"].checks[0].check_id == "project_check"
    assert certificate.rows["row-a"].checks[0].status == "fail"


def test_conformance_plugin_discovery_ignores_non_conformance_registrars() -> None:
    def experiment_registrar(registry: object) -> None:
        raise AssertionError("non-conformance registrar should not be called")

    class FakeEntryPoint:
        name = "experiment"

        def load(self) -> object:
            return experiment_registrar

    registry = CheckRegistry()
    load_conformance_check_plugins(registry=registry, entry_points=[FakeEntryPoint()])

    assert len(registry) == 0


def test_register_coupling_rejects_failing_certificate() -> None:
    certificate = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[_row()],
        registry=CheckRegistry({"bad": lambda row: CheckEntry(check_id="bad", status="fail")}),
        generated_at=GENERATED_AT,
    )

    with pytest.raises(ValueError, match="phase=completed"):
        assert_certificate_allows_completed_registration(certificate)
