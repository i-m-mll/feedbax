from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.contracts.manifest import TrainingRunManifest, spec_payload
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training.diagnostics import (
    TRAINING_DIAGNOSTICS_SCHEMA_ID,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4,
    MethodTrainingTrace,
    MethodTrainingTraceLoadError,
    MethodTrainingTraceRecord,
    TrainingDiagnostics,
    load_method_training_trace,
)
from tests.test_training_run_executor import _mapped_diagnostics_run_spec


def _stored_trace(
    tmp_path: Path,
    corruption: str | None = None,
    *,
    diagnostics_schema_version: str = TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
) -> tuple[TrainingRunManifest, ImmutableArtifactBlobProvider, MethodTrainingTrace]:
    run_spec = _mapped_diagnostics_run_spec()
    contract = run_spec.worker_execution.method_contract
    declaration = contract.training_diagnostics
    assert declaration is not None
    trace = MethodTrainingTrace(
        method_ref=contract.method_ref,
        trace_schema_id=declaration.trace_schema_id,
        trace_schema_version=declaration.trace_schema_version,
        measurement_basis=declaration.measurement_basis,
        metric_payload_slot=declaration.metric_payload_slot,
        replica_axis=declaration.replica_axis,
        records=[
            MethodTrainingTraceRecord(
                completed_batch=batch,
                replica_index=replica,
                value={"loss": float(batch + replica)},
            )
            for batch in (11, 12)
            for replica in range(5)
        ],
    )
    diagnostics = TrainingDiagnostics(
        schema_version=diagnostics_schema_version,
        manifest_id="feedbax-training-run:trace",
        run_id="trace",
        terminal_status="completed",
        completed_batches=12,
        segment_completed_batches=2,
        cumulative_completed_batches=12,
        method_trace=trace,
    )
    payload = diagnostics.model_dump(mode="json")
    records = payload["method_trace"]["records"]
    if corruption == "duplicate":
        records.append(records[0])
    elif corruption == "gap":
        records.pop()
    elif corruption == "out_of_range":
        records[0]["completed_batch"] = 10
    elif corruption == "schema":
        payload["method_trace"]["trace_schema_version"] = "unsupported"
    elif corruption == "provenance":
        payload["method_trace"]["measurement_basis"] = "different"
    elif corruption == "value":
        records[0]["value"] = "unsupported"
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    artifact = provider.store_bytes(
        json.dumps(payload).encode(),
        role="training_diagnostics",
        logical_name="training-diagnostics.json",
        media_type="application/json",
        metadata={
            "schema_id": TRAINING_DIAGNOSTICS_SCHEMA_ID,
            "schema_version": diagnostics_schema_version,
        },
    )
    manifest = TrainingRunManifest(
        id="feedbax-training-run:trace",
        job_id="trace",
        status="completed",
        training_spec=spec_payload(
            "TrainingRunSpec",
            run_spec.model_dump(mode="json"),
        ),
        artifacts=[artifact],
    )
    return manifest, provider, trace


@pytest.mark.parametrize(
    "diagnostics_schema_version",
    [TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3, TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4],
)
def test_load_method_training_trace_round_trips_governed_payload(
    tmp_path: Path,
    diagnostics_schema_version: str,
) -> None:
    manifest, provider, expected = _stored_trace(
        tmp_path,
        diagnostics_schema_version=diagnostics_schema_version,
    )

    assert load_method_training_trace(manifest, provider) == expected


@pytest.mark.parametrize(
    ("corruption", "kind"),
    [
        ("duplicate", "duplicate"),
        ("gap", "gap"),
        ("out_of_range", "out_of_range"),
        ("schema", "schema"),
        ("provenance", "provenance"),
        ("value", "schema"),
    ],
)
def test_load_method_training_trace_rejects_payload_corruption(
    tmp_path: Path,
    corruption: str,
    kind: str,
) -> None:
    manifest, provider, _ = _stored_trace(tmp_path, corruption)

    with pytest.raises(MethodTrainingTraceLoadError) as exc_info:
        load_method_training_trace(manifest, provider)

    assert exc_info.value.kind == kind


def test_load_method_training_trace_requires_manifest_authority(tmp_path: Path) -> None:
    manifest, provider, _ = _stored_trace(tmp_path)

    with pytest.raises(MethodTrainingTraceLoadError) as exc_info:
        load_method_training_trace(manifest.model_copy(update={"artifacts": []}), provider)

    assert exc_info.value.kind == "missing"


def test_load_method_training_trace_rejects_truncated_custody(tmp_path: Path) -> None:
    manifest, provider, _ = _stored_trace(tmp_path)
    artifact = manifest.artifacts[0]
    path = provider.root / provider.canonical_relative_path(artifact)
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(MethodTrainingTraceLoadError) as exc_info:
        load_method_training_trace(manifest, provider)

    assert exc_info.value.kind == "truncated"


@pytest.mark.parametrize(
    "value",
    [None, "text", b"bytes", complex(1, 2), float("inf"), {1: True}],
)
def test_method_trace_record_rejects_non_numeric_boolean_json(value: object) -> None:
    with pytest.raises(ValidationError):
        MethodTrainingTraceRecord(completed_batch=1, replica_index=0, value=value)


@pytest.mark.parametrize(
    ("value", "expected_json"),
    [
        (True, '{"completed_batch":1,"replica_index":0,"value":true}'),
        (3, '{"completed_batch":1,"replica_index":0,"value":3}'),
        (1.5, '{"completed_batch":1,"replica_index":0,"value":1.5}'),
        (
            {"accepted": True, "values": [2, 3.5]},
            (
                '{"completed_batch":1,"replica_index":0,'
                '"value":{"accepted":true,"values":[2,3.5]}}'
            ),
        ),
    ],
)
def test_method_trace_record_existing_valid_v3_bytes_are_unchanged(
    value: object,
    expected_json: str,
) -> None:
    record = MethodTrainingTraceRecord(completed_batch=1, replica_index=0, value=value)

    assert record.model_dump_json() == expected_json


def test_training_diagnostics_v3_existing_structured_value_bytes_are_unchanged() -> None:
    trace = MethodTrainingTrace(
        method_ref="m",
        trace_schema_id="s",
        trace_schema_version="s.v1",
        measurement_basis="basis",
        metric_payload_slot="slot",
        replica_axis="replica",
        records=[
            MethodTrainingTraceRecord(
                completed_batch=1,
                replica_index=0,
                value={"accepted": True, "values": [2, 3.5]},
            )
        ],
    )
    diagnostics = TrainingDiagnostics(
        schema_version=TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
        manifest_id="manifest",
        run_id="run",
        terminal_status="completed",
        completed_batches=1,
        segment_completed_batches=1,
        cumulative_completed_batches=1,
        method_trace=trace,
    )
    expected = (
        '{"kind":"TrainingDiagnostics","schema_id":"feedbax.manifest.training_diagnostics",'
        '"schema_version":"feedbax.manifest.training_diagnostics.v3","manifest_id":"manifest",'
        '"run_id":"run","terminal_status":"completed","completed_batches":1,'
        '"segment_completed_batches":1,"cumulative_completed_batches":1,"seeds":[],'
        '"lr_trace":[],"checkpoint_coordinates":[],"checkpoint_transactions":[],'
        '"resume_context":null,"optimizer_build_context":null,"method_trace":{'
        '"method_ref":"m","trace_schema_id":"s","trace_schema_version":"s.v1",'
        '"measurement_basis":"basis","metric_payload_slot":"slot","replica_axis":"replica",'
        '"records":[{"completed_batch":1,"replica_index":0,"value":{"accepted":true,'
        '"values":[2,3.5]}}]},"metadata":{}}'
    )

    assert diagnostics.model_dump_json() == expected
