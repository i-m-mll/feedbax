from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.contracts.manifest import TrainingRunManifest, spec_payload
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training.diagnostics import (
    TRAINING_DIAGNOSTICS_SCHEMA_ID,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
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
        schema_version=TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
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
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    artifact = provider.store_bytes(
        json.dumps(payload).encode(),
        role="training_diagnostics",
        logical_name="training-diagnostics.json",
        media_type="application/json",
        metadata={
            "schema_id": TRAINING_DIAGNOSTICS_SCHEMA_ID,
            "schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
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


def test_load_method_training_trace_round_trips_governed_payload(tmp_path: Path) -> None:
    manifest, provider, expected = _stored_trace(tmp_path)

    assert load_method_training_trace(manifest, provider) == expected


@pytest.mark.parametrize(
    ("corruption", "kind"),
    [
        ("duplicate", "duplicate"),
        ("gap", "gap"),
        ("out_of_range", "out_of_range"),
        ("schema", "schema"),
        ("provenance", "provenance"),
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
