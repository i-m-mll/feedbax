from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from dataclasses import replace

import pytest
from pydantic import ValidationError

from feedbax.contracts.graph import GRAPH_SPEC_SCHEMA_ID, GRAPH_SPEC_SCHEMA_VERSION_V3
from feedbax.contracts.training import default_training_method_registry
from feedbax.contracts.manifest import TrainingRunManifest, TrainingSweepAxis
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    RowLowererIdentity,
    TrainingRowLoweringResult,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.spec_storage import (
    TrainingRunExecutionCapsule,
    build_resolved_semantics_snapshot,
    training_run_intent_hash,
    training_spec_canonical_bytes,
    training_spec_sha256,
    store_canonical_json_artifact,
)
from feedbax.training.run_matrix import (
    materialize_adapted_run_matrix,
)
from feedbax.training.spec_storage import (
    compile_training_run_matrix,
    emit_training_run_spec_storage,
    stamp_training_run_manifest_identities,
)
from feedbax.orchestration.bundle import DeploymentPolicy

_METHOD_REGISTRY = default_training_method_registry()


def _matrix(base: dict[str, object]) -> TrainingRunMatrixSpec:
    return TrainingRunMatrixSpec.model_validate(
        {
            "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            "name": "cosmetic name",
            "base": {"kind": "inline", "inline": base},
            "rows": [{"row_id": "row", "overrides": []}],
        }
    )


def test_round_trip_numeric_canonicalization_and_structural_sharing() -> None:
    left = {"x": 1e-5, "nested": {"value": 2}, "zero": -0.0}
    right = {"nested": {"value": 2}, "x": 0.00001, "zero": 0.0}

    assert training_spec_canonical_bytes(left) == training_spec_canonical_bytes(right)
    assert training_spec_canonical_bytes({"large": 2**80}) == f'{{"large":{2**80}}}'.encode()
    assert b"e-" in training_spec_canonical_bytes({"small": 1e-100})
    assert training_spec_canonical_bytes({"integer": 2}) != training_spec_canonical_bytes(
        {"integer": 2.0}
    )
    with pytest.raises(TypeError, match="keys must all be strings"):
        training_spec_canonical_bytes({1: "integer", "1": "string"})
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            training_spec_canonical_bytes({"value": value})
    snapshot = build_resolved_semantics_snapshot({"a": left["nested"], "b": right["nested"]})
    object_nodes = [node for node in snapshot["nodes"].values() if node["type"] == "object"]
    assert len(object_nodes) == 2  # shared nested object plus root


def test_snapshot_decodes_with_pure_stdlib_decoder_alone() -> None:
    expected = {"rows": [{"x": 1.25}, {"x": 1.25}]}
    snapshot = build_resolved_semantics_snapshot(expected)
    decoder_path = (
        Path(__file__).parents[1] / "feedbax" / "contracts" / "resolved_snapshot_decoder.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "standalone_snapshot_decoder", decoder_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    assert module.decode_resolved_snapshot(snapshot) == expected

    root = snapshot["root_hash"]
    tampered = json.loads(json.dumps(snapshot))
    tampered["nodes"][root]["extra"] = True
    with pytest.raises(ValueError, match="extra fields|node fields"):
        module.decode_resolved_snapshot(tampered)


def test_decoder_rejects_invalid_variants_missing_refs_cycles_and_hash_mismatch() -> None:
    from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot

    snapshot = build_resolved_semantics_snapshot({"value": 1})
    scalar_hash = next(
        digest for digest, node in snapshot["nodes"].items() if node["type"] == "scalar"
    )
    missing_value = json.loads(json.dumps(snapshot))
    missing_value["nodes"][scalar_hash].pop("value")
    with pytest.raises(ValueError, match="scalar node fields"):
        decode_resolved_snapshot(missing_value)

    missing_ref = json.loads(json.dumps(snapshot))
    root = missing_ref["root_hash"]
    missing_ref["nodes"][root]["children"]["value"] = "f" * 64
    with pytest.raises(ValueError, match="missing resolved-semantics node"):
        decode_resolved_snapshot(missing_ref)

    cycle_hash = "a" * 64
    cyclic = {
        "schema_id": snapshot["schema_id"],
        "schema_version": snapshot["schema_version"],
        "root_hash": cycle_hash,
        "nodes": {cycle_hash: {"type": "array", "children": [cycle_hash]}},
    }
    with pytest.raises(ValueError, match="cycle"):
        decode_resolved_snapshot(cyclic)

    tampered = json.loads(json.dumps(snapshot))
    tampered["nodes"][scalar_hash]["value"] = 2
    with pytest.raises(ValueError, match="hash mismatch"):
        decode_resolved_snapshot(tampered)


def test_emission_round_trip_is_stable_and_materializer_drift_is_visible(
    tmp_path: Path,
) -> None:
    matrix = _matrix({"value": 1e-5})
    lock = tmp_path / "uv.lock"
    lock.write_text("locked", encoding="utf-8")
    with pytest.raises(ValueError, match="tests/fixtures only"):
        emit_training_run_spec_storage(
            matrix,
            method_registry=_METHOD_REGISTRY,
            repo_root=tmp_path,
            authored_path=tmp_path / "tracked" / "rejected.json",
            custody_root=tmp_path / "custody",
            materializer_commit="abc",
            dependency_lock_path=lock,
        )
    first = emit_training_run_spec_storage(
        matrix,
        method_registry=_METHOD_REGISTRY,
        repo_root=tmp_path,
        authored_path=tmp_path / "tracked" / "matrix.json",
        custody_root=tmp_path / "custody",
        materializer_commit="abc",
        dependency_lock_path=lock,
        environment_digest="e" * 64,
        input_data_identities=[{"role": "dataset", "sha256": "d" * 64}],
        allow_inline_base=True,
        row_validator=lambda _payload, _row_id: None,
    )
    second = emit_training_run_spec_storage(
        matrix,
        method_registry=_METHOD_REGISTRY,
        repo_root=tmp_path,
        authored_path=tmp_path / "tracked" / "matrix-again.json",
        custody_root=tmp_path / "custody",
        materializer_commit="def",
        dependency_lock_path=lock,
        environment_digest="e" * 64,
        input_data_identities=[{"role": "dataset", "sha256": "d" * 64}],
        allow_inline_base=True,
        row_validator=lambda _payload, _row_id: None,
    )
    assert first.intent_hash == second.intent_hash
    assert first.execution_hash == second.execution_hash
    stamped = stamp_training_run_manifest_identities(
        TrainingRunManifest(id="feedbax-training-run:test"),
        first,
    )
    assert stamped.intent_hash == first.intent_hash
    assert stamped.execution_hash == first.execution_hash

    def simulate_materializer_drift(materialized):
        row = materialized.rows[0]
        changed_row = replace(row, payload={**row.payload, "materializer_revision": 2})
        return replace(materialized, rows=[changed_row])

    drifted = emit_training_run_spec_storage(
        matrix,
        method_registry=_METHOD_REGISTRY,
        repo_root=tmp_path,
        authored_path=tmp_path / "tracked" / "matrix-drift.json",
        custody_root=tmp_path / "custody",
        materializer_commit="changed",
        dependency_lock_path=lock,
        environment_digest="e" * 64,
        input_data_identities=[{"role": "dataset", "sha256": "d" * 64}],
        allow_inline_base=True,
        row_validator=lambda _payload, _row_id: None,
        _test_materializer_transform=simulate_materializer_drift,
    )
    assert drifted.intent_hash == first.intent_hash
    assert drifted.execution_hash != first.execution_hash
    with pytest.raises(ValueError, match="materializer drift"):
        stamp_training_run_manifest_identities(stamped, drifted)
    with pytest.raises(ValidationError, match="materializer drift"):
        capsule_payload = drifted.capsule.model_dump(mode="json")
        capsule_payload["execution_hash"] = first.execution_hash
        TrainingRunExecutionCapsule.model_validate(capsule_payload)
    assert first.capsule.materializer_commit == "abc"
    assert first.capsule.environment_digest == "e" * 64
    assert first.capsule.input_data_identities[0]["sha256"] == "d" * 64
    assert first.capsule.dependency_lock_digest == hashlib.sha256(b"locked").hexdigest()
    assert "training_run_matrix" in first.capsule.relevant_schema_versions
    capsule_round_trip = TrainingRunExecutionCapsule.model_validate_json(
        Path(first.capsule_artifact.uri).read_bytes()
    )
    assert capsule_round_trip == first.capsule
    with pytest.raises(ValidationError, match="64-hex"):
        TrainingRunExecutionCapsule.model_validate(
            {**first.capsule.model_dump(mode="json"), "intent_hash": "BAD"}
        )
    bad_input = first.capsule.model_dump(mode="json")
    bad_input["input_data_identities"] = [{"sha256": "BAD"}]
    with pytest.raises(ValidationError, match="64-hex"):
        TrainingRunExecutionCapsule.model_validate(bad_input)
    with pytest.raises(ValidationError, match="64-hex"):
        TrainingRunManifest(
            id="feedbax-training-run:bad-hash",
            intent_hash="BAD",
        )


def test_snapshot_rows_are_complete_and_seed_changes_execution_identity(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("lock", encoding="utf-8")
    first_matrix = _matrix({"value": 1})
    first_payload = first_matrix.model_dump(mode="json")
    first_payload["rows"][0]["seed"] = 11
    second_payload = json.loads(json.dumps(first_payload))
    second_payload["rows"][0]["seed"] = 12

    def emit(payload, suffix):
        return emit_training_run_spec_storage(
            TrainingRunMatrixSpec.model_validate(payload),
            method_registry=_METHOD_REGISTRY,
            repo_root=tmp_path,
            authored_path=tmp_path / f"authored-{suffix}.json",
            custody_root=tmp_path / "custody",
            materializer_commit="abc",
            dependency_lock_path=lock,
            allow_inline_base=True,
            row_validator=lambda _payload, _row_id: None,
        )

    first = emit(first_payload, "first")
    second = emit(second_payload, "second")
    assert first.execution_hash != second.execution_hash

    from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot

    snapshot = json.loads(Path(first.snapshot_artifact.uri).read_text(encoding="utf-8"))
    decoded = decode_resolved_snapshot(snapshot)
    row = decoded["rows"][0]
    assert set(row) == {
        "row_id",
        "planned_run_id",
        "seed",
        "axis_coordinates",
        "overrides",
        "row_provenance",
        "payload",
    }
    assert row["seed"] == 11
    assert row["axis_coordinates"]["values"]["overrides"] == []


def test_typed_row_lowerer_is_authoritative_and_validation_cannot_mutate(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        TrainingRowLoweringResult(
            execution_payload={},
            lowerer_identities=[],
        )
    matrix_payload = _matrix({"compact": {"gain": 1}}).model_dump(mode="json")
    matrix_payload["rows"][0]["overrides"] = [{"path": "compact.gain", "op": "replace", "value": 2}]
    matrix = TrainingRunMatrixSpec.model_validate(matrix_payload)
    seen = []

    def lower(authored_row, _context):
        seen.append(authored_row)
        return TrainingRowLoweringResult(
            execution_payload={
                "schema_id": "example.execution",
                "schema_version": "example.execution.v1",
                "lowered_gain": authored_row.payload["compact"]["gain"] * 10,
            },
            lowerer_identities=[
                RowLowererIdentity(
                    lowerer_id="example.adapter",
                    lowerer_version="example.adapter.v1",
                ),
                RowLowererIdentity(
                    lowerer_id="example.gain-lowerer",
                    lowerer_version="example.gain-lowerer.v1",
                ),
            ],
        )

    def validate_only(payload, _row_id):
        payload["lowered_gain"] = -1
        return None

    materialized = materialize_adapted_run_matrix(
        matrix,
        repo_root=tmp_path,
        row_lowerer=lower,
        row_validator=validate_only,
    )
    row = materialized.rows[0]
    assert seen[0].payload == {"compact": {"gain": 2}}
    assert seen[0].payload_hash == row.provenance.authored_payload_hash
    assert row.authored_payload == {"compact": {"gain": 2}}
    assert row.payload["lowered_gain"] == 20
    assert row.provenance.planned_run_id == row.planned_run_id
    assert row.provenance.axis_coordinates["run_id"] == row.planned_run_id
    assert row.provenance.lowerer_identities == [
        RowLowererIdentity(
            lowerer_id="example.adapter",
            lowerer_version="example.adapter.v1",
        ),
        RowLowererIdentity(
            lowerer_id="example.gain-lowerer",
            lowerer_version="example.gain-lowerer.v1",
        ),
    ]


def test_planned_id_binds_complete_lowered_payload_and_ordered_authorship(
    tmp_path: Path,
) -> None:
    def materialize(
        authored_value: int,
        lowerer_version: str,
        *,
        unprojected_value: str | float = "baseline",
        graph_value: float | None = None,
        reverse_lowerers: bool = False,
    ):
        def lower(_authored_row, _context):
            lowerer_identities = [
                {
                    "lowerer_id": "example.adapter",
                    "lowerer_version": "example.adapter.v1",
                },
                {
                    "lowerer_id": "example.science",
                    "lowerer_version": lowerer_version,
                },
            ]
            return TrainingRowLoweringResult(
                execution_payload={
                    "schema_id": "example.execution",
                    "schema_version": "example.execution.v1",
                    "constant": True,
                    **({} if graph_value is None else {"graph_spec": {"signed_zero": graph_value}}),
                    # This field is outside the historical graph/training/task projections.
                    "extension_payload": {"value": unprojected_value},
                },
                lowerer_identities=(
                    list(reversed(lowerer_identities)) if reverse_lowerers else lowerer_identities
                ),
            )

        return materialize_adapted_run_matrix(
            _matrix({"authored_value": authored_value}),
            repo_root=tmp_path,
            row_lowerer=lower,
        ).rows[0]

    baseline = materialize(1, "example.science.v1")
    changed_authored = materialize(2, "example.science.v1")
    changed_lowerer = materialize(1, "example.science.v2")
    changed_unprojected_execution = materialize(
        1,
        "example.science.v1",
        unprojected_value="changed",
    )
    reversed_lowerer_order = materialize(
        1,
        "example.science.v1",
        reverse_lowerers=True,
    )
    negative_zero = materialize(
        1,
        "example.science.v1",
        graph_value=-0.0,
    )
    positive_zero = materialize(
        1,
        "example.science.v1",
        graph_value=0.0,
    )

    assert baseline.payload == changed_authored.payload == changed_lowerer.payload
    assert baseline.payload != changed_unprojected_execution.payload
    assert (
        len(
            {
                baseline.planned_run_id,
                changed_authored.planned_run_id,
                changed_lowerer.planned_run_id,
                changed_unprojected_execution.planned_run_id,
                reversed_lowerer_order.planned_run_id,
            }
        )
        == 5
    )
    assert baseline.provenance.lowered_execution_payload_hash == training_spec_sha256(
        baseline.payload
    )
    assert baseline.provenance.lowered_execution_payload_hash != (
        changed_unprojected_execution.provenance.lowered_execution_payload_hash
    )
    negative_zero_artifact = store_canonical_json_artifact(
        negative_zero.payload,
        root=tmp_path / "custody",
        role="compiled_execution_payload",
        logical_name="negative-zero.json",
    )
    positive_zero_artifact = store_canonical_json_artifact(
        positive_zero.payload,
        root=tmp_path / "custody",
        role="compiled_execution_payload",
        logical_name="positive-zero.json",
    )
    assert negative_zero.provenance.lowered_execution_payload_hash == (
        negative_zero_artifact.sha256
    )
    assert positive_zero.provenance.lowered_execution_payload_hash == (
        positive_zero_artifact.sha256
    )
    assert negative_zero_artifact.sha256 == positive_zero_artifact.sha256
    assert negative_zero.planned_run_id == positive_zero.planned_run_id
    assert [identity.lowerer_id for identity in baseline.provenance.lowerer_identities] == [
        "example.adapter",
        "example.science",
    ]


def test_sweep_planned_id_canonicalizes_payload_and_axis_coordinates(
    tmp_path: Path,
) -> None:
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            "name": "signed zero sweep",
            "base": {
                "kind": "inline",
                "inline": {"graph_spec": {"signed_zero": 1.0}},
            },
            "axes": [
                {
                    "id": "signed_zero",
                    "path": "graph_spec.signed_zero",
                    "variation": {"kind": "explicit", "values": [-0.0, 0.0]},
                }
            ],
            "combination": {"mode": "cross"},
        }
    )

    negative_zero, positive_zero = materialize_adapted_run_matrix(
        matrix,
        repo_root=tmp_path,
        row_validator=lambda _payload, _row_id: None,
    ).rows
    negative_zero_artifact = store_canonical_json_artifact(
        negative_zero.payload,
        root=tmp_path / "custody",
        role="compiled_execution_payload",
        logical_name="sweep-negative-zero.json",
    )
    positive_zero_artifact = store_canonical_json_artifact(
        positive_zero.payload,
        root=tmp_path / "custody",
        role="compiled_execution_payload",
        logical_name="sweep-positive-zero.json",
    )

    negative_coordinate = negative_zero.provenance.axis_coordinates["values"]["signed_zero"]
    positive_coordinate = positive_zero.provenance.axis_coordinates["values"]["signed_zero"]
    assert math.copysign(1.0, negative_coordinate) == -1.0
    assert math.copysign(1.0, positive_coordinate) == 1.0
    assert negative_zero.provenance.lowered_execution_payload_hash == (
        negative_zero_artifact.sha256
    )
    assert positive_zero.provenance.lowered_execution_payload_hash == (
        positive_zero_artifact.sha256
    )
    assert negative_zero_artifact.sha256 == positive_zero_artifact.sha256
    assert negative_zero.planned_run_id == positive_zero.planned_run_id


def test_lowered_payload_and_provenance_drive_storage_and_assembly(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot

    matrix = _matrix({"compact": {"gain": 3}})

    def lower(authored_row, _context):
        return TrainingRowLoweringResult(
            execution_payload={
                "schema_id": "example.execution",
                "schema_version": "example.execution.v1",
                "training_config": {"gain": authored_row.payload["compact"]["gain"]},
            },
            lowerer_identities=[
                {
                    "lowerer_id": "example.lowerer",
                    "lowerer_version": "example.lowerer.v2",
                }
            ],
        )

    lock = tmp_path / "uv.lock"
    lock.write_text("lock", encoding="utf-8")
    storage = emit_training_run_spec_storage(
        matrix,
        method_registry=_METHOD_REGISTRY,
        repo_root=tmp_path,
        authored_path=tmp_path / "authored.json",
        custody_root=tmp_path / "custody",
        materializer_commit="abc",
        dependency_lock_path=lock,
        allow_inline_base=True,
        row_lowerer=lower,
        row_validator=lambda _payload, _row_id: None,
    )
    decoded = decode_resolved_snapshot(
        json.loads(Path(storage.snapshot_artifact.uri).read_text(encoding="utf-8"))
    )
    stored_row = decoded["rows"][0]
    assert stored_row["payload"]["training_config"]["gain"] == 3
    assert stored_row["row_provenance"]["authored_payload_hash"]
    assert stored_row["row_provenance"]["lowerer_identities"] == [
        {
            "lowerer_id": "example.lowerer",
            "lowerer_version": "example.lowerer.v2",
        }
    ]

    compiled = compile_training_run_matrix(
        matrix,
        method_registry=_METHOD_REGISTRY,
        run_set_id="run-set",
        context=SimpleNamespace(repo_root=tmp_path, resolved_inputs=()),
        allow_inline_base=True,
        row_lowerer=lower,
        row_validator=lambda _payload, _row_id: None,
    )
    compiled_row = compiled.rows[0]
    assert compiled_row.payload == stored_row["payload"]
    assert compiled_row.resolved_semantics["payload"] == compiled_row.payload
    assert compiled_row.provenance is not None
    assert compiled_row.provenance.planned_run_id == stored_row["planned_run_id"]
    assert compiled_row.launch.collect == [
        "manifest.json",
        "training-diagnostics.json",
        "checkpoints",
        "manifests",
    ]


def test_assembled_bundle_carries_the_same_typed_row_provenance(tmp_path: Path) -> None:
    from feedbax.orchestration.assembly import (
        AssemblyCompilerRegistry,
        AssemblyContext,
        CompilerIdentity,
        RunAssemblyRequest,
        assemble_run_bundle,
    )
    from feedbax.orchestration.bundle import (
        BudgetPolicy,
        EnvironmentDeclaration,
        SchemaArtifactRef,
    )
    from feedbax.training.spec_storage import (
        TRAINING_RUN_MATRIX_COMPILER_ID,
        TRAINING_RUN_MATRIX_COMPILER_VERSION,
        register_training_run_matrix_compiler,
    )

    matrix = _matrix({"compact": {"gain": 4}})
    authored = matrix.model_dump(mode="json", exclude_none=True)
    authored_bytes = training_spec_canonical_bytes(authored)
    authored_path = tmp_path / "matrix.json"
    authored_path.write_bytes(authored_bytes)

    def lower(authored_row, _context):
        return TrainingRowLoweringResult(
            execution_payload={
                "schema_id": "example.execution",
                "schema_version": "example.execution.v1",
                "gain": authored_row.payload["compact"]["gain"],
            },
            lowerer_identities=[
                {
                    "lowerer_id": "example.lowerer",
                    "lowerer_version": "example.lowerer.v1",
                }
            ],
        )

    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            schema_version=TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            artifact_id="fixture:matrix",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
            compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
        ),
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        environment=EnvironmentDeclaration(python_version="3.13"),
        budget=BudgetPolicy(max_wall_clock_seconds=1),
    )
    registry = AssemblyCompilerRegistry()
    register_training_run_matrix_compiler(
        registry,
        method_registry=_METHOD_REGISTRY,
        allow_inline_base=True,
        row_lowerer=lower,
        row_validator=lambda _payload, _row_id: None,
    )
    bundle = assemble_run_bundle(
        request,
        run_set_id="run-set",
        context=AssemblyContext(
            custody_root=tmp_path / "custody",
            repo_root=tmp_path,
            materializer_commit="abc",
            dependency_lock_digest="d" * 64,
        ),
        registry=registry,
    )

    provenance = bundle.rows[0].execution.row_provenance
    assert provenance is not None
    assert provenance.planned_run_id.startswith("feedbax-training-run:")
    assert provenance.lowered_execution_payload_hash == bundle.rows[0].execution.payload.sha256
    assert provenance.lowerer_identities == [
        RowLowererIdentity(
            lowerer_id="example.lowerer",
            lowerer_version="example.lowerer.v1",
        )
    ]
    payload = json.loads(Path(bundle.rows[0].execution.payload.uri).read_text())
    assert payload["gain"] == 4

    mismatched = bundle.rows[0].execution.model_dump(mode="json")
    mismatched["row_provenance"]["lowered_execution_payload_hash"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="lowered_execution_payload_hash does not match",
    ):
        type(bundle.rows[0].execution).model_validate(mismatched)


def test_resolved_output_materializes_and_rejects_tampering(tmp_path: Path) -> None:
    from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot

    snapshot = build_resolved_semantics_snapshot({"value": 3})
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_bytes(training_spec_canonical_bytes(snapshot))
    payload = _matrix({"unused": True}).model_dump(mode="json")
    payload["base"] = {
        "kind": "resolved_output",
        "ref": "snapshot.json",
        "resolved_root_hash": snapshot["root_hash"],
    }
    materialized = materialize_adapted_run_matrix(
        TrainingRunMatrixSpec.model_validate(payload),
        repo_root=tmp_path,
        row_validator=lambda _payload, _row_id: None,
    )
    assert materialized.rows[0].payload == {"value": 3}

    tampered = json.loads(snapshot_path.read_text(encoding="utf-8"))
    scalar_hash = next(
        digest for digest, node in tampered["nodes"].items() if node["type"] == "scalar"
    )
    tampered["nodes"][scalar_hash]["value"] = 4
    snapshot_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        decode_resolved_snapshot(tampered)
    with pytest.raises(ValueError, match="hash mismatch"):
        materialize_adapted_run_matrix(
            TrainingRunMatrixSpec.model_validate(payload),
            repo_root=tmp_path,
            row_validator=lambda _payload, _row_id: None,
        )


def test_custody_reuse_verifies_existing_bytes_and_public_exports(tmp_path: Path) -> None:
    import feedbax.contracts as contracts_exports
    import feedbax.training as training_exports

    value = {"value": 1}
    artifact = store_canonical_json_artifact(
        value,
        root=tmp_path,
        role="test",
        logical_name="test.json",
    )
    Path(artifact.uri).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatched bytes"):
        store_canonical_json_artifact(
            value,
            root=tmp_path,
            role="test",
            logical_name="test.json",
        )
    assert "training_spec_canonical_bytes" in contracts_exports.__all__
    assert "emit_training_run_spec_storage" in training_exports.__all__
    assert "TrainingRowLoweringResult" in contracts_exports.__all__
    assert "TrainingRowProvenance" in training_exports.__all__
    assert training_exports.TrainingRowLoweringResult is TrainingRowLoweringResult
    exported_canonical = getattr(contracts_exports, "training_spec_canonical_bytes")
    training_spec_storage = importlib.import_module("feedbax.training.spec_storage")
    exported_emitter = getattr(training_spec_storage, "emit_training_run_spec_storage")
    assert exported_canonical(value) == training_spec_canonical_bytes(value)
    assert (exported_canonical.__module__, exported_canonical.__qualname__) == (
        training_spec_canonical_bytes.__module__,
        training_spec_canonical_bytes.__qualname__,
    )
    assert (exported_emitter.__module__, exported_emitter.__qualname__) == (
        emit_training_run_spec_storage.__module__,
        emit_training_run_spec_storage.__qualname__,
    )


def test_reference_variants_are_distinct_and_recursive_authored_base_composes(
    tmp_path: Path,
) -> None:
    parent = _matrix({"value": 1}).model_dump(mode="json", exclude_none=True)
    parent_path = tmp_path / "parent.json"
    parent_path.write_bytes(training_spec_canonical_bytes(parent))
    authored_payload = _matrix({"unused": True}).model_dump(mode="json")
    authored_payload["base"] = {
        "kind": "authored_intent",
        "ref": "parent.json",
        "content_hash": training_spec_sha256(parent),
    }
    authored = TrainingRunMatrixSpec.model_validate(authored_payload)
    snapshot = build_resolved_semantics_snapshot({"value": 1})
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    resolved_payload = authored.model_dump(mode="json")
    resolved_payload["base"] = {
        "kind": "resolved_output",
        "ref": "snapshot.json",
        "resolved_root_hash": snapshot["root_hash"],
    }
    resolved = TrainingRunMatrixSpec.model_validate(resolved_payload)
    assert authored.base.kind == "authored_intent"
    assert resolved.base.kind == "resolved_output"

    recursive_parent = parent.copy()
    recursive_parent["base"] = resolved_payload["base"]
    recursive_parent["deltas"] = [
        {
            "layer_id": "parent",
            "patches": [{"path": "value", "op": "replace", "value": 2}],
        }
    ]
    parent_path.write_bytes(training_spec_canonical_bytes(recursive_parent))
    authored_payload = authored.model_dump(mode="json")
    authored_payload["base"]["content_hash"] = training_spec_sha256(recursive_parent)
    recursive = TrainingRunMatrixSpec.model_validate(authored_payload)
    seen: list[dict[str, object]] = []
    materialize_adapted_run_matrix(
        recursive,
        repo_root=tmp_path,
        row_validator=lambda payload, _row_id: seen.append(payload) or None,
    )
    assert seen == [{"value": 2}]


def test_v3_graph_base_is_migrated_before_row_validation(tmp_path: Path) -> None:
    v3_graph = {
        "schema_id": GRAPH_SPEC_SCHEMA_ID,
        "schema_version": GRAPH_SPEC_SCHEMA_VERSION_V3,
        "nodes": {},
        "wires": [],
        "derived_dimensions": [],
    }
    base_document = {"graph": {"kind": "GraphSpec", "inline": v3_graph}}
    (tmp_path / "tracked-v3-base.json").write_bytes(training_spec_canonical_bytes(base_document))
    matrix_payload = _matrix({"unused": True}).model_dump(mode="json")
    matrix_payload["base"] = {
        "kind": "authored_intent",
        "ref": "tracked-v3-base.json",
        "content_hash": training_spec_sha256(base_document),
    }
    matrix = TrainingRunMatrixSpec.model_validate(matrix_payload)
    seen: list[dict[str, object]] = []
    materialize_adapted_run_matrix(
        matrix,
        repo_root=tmp_path,
        row_validator=lambda payload, _row_id: seen.append(payload) or None,
    )
    assert seen[0]["graph"]["inline"]["schema_version"].endswith(".v4")


def test_intent_identity_excludes_symbolic_names_but_retains_pins() -> None:
    first = {
        "name": "display a",
        "base": {
            "kind": "authored_intent",
            "symbolic_name": "a",
            "ref": "mutable-a.json",
            "content_hash": "1",
        },
        "sources": [
            {
                "alias": "data",
                "kind": "manifest",
                "uri": "mutable-a.json",
                "adoption_note": "prose a",
                "payload_query": {"item": "data", "path": "value"},
            }
        ],
        "rows": [{"row_id": "row", "seed": 1, "notes": "a", "metadata": {"x": 1}}],
    }
    second = json.loads(json.dumps(first))
    second["name"] = "display b"
    second["base"]["symbolic_name"] = "b"
    second["base"]["ref"] = "mutable-b.json"
    second["sources"][0]["uri"] = "mutable-b.json"
    second["sources"][0]["adoption_note"] = "prose b"
    second["rows"][0]["notes"] = "b"
    second["rows"][0]["metadata"] = {"x": 2}
    assert training_run_intent_hash(first) == training_run_intent_hash(second)
    second["base"]["content_hash"] = "2"
    assert training_run_intent_hash(first) != training_run_intent_hash(second)
    second = json.loads(json.dumps(first))
    second["sources"][0]["payload_query"]["path"] = "other"
    assert training_run_intent_hash(first) != training_run_intent_hash(second)


def test_axis_authored_parameter_and_run_completion_are_declarative() -> None:
    axis = TrainingSweepAxis.model_validate(
        {
            "id": "objective_lambda",
            "path": "objective.lambda",
            "variation": {"kind": "explicit", "values": [0.2]},
            "authored_parameter": {"name": "beta", "value": 1.4},
        }
    )
    manifest = TrainingRunManifest(
        id="feedbax-training-run:stopped",
        status="cancelled",
        completed_at="2026-07-11T00:00:00Z",
        completed_batches=42,
        stopped=True,
        stop_reason="early-stop policy",
    )
    assert axis.authored_parameter == {"name": "beta", "value": 1.4}
    assert manifest.completed_batches == 42
    assert manifest.stopped
