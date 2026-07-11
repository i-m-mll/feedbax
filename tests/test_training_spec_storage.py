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
from feedbax.contracts.manifest import TrainingRunManifest, TrainingSweepAxis
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
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
    RunMatrixError,
    materialize_adapted_run_matrix,
)
from feedbax.training.spec_storage import (
    emit_training_run_spec_storage,
    stamp_training_run_manifest_identities,
)


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
        Path(__file__).parents[1]
        / "feedbax"
        / "contracts"
        / "resolved_snapshot_decoder.py"
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
            repo_root=tmp_path,
            authored_path=tmp_path / "tracked" / "rejected.json",
            custody_root=tmp_path / "custody",
            materializer_commit="abc",
            dependency_lock_path=lock,
        )
    first = emit_training_run_spec_storage(
        matrix,
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
        "payload",
    }
    assert row["seed"] == 11
    assert row["axis_coordinates"]["values"]["overrides"] == []


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
    (tmp_path / "tracked-v3-base.json").write_bytes(
        training_spec_canonical_bytes(base_document)
    )
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
