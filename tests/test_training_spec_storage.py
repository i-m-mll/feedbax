from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.contracts.graph import GRAPH_SPEC_SCHEMA_ID, GRAPH_SPEC_SCHEMA_VERSION_V3
from tests.support.training_fixtures import synthetic_training_program_registry
from feedbax.contracts.manifest import TrainingRunManifest, TrainingSweepAxis
from feedbax.contracts.run_composition import ResolvedOutputParent
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    RowLowererIdentity,
    TrainingRowParentProvenance,
    TrainingRowLoweringResult,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.spec_storage import (
    build_resolved_semantics_snapshot,
    training_run_intent_hash,
    training_spec_canonical_bytes,
    training_spec_sha256,
    store_canonical_json_artifact,
)
from feedbax.training.run_matrix import (
    materialize_adapted_run_matrix,
)
from feedbax.training.row_lowering import (
    GovernedTrainingRowParent,
    TrainingRowLoweringContext,
)
from feedbax.training.spec_storage import (
    compile_training_run_matrix,
)
from feedbax.orchestration.bundle import DeploymentPolicy
from feedbax.orchestration.revision import resolve_feedbax_revision

_METHOD_REGISTRY = synthetic_training_program_registry()


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


def test_lowered_payload_and_provenance_drive_assembly(tmp_path: Path) -> None:
    from types import SimpleNamespace


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
    assert compiled_row.payload["training_config"]["gain"] == 3
    assert compiled_row.resolved_semantics["payload"] == compiled_row.payload
    assert compiled_row.provenance is not None
    assert compiled_row.provenance.authored_payload_hash
    assert compiled_row.provenance.lowerer_identities == [
        RowLowererIdentity(lowerer_id="example.lowerer", lowerer_version="example.lowerer.v2")
    ]
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
        feedbax_revision=resolve_feedbax_revision(),
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
    from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
    from feedbax.contracts.spec_storage import TrainingRunExecutionCapsule

    execution = bundle.rows[0].execution
    snapshot = json.loads(Path(execution.resolved_snapshot.uri).read_bytes())
    assert decode_resolved_snapshot(snapshot)["payload"] == payload
    capsule = TrainingRunExecutionCapsule.model_validate_json(
        Path(execution.execution_capsule.uri).read_bytes()
    )
    assert capsule.execution_hash == execution.execution_capsule.execution_hash
    assert capsule.resolved_root_hash == execution.resolved_snapshot.root_hash
    with pytest.raises(ValidationError, match="materializer drift"):
        TrainingRunExecutionCapsule.model_validate(
            {**capsule.model_dump(mode="json"), "execution_hash": "0" * 64}
        )

    mismatched = bundle.rows[0].execution.model_dump(mode="json")
    mismatched["row_provenance"]["lowered_execution_payload_hash"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="lowered_execution_payload_hash does not match",
    ):
        type(bundle.rows[0].execution).model_validate(mismatched)


def test_resolved_output_materializes_and_rejects_tampering(tmp_path: Path) -> None:
    from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot

    resolved_payload = {"value": 3}
    snapshot = build_resolved_semantics_snapshot(resolved_payload)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_bytes(training_spec_canonical_bytes(snapshot))
    parent = ResolvedOutputParent(
        ref="snapshot.json",
        resolved_root_hash=training_spec_sha256(resolved_payload),
        row_id="source-row",
        checkpoint_transaction_id="source-transaction",
    )
    context = TrainingRowLoweringContext(
        (
            GovernedTrainingRowParent(
                provenance=TrainingRowParentProvenance(
                    role="terminal",
                    parent_kind="resolved_output",
                    ref=parent.ref,
                    semantic_hash=parent.resolved_root_hash,
                    artifact_id="terminal-artifact",
                    artifact_sha256=training_spec_sha256(resolved_payload),
                    schema_id="example.intent",
                    schema_version="example.intent.v1",
                ),
                parent=parent,
                payload=resolved_payload,
            ),
        )
    )
    payload = _matrix({"unused": True}).model_dump(mode="json")
    payload["base"] = {
        "kind": "resolved_output",
        "ref": parent.ref,
        "resolved_root_hash": parent.resolved_root_hash,
        "row_id": parent.row_id,
        "checkpoint_transaction_id": parent.checkpoint_transaction_id,
    }
    materialized = materialize_adapted_run_matrix(
        TrainingRunMatrixSpec.model_validate(payload),
        repo_root=tmp_path,
        row_validator=lambda _payload, _row_id: None,
        row_lowering_context=context,
    )
    assert materialized.rows[0].payload == resolved_payload

    tampered = json.loads(snapshot_path.read_text(encoding="utf-8"))
    scalar_hash = next(
        digest for digest, node in tampered["nodes"].items() if node["type"] == "scalar"
    )
    tampered["nodes"][scalar_hash]["value"] = 4
    snapshot_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        decode_resolved_snapshot(tampered)
    with pytest.raises(ValueError, match="semantic hash drifted"):
        materialize_adapted_run_matrix(
            TrainingRunMatrixSpec.model_validate(payload),
            repo_root=tmp_path,
            row_validator=lambda _payload, _row_id: None,
            row_lowering_context=TrainingRowLoweringContext(
                (
                    GovernedTrainingRowParent(
                        provenance=context.parents[0].provenance,
                        parent=parent,
                        payload={"value": 4},
                    ),
                )
            ),
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
    assert "TrainingRowLoweringResult" in contracts_exports.__all__
    assert "TrainingRowProvenance" in training_exports.__all__
    assert training_exports.TrainingRowLoweringResult is TrainingRowLoweringResult
    exported_canonical = getattr(contracts_exports, "training_spec_canonical_bytes")
    assert exported_canonical(value) == training_spec_canonical_bytes(value)
    assert (exported_canonical.__module__, exported_canonical.__qualname__) == (
        training_spec_canonical_bytes.__module__,
        training_spec_canonical_bytes.__qualname__,
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
    resolved_payload = {"value": 1}
    parent_ref = ResolvedOutputParent(
        ref="snapshot.json",
        resolved_root_hash=training_spec_sha256(resolved_payload),
        row_id="source-row",
        checkpoint_transaction_id="source-transaction",
    )
    resolved_payload = authored.model_dump(mode="json")
    resolved_payload["base"] = {
        "kind": "resolved_output",
        "ref": parent_ref.ref,
        "resolved_root_hash": parent_ref.resolved_root_hash,
        "row_id": parent_ref.row_id,
        "checkpoint_transaction_id": parent_ref.checkpoint_transaction_id,
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
        row_lowering_context=TrainingRowLoweringContext(
            (
                GovernedTrainingRowParent(
                    provenance=TrainingRowParentProvenance(
                        role="terminal",
                        parent_kind="resolved_output",
                        ref=parent_ref.ref,
                        semantic_hash=parent_ref.resolved_root_hash,
                        artifact_id="terminal-artifact",
                        artifact_sha256=parent_ref.resolved_root_hash,
                        schema_id="example.intent",
                        schema_version="example.intent.v1",
                    ),
                    parent=parent_ref,
                    payload={"value": 1},
                ),
            )
        ),
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
    assert seen[0]["graph"]["inline"]["schema_version"].endswith(".v6")


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
