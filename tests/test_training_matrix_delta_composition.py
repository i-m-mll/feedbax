from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import feedbax.contracts as public_contracts
import feedbax.contracts.training_matrix_composition as training_composition
from feedbax.contracts.manifest import (
    TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.training_matrix_composition import (
    TrainingRunMatrixDeltaSpec,
    flatten_training_run_matrix_delta,
    training_matrix_delta_envelope_hash,
)
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    CompilerIdentity,
    RunAssemblyRequest,
    assemble_run_bundle,
)
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    DeploymentPolicy,
    EnvironmentDeclaration,
    SchemaArtifactRef,
)
from feedbax.orchestration.conformance import _validate_training_matrix_delta_parent_chain
from feedbax.integrations.provider import provider_manifest
from feedbax.training.spec_storage import (
    TRAINING_RUN_MATRIX_COMPILER_ID,
    TRAINING_RUN_MATRIX_COMPILER_VERSION,
    compile_training_run_matrix,
    emit_training_run_spec_storage,
    register_training_run_matrix_compiler,
)
from feedbax.orchestration.revision import resolve_feedbax_revision


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> str:
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return sha256_bytes(canonical_json_bytes(payload))


def _parent() -> dict[str, Any]:
    return {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "discrete",
        "base": {
            "kind": "inline",
            "inline": {
                "schema_id": "example.training",
                "schema_version": "example.training.v1",
                "positive_arm": "discrete",
            },
        },
        "rows": [{"row_id": "discrete", "label": "Discrete"}],
        "tags": ["discrete"],
    }


def _child(parent_sha: str, *, layer_id: str = "continuous") -> dict[str, Any]:
    return {
        "schema_id": TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
        "parent": {"ref": "parent.json", "sha256": parent_sha},
        "deltas": [
            {
                "layer_id": layer_id,
                "patches": [
                    {"path": "name", "value": "continuous"},
                    {"path": "base.inline.positive_arm", "value": "continuous"},
                    {"path": "rows.0.row_id", "value": "continuous"},
                    {"path": "rows.0.label", "value": "Continuous"},
                    {"path": "tags.0", "value": "continuous"},
                ],
            }
        ],
    }


def test_matrix_level_delta_flattens_and_compiles_without_restating_parent(
    tmp_path: Path, application_registry_bundle
) -> None:
    parent_sha = _write(tmp_path, "parent.json", _parent())
    child = _child(parent_sha)
    spec = TrainingRunMatrixDeltaSpec.model_validate(child)

    flattened = flatten_training_run_matrix_delta(spec, repo_root=tmp_path)
    compiled = compile_training_run_matrix(
        child,
        run_set_id="example",
        context=SimpleNamespace(
            repo_root=tmp_path,
            resolved_inputs=(),
            training_row_lowering_context=None,
        ),
        method_registry=application_registry_bundle.training_methods,
        allow_inline_base=True,
        row_validator=lambda _payload, _row_id: None,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )

    assert set(child) == {"schema_id", "schema_version", "parent", "deltas"}
    assert flattened.authored_envelope_sha256 == training_matrix_delta_envelope_hash(spec)
    assert flattened.root_matrix.parent_sha256 == parent_sha
    assert flattened.attribution == {
        "name": "continuous",
        "base.inline.positive_arm": "continuous",
        "rows.0.row_id": "continuous",
        "rows.0.label": "continuous",
        "tags.0": "continuous",
    }
    assert flattened.payload["name"] == "continuous"
    assert flattened.payload["rows"][0]["row_id"] == "continuous"
    assert flattened.payload["rows"][0]["label"] == "Continuous"
    assert [row.row_id for row in compiled.rows] == ["continuous"]
    assert compiled.rows[0].payload["positive_arm"] == "continuous"


def test_nested_delta_requires_explicit_ancestor_override_acknowledgement(
    tmp_path: Path,
) -> None:
    parent_sha = _write(tmp_path, "parent.json", _parent())
    middle = _child(parent_sha, layer_id="middle")
    middle_sha = _write(tmp_path, "middle.json", middle)
    leaf = {
        "schema_id": TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
        "parent": {"ref": "middle.json", "sha256": middle_sha},
        "deltas": [
            {
                "layer_id": "leaf",
                "patches": [{"path": "name", "value": "leaf"}],
            }
        ],
    }

    with pytest.raises(ValueError, match="without explicit acknowledgement"):
        flatten_training_run_matrix_delta(
            TrainingRunMatrixDeltaSpec.model_validate(leaf), repo_root=tmp_path
        )

    leaf["deltas"][0]["acknowledges_ancestor_paths"] = ["name"]
    flattened = flatten_training_run_matrix_delta(
        TrainingRunMatrixDeltaSpec.model_validate(leaf), repo_root=tmp_path
    )
    assert [layer.layer_ids for layer in flattened.layers] == [["middle"], ["leaf"]]
    assert flattened.payload["name"] == "leaf"


def test_parent_resolution_and_terminal_validation_fail_closed(tmp_path: Path) -> None:
    parent = _parent()
    parent_sha = _write(tmp_path, "parent.json", parent)
    child = _child(parent_sha)

    parent["rows"] = []
    _write(tmp_path, "parent.json", parent)
    with pytest.raises(ValueError, match="hash mismatch"):
        flatten_training_run_matrix_delta(
            TrainingRunMatrixDeltaSpec.model_validate(child), repo_root=tmp_path
        )

    parent_sha = _write(tmp_path, "parent.json", _parent())
    escaping = _child(parent_sha)
    escaping["parent"]["ref"] = "../parent.json"
    with pytest.raises(ValueError, match="escapes repo_root"):
        flatten_training_run_matrix_delta(
            TrainingRunMatrixDeltaSpec.model_validate(escaping),
            repo_root=tmp_path / "nested",
        )

    invalid = _child(parent_sha)
    invalid["deltas"][0]["patches"][2]["value"] = "bad/id"
    with pytest.raises((ValueError, ValidationError), match="path-safe"):
        flatten_training_run_matrix_delta(
            TrainingRunMatrixDeltaSpec.model_validate(invalid), repo_root=tmp_path
        )


def test_delta_schema_is_registered_and_rejects_unknown_versions() -> None:
    family = next(
        family
        for family in default_spec_registry.families()
        if family.identity == TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID
    )
    assert family.current_version == TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION

    unsupported = _child("0" * 64)
    unsupported["schema_version"] = "feedbax.spec.training_run_matrix_delta.v0"
    with pytest.raises(ValidationError, match="unsupported TrainingRunMatrixDeltaSpec"):
        TrainingRunMatrixDeltaSpec.model_validate(unsupported)


def test_optional_metadata_carrier_accepts_absence_round_trips_and_stays_strict() -> None:
    """``metadata`` is an additive optional field on the current delta spec version.

    The migration story is additive-optional: documents emitted before the field
    existed omit it entirely and still validate, defaulting to an empty mapping, so
    no versioned migration rule is required and the schema version is unchanged.
    """
    without_metadata = _child("0" * 64)
    assert "metadata" not in without_metadata

    baseline = TrainingRunMatrixDeltaSpec.model_validate(without_metadata)
    assert baseline.metadata == {}

    annotated = TrainingRunMatrixDeltaSpec.model_validate(
        {**without_metadata, "metadata": {"authoring_status": "draft"}}
    )
    assert annotated.metadata == {"authoring_status": "draft"}
    dumped = annotated.model_dump(mode="json", exclude_none=True)
    assert dumped["metadata"] == {"authoring_status": "draft"}
    assert TrainingRunMatrixDeltaSpec.model_validate(dumped).metadata == annotated.metadata
    # The carrier is not part of the authored envelope, so pinned hashes are unaffected.
    assert training_matrix_delta_envelope_hash(annotated) == training_matrix_delta_envelope_hash(
        baseline
    )

    with pytest.raises(ValidationError):
        TrainingRunMatrixDeltaSpec.model_validate({**without_metadata, "unexpected": 1})


def test_repeated_parent_document_is_rejected_as_a_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_sha = _write(tmp_path, "parent.json", _parent())
    child = _child(parent_sha)
    monkeypatch.setattr(
        training_composition,
        "load_content_pinned_json_base",
        lambda _base, *, repo_root: dict(child),
    )

    with pytest.raises(ValueError, match="cycle detected"):
        flatten_training_run_matrix_delta(
            TrainingRunMatrixDeltaSpec.model_validate(child), repo_root=tmp_path
        )


def test_assembly_dispatch_retains_delta_artifact_and_authored_identity(
    tmp_path: Path, application_registry_bundle
) -> None:
    parent_sha = _write(tmp_path, "parent.json", _parent())
    child = _child(parent_sha)
    child_bytes = canonical_json_bytes(child)
    child_path = tmp_path / "child.json"
    child_path.write_bytes(child_bytes)
    request = RunAssemblyRequest(
        feedbax_revision=resolve_feedbax_revision(),
        authored=SchemaArtifactRef(
            schema_id=TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
            schema_version=TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
            artifact_id="artifact://training-matrix/delta",
            sha256=hashlib.sha256(child_bytes).hexdigest(),
            uri=str(child_path),
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
        budget=BudgetPolicy(max_wall_clock_seconds=60),
    )
    registry = AssemblyCompilerRegistry()
    register_training_run_matrix_compiler(
        registry,
        method_registry=application_registry_bundle.training_methods,
        allow_inline_base=True,
        row_validator=lambda _payload, _row_id: None,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )

    bundle = assemble_run_bundle(
        request,
        run_set_id="delta",
        context=AssemblyContext(custody_root=tmp_path / "custody", repo_root=tmp_path),
        registry=registry,
    )

    authored = bundle.rows[0].execution.authored_intent
    assert authored.schema_id == TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID
    assert authored.intent_hash == training_matrix_delta_envelope_hash(
        TrainingRunMatrixDeltaSpec.model_validate(child)
    )
    capsule_path = Path(bundle.rows[0].execution.execution_capsule.uri or "")
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    assert (
        capsule["relevant_schema_versions"]["training_run_matrix_delta"]
        == TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION
    )

    _validate_training_matrix_delta_parent_chain(authored, child, repo_root=tmp_path)
    with pytest.raises(ValueError, match="requires authored_repo_root"):
        _validate_training_matrix_delta_parent_chain(authored, child, repo_root=None)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside authored_repo_root"):
        _validate_training_matrix_delta_parent_chain(authored, child, repo_root=outside)
    parent = _parent()
    parent["name"] = "tampered"
    (tmp_path / "parent.json").write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="parent chain cannot be validated"):
        _validate_training_matrix_delta_parent_chain(authored, child, repo_root=tmp_path)


def test_public_emitter_and_schema_discovery_preserve_delta_authority(
    tmp_path: Path, application_registry_bundle
) -> None:
    parent_sha = _write(tmp_path, "parent.json", _parent())
    child = _child(parent_sha)
    lock = tmp_path / "uv.lock"
    lock.write_text("locked", encoding="utf-8")
    authored_path = tmp_path / "authored-delta.json"

    storage = emit_training_run_spec_storage(
        child,
        repo_root=tmp_path,
        authored_path=authored_path,
        custody_root=tmp_path / "custody",
        materializer_commit="abc",
        dependency_lock_path=lock,
        method_registry=application_registry_bundle.training_methods,
        allow_inline_base=True,
        row_validator=lambda _payload, _row_id: None,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )

    assert json.loads(authored_path.read_text(encoding="utf-8")) == (
        TrainingRunMatrixDeltaSpec.model_validate(child).model_dump(
            mode="json", exclude_none=True
        )
    )
    assert storage.intent_hash == training_matrix_delta_envelope_hash(
        TrainingRunMatrixDeltaSpec.model_validate(child)
    )
    assert (
        storage.capsule.relevant_schema_versions["training_run_matrix_delta"]
        == TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION
    )
    assert public_contracts.TrainingRunMatrixDeltaSpec is TrainingRunMatrixDeltaSpec
    assert "TrainingRunMatrixDeltaSpec" in provider_manifest().schemas
