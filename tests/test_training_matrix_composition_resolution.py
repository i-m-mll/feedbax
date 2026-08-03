import json
from pathlib import Path

import pytest

from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    CompiledTrainingRowParent,
    CompositionDelta,
    CompositionNode,
    CompositionNodeV2,
    InlineIntentParent,
    ResolvedOutputParent,
    authored_envelope_hash,
    composition_identity_projection,
    parse_composition_node,
)
from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.experiment_compile_lock import (
    CompileLockInputs,
    CompilerContract,
    CompilerImplementation,
    ContentPinReference,
    build_compile_lock,
)
from feedbax.contracts.run_matrix import (
    TrainingRowParentProvenance,
    TrainingRunMatrixSpec,
    apply_composition_deltas,
)
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.training.run_matrix import (
    RunMatrixError,
    materialize_adapted_run_matrix,
    _resolve_compiled_training_row_parent,
    resolve_base_payload_with_attribution,
    _verify_compiled_training_row_parent,
)
from feedbax.training.row_lowering import (
    GovernedTrainingRowParent,
    TrainingRowLoweringContext,
)
from feedbax.envelope.compile import _composition_root_pins


def _replace(
    layer_id: str,
    path: str,
    value: object,
    *,
    acknowledge: bool = False,
) -> CompositionDelta:
    return CompositionDelta(
        layer_id=layer_id,
        patches=[{"op": "replace", "path": path, "value": value}],
        acknowledges_ancestor_paths=[path] if acknowledge else [],
    )


def _write_node(path: Path, node: CompositionNode) -> dict[str, object]:
    document = node.model_dump(mode="json", exclude_none=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return document


def _matrix(child_document: dict[str, object]) -> TrainingRunMatrixSpec:
    return TrainingRunMatrixSpec.model_validate(
        {
            "name": "composed-matrix",
            "base": {
                "kind": "authored_intent",
                "ref": "child.json",
                "content_hash": training_spec_sha256(child_document),
            },
            "deltas": [
                {
                    "layer_id": "matrix",
                    "patches": [{"op": "replace", "path": "width", "value": 32}],
                    "acknowledges_ancestor_paths": ["width"],
                }
            ],
            "rows": [
                {
                    "row_id": "row",
                    "overrides": [{"op": "replace", "path": "gain", "value": 4}],
                }
            ],
        }
    )


def _resolved_context(
    parent: ResolvedOutputParent, payload: dict[str, object]
) -> TrainingRowLoweringContext:
    return TrainingRowLoweringContext(
        (
            GovernedTrainingRowParent(
                provenance=TrainingRowParentProvenance(
                    role="terminal",
                    parent_kind="resolved_output",
                    ref=parent.ref,
                    semantic_hash=parent.resolved_root_hash,
                    artifact_id="terminal-artifact",
                    artifact_sha256=training_spec_sha256(payload),
                    schema_id="example.intent",
                    schema_version="example.intent.v1",
                ),
                parent=parent,
                payload=payload,
            ),
        )
    )


def _write_compiled_matrix(
    root: Path,
    *,
    base: dict[str, object] | None = None,
    references: list[ContentPinReference] | None = None,
) -> CompiledTrainingRowParent:
    matrix_base = (
        {
            "kind": "inline",
            "inline": {
                "schema_id": "example.intent",
                "schema_version": "example.intent.v1",
                "gain": 1,
                "width": 8,
            },
        }
        if base is None
        else base
    )
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "name": "mapped-composition",
            "base": matrix_base,
            "deltas": [
                {
                    "layer_id": "mapped",
                    "patches": [{"op": "replace", "path": "width", "value": 16}],
                }
            ],
            "rows": [
                {
                    "row_id": "mapped-row",
                    "overrides": [{"op": "replace", "path": "gain", "value": 4}],
                }
            ],
        }
    )
    matrix_document = matrix.model_dump(mode="json", exclude_none=True)
    lock = build_compile_lock(
        CompileLockInputs(
            envelope_ref="specs/mapped.envelope.json",
            envelope_document={
                "schema": "feedbax.experiment_envelope.v3",
                "name": "mapped",
            },
            envelope_schema="feedbax.experiment_envelope.v3",
            name="mapped",
            family="training_run_matrix",
            compiled_document=matrix_document,
            contract=CompilerContract(
                "feedbax.experiment_envelope.compiler",
                "feedbax.experiment_envelope.compiler.v2",
            ),
            implementation=CompilerImplementation(
                code_unit="tests.test_training_matrix_composition_resolution"
            ),
            references=references or [],
            identity_contributions={
                "training_root": {
                    "kind": "composition",
                    "rows": [{"id": "mapped-row"}],
                }
            },
        )
    )
    (root / "mapped.training_run_matrix.json").write_text(
        json.dumps(matrix_document), encoding="utf-8"
    )
    (root / "mapped.compile-lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return CompiledTrainingRowParent(
        matrix={
            "ref": "mapped.training_run_matrix.json",
            "sha256": canonical_sha256(matrix_document),
        },
        compile_lock={
            "ref": "mapped.compile-lock.json",
            "sha256": canonical_sha256(lock),
        },
        row_id="mapped-row",
        symbolic_name="mapped locator",
    )


def test_matrix_flattens_canonical_child_and_envelope_pinned_parent_before_rows(
    tmp_path: Path,
) -> None:
    root = CompositionNode(
        name="root",
        parent=InlineIntentParent(
            payload={
                "schema_id": "example.intent",
                "schema_version": "example.intent.v1",
                "depth": 0,
                "gain": 1,
                "width": 8,
            },
            schema_id="example.intent",
            schema_version="example.intent.v1",
        ),
        deltas=[_replace("root", "gain", 2)],
    )
    root_document = _write_node(tmp_path / "root.json", root)
    middle = CompositionNode(
        name="middle",
        parent=AuthoredIntentParent(
            ref="root.json",
            content_hash=authored_envelope_hash(root),
        ),
        deltas=[_replace("middle", "width", 16)],
    )
    _write_node(tmp_path / "middle.json", middle)
    child = CompositionNode(
        name="child",
        parent=AuthoredIntentParent(
            ref="middle.json",
            content_hash=authored_envelope_hash(middle),
        ),
        deltas=[_replace("child", "depth", 1)],
    )
    child_document = _write_node(tmp_path / "child.json", child)

    child_canonical_hash = training_spec_sha256(child_document)
    root_envelope_hash = authored_envelope_hash(root)
    assert child_canonical_hash != root_envelope_hash
    assert training_spec_sha256(root_document) != root_envelope_hash

    matrix = _matrix(child_document)
    payload, attribution = resolve_base_payload_with_attribution(
        matrix,
        repo_root=tmp_path,
    )
    assert payload == {
        "schema_id": "example.intent",
        "schema_version": "example.intent.v1",
        "depth": 1,
        "gain": 2,
        "width": 32,
    }
    assert attribution == {"depth": "child", "gain": "root", "width": "matrix"}

    materialized = materialize_adapted_run_matrix(
        matrix,
        repo_root=tmp_path,
        row_validator=lambda _payload, _row_id: None,
    )
    assert materialized.rows[0].authored_payload["gain"] == 4
    assert materialized.rows[0].authored_payload["width"] == 32


def test_matrix_rejects_child_canonical_document_drift(tmp_path: Path) -> None:
    child = CompositionNode(
        name="child",
        parent=InlineIntentParent(
            payload={"value": 1},
            schema_id="example.intent",
            schema_version="example.intent.v1",
        ),
    )
    child_document = _write_node(tmp_path / "child.json", child)
    matrix_payload = _matrix(child_document).model_dump(mode="json", exclude_none=True)
    matrix_payload["base"]["content_hash"] = "0" * 64
    matrix = TrainingRunMatrixSpec.model_validate(matrix_payload)

    with pytest.raises(RunMatrixError, match="canonical content hash mismatch"):
        resolve_base_payload_with_attribution(matrix, repo_root=tmp_path)


def test_matrix_rejects_parent_authored_envelope_drift(tmp_path: Path) -> None:
    parent = CompositionNode(
        name="parent",
        parent=InlineIntentParent(
            payload={"gain": 1, "width": 8},
            schema_id="example.intent",
            schema_version="example.intent.v1",
        ),
    )
    _write_node(tmp_path / "parent.json", parent)
    child = CompositionNode(
        name="child",
        parent=AuthoredIntentParent(
            ref="parent.json",
            content_hash=authored_envelope_hash(parent),
        ),
    )
    child_document = _write_node(tmp_path / "child.json", child)
    drifted_parent = parent.model_copy(update={"deltas": [_replace("drift", "gain", 2)]})
    _write_node(tmp_path / "parent.json", drifted_parent)

    with pytest.raises(RunMatrixError, match="/parent/content_hash mismatch"):
        resolve_base_payload_with_attribution(
            _matrix(child_document),
            repo_root=tmp_path,
        )


def test_matrix_rejects_composition_source_reference_cycle(tmp_path: Path) -> None:
    child = CompositionNode(
        name="child",
        parent=AuthoredIntentParent(
            ref="child.json",
            content_hash="0" * 64,
        ),
    )
    child_document = _write_node(tmp_path / "child.json", child)

    with pytest.raises(RunMatrixError, match="authored composition cycle"):
        resolve_base_payload_with_attribution(
            _matrix(child_document),
            repo_root=tmp_path,
        )


def test_resolved_output_base_requires_governed_custody_and_applies_rows_after_deltas(
    tmp_path: Path,
) -> None:
    terminal = {
        "schema_id": "example.intent",
        "schema_version": "example.intent.v1",
        "gain": 1,
        "width": 8,
    }
    parent = ResolvedOutputParent(
        ref="artifact-blob:terminal",
        resolved_root_hash=training_spec_sha256(terminal),
        row_id="source-row",
        checkpoint_transaction_id="source-transaction",
    )
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "name": "resolved-terminal",
            "base": {
                "kind": "resolved_output",
                "ref": parent.ref,
                "resolved_root_hash": parent.resolved_root_hash,
                "row_id": parent.row_id,
                "checkpoint_transaction_id": parent.checkpoint_transaction_id,
            },
            "deltas": [
                {
                    "layer_id": "matrix",
                    "patches": [_replace("unused", "width", 16).patches[0]],
                }
            ],
            "rows": [
                {
                    "row_id": "condition-a",
                    "overrides": [{"op": "replace", "path": "gain", "value": 4}],
                }
            ],
        }
    )

    with pytest.raises(RunMatrixError, match="no governed lowering custody"):
        materialize_adapted_run_matrix(
            matrix,
            repo_root=tmp_path,
            row_validator=lambda _payload, _row_id: None,
        )

    materialized = materialize_adapted_run_matrix(
        matrix,
        repo_root=tmp_path,
        row_validator=lambda _payload, _row_id: None,
        row_lowering_context=_resolved_context(parent, terminal),
    )
    assert materialized.rows[0].authored_payload["width"] == 16
    assert materialized.rows[0].authored_payload["gain"] == 4


def test_authored_composition_uses_governed_resolved_terminal_without_repo_loading(
    tmp_path: Path,
) -> None:
    terminal = {
        "schema_id": "example.intent",
        "schema_version": "example.intent.v1",
        "gain": 1,
        "width": 8,
    }
    parent = ResolvedOutputParent(
        ref="artifact-blob:terminal",
        resolved_root_hash=training_spec_sha256(terminal),
    )
    child = CompositionNode(
        name="resolved-child",
        parent=parent,
        deltas=[_replace("child", "gain", 2)],
    )
    child_document = _write_node(tmp_path / "child.json", child)
    matrix = _matrix(child_document)

    with pytest.raises(RunMatrixError, match="no governed lowering custody"):
        resolve_base_payload_with_attribution(matrix, repo_root=tmp_path)

    materialized = materialize_adapted_run_matrix(
        matrix,
        repo_root=tmp_path,
        row_validator=lambda _payload, _row_id: None,
        row_lowering_context=_resolved_context(parent, terminal),
    )
    assert materialized.rows[0].authored_payload == {
        "schema_id": "example.intent",
        "schema_version": "example.intent.v1",
        "gain": 4,
        "width": 32,
    }


def test_v2_compiled_row_parent_materializes_exact_effective_payload(tmp_path: Path) -> None:
    parent = _write_compiled_matrix(tmp_path)
    direct = _resolve_compiled_training_row_parent(parent, repo_root=tmp_path)
    node = CompositionNodeV2(
        name="frozen-mapped-row",
        parent=parent,
        deltas=[_replace("child", "gain", 5, acknowledge=True)],
    )
    document = node.model_dump(mode="json", exclude_none=True)
    (tmp_path / "child.json").write_text(json.dumps(document), encoding="utf-8")
    outer = TrainingRunMatrixSpec.model_validate(
        {
            "name": "outer",
            "base": {
                "kind": "authored_intent",
                "ref": "child.json",
                "content_hash": training_spec_sha256(document),
            },
            "rows": [{"row_id": "outer-row"}],
        }
    )

    materialized = materialize_adapted_run_matrix(
        outer,
        repo_root=tmp_path,
        row_validator=lambda _payload, _row_id: None,
    )

    expected = {
        "schema_id": "example.intent",
        "schema_version": "example.intent.v1",
        "gain": 5,
        "width": 16,
    }
    assert direct == {**expected, "gain": 4}
    assert materialized.rows[0].authored_payload == expected

    normalized_direct = {**direct, "graph": {"metadata": {}, "schema_id": None}}
    normalized_selected = {
        **_resolve_compiled_training_row_parent(parent, repo_root=tmp_path),
        "graph": {"metadata": {}, "schema_id": None},
    }
    assert normalized_selected == normalized_direct
    unchanged_damage = _replace("applied-damage", "gain", 3)
    assert apply_composition_deltas(normalized_selected, [unchanged_damage])[0] == (
        apply_composition_deltas(normalized_direct, [unchanged_damage])[0]
    )


def test_v2_compile_verifies_matrix_and_lock_without_resolving_output_custody(
    tmp_path: Path,
) -> None:
    terminal = {
        "schema_id": "example.intent",
        "schema_version": "example.intent.v1",
        "gain": 1,
        "width": 8,
    }
    parent = _write_compiled_matrix(
        tmp_path,
        base={
            "kind": "resolved_output",
            "ref": "artifact-blob:terminal",
            "resolved_root_hash": training_spec_sha256(terminal),
            "row_id": "source-row",
            "checkpoint_transaction_id": "source-transaction",
        },
    )
    node = CompositionNodeV2(name="compiled-row", parent=parent)
    node_document = node.model_dump(mode="json", exclude_none=True)
    (tmp_path / "compiled-row.json").write_text(json.dumps(node_document), encoding="utf-8")

    pins = _composition_root_pins(
        tmp_path,
        AuthoredIntentParent(
            ref="compiled-row.json",
            content_hash=authored_envelope_hash(node),
        ),
        field="training.root.parent",
    )

    assert [pin.ref for pin in pins] == [
        "compiled-row.json",
        "mapped.training_run_matrix.json",
        "mapped.compile-lock.json",
    ]


def test_v2_identity_projection_has_one_hash_domain_and_ignores_locators(tmp_path: Path) -> None:
    parent = _write_compiled_matrix(tmp_path)
    node = CompositionNodeV2(name="first", parent=parent)
    relocated = node.model_copy(
        update={
            "name": "second",
            "parent": parent.model_copy(
                update={
                    "symbolic_name": "other locator",
                    "matrix": parent.matrix.model_copy(update={"ref": "elsewhere.json"}),
                    "compile_lock": parent.compile_lock.model_copy(
                        update={"ref": "elsewhere.lock.json"}
                    ),
                }
            ),
        }
    )
    changed_row = node.model_copy(update={"parent": parent.model_copy(update={"row_id": "b"})})
    changed_pin = node.model_copy(
        update={
            "parent": parent.model_copy(
                update={"matrix": parent.matrix.model_copy(update={"sha256": "a" * 64})}
            )
        }
    )

    assert authored_envelope_hash(node) == authored_envelope_hash(relocated)
    assert authored_envelope_hash(node) != authored_envelope_hash(changed_row)
    assert authored_envelope_hash(node) != authored_envelope_hash(changed_pin)
    assert composition_identity_projection(node)["parent"] == {
        "kind": "compiled_training_row",
        "matrix": {
            "sha256": parent.matrix.sha256,
            "pin_algorithm": "canonical_json_v1",
        },
        "compile_lock": {
            "sha256": parent.compile_lock.sha256,
            "pin_algorithm": "canonical_json_v1",
        },
        "row_id": "mapped-row",
    }


def test_v1_parent_union_and_bytes_remain_closed_and_unchanged() -> None:
    document = {
        "schema_id": "feedbax.spec.training_run_composition",
        "schema_version": "feedbax.spec.training_run_composition.v1",
        "name": "v1-golden",
        "parent": {
            "kind": "inline",
            "payload": {"schema_id": "example.intent", "schema_version": "example.intent.v1"},
            "schema_id": "example.intent",
            "schema_version": "example.intent.v1",
        },
        "deltas": [],
        "sources": [],
        "selectors": [],
        "seeds": [],
    }
    parsed = parse_composition_node(document)
    assert isinstance(parsed, CompositionNode)
    assert parsed.model_dump(mode="json", exclude_none=True) == document
    invalid = {**document, "parent": {"kind": "compiled_training_row"}}
    with pytest.raises(ValueError, match="union_tag_invalid"):
        CompositionNode.model_validate(invalid)
    with pytest.raises(ValueError, match="unsupported.*schema_version"):
        parse_composition_node({**document, "schema_version": "feedbax.spec.training_run_composition.v3"})


def test_compiled_row_parent_refuses_pin_lock_row_and_reference_drift(tmp_path: Path) -> None:
    source = {"schema_id": "example.source", "schema_version": "example.source.v1"}
    (tmp_path / "source.json").write_text(json.dumps(source), encoding="utf-8")
    reference = ContentPinReference(
        ref="source.json",
        content_hash=canonical_sha256(source),
    )
    parent = _write_compiled_matrix(tmp_path, references=[reference])
    _verify_compiled_training_row_parent(parent, repo_root=tmp_path)

    with pytest.raises(RunMatrixError, match="matrix.*pin mismatch"):
        _verify_compiled_training_row_parent(
            parent.model_copy(
                update={"matrix": parent.matrix.model_copy(update={"sha256": "0" * 64})}
            ),
            repo_root=tmp_path,
        )
    with pytest.raises(RunMatrixError, match="row_id is absent"):
        _resolve_compiled_training_row_parent(
            parent.model_copy(update={"row_id": "absent"}),
            repo_root=tmp_path,
        )

    lock_path = tmp_path / parent.compile_lock.ref
    original_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    mismatched_lock = json.loads(json.dumps(original_lock))
    mismatched_lock["compiled_document"]["content_hash"] = "f" * 64
    contribution = mismatched_lock["identity_contributions"]["training_root"]
    mismatched_lock["execution_identity"]["sha256"] = canonical_sha256(
        {
            "compiled_document": "f" * 64,
            "training_root": canonical_sha256(contribution),
        }
    )
    lock_path.write_text(json.dumps(mismatched_lock), encoding="utf-8")
    mismatched_parent = parent.model_copy(
        update={
            "compile_lock": parent.compile_lock.model_copy(
                update={"sha256": canonical_sha256(mismatched_lock)}
            )
        }
    )
    with pytest.raises(RunMatrixError, match="content_hash does not match matrix"):
        _verify_compiled_training_row_parent(mismatched_parent, repo_root=tmp_path)
    lock_path.write_text(json.dumps(original_lock), encoding="utf-8")

    source["changed"] = True
    (tmp_path / "source.json").write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(RunMatrixError, match="references/0.*pin mismatch"):
        _verify_compiled_training_row_parent(parent, repo_root=tmp_path)


def test_compiled_row_parent_requires_governed_resolved_output_custody(tmp_path: Path) -> None:
    terminal = {
        "schema_id": "example.intent",
        "schema_version": "example.intent.v1",
        "gain": 1,
        "width": 8,
    }
    resolved = ResolvedOutputParent(
        ref="artifact-blob:terminal",
        resolved_root_hash=training_spec_sha256(terminal),
        row_id="source-row",
        checkpoint_transaction_id="source-transaction",
    )
    parent = _write_compiled_matrix(
        tmp_path,
        base={
            "kind": "resolved_output",
            "ref": resolved.ref,
            "resolved_root_hash": resolved.resolved_root_hash,
            "row_id": resolved.row_id,
            "checkpoint_transaction_id": resolved.checkpoint_transaction_id,
            "symbolic_name": "locator-only",
        },
    )
    with pytest.raises(RunMatrixError, match="no governed lowering custody"):
        _resolve_compiled_training_row_parent(parent, repo_root=tmp_path)
    assert _resolve_compiled_training_row_parent(
        parent,
        repo_root=tmp_path,
        row_lowering_context=_resolved_context(resolved, terminal),
    )["gain"] == 4
