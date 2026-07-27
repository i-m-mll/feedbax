import json
from pathlib import Path

import pytest

from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    CompositionDelta,
    CompositionNode,
    InlineIntentParent,
    authored_envelope_hash,
)
from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.training.run_matrix import (
    RunMatrixError,
    materialize_adapted_run_matrix,
    resolve_base_payload_with_attribution,
)


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
