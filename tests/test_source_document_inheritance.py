"""Focused tests for content-pinned sub-document inheritance in source documents.

A source document consumed via ``SourceBinding`` may declare, under the reserved
``__inherit__`` key, that a subtree is inherited content-pinned from another file
(whole-file digest, then ``payload_path`` sub-document selection). The inheritance
is materialized into the effective document before any dotted-path query runs, so
a consumer no longer has to physically duplicate the shared subtree.

These tests cover effective-document equivalence with a physically-duplicated
equivalent, each fail-closed case (digest mismatch, missing ``payload_path``,
local/inherited key collision), byte-for-byte unchanged behavior when no
inheritance is declared, and resolution ordering (inheritance before queries).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.contracts.expressions import ValueQuery
from feedbax.contracts.extraction import SourceBinding, load_expression_context
from feedbax.contracts.manifest import canonical_json_bytes, sha256_bytes
from feedbax.contracts.matrix_core import (
    SOURCE_DOCUMENT_INHERITANCE_KEY,
    materialize_inherited_document,
)


_PARENT = {
    "schema_id": "example.parent",
    "harmonized_task": {
        "target_support": {"kind": "disk", "radius": 3, "values": [1, 2, 3]},
        "variants": [{"id": "a"}, {"id": "b"}],
    },
    "unused": {"ignored": True},
}


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> str:
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return sha256_bytes(canonical_json_bytes(payload))


def _child_with_inheritance(sha: str) -> dict[str, Any]:
    """A source document that inherits ``harmonized_task`` from the parent file."""
    return {
        "schema_id": "example.child",
        "native_baseline": {"lambda": 0.5},
        SOURCE_DOCUMENT_INHERITANCE_KEY: {
            "schema_id": "feedbax.spec.source_document_inheritance",
            "schema_version": "feedbax.spec.source_document_inheritance.v1",
            "inherit": [
                {
                    "target": "harmonized_task",
                    "parent": {
                        "ref": "parent.json",
                        "sha256": sha,
                        "payload_path": ["harmonized_task"],
                    },
                }
            ],
        },
    }


def _duplicated_equivalent() -> dict[str, Any]:
    """The physically-duplicated document the inheritance must reproduce."""
    return {
        "schema_id": "example.child",
        "native_baseline": {"lambda": 0.5},
        "harmonized_task": _PARENT["harmonized_task"],
    }


def _strip_provenance(document: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in document.items() if k != SOURCE_DOCUMENT_INHERITANCE_KEY}


def _list_child(sha: str, target: str = "trace_families.0.index") -> dict[str, Any]:
    return {
        "trace_families": [{"name": "family"}],
        SOURCE_DOCUMENT_INHERITANCE_KEY: {
            "schema_id": "feedbax.spec.source_document_inheritance",
            "schema_version": "feedbax.spec.source_document_inheritance.v1",
            "inherit": [
                {
                    "target": target,
                    "parent": {
                        "ref": "parent.json",
                        "sha256": sha,
                        "payload_path": ["harmonized_task", "target_support"],
                    },
                }
            ],
        },
    }


def test_effective_document_matches_physically_duplicated_equivalent(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    effective = materialize_inherited_document(
        _child_with_inheritance(sha), repo_root=tmp_path
    )
    assert _strip_provenance(effective) == _duplicated_equivalent()


def test_absent_declaration_is_byte_for_byte_unchanged(tmp_path: Path) -> None:
    document = {"schema_id": "example.child", "harmonized_task": {"kind": "disk"}}
    result = materialize_inherited_document(document, repo_root=tmp_path)
    assert result == document
    assert result is document  # untouched documents are returned as-is


def test_non_mapping_document_is_returned_unchanged(tmp_path: Path) -> None:
    assert materialize_inherited_document([1, 2, 3], repo_root=tmp_path) == [1, 2, 3]


def test_provenance_records_pinned_parent_and_pointer(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    effective = materialize_inherited_document(
        _child_with_inheritance(sha), repo_root=tmp_path
    )
    provenance = effective[SOURCE_DOCUMENT_INHERITANCE_KEY]["inherit"]
    assert provenance == [
        {
            "target": "harmonized_task",
            "parent": {
                "ref": "parent.json",
                "sha256": sha,
                "payload_path": ["harmonized_task"],
            },
        }
    ]


def test_canonical_list_index_target_preserves_pin_and_lineage(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    child = _list_child(sha)
    declaration = canonical_json_bytes(child[SOURCE_DOCUMENT_INHERITANCE_KEY])

    effective = materialize_inherited_document(child, repo_root=tmp_path)

    assert effective["trace_families"][0]["index"] == _PARENT["harmonized_task"][
        "target_support"
    ]
    assert canonical_json_bytes(effective[SOURCE_DOCUMENT_INHERITANCE_KEY]) == declaration
    assert effective[SOURCE_DOCUMENT_INHERITANCE_KEY]["inherit"][0]["parent"][
        "sha256"
    ] == sha


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("trace_families.00.index", "canonical non-negative decimal array index"),
        ("trace_families.-1.index", "canonical non-negative decimal array index"),
        ("trace_families.1.index", "array index out of range for length 1"),
    ],
)
def test_invalid_list_index_target_fails_closed(
    tmp_path: Path,
    target: str,
    message: str,
) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    with pytest.raises(ValueError, match=message):
        materialize_inherited_document(_list_child(sha, target), repo_root=tmp_path)


def test_list_target_traversal_into_scalar_fails_closed(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    child = _list_child(sha, "trace_families.0.name.value.index")
    with pytest.raises(ValueError, match="traverses scalar segment"):
        materialize_inherited_document(child, repo_root=tmp_path)


def test_list_target_cannot_overwrite_existing_leaf(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    child = _list_child(sha)
    child["trace_families"][0]["index"] = {"local": True}
    with pytest.raises(ValueError, match="collides with a locally-present key"):
        materialize_inherited_document(child, repo_root=tmp_path)


def test_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, "parent.json", _PARENT)
    with pytest.raises(ValueError, match="hash mismatch"):
        materialize_inherited_document(
            _child_with_inheritance("a" * 64), repo_root=tmp_path
        )


def test_missing_payload_path_fails_closed(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    child = _child_with_inheritance(sha)
    child[SOURCE_DOCUMENT_INHERITANCE_KEY]["inherit"][0]["parent"]["payload_path"] = [
        "harmonized_task",
        "absent",
    ]
    with pytest.raises(ValueError, match="missing object key"):
        materialize_inherited_document(child, repo_root=tmp_path)


def test_collision_with_local_key_fails_closed(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    child = _child_with_inheritance(sha)
    child["harmonized_task"] = {"locally": "present"}  # already present locally
    with pytest.raises(ValueError, match="collides with a locally-present key"):
        materialize_inherited_document(child, repo_root=tmp_path)


def test_duplicate_targets_rejected(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    child = _child_with_inheritance(sha)
    entry = child[SOURCE_DOCUMENT_INHERITANCE_KEY]["inherit"][0]
    child[SOURCE_DOCUMENT_INHERITANCE_KEY]["inherit"].append(dict(entry))
    with pytest.raises(ValueError, match="targets must be unique"):
        materialize_inherited_document(child, repo_root=tmp_path)


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    child = _child_with_inheritance(sha)
    child[SOURCE_DOCUMENT_INHERITANCE_KEY]["schema_version"] = "unknown.v9"
    with pytest.raises(ValueError, match="schema_version"):
        materialize_inherited_document(child, repo_root=tmp_path)


def test_inheritance_resolves_before_dotted_path_queries(tmp_path: Path) -> None:
    """A binding query over an inherited path resolves: inheritance runs first."""
    sha = _write(tmp_path, "parent.json", _PARENT)
    (tmp_path / "child.json").write_text(
        json.dumps(_child_with_inheritance(sha)), encoding="utf-8"
    )
    binding = SourceBinding(
        alias="sisu",
        kind="lock",
        uri="child.json",
        payload_query=ValueQuery(item="source", path="harmonized_task.target_support.values"),
    )
    context = load_expression_context([binding], tmp_path)
    assert context.items["sisu"].payload == [1, 2, 3]


def test_binding_without_query_exposes_effective_inherited_subtree(tmp_path: Path) -> None:
    sha = _write(tmp_path, "parent.json", _PARENT)
    (tmp_path / "child.json").write_text(
        json.dumps(_child_with_inheritance(sha)), encoding="utf-8"
    )
    binding = SourceBinding(alias="sisu", kind="lock", uri="child.json")
    context = load_expression_context([binding], tmp_path)
    payload = context.items["sisu"].payload
    assert _strip_provenance(payload) == _duplicated_equivalent()
