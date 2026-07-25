"""Focused tests for ``ContentPinnedJsonBase`` sub-document selection.

The pin ``sha256`` always covers the whole referenced file; ``payload_path`` is a
JSON-pointer-lite selector applied strictly after whole-file hash verification.
These tests cover object-key and array-index selection, every fail-closed error
case, and byte-for-byte unchanged behavior when ``payload_path`` is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.contracts.manifest import canonical_json_bytes, sha256_bytes
from feedbax.contracts.matrix_core import (
    ContentPinnedJsonBase,
    load_content_pinned_json_base,
)


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> str:
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return sha256_bytes(canonical_json_bytes(payload))


_WRAPPER = {
    "analysis": {"aggregation": "mean"},
    "harmonized_task": {
        "target_support": {"kind": "disk", "radius": 3},
        "variants": [
            {"id": "a", "value": 1},
            {"id": "b", "value": 2},
        ],
    },
}


def test_absent_payload_path_returns_the_whole_document(tmp_path: Path) -> None:
    sha = _write(tmp_path, "doc.json", _WRAPPER)
    base = ContentPinnedJsonBase(ref="doc.json", sha256=sha)
    assert base.payload_path is None
    assert load_content_pinned_json_base(base, repo_root=tmp_path) == _WRAPPER


def test_absent_payload_path_is_byte_for_byte_unchanged_dump() -> None:
    """An absent selector never adds a ``payload_path`` key to identity dumps."""
    base = ContentPinnedJsonBase(ref="doc.json", sha256="0" * 64)
    dumped = base.model_dump(mode="json", exclude_none=True)
    assert dumped == {"ref": "doc.json", "sha256": "0" * 64}


def test_object_key_selection(tmp_path: Path) -> None:
    sha = _write(tmp_path, "doc.json", _WRAPPER)
    base = ContentPinnedJsonBase(
        ref="doc.json", sha256=sha, payload_path=("harmonized_task", "target_support")
    )
    assert load_content_pinned_json_base(base, repo_root=tmp_path) == {
        "kind": "disk",
        "radius": 3,
    }


def test_array_index_selection(tmp_path: Path) -> None:
    sha = _write(tmp_path, "doc.json", _WRAPPER)
    base = ContentPinnedJsonBase(
        ref="doc.json", sha256=sha, payload_path=("harmonized_task", "variants", "1")
    )
    assert load_content_pinned_json_base(base, repo_root=tmp_path) == {"id": "b", "value": 2}


def test_whole_file_hash_still_covers_the_full_document(tmp_path: Path) -> None:
    """A selector does not narrow what the pin verifies: mismatch fails closed."""
    _write(tmp_path, "doc.json", _WRAPPER)
    base = ContentPinnedJsonBase(
        ref="doc.json", sha256="a" * 64, payload_path=("harmonized_task",)
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_content_pinned_json_base(base, repo_root=tmp_path)


def test_missing_object_key_fails_closed(tmp_path: Path) -> None:
    sha = _write(tmp_path, "doc.json", _WRAPPER)
    base = ContentPinnedJsonBase(
        ref="doc.json", sha256=sha, payload_path=("harmonized_task", "absent")
    )
    with pytest.raises(ValueError, match="missing object key"):
        load_content_pinned_json_base(base, repo_root=tmp_path)


def test_array_index_out_of_range_fails_closed(tmp_path: Path) -> None:
    sha = _write(tmp_path, "doc.json", _WRAPPER)
    base = ContentPinnedJsonBase(
        ref="doc.json", sha256=sha, payload_path=("harmonized_task", "variants", "9")
    )
    with pytest.raises(ValueError, match="out of range"):
        load_content_pinned_json_base(base, repo_root=tmp_path)


@pytest.mark.parametrize("segment", ["01", "-1", "1.0", "+1", " 1", "0x1"])
def test_malformed_array_index_segment_fails_closed(tmp_path: Path, segment: str) -> None:
    sha = _write(tmp_path, "doc.json", _WRAPPER)
    base = ContentPinnedJsonBase(
        ref="doc.json", sha256=sha, payload_path=("harmonized_task", "variants", segment)
    )
    with pytest.raises(ValueError, match="canonical non-negative decimal integer"):
        load_content_pinned_json_base(base, repo_root=tmp_path)


def test_traversal_into_scalar_fails_closed(tmp_path: Path) -> None:
    sha = _write(tmp_path, "doc.json", _WRAPPER)
    base = ContentPinnedJsonBase(
        ref="doc.json",
        sha256=sha,
        payload_path=("harmonized_task", "target_support", "kind", "deeper"),
    )
    with pytest.raises(ValueError, match="cannot traverse into str value"):
        load_content_pinned_json_base(base, repo_root=tmp_path)


def test_selection_of_non_object_where_object_required_fails_closed(tmp_path: Path) -> None:
    """A selector resolving to a list/scalar cannot satisfy the object contract."""
    sha = _write(tmp_path, "doc.json", _WRAPPER)
    array_base = ContentPinnedJsonBase(
        ref="doc.json", sha256=sha, payload_path=("harmonized_task", "variants")
    )
    with pytest.raises(ValueError, match="must select a JSON object"):
        load_content_pinned_json_base(array_base, repo_root=tmp_path)
    scalar_base = ContentPinnedJsonBase(
        ref="doc.json", sha256=sha, payload_path=("harmonized_task", "target_support", "radius")
    )
    with pytest.raises(ValueError, match="must select a JSON object"):
        load_content_pinned_json_base(scalar_base, repo_root=tmp_path)


def test_empty_payload_path_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError, match="non-empty sequence"):
        ContentPinnedJsonBase(ref="doc.json", sha256="0" * 64, payload_path=())


def test_empty_segment_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError, match="non-empty strings"):
        ContentPinnedJsonBase(ref="doc.json", sha256="0" * 64, payload_path=("ok", ""))
