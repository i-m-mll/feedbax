"""Destination verification for the manifest artifact storage helpers.

`store_artifact` and `store_json_artifact` must never return an `ArtifactRef`
whose digest and size describe bytes other than the ones actually published at
the canonical content-addressed path. These tests pin the write-side
time-of-check/time-of-use case, the already-existing-destination case, and the
unchanged reference shape for good inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from feedbax.contracts import manifest as manifest_module
from feedbax.contracts.manifest import (
    ArtifactStoreIntegrityError,
    _artifact_path,
    store_artifact,
    store_json_artifact,
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_store_artifact_returns_verified_identity_of_published_bytes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    source = tmp_path / "figure.png"
    data = b"\x89PNG\r\n\x1a\n binary payload"
    source.write_bytes(data)

    artifact = store_artifact(source, root=root, role="figure", media_type="image/png")

    destination = _artifact_path(root, _digest(data), ".png")
    assert Path(artifact.uri or "") == destination
    assert destination.read_bytes() == data
    assert artifact.sha256 == _digest(data)
    assert artifact.size_bytes == len(data)
    assert artifact.role == "figure"
    assert artifact.logical_name == "figure.png"
    assert artifact.artifact_id == f"artifact://sha256/{_digest(data)}"
    assert artifact.media_type == "image/png"
    assert artifact.storage_backend == "feedbax-local"
    assert artifact.metadata == {
        "original_uri": str(source),
        "relative_path": str(destination.relative_to(root)),
    }


def test_store_artifact_is_idempotent_and_stable_for_repeated_stores(tmp_path: Path) -> None:
    root = tmp_path / "store"
    source = tmp_path / "payload.bin"
    source.write_bytes(b"stable bytes")

    first = store_artifact(source, root=root, role="payload", logical_name="named")
    second = store_artifact(source, root=root, role="payload", logical_name="named")

    assert first == second
    assert Path(first.uri or "").read_bytes() == b"stable bytes"


def test_store_artifact_streams_payloads_larger_than_one_chunk(tmp_path: Path) -> None:
    root = tmp_path / "store"
    source = tmp_path / "large.bin"
    data = (b"feedbax-artifact-chunk" * 60_000)[: manifest_module._ARTIFACT_STREAM_CHUNK_BYTES + 7]
    source.write_bytes(data)

    artifact = store_artifact(source, root=root, role="payload")

    assert artifact.sha256 == _digest(data)
    assert artifact.size_bytes == len(data)
    assert Path(artifact.uri or "").read_bytes() == data


def test_store_artifact_refuses_a_source_that_changes_after_naming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy must not publish bytes that differ from the recorded digest."""
    root = tmp_path / "store"
    source = tmp_path / "payload.bin"
    original = b"original bytes"
    source.write_bytes(original)
    real_identity = manifest_module._file_content_identity

    def identity_then_tamper(path: Path | str) -> tuple[str, int]:
        identity = real_identity(path)
        Path(path).write_bytes(b"tampered bytes of a different length")
        return identity

    monkeypatch.setattr(manifest_module, "_file_content_identity", identity_then_tamper)

    with pytest.raises(ArtifactStoreIntegrityError, match="source bytes changed during store"):
        store_artifact(source, root=root, role="payload")

    assert not _artifact_path(root, _digest(original), ".bin").exists()


def test_store_artifact_refuses_existing_canonical_file_with_wrong_bytes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    source = tmp_path / "payload.bin"
    intended = b"right bytes"
    source.write_bytes(intended)
    destination = _artifact_path(root, _digest(intended), ".bin")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong bytes")

    with pytest.raises(ArtifactStoreIntegrityError, match="do not match content identity"):
        store_artifact(source, root=root, role="payload")

    assert destination.read_bytes() == b"wrong bytes"


def test_store_artifact_refuses_existing_canonical_file_with_matching_size(
    tmp_path: Path,
) -> None:
    """Equal size is not equal content; the digest comparison must still refuse."""
    root = tmp_path / "store"
    source = tmp_path / "payload.bin"
    intended = b"aaaaaaaa"
    source.write_bytes(intended)
    destination = _artifact_path(root, _digest(intended), ".bin")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"bbbbbbbb")

    with pytest.raises(ArtifactStoreIntegrityError, match="do not match content identity"):
        store_artifact(source, root=root, role="payload")

    assert destination.read_bytes() == b"bbbbbbbb"


def test_store_json_artifact_returns_verified_identity_of_published_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    value = {"b": 2, "a": [1, 2, 3]}
    data = _json_bytes(value)

    artifact = store_json_artifact(value, root=root, role="report", logical_name="report.json")

    destination = _artifact_path(root, _digest(data), ".json")
    assert Path(artifact.uri or "") == destination
    assert destination.read_bytes() == data
    assert artifact.sha256 == _digest(data)
    assert artifact.size_bytes == len(data)
    assert artifact.role == "report"
    assert artifact.logical_name == "report.json"
    assert artifact.artifact_id == f"artifact://sha256/{_digest(data)}"
    assert artifact.media_type == "application/json"
    assert artifact.storage_backend == "feedbax-local"
    assert artifact.metadata == {"relative_path": str(destination.relative_to(root))}


def test_store_json_artifact_preserves_caller_metadata_and_is_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    value = {"kind": "analysis"}

    first = store_json_artifact(
        value,
        root=root,
        role="analysis",
        logical_name="analysis.json",
        metadata={"note": "first"},
    )
    second = store_json_artifact(
        value,
        root=root,
        role="analysis",
        logical_name="analysis.json",
        metadata={"note": "first"},
    )

    assert first == second
    assert first.metadata["note"] == "first"
    assert Path(first.uri or "").read_bytes() == _json_bytes(value)


def test_store_json_artifact_refuses_existing_canonical_file_with_wrong_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    value = {"a": 1}
    data = _json_bytes(value)
    destination = _artifact_path(root, _digest(data), ".json")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'{"a": 2}\n')

    with pytest.raises(ArtifactStoreIntegrityError, match="do not match content identity"):
        store_json_artifact(value, root=root, role="report", logical_name="report.json")

    assert destination.read_bytes() == b'{"a": 2}\n'


def test_store_json_artifact_refuses_truncated_existing_canonical_file(tmp_path: Path) -> None:
    root = tmp_path / "store"
    value = {"a": 1}
    data = _json_bytes(value)
    destination = _artifact_path(root, _digest(data), ".json")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(data[:-1])

    with pytest.raises(ArtifactStoreIntegrityError, match="do not match content identity"):
        store_json_artifact(value, root=root, role="report", logical_name="report.json")

    assert destination.read_bytes() == data[:-1]


def test_store_artifact_still_reports_a_missing_source_as_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store_artifact(tmp_path / "absent.bin", root=tmp_path / "store", role="payload")
