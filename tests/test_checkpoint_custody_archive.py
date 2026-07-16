from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from pydantic import ValidationError

import feedbax.training.checkpoint_custody as custody
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training import (
    CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
    CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_ID,
    CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_VERSION,
    CheckpointReferenceResolutionError,
    produce_checkpoint_custody_archive,
)
from tests.test_checkpoint_custody import (
    _resolver_parent_ref,
    _rewrite_manifest_and_latest,
    _slot_blob_path,
    _write_json,
    _write_resolver_checkpoint,
)


def _setup(tmp_path: Path):
    source = tmp_path / "source"
    result = _write_resolver_checkpoint(source)
    ref = _resolver_parent_ref(result)
    provider = ImmutableArtifactBlobProvider(tmp_path / "artifacts")
    return result, ref, provider


def test_checkpoint_custody_archive_is_canonical_and_byte_identical(tmp_path: Path) -> None:
    result, ref, provider = _setup(tmp_path)
    first = produce_checkpoint_custody_archive(
        ref, allowed_root=result.root, artifact_provider=provider
    )
    second = produce_checkpoint_custody_archive(
        ref, allowed_root=result.root, artifact_provider=provider
    )
    first_bytes = provider.get_bytes(first.artifact_ref)
    assert first_bytes == provider.get_bytes(second.artifact_ref)
    assert first_bytes[3] & 0x08 == 0
    assert first_bytes[4:8] == b"\0\0\0\0"

    with tarfile.open(fileobj=io.BytesIO(first_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        expected = [
            "archive.json",
            "checkpoint/latest.json",
            f"checkpoint/{ref.uri}",
            *[
                f"checkpoint/{Path(ref.uri).parent.as_posix()}/{slot.relative_path}"
                for slot in result.manifest.slots
            ],
        ]
        assert names == expected
        assert all(member.isfile() for member in members)
        assert all(
            (member.mode, member.uid, member.gid, member.mtime, member.uname, member.gname)
            == (0o644, 0, 0, 0, "", "")
            for member in members
        )
        metadata = json.load(archive.extractfile("archive.json"))
        payload_bytes = sum(member.size for member in members[1:])
        assert {
            member.name: archive.extractfile(member).read() for member in members[1:]
        } == {
            "checkpoint/latest.json": result.latest_pointer_path.read_bytes(),
            f"checkpoint/{ref.uri}": result.manifest_path.read_bytes(),
            **{
                expected_name: _slot_blob_path(result.manifest_path, slot.slot).read_bytes()
                for expected_name, slot in zip(expected[3:], result.manifest.slots, strict=True)
            },
        }

    assert metadata == {
        "schema_id": CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_ID,
        "schema_version": CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_VERSION,
        "media_type": CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
        "parent_ref": ref.model_dump(mode="json", exclude_none=True),
        "transaction_root_sha256": (
            result.manifest.content_integrity_digest.transaction_root_sha256
        ),
        "payload_member_count": len(expected) - 1,
        "expanded_payload_size_bytes": payload_bytes,
    }
    assert first.artifact_ref.media_type == first.evidence.media_type
    assert first.artifact_ref.sha256 == first.evidence.archive_sha256
    assert first.artifact_ref.size_bytes == first.evidence.archive_size_bytes == len(first_bytes)
    assert first.evidence.payload_member_count == len(expected) - 1
    assert first.evidence.expanded_payload_size_bytes == payload_bytes
    with pytest.raises(ValidationError):
        first.artifact_ref.role = "changed"
    with pytest.raises(ValidationError):
        first.evidence.parent_ref.id = "changed"


def test_checkpoint_custody_archive_rejects_stale_latest_without_storage(
    tmp_path: Path,
) -> None:
    result, ref, provider = _setup(tmp_path)
    payload = json.loads(result.latest_pointer_path.read_text())
    payload["transaction_id"] = "stale-transaction"
    _write_json(result.latest_pointer_path, payload)

    with pytest.raises(CheckpointReferenceResolutionError, match="latest pointer"):
        produce_checkpoint_custody_archive(
            ref, allowed_root=result.root, artifact_provider=provider
        )
    assert not provider.root.exists()


def test_checkpoint_custody_archive_rechecks_blob_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, ref, provider = _setup(tmp_path)
    blob_path = _slot_blob_path(result.manifest_path, result.manifest.slots[0].slot)
    original_resolver = custody.resolve_checkpoint_custody_ref

    def resolve_then_replace(*args: object, **kwargs: object):
        resolved = original_resolver(*args, **kwargs)
        blob_path.write_bytes(b"replaced after resolution")
        return resolved

    monkeypatch.setattr(custody, "resolve_checkpoint_custody_ref", resolve_then_replace)
    with pytest.raises(CheckpointReferenceResolutionError, match="size mismatch|hash mismatch"):
        produce_checkpoint_custody_archive(
            ref, allowed_root=result.root, artifact_provider=provider
        )
    assert not provider.root.exists()


def test_checkpoint_custody_archive_rejects_unsafe_member_path(
    tmp_path: Path,
) -> None:
    result, _, provider = _setup(tmp_path)
    payload = json.loads(result.manifest_path.read_text())
    slot = payload["slots"][0]
    slot["relative_path"] = slot["relative_path"].replace("/", "//", 1)
    _rewrite_manifest_and_latest(result, payload)
    ref = _resolver_parent_ref(result)

    with pytest.raises(CheckpointReferenceResolutionError, match="canonical relative path"):
        produce_checkpoint_custody_archive(
            ref, allowed_root=result.root, artifact_provider=provider
        )
    assert not provider.root.exists()


def test_checkpoint_custody_archive_rejects_blob_change_before_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, ref, provider = _setup(tmp_path)
    blob_path = _slot_blob_path(result.manifest_path, result.manifest.slots[0].slot)
    encoder = custody._checkpoint_custody_archive_bytes

    def encode_then_change(*args: object, **kwargs: object) -> bytes:
        data = encoder(*args, **kwargs)
        blob_path.write_bytes(b"changed after archive assembly")
        return data

    monkeypatch.setattr(custody, "_checkpoint_custody_archive_bytes", encode_then_change)
    with pytest.raises(CheckpointReferenceResolutionError, match="changed before storage"):
        produce_checkpoint_custody_archive(
            ref, allowed_root=result.root, artifact_provider=provider
        )
    assert not provider.root.exists()
