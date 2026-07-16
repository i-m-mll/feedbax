from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from pydantic import ValidationError

import feedbax.training.checkpoint_custody as custody
from feedbax.contracts.checkpoints import TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION_V2
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training import (
    CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
    CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_ID,
    CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_VERSION,
    CheckpointReferenceResolutionError,
    materialize_checkpoint_custody_archive,
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


def _produce(tmp_path: Path):
    result, ref, provider = _setup(tmp_path)
    produced = produce_checkpoint_custody_archive(
        ref, allowed_root=result.root, artifact_provider=provider
    )
    transaction_root = result.manifest.content_integrity_digest.transaction_root_sha256
    return result, ref, provider, produced, transaction_root


def _materialize(tmp_path: Path):
    result, ref, provider, produced, transaction_root = _produce(tmp_path)
    destination = tmp_path / "materialized"
    materialized = materialize_checkpoint_custody_archive(
        provider,
        produced.artifact_ref,
        destination,
        expected_parent_ref=ref,
        expected_transaction_root_sha256=transaction_root,
    )
    return result, ref, provider, produced, materialized, destination


def test_materialize_checkpoint_custody_archive_round_trip(tmp_path: Path) -> None:
    result, ref, _, produced, materialized, destination = _materialize(tmp_path)

    assert destination.joinpath("latest.json").read_bytes() == result.latest_pointer_path.read_bytes()
    assert destination.joinpath(ref.uri).read_bytes() == result.manifest_path.read_bytes()
    assert materialized.artifact_ref == produced.artifact_ref
    assert materialized.archive_evidence == produced.evidence
    assert materialized.destination == destination
    assert materialized.manifest_sha256 == ref.metadata["manifest_sha256"]
    assert materialized.resolved_transaction.parent_ref == ref
    with pytest.raises(ValidationError):
        materialized.resolved_transaction.parent_ref.id = "changed"
    with pytest.raises(TypeError):
        materialized.resolved_transaction.slots["extra"] = object()


def test_materialize_authenticates_blob_before_tar_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    blob = provider._canonical_path(produced.artifact_ref.sha256)
    blob.write_bytes(b"not the authenticated archive")
    opened = False

    def observe_open(*args: object, **kwargs: object):
        nonlocal opened
        opened = True
        raise AssertionError("tar parsing must not occur")

    monkeypatch.setattr(tarfile, "open", observe_open)
    with pytest.raises(Exception, match="size mismatch|sha256 mismatch"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert not opened


@pytest.mark.parametrize("destination_kind", ["directory", "file", "symlink"])
def test_materialize_rejects_existing_destination_without_touching_it(
    tmp_path: Path, destination_kind: str
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    destination = tmp_path / "destination"
    foreign = tmp_path / "foreign"
    if destination_kind == "directory":
        destination.mkdir()
        foreign = destination / "foreign"
        foreign.write_text("keep")
    elif destination_kind == "file":
        destination.write_text("keep")
        foreign = destination
    else:
        foreign.write_text("keep")
        destination.symlink_to(foreign)

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            destination,
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert foreign.read_text() == "keep"
    if destination_kind == "symlink":
        assert destination.is_symlink()
    assert not list(tmp_path.glob(".destination.checkpoint-archive-*"))


def test_materialize_preserves_publication_race_winner_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    destination = tmp_path / "destination"

    def lose_publication_race(*args: object, **kwargs: object) -> None:
        destination.mkdir()
        destination.joinpath("winner").write_text("foreign")
        raise FileExistsError("destination won publication race")

    monkeypatch.setattr(custody, "_publish_directory_no_replace", lose_publication_race)
    with pytest.raises(FileExistsError, match="publication race"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            destination,
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert destination.joinpath("winner").read_text() == "foreign"
    assert not list(tmp_path.glob(".destination.checkpoint-archive-*"))


@pytest.mark.parametrize("identity", ["parent", "transaction-root"])
def test_materialize_rejects_expected_identity_mismatch(
    tmp_path: Path, identity: str
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    expected_ref = ref.model_copy(update={"id": "wrong"}) if identity == "parent" else ref
    expected_root = "0" * 64 if identity == "transaction-root" else transaction_root

    with pytest.raises(CheckpointReferenceResolutionError, match="document identity"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            tmp_path / "destination",
            expected_parent_ref=expected_ref,
            expected_transaction_root_sha256=expected_root,
        )
    assert not (tmp_path / "destination").exists()
    assert not list(tmp_path.glob(".destination.checkpoint-archive-*"))


def _rewritten_archive(
    provider: ImmutableArtifactBlobProvider,
    produced: object,
    rewrite: callable,
):
    original = provider.get_bytes(produced.artifact_ref)
    members: list[tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(original), mode="r:gz") as archive:
        for member in archive.getmembers():
            members.append((member, archive.extractfile(member).read()))
    rewrite(members)
    output = io.BytesIO()
    archive_format = tarfile.USTAR_FORMAT
    if any(member.pax_headers for member, _ in members):
        archive_format = tarfile.PAX_FORMAT
    elif any(member.name.startswith("checkpoint/long-name-") for member, _ in members):
        archive_format = tarfile.GNU_FORMAT
    with custody.gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=archive_format) as archive:
            for member, data in members:
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return provider.store_bytes(
        output.getvalue(),
        role="training_checkpoint_custody_archive",
        logical_name="rewritten.tar.gz",
        media_type=CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
    )


@pytest.mark.parametrize(
    "mutation", ["unexpected", "duplicate", "case-collision", "link", "special", "pax"]
)
def test_materialize_rejects_ungoverned_or_unsafe_members(
    tmp_path: Path, mutation: str
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)

    def rewrite(members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
        if mutation == "unexpected":
            member = tarfile.TarInfo("checkpoint/unexpected.bin")
            members.append((member, b"extra"))
        elif mutation == "duplicate":
            member = tarfile.TarInfo(members[1][0].name)
            members.append((member, b"duplicate"))
        elif mutation == "case-collision":
            member = tarfile.TarInfo("checkpoint/LATEST.json")
            members.append((member, b"extra"))
        elif mutation == "link":
            members[1][0].type = tarfile.SYMTYPE
            members[1][0].linkname = "elsewhere"
        elif mutation == "special":
            members[1][0].type = tarfile.CHRTYPE
        else:
            members[1][0].pax_headers = {"comment": "forbidden"}

    rewritten = _rewritten_archive(provider, produced, rewrite)
    with pytest.raises(CheckpointReferenceResolutionError, match="unsafe|unexpected|mismatch"):
        materialize_checkpoint_custody_archive(
            provider,
            rewritten,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert not (tmp_path / "destination").exists()
    assert not list(tmp_path.glob(".destination.checkpoint-archive-*"))


def test_materialize_rejects_hidden_gnu_longname_header(tmp_path: Path) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    def rewrite(members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
        members[1][0].name = "checkpoint/" + "long-name-" * 12
    rewritten = _rewritten_archive(provider, produced, rewrite)
    with pytest.raises(CheckpointReferenceResolutionError, match="unsafe"):
        materialize_checkpoint_custody_archive(
            provider,
            rewritten,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )


@pytest.mark.parametrize("mutation", ["stale", "legacy-schema"])
def test_materialize_rejects_stale_or_migrated_latest_pointer(
    tmp_path: Path, mutation: str
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)

    def rewrite(members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
        latest = json.loads(members[1][1])
        if mutation == "stale":
            data = members[1][1].replace(
                latest["transaction_id"].encode(), b"tx-" + b"0" * 32
            )
        else:
            latest["schema_version"] = TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION_V2
            latest["completed_coordinate"]["global_step"] = latest[
                "completed_coordinate"
            ].pop("program_step")
            data = custody.canonical_json_bytes(latest)
            document = json.loads(members[0][1])
            document["expanded_payload_size_bytes"] += len(data) - len(members[1][1])
            members[0] = (members[0][0], custody.canonical_json_bytes(document))
        members[1] = (members[1][0], data)

    rewritten = _rewritten_archive(provider, produced, rewrite)
    with pytest.raises(CheckpointReferenceResolutionError, match="latest pointer|current schemas"):
        materialize_checkpoint_custody_archive(
            provider,
            rewritten,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert not (tmp_path / "destination").exists()
    assert not list(tmp_path.glob(".destination.checkpoint-archive-*"))


def test_materialize_rejects_unsupported_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    def unsupported(*args: object, **kwargs: object) -> None:
        raise CheckpointReferenceResolutionError("atomic no-replace publication unsupported")

    monkeypatch.setattr(custody, "_publish_directory_no_replace", unsupported)
    with pytest.raises(CheckpointReferenceResolutionError, match="unsupported"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert not (tmp_path / "destination").exists()
    assert not list(tmp_path.glob(".destination.checkpoint-archive-*"))


def test_materialize_rejects_noncanonical_archive_document(tmp_path: Path) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)

    def rewrite(members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
        document = json.loads(members[0][1])
        members[0] = (members[0][0], json.dumps(document, indent=2).encode())

    rewritten = _rewritten_archive(provider, produced, rewrite)
    with pytest.raises(CheckpointReferenceResolutionError, match="canonical JSON"):
        materialize_checkpoint_custody_archive(
            provider,
            rewritten,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )


def test_materialize_rejects_incidental_gzip_packet(tmp_path: Path) -> None:
    _, ref, provider, _, transaction_root = _produce(tmp_path)
    artifact = provider.store_bytes(
        custody.gzip.compress(b"rlrmp2 checkpoint packet"),
        role="training_checkpoint_custody_archive",
        logical_name="packet.gz",
        media_type=CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
    )
    with pytest.raises(CheckpointReferenceResolutionError, match="tar stream"):
        materialize_checkpoint_custody_archive(
            provider,
            artifact,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )


@pytest.mark.parametrize("suffix_kind", ["tar-record", "gzip-member", "trailing-bytes"])
def test_materialize_rejects_bytes_beyond_canonical_archive(
    tmp_path: Path, suffix_kind: str
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    original = provider.get_bytes(produced.artifact_ref)
    if suffix_kind == "tar-record":
        raw_tar = custody.gzip.decompress(original)
        member = tarfile.TarInfo("after-logical-eof")
        member.size = 1
        raw_tar += member.tobuf(format=tarfile.USTAR_FORMAT) + b"x" + b"\0" * 511
        output = io.BytesIO()
        with custody.gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as gz:
            gz.write(raw_tar)
        altered = output.getvalue()
    elif suffix_kind == "gzip-member":
        altered = original + custody.gzip.compress(b"concatenated")
    else:
        altered = original + b"trailing"
    artifact = provider.store_bytes(
        altered,
        role="training_checkpoint_custody_archive",
        logical_name=f"{suffix_kind}.tar.gz",
        media_type=CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
    )
    with pytest.raises(CheckpointReferenceResolutionError, match="noncanonical|trailing"):
        materialize_checkpoint_custody_archive(
            provider,
            artifact,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )


def test_materialize_rejects_parent_path_swap_without_foreign_damage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    parent = tmp_path / "publication"
    parent.mkdir()
    destination = parent / "destination"
    detached = tmp_path / "detached-publication"
    original_publish = custody._publish_directory_no_replace

    def swap_parent(*args: object, **kwargs: object) -> None:
        parent.rename(detached)
        parent.mkdir()
        destination.mkdir()
        destination.joinpath("foreign").write_text("keep")
        original_publish(*args, **kwargs)

    monkeypatch.setattr(custody, "_publish_directory_no_replace", swap_parent)
    with pytest.raises(CheckpointReferenceResolutionError, match="mapping changed"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            destination,
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert destination.joinpath("foreign").read_text() == "keep"
    assert not detached.joinpath("destination").exists()


def test_materialize_rejects_staging_name_substitution_without_foreign_damage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    destination = tmp_path / "destination"
    original_publish = custody._publish_directory_no_replace
    replacement: dict[str, str] = {}

    def substitute_staging(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
        **kwargs: object,
    ) -> None:
        stolen_name = f"stolen-{source_name}"
        os.rename(
            source_name,
            stolen_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.mkdir(source_name, mode=0o700, dir_fd=parent_descriptor)
        foreign = tmp_path / source_name / "foreign"
        foreign.write_text("keep")
        replacement["path"] = str(foreign)
        original_publish(
            parent_descriptor, source_name, destination_name, **kwargs
        )

    monkeypatch.setattr(custody, "_publish_directory_no_replace", substitute_staging)
    with pytest.raises(CheckpointReferenceResolutionError, match="staging identity changed"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            destination,
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert Path(replacement["path"]).read_text() == "keep"
    assert not destination.exists()
    assert not list(tmp_path.glob("stolen-.destination.checkpoint-archive-*"))


def test_materialize_cleans_staging_when_first_identity_acquisition_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    original_fstat = custody.os.fstat
    calls = 0

    def fail_first_staging_identity(descriptor: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected first staging identity failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(custody.os, "fstat", fail_first_staging_identity)
    with pytest.raises(OSError, match="first staging identity failure"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert not list(tmp_path.glob(".destination.checkpoint-archive-*"))


def test_cleanup_preserves_foreign_swapped_between_stat_and_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    original_rename = custody._perform_archive_rename_no_replace
    replacement: dict[str, Path] = {}
    swapped = False

    def fail_publication(*args: object, **kwargs: object) -> None:
        raise CheckpointReferenceResolutionError("injected publication failure")

    def swap_before_quarantine(
        parent_descriptor: int, source_name: str, destination_name: str
    ) -> None:
        nonlocal swapped
        if not swapped and destination_name.startswith(".checkpoint-archive-cleanup-"):
            swapped = True
            stolen_name = f"stolen-{source_name}"
            os.rename(
                source_name,
                stolen_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.mkdir(source_name, mode=0o700, dir_fd=parent_descriptor)
            foreign = tmp_path / source_name / "foreign"
            foreign.write_text("keep")
            replacement["path"] = foreign
        original_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(custody, "_publish_directory_no_replace", fail_publication)
    monkeypatch.setattr(custody, "_perform_archive_rename_no_replace", swap_before_quarantine)
    with pytest.raises(CheckpointReferenceResolutionError, match="publication failure"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert replacement["path"].read_text() == "keep"
    assert not list(tmp_path.glob("stolen-.destination.checkpoint-archive-*"))
