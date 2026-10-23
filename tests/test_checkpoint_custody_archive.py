from __future__ import annotations

import io
import json
import os
import tarfile
from errno import EBADF
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
            f"checkpoint/{Path(ref.uri).parent.as_posix()}/checkpoint-set.json",
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
        assert {member.name: archive.extractfile(member).read() for member in members[1:]} == {
            "checkpoint/latest.json": result.latest_pointer_path.read_bytes(),
            f"checkpoint/{ref.uri}": result.manifest_path.read_bytes(),
            f"checkpoint/{Path(ref.uri).parent.as_posix()}/checkpoint-set.json": (
                result.checkpoint_set_path.read_bytes()
            ),
            **{
                expected_name: _slot_blob_path(result.manifest_path, slot.slot).read_bytes()
                for expected_name, slot in zip(expected[4:], result.manifest.slots, strict=True)
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


def test_checkpoint_custody_archive_evidence_preserves_parent_ref_equality(
    tmp_path: Path,
) -> None:
    result, ref, provider = _setup(tmp_path)
    assert len(result.manifest.slots) > 1

    produced = produce_checkpoint_custody_archive(
        ref, allowed_root=result.root, artifact_provider=provider
    )

    assert produced.evidence.parent_ref == ref
    assert ref == produced.evidence.parent_ref


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


def _retained_staging(tmp_path: Path, destination_name: str = "destination") -> Path:
    residues = list(tmp_path.glob(f".{destination_name}.checkpoint-archive-*"))
    assert len(residues) == 1
    assert residues[0].is_dir()
    return residues[0]


def test_materialize_checkpoint_custody_archive_round_trip(tmp_path: Path) -> None:
    result, ref, _, produced, materialized, destination = _materialize(tmp_path)

    assert (
        destination.joinpath("latest.json").read_bytes() == result.latest_pointer_path.read_bytes()
    )
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


def test_materialize_publication_is_final_fallible_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    original_mapping = custody._require_external_archive_mapping
    original_result = custody.MaterializedCheckpointCustodyArchive
    original_publish = custody._publish_directory_no_replace
    result_constructed = False
    published = False

    def require_mapping(*args: object, **kwargs: object) -> None:
        assert not published
        original_mapping(*args, **kwargs)

    def construct_result(*args: object, **kwargs: object):
        nonlocal result_constructed
        result_constructed = True
        return original_result(*args, **kwargs)

    def publish(*args: object, **kwargs: object) -> None:
        nonlocal published
        assert result_constructed
        original_publish(*args, **kwargs)
        published = True

    monkeypatch.setattr(custody, "_require_external_archive_mapping", require_mapping)
    monkeypatch.setattr(custody, "MaterializedCheckpointCustodyArchive", construct_result)
    monkeypatch.setattr(custody, "_publish_directory_no_replace", publish)
    materialized = materialize_checkpoint_custody_archive(
        provider,
        produced.artifact_ref,
        tmp_path / "destination",
        expected_parent_ref=ref,
        expected_transaction_root_sha256=transaction_root,
    )
    assert published
    assert materialized.destination == tmp_path / "destination"


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
    assert not list(tmp_path.glob(".destination.checkpoint-archive-*"))


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


def test_materialize_preserves_publication_race_winner_and_retains_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    destination = tmp_path / "destination"
    perform_rename = custody._perform_archive_rename_no_replace

    def create_winner_then_perform_real_rename(
        parent_descriptor: int, source_name: str, destination_name: str
    ) -> None:
        os.mkdir(destination_name, dir_fd=parent_descriptor)
        winner_descriptor = os.open(
            f"{destination_name}/winner",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            os.write(winner_descriptor, b"foreign")
        finally:
            os.close(winner_descriptor)
        perform_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(
        custody,
        "_perform_archive_rename_no_replace",
        create_winner_then_perform_real_rename,
    )
    with pytest.raises(FileExistsError, match="publication race"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            destination,
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert destination.joinpath("winner").read_text() == "foreign"
    _retained_staging(tmp_path)


@pytest.mark.parametrize(
    "failure_point",
    ["directory-fstat", "directory-stat", "file-fstat", "file-stat"],
)
def test_open_archive_member_closes_unadopted_descriptor_on_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    original_open = os.open
    original_close = os.close
    original_dup = os.dup
    original_fstat = os.fstat
    original_stat = os.stat
    staging_descriptor = original_open(staging, custody._archive_directory_flags())
    observed: dict[str, int] = {}

    def track_dup(descriptor: int) -> int:
        duplicate = original_dup(descriptor)
        observed["root"] = duplicate
        return duplicate

    def track_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "nested":
            observed["directory"] = descriptor
        elif path == "member":
            observed["file"] = descriptor
        return descriptor

    def fail_fstat(descriptor: int):
        if (failure_point == "directory-fstat" and descriptor == observed.get("directory")) or (
            failure_point == "file-fstat" and descriptor == observed.get("file")
        ):
            raise OSError(f"injected {failure_point} failure")
        return original_fstat(descriptor)

    def fail_stat(path: object, *args: object, **kwargs: object):
        if (failure_point == "directory-stat" and path == "nested" and "directory" in observed) or (
            failure_point == "file-stat" and path == "member"
        ):
            raise OSError(f"injected {failure_point} failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(custody.os, "dup", track_dup)
    monkeypatch.setattr(custody.os, "open", track_open)
    monkeypatch.setattr(custody.os, "fstat", fail_fstat)
    monkeypatch.setattr(custody.os, "stat", fail_stat)
    try:
        with pytest.raises(OSError, match=f"injected {failure_point} failure"):
            custody._open_archive_member(staging_descriptor, ("nested", "member"), {})
        for descriptor in observed.values():
            with pytest.raises(OSError) as closed:
                original_fstat(descriptor)
            assert closed.value.errno == EBADF
    finally:
        original_close(staging_descriptor)


def test_materialize_closes_member_descriptor_when_fdopen_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    original_fdopen = os.fdopen
    original_fstat = os.fstat
    member_descriptor: int | None = None

    def fail_fdopen(descriptor: int, *args: object, **kwargs: object):
        nonlocal member_descriptor
        member_descriptor = descriptor
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(custody.os, "fdopen", fail_fdopen)
    with pytest.raises(CheckpointReferenceResolutionError, match="fdopen failure"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert member_descriptor is not None
    with pytest.raises(OSError) as closed:
        original_fstat(member_descriptor)
    assert closed.value.errno == EBADF
    _retained_staging(tmp_path)
    monkeypatch.setattr(custody.os, "fdopen", original_fdopen)


@pytest.mark.parametrize("identity", ["parent", "transaction-root"])
def test_materialize_rejects_expected_identity_mismatch(tmp_path: Path, identity: str) -> None:
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
    _retained_staging(tmp_path)


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
def test_materialize_rejects_ungoverned_or_unsafe_members(tmp_path: Path, mutation: str) -> None:
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
    _retained_staging(tmp_path)


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
            data = members[1][1].replace(latest["transaction_id"].encode(), b"tx-" + b"0" * 32)
        else:
            latest["schema_version"] = TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION_V2
            latest["completed_coordinate"]["global_step"] = latest["completed_coordinate"].pop(
                "program_step"
            )
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
    _retained_staging(tmp_path)


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
    _retained_staging(tmp_path)


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
    _retained_staging(tmp_path)


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
    _retained_staging(tmp_path)


def test_materialize_rejects_cooperative_parent_mapping_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    parent = tmp_path / "publication"
    parent.mkdir()
    destination = parent / "destination"
    detached = tmp_path / "detached-publication"
    original_resolve = custody.resolve_checkpoint_custody_ref

    def resolve_then_swap_parent(*args: object, **kwargs: object):
        resolved = original_resolve(*args, **kwargs)
        parent.rename(detached)
        parent.mkdir()
        destination.mkdir()
        destination.joinpath("foreign").write_text("keep")
        return resolved

    monkeypatch.setattr(custody, "resolve_checkpoint_custody_ref", resolve_then_swap_parent)
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
    _retained_staging(detached)


def test_materialize_rejects_cooperative_staging_mapping_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    destination = tmp_path / "destination"
    original_resolve = custody.resolve_checkpoint_custody_ref
    replacement: dict[str, Path] = {}

    def resolve_then_substitute_staging(*args: object, **kwargs: object):
        resolved = original_resolve(*args, **kwargs)
        staging = _retained_staging(tmp_path)
        stolen = tmp_path / f"stolen-{staging.name}"
        staging.rename(stolen)
        staging.mkdir()
        foreign = staging / "foreign"
        foreign.write_text("keep")
        replacement["foreign"] = foreign
        replacement["stolen"] = stolen
        return resolved

    monkeypatch.setattr(custody, "resolve_checkpoint_custody_ref", resolve_then_substitute_staging)
    with pytest.raises(CheckpointReferenceResolutionError, match="mapping changed"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            destination,
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    assert replacement["foreign"].read_text() == "keep"
    assert replacement["stolen"].is_dir()
    assert not destination.exists()


def test_materialize_retains_staging_when_first_identity_acquisition_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    original_fstat = custody.os.fstat
    provider_type = type(provider)
    original_get = provider_type.get_bytes
    calls = 0

    def fail_first_staging_identity(descriptor: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected first staging identity failure")
        return original_fstat(descriptor)

    def authenticate_then_arm(self: object, artifact_ref: object) -> bytes:
        data = original_get(self, artifact_ref)
        monkeypatch.setattr(custody.os, "fstat", fail_first_staging_identity)
        return data

    monkeypatch.setattr(provider_type, "get_bytes", authenticate_then_arm)
    with pytest.raises(OSError, match="first staging identity failure"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    _retained_staging(tmp_path)


def test_materialize_rejects_descendant_directory_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    original_open = custody.os.open
    provider_type = type(provider)
    original_get = provider_type.get_bytes
    replacement: dict[str, Path] = {}
    swapped = False

    def swap_created_descendant(path: object, flags: int, *args: object, **kwargs: object):
        nonlocal swapped
        directory_fd = kwargs.get("dir_fd")
        if path == "transactions" and directory_fd is not None and not swapped:
            swapped = True
            os.rename(
                "transactions",
                "owned-transactions",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.mkdir("transactions", mode=0o700, dir_fd=directory_fd)
            foreign_directory = original_open(
                "transactions", custody._archive_directory_flags(), dir_fd=directory_fd
            )
            try:
                marker_fd = original_open(
                    "foreign",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=foreign_directory,
                )
                os.close(marker_fd)
            finally:
                os.close(foreign_directory)
        return original_open(path, flags, *args, **kwargs)

    def authenticate_then_arm(self: object, artifact_ref: object) -> bytes:
        data = original_get(self, artifact_ref)
        monkeypatch.setattr(custody.os, "open", swap_created_descendant)
        return data

    monkeypatch.setattr(provider_type, "get_bytes", authenticate_then_arm)
    with pytest.raises(CheckpointReferenceResolutionError, match="directory identity changed"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    staging = _retained_staging(tmp_path)
    replacement["marker"] = staging / "transactions" / "foreign"
    assert replacement["marker"].read_bytes() == b""
    assert list((staging / "transactions").iterdir()) == [replacement["marker"]]


def test_materialize_retains_staging_when_open_fails_immediately_after_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    original_open = custody.os.open
    provider_type = type(provider)
    original_get = provider_type.get_bytes
    failed = False

    def fail_first_staging_open(path: object, *args: object, **kwargs: object):
        nonlocal failed
        if (
            isinstance(path, str)
            and path.startswith(".destination.checkpoint-archive-")
            and not failed
        ):
            failed = True
            raise OSError("injected staging open failure")
        return original_open(path, *args, **kwargs)

    def authenticate_then_arm(self: object, artifact_ref: object) -> bytes:
        data = original_get(self, artifact_ref)
        monkeypatch.setattr(custody.os, "open", fail_first_staging_open)
        return data

    monkeypatch.setattr(provider_type, "get_bytes", authenticate_then_arm)
    with pytest.raises(OSError, match="staging open failure"):
        materialize_checkpoint_custody_archive(
            provider,
            produced.artifact_ref,
            tmp_path / "destination",
            expected_parent_ref=ref,
            expected_transaction_root_sha256=transaction_root,
        )
    _retained_staging(tmp_path)


def test_materialize_streaming_acceptance_does_not_reencode_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ref, provider, produced, transaction_root = _produce(tmp_path)
    raw_tar = custody.gzip.decompress(provider.get_bytes(produced.artifact_ref))
    alternate = io.BytesIO()
    with custody.gzip.GzipFile(
        fileobj=alternate, mode="wb", filename="", mtime=7, compresslevel=1
    ) as compressed:
        compressed.write(raw_tar)
    alternate_ref = provider.store_bytes(
        alternate.getvalue(),
        role="training_checkpoint_custody_archive",
        logical_name="alternate-compression.tar.gz",
        media_type=CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
    )

    def reject_reencoding(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("materialization must not re-encode authenticated bytes")

    monkeypatch.setattr(custody, "_checkpoint_custody_archive_bytes", reject_reencoding)
    result = materialize_checkpoint_custody_archive(
        provider,
        alternate_ref,
        tmp_path / "destination",
        expected_parent_ref=ref,
        expected_transaction_root_sha256=transaction_root,
    )
    assert result.archive_evidence.archive_sha256 == alternate_ref.sha256
