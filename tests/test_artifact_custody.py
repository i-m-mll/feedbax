from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import feedbax.persistence.artifact_custody as custody_module
from feedbax.contracts.manifest import ArtifactRef, store_bytes_artifact
from feedbax.persistence import (
    ArtifactBlobContainmentError,
    ArtifactBlobCustodyError,
    ArtifactBlobIntegrityError,
    ArtifactBlobReferenceError,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION,
    IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND,
    ImmutableArtifactBlobProvider,
    ImmutableArtifactBlobProviderConfig,
    ImmutableArtifactBlobProviderSpec,
    open_immutable_artifact_blob_provider,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider as CanonicalProvider


pytestmark = pytest.mark.feedbax_contract

_VALID_DIGEST = "a" * 64
_VALID_ARTIFACT_ID = f"artifact://sha256/{_VALID_DIGEST}"


def _artifact_path(root: Path, digest: str) -> Path:
    return root / "artifacts" / "sha256" / digest[:2] / digest


def test_store_uses_existing_suffixless_cas_and_round_trips_exact_binary_bytes(
    tmp_path: Path,
) -> None:
    data = b"\x00\xffbinary\x00payload\n"
    root = tmp_path / "custody"
    provider = ImmutableArtifactBlobProvider(root)
    artifact = provider.store_bytes(
        data,
        role="opaque_payload",
        logical_name="payload.bin",
        metadata={"purpose": "round-trip"},
    )
    digest = hashlib.sha256(data).hexdigest()
    canonical_path = _artifact_path(root, digest)

    assert artifact.artifact_id == f"artifact://sha256/{digest}"
    assert artifact.uri == artifact.artifact_id
    assert artifact.sha256 == digest
    assert artifact.size_bytes == len(data)
    assert artifact.metadata == {"purpose": "round-trip"}
    assert canonical_path.read_bytes() == data
    assert [path for path in root.rglob("*") if path.is_file()] == [canonical_path]
    assert provider.get_bytes(artifact) == data
    assert provider.get_bytes(artifact.artifact_id, size_bytes=len(data)) == data


def test_exact_json_bytes_and_trailing_newline_are_not_canonicalized(tmp_path: Path) -> None:
    data = b'{"z": 1, "a": [3, 2, 1]}  \n'
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")

    artifact = provider.store_bytes(
        data,
        role="manifest",
        logical_name="manifest.json",
        media_type="application/json",
    )

    assert provider.get_bytes(artifact) == data


def test_repeated_store_has_stable_identity_and_does_not_rewrite(tmp_path: Path) -> None:
    data = b"stored exactly once"
    root = tmp_path / "custody"
    provider = ImmutableArtifactBlobProvider(root)
    first = provider.store_bytes(data, role="first", logical_name="first.bin")
    path = _artifact_path(root, first.sha256 or "")
    first_stat = path.stat()

    second = provider.store_bytes(data, role="second", logical_name="second.bin")
    second_stat = path.stat()

    assert second.artifact_id == first.artifact_id
    assert second.sha256 == first.sha256
    assert second_stat.st_ino == first_stat.st_ino
    assert second_stat.st_mtime_ns == first_stat.st_mtime_ns
    assert path.read_bytes() == data


def test_custody_survives_source_directory_deletion_and_materializes_copy(
    tmp_path: Path,
) -> None:
    source_directory = tmp_path / "removable-run-set"
    source_directory.mkdir()
    source = source_directory / "manifest.json"
    data = b'{\n  "schema": "exact"\n}\n'
    source.write_bytes(data)
    root = tmp_path / "durable-custody"
    provider = ImmutableArtifactBlobProvider(root)
    artifact = provider.store_bytes(
        source.read_bytes(),
        role="manifest",
        logical_name=source.name,
        media_type="application/json",
    )
    custody_path = _artifact_path(root, artifact.sha256 or "")

    shutil.rmtree(source_directory)
    destination = tmp_path / "materialized" / "manifest.json"
    result = provider.materialize(artifact, destination)

    assert result == destination
    assert provider.get_bytes(artifact) == data
    assert destination.read_bytes() == data
    assert custody_path.read_bytes() == data
    assert os.stat(destination).st_ino != os.stat(custody_path).st_ino
    assert stat.S_IMODE(destination.stat().st_mode) & 0o077 == 0


def test_public_imports_do_not_load_web_package(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from feedbax.persistence import (ImmutableArtifactBlobProvider, "
                "ImmutableArtifactBlobProviderSpec, "
                "open_immutable_artifact_blob_provider); "
                "from feedbax.contracts.artifact_custody import "
                "ImmutableArtifactBlobProviderConfig; "
                "from feedbax.persistence.artifact_custody import "
                "ArtifactBlobIntegrityError; "
                "assert ImmutableArtifactBlobProvider; "
                "assert ImmutableArtifactBlobProviderSpec; "
                "assert ImmutableArtifactBlobProviderConfig; "
                "assert open_immutable_artifact_blob_provider; "
                "assert ArtifactBlobIntegrityError; "
                "assert not any(n == 'feedbax.web' or n.startswith('feedbax.web.') "
                "for n in sys.modules)"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert CanonicalProvider is ImmutableArtifactBlobProvider


def test_portable_provider_spec_has_exact_root_free_json_and_round_trips() -> None:
    expected = {
        "schema_id": "feedbax.spec.immutable_artifact_blob_provider",
        "schema_version": "feedbax.spec.immutable_artifact_blob_provider.v1",
        "kind": "feedbax-local-sha256-cas",
        "config": {"storage_backend": "feedbax-local"},
    }
    spec = ImmutableArtifactBlobProviderSpec()

    assert spec.model_dump(mode="json") == expected
    assert spec.model_dump_json() == json.dumps(expected, separators=(",", ":"))
    assert ImmutableArtifactBlobProviderSpec.model_validate_json(spec.model_dump_json()) == spec
    assert "root" not in spec.model_dump_json()
    assert spec.config == ImmutableArtifactBlobProviderConfig()
    assert IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID == expected["schema_id"]
    assert IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION == expected["schema_version"]
    assert IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND == expected["kind"]
    assert IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND == "feedbax-local"
    assert ImmutableArtifactBlobProviderConfig.__module__ == ("feedbax.contracts.artifact_custody")
    assert ImmutableArtifactBlobProviderSpec.__module__ == "feedbax.contracts.artifact_custody"


def test_mapping_and_model_factory_bind_the_same_explicit_root(tmp_path: Path) -> None:
    explicit_root = tmp_path / "custody"
    spec = ImmutableArtifactBlobProviderSpec()

    from_model = open_immutable_artifact_blob_provider(spec, explicit_root=explicit_root)
    from_mapping = open_immutable_artifact_blob_provider(
        spec.model_dump(mode="json"),
        explicit_root=explicit_root,
    )

    assert from_model == from_mapping
    assert from_model.root == explicit_root.resolve()
    assert from_model.storage_backend == "feedbax-local"


@pytest.mark.parametrize(
    "explicit_root",
    [
        "",
        "relative/custody",
        "~/custody",
        "$FEEDBAX_CUSTODY/custody",
        "/tmp/../custody",
        "/tmp/custody\0escape",
    ],
)
def test_provider_and_factory_reject_ambient_or_unsafe_roots(explicit_root: str) -> None:
    with pytest.raises(ArtifactBlobReferenceError):
        ImmutableArtifactBlobProvider(explicit_root)
    with pytest.raises(ArtifactBlobReferenceError):
        open_immutable_artifact_blob_provider({}, explicit_root=explicit_root)


def test_factory_does_not_expand_environment_or_use_ambient_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    literal_root = tmp_path / "$FEEDBAX_CUSTODY"
    monkeypatch.chdir(ambient)
    monkeypatch.setenv("FEEDBAX_CUSTODY", str(tmp_path / "expanded"))

    provider = open_immutable_artifact_blob_provider({}, explicit_root=literal_root)

    assert provider.root == literal_root
    assert provider.root != Path(os.environ["FEEDBAX_CUSTODY"])


def test_provider_canonicalizes_explicit_macos_alias_root() -> None:
    alias_root = Path("/tmp")
    if not alias_root.is_symlink():
        pytest.skip("platform has no canonical /tmp symlink alias")

    provider = ImmutableArtifactBlobProvider(alias_root / "feedbax-custody")

    assert provider.root == alias_root.resolve() / "feedbax-custody"


def test_provider_rejects_arbitrary_explicit_root_symlink(tmp_path: Path) -> None:
    target = tmp_path / "attacker-selected-target"
    target.mkdir()
    root_alias = tmp_path / "custody-alias"
    root_alias.symlink_to(target, target_is_directory=True)
    provider = ImmutableArtifactBlobProvider(root_alias)

    with pytest.raises(ArtifactBlobContainmentError):
        provider.store_bytes(b"must-not-escape", role="payload", logical_name="blob")

    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    ("update", "diagnostic"),
    [
        ({"schema_id": "feedbax.spec.other"}, "schema_id"),
        ({"root": "/tmp/ambient-root"}, "root"),
        (
            {"schema_version": "feedbax.spec.immutable_artifact_blob_provider.v0"},
            "current_version",
        ),
        ({"kind": "python:dynamic.provider"}, "kind"),
        ({"config": {"storage_backend": "s3"}}, "storage_backend"),
    ],
)
def test_factory_rejects_unsupported_portable_provider_fields(
    tmp_path: Path,
    update: dict[str, object],
    diagnostic: str,
) -> None:
    payload = ImmutableArtifactBlobProviderSpec().model_dump(mode="json")
    payload.update(update)

    with pytest.raises(ArtifactBlobReferenceError, match=diagnostic):
        open_immutable_artifact_blob_provider(payload, explicit_root=tmp_path / "custody")


@pytest.mark.parametrize(
    "artifact_id",
    [
        "",
        "/tmp/blob",
        "file:///tmp/blob",
        "http://sha256/" + "a" * 64,
        "artifact://other/" + "a" * 64,
        "artifact://sha256/abc",
        "artifact://sha256/" + "g" * 64,
        "artifact://sha256/" + "A" * 64,
        "artifact://sha256/" + "a" * 64 + ".json",
        "artifact://sha256/" + "a" * 64 + "/extra",
        "artifact://sha256/" + "a" * 64 + "/",
        "artifact://sha256/" + "a" * 64 + "?download=1",
        "artifact://sha256/" + "a" * 64 + "#fragment",
        "artifact://user@sha256/" + "a" * 64,
        "artifact://sha256:443/" + "a" * 64,
        "artifact://sha256/" + "%61" * 64,
    ],
)
def test_malformed_or_mutable_artifact_ids_are_rejected(
    tmp_path: Path,
    artifact_id: str,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")

    with pytest.raises(ArtifactBlobReferenceError):
        provider.get_bytes(artifact_id, size_bytes=0)


@pytest.mark.parametrize(
    "reference",
    [
        ArtifactRef(role="payload", logical_name="blob", sha256="a" * 64, size_bytes=0),
        ArtifactRef(
            role="payload",
            logical_name="blob",
            artifact_id=_VALID_ARTIFACT_ID,
            uri=_VALID_ARTIFACT_ID,
            size_bytes=0,
        ),
        ArtifactRef(
            role="payload",
            logical_name="blob",
            artifact_id=_VALID_ARTIFACT_ID,
            sha256=_VALID_DIGEST,
            uri=_VALID_ARTIFACT_ID,
        ),
        ArtifactRef(
            role="payload",
            logical_name="blob",
            artifact_id=_VALID_ARTIFACT_ID,
            sha256="b" * 64,
            uri=_VALID_ARTIFACT_ID,
            size_bytes=0,
        ),
        ArtifactRef(
            role="payload",
            logical_name="blob",
            artifact_id=_VALID_ARTIFACT_ID,
            sha256=_VALID_DIGEST,
            uri=_VALID_ARTIFACT_ID,
            size_bytes=-1,
        ),
        ArtifactRef(
            role="payload",
            logical_name="blob",
            artifact_id=_VALID_ARTIFACT_ID,
            sha256=_VALID_DIGEST,
            uri=_VALID_ARTIFACT_ID,
            size_bytes=0,
            storage_backend="unsupported",
        ),
        ArtifactRef.model_construct(
            role="payload",
            logical_name="blob",
            artifact_id=_VALID_ARTIFACT_ID,
            sha256=_VALID_DIGEST,
            uri=_VALID_ARTIFACT_ID,
            size_bytes=True,
            storage_backend="feedbax-local",
        ),
    ],
)
def test_invalid_artifact_ref_fields_are_rejected(
    tmp_path: Path,
    reference: ArtifactRef,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")

    with pytest.raises(ArtifactBlobReferenceError):
        provider.get_bytes(reference)


def test_raw_id_requires_valid_size_and_ref_override_must_agree(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    data = b"size-bound"
    artifact = provider.store_bytes(data, role="payload", logical_name="blob")

    with pytest.raises(ArtifactBlobReferenceError, match="required"):
        provider.get_bytes(artifact.artifact_id or "")
    with pytest.raises(ArtifactBlobReferenceError):
        provider.get_bytes(artifact.artifact_id or "", size_bytes=-1)
    with pytest.raises(ArtifactBlobReferenceError):
        provider.get_bytes(artifact.artifact_id or "", size_bytes=True)
    with pytest.raises(ArtifactBlobReferenceError, match="does not match"):
        provider.get_bytes(artifact, size_bytes=len(data) + 1)


@pytest.mark.parametrize(
    "uri",
    [
        None,
        "/tmp/local-blob",
        "artifact://sha256/abc",
        "artifact://sha256/" + "b" * 64,
    ],
)
def test_artifact_ref_requires_exact_canonical_uri(tmp_path: Path, uri: str | None) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    artifact = provider.store_bytes(b"data", role="payload", logical_name="blob")

    with pytest.raises(ArtifactBlobReferenceError):
        provider.get_bytes(artifact.model_copy(update={"uri": uri}))


def test_fixed_provider_identity_and_reserved_metadata_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(ArtifactBlobReferenceError, match="unsupported"):
        ImmutableArtifactBlobProvider(
            tmp_path / "named-custody",
            storage_backend="feedbax-archive",
        )
    with pytest.raises(ArtifactBlobCustodyError, match="explicit"):
        ImmutableArtifactBlobProvider(None)  # type: ignore[arg-type]

    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    artifact = provider.store_bytes(b"data", role="payload", logical_name="blob")
    assert artifact.storage_backend == "feedbax-local"
    with pytest.raises(ArtifactBlobReferenceError, match="relative_path"):
        provider.store_bytes(
            b"data",
            role="payload",
            logical_name="blob",
            metadata={"relative_path": "elsewhere"},
        )
    with pytest.raises(ArtifactBlobReferenceError, match="relative_path"):
        provider.get_bytes(artifact.model_copy(update={"metadata": {"relative_path": "elsewhere"}}))


def test_missing_blob_is_rejected(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    missing_id = "artifact://sha256/" + "a" * 64

    with pytest.raises(FileNotFoundError, match="missing"):
        provider.get_bytes(missing_id, size_bytes=1)


@pytest.mark.parametrize("tampered", [b"VWXYZ", b"abc", b"abcdefgh"])
def test_hash_truncation_and_extension_tampering_are_rejected(
    tmp_path: Path,
    tampered: bytes,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / tampered.hex())
    original = b"abcde"
    artifact = provider.store_bytes(original, role="payload", logical_name="blob")
    path = _artifact_path(provider.root, artifact.sha256 or "")
    path.write_bytes(tampered)

    with pytest.raises(ArtifactBlobIntegrityError):
        provider.get_bytes(artifact)


def test_corrupt_preexisting_canonical_blob_is_not_overwritten(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    intended = b"right"
    corrupt = b"wrong"
    digest = hashlib.sha256(intended).hexdigest()
    path = _artifact_path(provider.root, digest)
    path.parent.mkdir(parents=True)
    path.write_bytes(corrupt)

    with pytest.raises(ArtifactBlobIntegrityError):
        provider.store_bytes(intended, role="payload", logical_name="blob")

    assert path.read_bytes() == corrupt


def test_shared_writer_parent_swap_cannot_write_outside_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "custody"
    outside = tmp_path / "outside"
    outside.mkdir()
    data = b"race-safe"
    digest = hashlib.sha256(data).hexdigest()
    canonical_parent = root / "artifacts" / "sha256" / digest[:2]
    moved_parent = tmp_path / "pinned-original-parent"
    original_link = custody_module._link_materialized_file
    swapped = False

    def swap_parent_then_link(
        temporary_name: str,
        final_name: str,
        *,
        temporary_parent_descriptor: int,
        parent_descriptor: int,
    ) -> None:
        nonlocal swapped
        canonical_parent.rename(moved_parent)
        canonical_parent.symlink_to(outside, target_is_directory=True)
        swapped = True
        original_link(
            temporary_name,
            final_name,
            temporary_parent_descriptor=temporary_parent_descriptor,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(custody_module, "_link_materialized_file", swap_parent_then_link)

    with pytest.raises(ArtifactBlobContainmentError, match="identity changed"):
        store_bytes_artifact(
            data,
            root=root,
            role="payload",
            logical_name="blob",
        )

    assert swapped
    assert not (outside / digest).exists()
    assert (moved_parent / digest).read_bytes() == data


def test_shared_writer_never_cleanup_deletes_late_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "custody"
    data = b"owned-complete-bytes"
    foreign = b"foreign-replacement-must-survive"
    digest = hashlib.sha256(data).hexdigest()
    canonical_path = _artifact_path(root, digest)
    original_recheck = custody_module._recheck_directory_chain

    def replace_after_verification(records: object) -> None:
        original_recheck(records)  # type: ignore[arg-type]
        canonical_path.unlink()
        canonical_path.write_bytes(foreign)
        raise ArtifactBlobContainmentError("forced failure after foreign replacement")

    monkeypatch.setattr(
        custody_module,
        "_recheck_directory_chain",
        replace_after_verification,
    )

    with pytest.raises(ArtifactBlobContainmentError, match="forced failure"):
        store_bytes_artifact(data, root=root, role="payload", logical_name="blob")

    assert canonical_path.read_bytes() == foreign
    assert not list(canonical_path.parent.glob(".feedbax-artifact-*.tmp"))


def test_shared_writer_is_atomic_and_idempotent_under_concurrent_calls(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    data = b"concurrent exact bytes"

    def store_once(index: int) -> ArtifactRef:
        return store_bytes_artifact(
            data,
            root=root,
            role=f"payload-{index}",
            logical_name=f"blob-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        refs = list(executor.map(store_once, range(16)))

    digest = hashlib.sha256(data).hexdigest()
    canonical_path = _artifact_path(root, digest)
    assert {ref.artifact_id for ref in refs} == {f"artifact://sha256/{digest}"}
    assert canonical_path.read_bytes() == data
    assert list(canonical_path.parent.glob(".feedbax-materialization-staging/*")) == []


def test_shared_writer_preserves_replaced_staging_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "custody"
    data = b"staging-container-race"
    digest = hashlib.sha256(data).hexdigest()
    canonical_parent = root / "artifacts" / "sha256" / digest[:2]
    staging_path = canonical_parent / ".feedbax-materialization-staging"
    moved_staging = tmp_path / "moved-blob-staging"
    original_link = custody_module._link_materialized_file
    foreign_identity: tuple[int, int] | None = None

    def replace_staging_then_link(
        temporary_name: str,
        final_name: str,
        *,
        temporary_parent_descriptor: int,
        parent_descriptor: int,
    ) -> None:
        nonlocal foreign_identity
        staging_path.rename(moved_staging)
        staging_path.mkdir(mode=0o700)
        foreign_stat = staging_path.stat()
        foreign_identity = (foreign_stat.st_dev, foreign_stat.st_ino)
        original_link(
            temporary_name,
            final_name,
            temporary_parent_descriptor=temporary_parent_descriptor,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(custody_module, "_link_materialized_file", replace_staging_then_link)

    artifact = store_bytes_artifact(data, root=root, role="payload", logical_name="blob")

    assert artifact.sha256 == digest
    assert foreign_identity is not None
    surviving_stat = staging_path.stat()
    assert (surviving_stat.st_dev, surviving_stat.st_ino) == foreign_identity
    assert list(staging_path.iterdir()) == []
    assert list(moved_staging.iterdir()) == []


def test_shared_writer_suffix_works_through_trusted_darwin_var_alias(tmp_path: Path) -> None:
    if sys.platform != "darwin" or not Path("/var").is_symlink():
        pytest.skip("requires Darwin's canonical /var alias")
    canonical_prefix = "/private/var/"
    if not str(tmp_path).startswith(canonical_prefix):
        pytest.skip("pytest temporary root is not beneath /private/var")
    alias_root = Path("/var") / Path(str(tmp_path)[len(canonical_prefix) :]) / "suffix-root"
    data = b"suffix-compatible"

    artifact = store_bytes_artifact(
        data,
        root=alias_root,
        role="payload",
        logical_name="blob.bin",
    )

    digest = hashlib.sha256(data).hexdigest()
    assert artifact.sha256 == digest
    assert Path(artifact.uri or "").read_bytes() == data
    assert (
        alias_root.resolve() / "artifacts" / "sha256" / digest[:2] / digest
    ).read_bytes() == data


def test_parent_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    data = b"outside"
    digest = hashlib.sha256(data).hexdigest()
    outside_artifacts = tmp_path / "outside-artifacts"
    outside_path = outside_artifacts / "sha256" / digest[:2] / digest
    outside_path.parent.mkdir(parents=True)
    outside_path.write_bytes(data)
    (root / "artifacts").symlink_to(outside_artifacts, target_is_directory=True)
    provider = ImmutableArtifactBlobProvider(root)

    with pytest.raises(ArtifactBlobContainmentError):
        provider.get_bytes(f"artifact://sha256/{digest}", size_bytes=len(data))


def test_canonical_file_symlink_is_rejected(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    data = b"outside"
    digest = hashlib.sha256(data).hexdigest()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(data)
    path = _artifact_path(provider.root, digest)
    path.parent.mkdir(parents=True)
    path.symlink_to(outside)

    with pytest.raises(ArtifactBlobContainmentError):
        provider.get_bytes(f"artifact://sha256/{digest}", size_bytes=len(data))


def test_non_regular_blob_and_hardlink_alias_are_rejected(tmp_path: Path) -> None:
    non_regular_provider = ImmutableArtifactBlobProvider(tmp_path / "non-regular")
    data = b"payload"
    digest = hashlib.sha256(data).hexdigest()
    directory_blob = _artifact_path(non_regular_provider.root, digest)
    directory_blob.mkdir(parents=True)
    with pytest.raises(ArtifactBlobIntegrityError, match="regular"):
        non_regular_provider.get_bytes(
            f"artifact://sha256/{digest}",
            size_bytes=len(data),
        )

    alias_provider = ImmutableArtifactBlobProvider(tmp_path / "hardlink")
    artifact = alias_provider.store_bytes(data, role="payload", logical_name="blob")
    canonical_path = _artifact_path(alias_provider.root, artifact.sha256 or "")
    os.link(canonical_path, tmp_path / "mutable-alias.bin")
    with pytest.raises(ArtifactBlobIntegrityError, match="hard-link"):
        alias_provider.get_bytes(artifact)


def test_materialize_rejects_existing_symlink_and_cas_destinations(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    data = b"payload"
    artifact = provider.store_bytes(data, role="payload", logical_name="blob")

    existing = tmp_path / "existing.bin"
    existing.write_bytes(b"do not replace")
    with pytest.raises(FileExistsError):
        provider.materialize(artifact, existing)
    assert existing.read_bytes() == b"do not replace"

    symlink = tmp_path / "destination-link"
    target = tmp_path / "symlink-target"
    symlink.symlink_to(target)
    with pytest.raises(FileExistsError):
        provider.materialize(artifact, symlink)

    inside_cas = provider.root / "artifacts" / "sha256" / "materialized.bin"
    with pytest.raises(ArtifactBlobContainmentError):
        provider.materialize(artifact, inside_cas)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ArtifactBlobContainmentError):
        provider.materialize(artifact, symlink_parent / "output.bin")


def test_materialize_works_through_trusted_darwin_tmp_alias(tmp_path: Path) -> None:
    if sys.platform != "darwin" or not Path("/tmp").is_symlink():
        pytest.skip("requires Darwin's canonical /tmp alias")
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    data = b"trusted-alias-materialization"
    artifact = provider.store_bytes(data, role="payload", logical_name="blob")
    alias_parent = Path(tempfile.mkdtemp(prefix="feedbax-materialize-", dir="/tmp"))
    destination = alias_parent / "output.bin"
    try:
        result = provider.materialize(artifact, destination)

        assert result == destination
        assert destination.read_bytes() == data
        custody_path = _artifact_path(provider.root, artifact.sha256 or "")
        assert destination.stat().st_ino != custody_path.stat().st_ino
    finally:
        shutil.rmtree(alias_parent)


@pytest.mark.parametrize(
    "destination",
    ["relative.bin", "~/output.bin", "$FEEDBAX_OUTPUT/output.bin", "/tmp/../output.bin"],
)
def test_materialize_rejects_ambient_or_unsafe_destinations(
    tmp_path: Path,
    destination: str,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    artifact = provider.store_bytes(b"payload", role="payload", logical_name="blob")

    with pytest.raises(ArtifactBlobReferenceError):
        provider.materialize(artifact, destination)


def test_materialize_parent_swap_fails_closed_and_preserves_complete_owned_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    artifact = provider.store_bytes(b"payload", role="payload", logical_name="blob")
    destination_parent = tmp_path / "destination"
    destination_parent.mkdir()
    moved_parent = tmp_path / "pinned-destination"
    cas_alias_target = provider.root / "artifacts" / "sha256" / "alias-target"
    cas_alias_target.mkdir()
    destination = destination_parent / "output.bin"
    original_link = custody_module._link_materialized_file
    swapped = False

    def swap_parent_then_link(
        temporary_name: str,
        destination_name: str,
        *,
        temporary_parent_descriptor: int,
        parent_descriptor: int,
    ) -> None:
        nonlocal swapped
        destination_parent.rename(moved_parent)
        destination_parent.symlink_to(cas_alias_target, target_is_directory=True)
        swapped = True
        original_link(
            temporary_name,
            destination_name,
            temporary_parent_descriptor=temporary_parent_descriptor,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(custody_module, "_link_materialized_file", swap_parent_then_link)

    with pytest.raises(ArtifactBlobContainmentError, match="identity changed"):
        provider.materialize(artifact, destination)

    assert swapped
    assert (moved_parent / destination.name).read_bytes() == b"payload"
    assert not (cas_alias_target / destination.name).exists()


def test_materialize_never_cleanup_deletes_late_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    artifact = provider.store_bytes(b"payload", role="payload", logical_name="blob")
    destination = tmp_path / "destination" / "output.bin"
    foreign = b"foreign-materialization-replacement"
    original_recheck = custody_module._recheck_directory_chain

    def replace_after_verification(records: object) -> None:
        original_recheck(records)  # type: ignore[arg-type]
        if not destination.exists():
            return
        destination.unlink()
        destination.write_bytes(foreign)
        raise ArtifactBlobContainmentError("forced failure after foreign replacement")

    monkeypatch.setattr(custody_module, "_recheck_directory_chain", replace_after_verification)

    with pytest.raises(ArtifactBlobContainmentError, match="forced failure"):
        provider.materialize(artifact, destination)

    assert destination.read_bytes() == foreign
    assert not list(destination.parent.glob(".feedbax-materialize-*.tmp"))


def test_materialize_preserves_replaced_staging_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    artifact = provider.store_bytes(b"payload", role="payload", logical_name="blob")
    destination_parent = tmp_path / "destination"
    destination = destination_parent / "output.bin"
    staging_path = destination_parent / ".feedbax-materialization-staging"
    moved_staging = tmp_path / "moved-materialization-staging"
    original_link = custody_module._link_materialized_file
    foreign_identity: tuple[int, int] | None = None

    def replace_staging_then_link(
        temporary_name: str,
        destination_name: str,
        *,
        temporary_parent_descriptor: int,
        parent_descriptor: int,
    ) -> None:
        nonlocal foreign_identity
        staging_path.rename(moved_staging)
        staging_path.mkdir(mode=0o700)
        foreign_stat = staging_path.stat()
        foreign_identity = (foreign_stat.st_dev, foreign_stat.st_ino)
        original_link(
            temporary_name,
            destination_name,
            temporary_parent_descriptor=temporary_parent_descriptor,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(custody_module, "_link_materialized_file", replace_staging_then_link)

    assert provider.materialize(artifact, destination) == destination

    assert destination.read_bytes() == b"payload"
    assert foreign_identity is not None
    surviving_stat = staging_path.stat()
    assert (surviving_stat.st_dev, surviving_stat.st_ino) == foreign_identity
    assert list(staging_path.iterdir()) == []
    assert list(moved_staging.iterdir()) == []


def test_missing_descriptor_capability_fails_closed_with_public_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
    artifact = provider.store_bytes(b"payload", role="payload", logical_name="blob")
    monkeypatch.setattr(custody_module.os, "O_NOFOLLOW", 0)

    with pytest.raises(ArtifactBlobContainmentError, match="no-follow"):
        provider.get_bytes(artifact)
