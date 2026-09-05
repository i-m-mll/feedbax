from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from feedbax.contracts.artifact_custody import LegacyContentFormatError
from feedbax.orchestration.staged_root_custody import (
    StagedRootSourceBinding,
    migrate_run_bundle_staged_roots_v12,
    migrate_staged_root_v1,
    reject_legacy_run_bundle_staged_roots,
    require_content_binding,
    seal_staged_root,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def test_staged_root_v1_migrates_offline_and_runtime_requires_content_binding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "empty").mkdir(parents=True)
    (source / "data.bin").write_bytes(b"migrated exact bytes")
    sealed = seal_staged_root(
        StagedRootSourceBinding(name="training", kind="manifest-store", root=source),
        snapshot_parent=tmp_path / "snapshots",
    )

    binding, receipt = migrate_staged_root_v1(
        sealed.custody,
        source_root=sealed.staging_root,
        destination_store=tmp_path / "store",
    )
    saved_bundle = {
        "schema_id": "feedbax.orchestration.run_bundle",
        "schema_version": "feedbax.orchestration.run_bundle.v12",
        "staged_roots": [sealed.custody.model_dump(mode="json")],
    }
    bundle_bindings, bundle_receipt = migrate_run_bundle_staged_roots_v12(
        saved_bundle,
        source_roots={"manifest-store": {"training": sealed.staging_root}},
        destination_store=tmp_path / "store",
    )

    assert bundle_bindings == (binding,)
    assert bundle_receipt.source_schema_version == "feedbax.orchestration.run_bundle.v12"
    with pytest.raises(LegacyContentFormatError, match="--run-bundle-v12"):
        reject_legacy_run_bundle_staged_roots(saved_bundle)

    shutil.rmtree(source)
    sealed.staging_root.rename(sealed.staging_root.with_name("legacy-source-hidden"))
    assert require_content_binding(binding) == binding
    assert receipt.source_schema_version == "feedbax.orchestration.staged_root_custody.v1"
    assert {mapping.source_ref for mapping in receipt.mappings} == {
        sealed.custody.custody_ref,
        f"{sealed.custody.custody_ref}#data.bin",
    }
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    tree = provider.get_tree(binding.tree)
    assert tree.directories == ("empty",)
    assert provider.get_object(tree.files[0].object) == b"migrated exact bytes"
    with pytest.raises(LegacyContentFormatError, match="migrate-content-store"):
        require_content_binding(sealed.custody)
