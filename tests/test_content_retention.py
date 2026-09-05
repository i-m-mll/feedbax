from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import feedbax.persistence.content_retention as retention_module
from feedbax.contracts.artifact_custody import (
    ContentGcRequest,
    ContentHoldRecord,
    ContentLeaseRecord,
    ContentRetentionTarget,
    ContentRootRecord,
)
from feedbax.persistence import ContentRetentionError, ImmutableArtifactBlobProvider


pytestmark = pytest.mark.feedbax_contract


def _object_target(reference) -> ContentRetentionTarget:
    return ContentRetentionTarget(ref_kind="object", object=reference)


def _tree_target(reference) -> ContentRetentionTarget:
    return ContentRetentionTarget(ref_kind="tree", tree=reference)


def _path(provider: ImmutableArtifactBlobProvider, digest: str) -> Path:
    return provider.root / "artifacts" / "sha256" / digest[:2] / digest


def _initialize(provider: ImmutableArtifactBlobProvider, reference) -> None:
    provider.register_root(
        ContentRootRecord(
            root_id="initialization",
            root_kind="logical-ref",
            targets=(_object_target(reference),),
        )
    )
    provider.release_root("initialization")


def test_tree_root_keeps_its_full_closure_while_unreferenced_objects_sweep(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one").write_bytes(b"one")
    (source / "two").write_bytes(b"two")
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    tree_ref = provider.ingest_tree(source)
    orphan = provider.store_object(b"orphan")
    provider.register_root(
        ContentRootRecord(
            root_id="export:one",
            root_kind="export",
            targets=(_tree_target(tree_ref),),
        )
    )

    report = provider.collect_garbage()
    assert report.stage == "reported"
    assert {item.sha256 for item in report.decisions if item.decision == "live"} == {
        tree_ref.descriptor.sha256,
        *(entry.object.sha256 for entry in provider.get_tree(tree_ref).files),
    }
    result = provider.collect_garbage(
        ContentGcRequest(destructive=True, expected_generation=report.generation)
    )

    assert result.stage == "swept"
    assert [item.sha256 for item in result.decisions] == [orphan.sha256]
    assert result.domain == "canonical"
    assert result.generation == report.generation + 1
    assert result.root_ids == ("export:one",)
    assert result.byte_count == orphan.size_bytes
    receipt_root = provider.root / "retention" / "receipts" / result.receipt_id
    assert {path.name for path in receipt_root.iterdir()} == {
        "marked.json",
        "quarantined.json",
        "swept.json",
    }
    assert not _path(provider, orphan.sha256).exists()
    assert provider.get_tree(tree_ref).files


def test_dry_run_does_not_mutate_store(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    reference = provider.store_object(b"candidate")
    _initialize(provider, reference)
    before = {
        path.relative_to(provider.root): path.read_bytes()
        for path in provider.root.rglob("*")
        if path.is_file()
    }

    report = provider.collect_garbage()

    after = {
        path.relative_to(provider.root): path.read_bytes()
        for path in provider.root.rglob("*")
        if path.is_file()
    }
    assert report.stage == "reported"
    assert before == after


def test_lease_mutation_invalidates_a_reported_generation_and_preserves_content(
    tmp_path: Path,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    reference = provider.store_object(b"leased")
    _initialize(provider, reference)
    report = provider.collect_garbage()
    lease = ContentLeaseRecord(
        lease_id="attempt:one",
        domain="remote-attempt-cache",
        owner="remote-worker",
        targets=(_object_target(reference),),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        acquired = executor.submit(provider.acquire_lease, lease).result()

    stale = provider.collect_garbage(
        ContentGcRequest(destructive=True, expected_generation=report.generation)
    )
    assert acquired.domain == "remote-attempt-cache"
    assert stale.stage == "no-op"
    assert "generation drifted" in stale.errors[0]
    assert provider.get_object(reference) == b"leased"
    provider.release_lease("attempt:one")
    remote_report = provider.collect_garbage(ContentGcRequest(domain="remote-attempt-cache"))
    remote_result = provider.collect_garbage(
        ContentGcRequest(
            domain="remote-attempt-cache",
            destructive=True,
            expected_generation=remote_report.generation,
        )
    )
    assert remote_result.stage == "swept"
    assert remote_result.domain == "remote-attempt-cache"


def test_object_publication_serializes_against_destructive_gc_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    old = provider.store_object(b"old candidate")
    _initialize(provider, old)
    report = provider.collect_garbage()
    inventory_entered = threading.Event()
    release_inventory = threading.Event()
    original_inventory = retention_module._inventory

    def blocked_inventory(current_provider):
        inventory_entered.set()
        assert release_inventory.wait(timeout=5)
        return original_inventory(current_provider)

    monkeypatch.setattr(retention_module, "_inventory", blocked_inventory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        sweep = executor.submit(
            provider.collect_garbage,
            ContentGcRequest(destructive=True, expected_generation=report.generation),
        )
        assert inventory_entered.wait(timeout=5)
        publication = executor.submit(provider.store_object, b"published after mark")
        assert not publication.done()
        release_inventory.set()
        assert sweep.result().stage == "swept"
        published = publication.result()
    assert provider.get_object(published) == b"published after mark"


def test_expired_lease_is_ambiguous_until_explicit_release(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    reference = provider.store_object(b"expired-but-live")
    provider.acquire_lease(
        ContentLeaseRecord(
            lease_id="lease:expired",
            owner="worker",
            targets=(_object_target(reference),),
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
    )

    report = provider.collect_garbage()
    assert report.stage == "no-op"
    assert "expired without explicit release" in report.errors[0]
    generation = report.generation
    destructive = provider.collect_garbage(
        ContentGcRequest(destructive=True, expected_generation=generation)
    )
    assert destructive.stage == "no-op"
    assert provider.get_object(reference) == b"expired-but-live"


def test_nonterminal_restartable_run_requires_durable_terminal_release(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    reference = provider.store_object(b"checkpoint")
    provider.register_root(
        ContentRootRecord(
            root_id="run:one",
            root_kind="run",
            targets=(_object_target(reference),),
            nonterminal=True,
            restartable=True,
        )
    )
    with pytest.raises(ContentRetentionError, match="cannot be released"):
        provider.release_root("run:one")
    blocked = provider.collect_garbage()
    assert blocked.stage == "no-op"
    assert "nonterminal or restartable" in blocked.errors[0]
    provider.set_run_state("run:one", nonterminal=False, restartable=False)
    provider.release_root("run:one")
    report = provider.collect_garbage()
    result = provider.collect_garbage(
        ContentGcRequest(destructive=True, expected_generation=report.generation)
    )
    assert result.stage == "swept"
    assert not _path(provider, reference.sha256).exists()


def test_incomplete_migration_blocks_collection_globally(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    migrating = provider.store_object(b"migrating")
    orphan = provider.store_object(b"unrelated")
    provider.register_root(
        ContentRootRecord(
            root_id="migration:legacy-custody",
            root_kind="migration",
            targets=(_object_target(migrating),),
        )
    )

    report = provider.collect_garbage()
    assert report.stage == "no-op"
    assert "migration root remains incomplete" in report.errors[0]
    destructive = provider.collect_garbage(
        ContentGcRequest(destructive=True, expected_generation=report.generation)
    )
    assert destructive.stage == "no-op"
    assert provider.get_object(orphan) == b"unrelated"


def test_holds_and_remote_cache_non_authority_are_explicit(tmp_path: Path) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    reference = provider.store_object(b"scientific-evidence")
    provider.place_hold(
        ContentHoldRecord(
            hold_id="hold:science",
            hold_kind="scientific",
            reason="awaiting owner disposition",
            targets=(_object_target(reference),),
        )
    )
    with pytest.raises(ContentRetentionError, match="non-authoritative"):
        provider.register_root(
            ContentRootRecord(
                root_id="remote-copy",
                domain="remote-attempt-cache",
                root_kind="logical-ref",
                targets=(_object_target(reference),),
            )
        )
    assert provider.collect_garbage().decisions[0].decision == "live"


@pytest.mark.parametrize(
    "damage",
    ["corrupt-index", "hard-link", "symlink", "missing-live-object"],
)
def test_uncertain_store_state_makes_destructive_gc_a_no_op(
    tmp_path: Path,
    damage: str,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    reference = provider.store_object(b"uncertain")
    orphan = provider.store_object(b"must-not-delete-under-uncertainty")
    provider.register_root(
        ContentRootRecord(
            root_id="live:uncertain",
            root_kind="logical-ref",
            targets=(_object_target(reference),),
        )
    )
    index = provider.root / "retention" / "index.json"
    object_path = _path(provider, reference.sha256)
    if damage == "corrupt-index":
        index.write_bytes(b"not-json")
    elif damage == "hard-link":
        os.link(object_path, tmp_path / "alias")
    elif damage == "symlink":
        object_path.unlink()
        object_path.symlink_to(tmp_path / "outside")
    else:
        object_path.unlink()

    result = provider.collect_garbage(ContentGcRequest(destructive=True, expected_generation=1))
    assert result.stage == "no-op"
    assert result.errors
    assert provider.get_object(orphan) == b"must-not-delete-under-uncertainty"


def test_mark_and_quarantine_recover_after_interrupted_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    reference = provider.store_object(b"recover-after-move")
    _initialize(provider, reference)
    report = provider.collect_garbage()
    original_replace = retention_module.os.replace
    interrupted = False

    def replace_then_interrupt(source, destination):
        nonlocal interrupted
        original_replace(source, destination)
        if Path(source) == _path(provider, reference.sha256) and not interrupted:
            interrupted = True
            raise OSError("simulated crash after quarantine move")

    monkeypatch.setattr(retention_module.os, "replace", replace_then_interrupt)
    first = provider.collect_garbage(
        ContentGcRequest(destructive=True, expected_generation=report.generation)
    )
    assert first.stage == "no-op"
    assert interrupted
    monkeypatch.setattr(retention_module.os, "replace", original_replace)

    recovered = provider.recover_garbage_collection()
    assert recovered is not None and recovered.stage == "swept"
    assert recovered.receipt_id == first.receipt_id
    assert provider.recover_garbage_collection() is None


def test_final_sweep_recovery_is_idempotent_after_deleted_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "store")
    reference = provider.store_object(b"recover-after-unlink")
    _initialize(provider, reference)
    report = provider.collect_garbage()
    original_write = retention_module._write_index
    interrupted = False

    def fail_completed_index(current_provider, index):
        nonlocal interrupted
        if index.pending_gc is None and index.generation > report.generation and not interrupted:
            interrupted = True
            raise OSError("simulated crash before final index commit")
        original_write(current_provider, index)

    monkeypatch.setattr(retention_module, "_write_index", fail_completed_index)
    first = provider.collect_garbage(
        ContentGcRequest(destructive=True, expected_generation=report.generation)
    )
    assert first.stage == "no-op"
    assert not _path(provider, reference.sha256).exists()
    monkeypatch.setattr(retention_module, "_write_index", original_write)

    recovered = provider.recover_garbage_collection()
    assert recovered is not None and recovered.stage == "swept"
    assert recovered.receipt_id == first.receipt_id
