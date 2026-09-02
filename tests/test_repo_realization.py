from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.orchestration.repo_realization import (
    EditableSourceResolution,
    RepoRealizationEntry,
    RepoRealizationError,
    RepoRealizationPlan,
    seal_local_repo_realizations,
    sealed_lock_digest,
    validate_non_overlapping_remote_roots,
)
from feedbax.orchestration.repo_snapshot import RepoSnapshotManifest, RepoSnapshotRecord


def _record(content: str = "1" * 64) -> RepoSnapshotRecord:
    return RepoSnapshotRecord(
        commit="a" * 40,
        dirty=False,
        content_sha256=content,
        source_state_sha256="b" * 64,
        file_count=2,
    )


def _plan(
    *, local_root: str, staging_root: str, remote_root: str = "/work/consumer"
) -> RepoRealizationPlan:
    record = _record()
    lock_digest = hashlib.sha256(b"version = 1\n").hexdigest()
    return RepoRealizationPlan.create(
        primary_repo="consumer",
        repos={
            "consumer": RepoRealizationEntry(
                local_root=local_root,
                staging_root=staging_root,
                remote_root=remote_root,
                snapshot=record,
                sealed_lock_digests={"uv.lock": lock_digest},
            )
        },
        editable_source_resolutions=[],
        snapshot_manifest=RepoSnapshotManifest(repos={"consumer": record}),
    )


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )


def test_plan_digest_excludes_diagnostic_roots_and_detects_tampering() -> None:
    first = _plan(local_root="/live/a", staging_root="/staging/a")
    relocated = _plan(local_root="/live/b", staging_root="/staging/b")

    assert first.plan_digest == relocated.plan_digest
    loaded = RepoRealizationPlan.model_validate_json(first.model_dump_json())
    assert loaded == first

    payload = first.model_dump(mode="json")
    payload["repos"]["consumer"]["remote_root"] = "/work/changed"
    with pytest.raises(ValidationError, match="plan digest mismatch"):
        RepoRealizationPlan.model_validate(payload)


def test_plan_canonicalizes_repos_locks_and_fully_keyed_resolutions() -> None:
    consumer_record = _record("1" * 64)
    provider_record = _record("2" * 64)
    digest = "3" * 64
    repeated_forms = [
        EditableSourceResolution(
            consumer_repo="consumer",
            lock_relative_path="uv.lock",
            source_form="editable",
            spelling="../provider",
            target_repo="provider",
            target_subpath=".",
        ),
        EditableSourceResolution(
            consumer_repo="consumer",
            lock_relative_path="other.lock",
            source_form="editable",
            spelling="../provider",
            target_repo="provider",
            target_subpath=".",
        ),
    ]
    plan = RepoRealizationPlan.create(
        primary_repo="consumer",
        repos={
            "provider": RepoRealizationEntry(
                local_root="/live/provider",
                staging_root="/sealed/provider",
                remote_root="/work/provider",
                snapshot=provider_record,
            ),
            "consumer": RepoRealizationEntry(
                local_root="/live/consumer",
                staging_root="/sealed/consumer",
                remote_root="/work/consumer",
                snapshot=consumer_record,
                sealed_lock_digests={"uv.lock": digest, "other.lock": digest},
            ),
        },
        editable_source_resolutions=list(reversed(repeated_forms)),
        snapshot_manifest=RepoSnapshotManifest(
            repos={"provider": provider_record, "consumer": consumer_record}
        ),
    )

    assert list(plan.repos) == ["consumer", "provider"]
    assert [item.lock_relative_path for item in plan.editable_source_resolutions] == [
        "other.lock",
        "uv.lock",
    ]
    assert len(plan.editable_source_resolutions) == 2


def test_sealed_lock_digest_rejects_symlink_and_uses_sealed_bytes(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    external = tmp_path / "external.lock"
    external.write_bytes(b"external")
    (sealed / "uv.lock").symlink_to(external)

    with pytest.raises(RepoRealizationError, match=r"symlink: .*uv\.lock"):
        sealed_lock_digest(sealed, "uv.lock")

    (sealed / "uv.lock").unlink()
    (sealed / "uv.lock").write_bytes(b"sealed")
    assert sealed_lock_digest(sealed, "uv.lock") == hashlib.sha256(b"sealed").hexdigest()


def test_shared_component_rejects_tracked_lock_symlink_to_external_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "realization@example.invalid")
    _git(root, "config", "user.name", "Realization Test")
    external = tmp_path / "external.lock"
    external.write_bytes(b"version = 1\n")
    (root / "uv.lock").symlink_to(external)
    _git(root, "add", "uv.lock")
    _git(root, "commit", "-m", "tracked symlink")

    snapshot_parent = tmp_path / "snapshots"
    with pytest.raises(RepoRealizationError, match=r"symlink: .*uv\.lock"):
        seal_local_repo_realizations(
            {"repo": root},
            lock_relative_paths={"repo": ["uv.lock"]},
            expected_lock_digests={
                "repo": {"uv.lock": hashlib.sha256(external.read_bytes()).hexdigest()}
            },
            snapshot_parent=snapshot_parent,
        )
    subprocess.run(["chmod", "-R", "u+w", snapshot_parent], check=True)
    shutil.rmtree(snapshot_parent)


@pytest.mark.parametrize(
    "roots",
    [
        {"outer": "/work/a", "inner": "/work/a/b"},
        {"inner": "/work/a/b", "outer": "/work/a"},
    ],
)
def test_remote_roots_reject_nesting_in_both_orders(roots: dict[str, str]) -> None:
    with pytest.raises(RepoRealizationError, match="overlapping remote repo roots"):
        validate_non_overlapping_remote_roots(roots)


def test_shared_component_hashes_only_sealed_lock_and_checks_declaration(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "realization@example.invalid")
    _git(root, "config", "user.name", "Realization Test")
    lock = root / "uv.lock"
    lock.write_bytes(b"version = 1\n")
    _git(root, "add", "uv.lock")
    _git(root, "commit", "-m", "fixture")
    expected = hashlib.sha256(lock.read_bytes()).hexdigest()

    manifest, realized = seal_local_repo_realizations(
        {"repo": root},
        lock_relative_paths={"repo": ["uv.lock"]},
        expected_lock_digests={"repo": {"uv.lock": expected}},
        snapshot_parent=tmp_path / "snapshots",
    )
    lock.write_bytes(b"mutated after sealing")

    assert realized["repo"].sealed_lock_digests["uv.lock"] == expected
    assert manifest.repos["repo"] == realized["repo"].snapshot.record

    with pytest.raises(RepoRealizationError, match="declared lockfile hash mismatch"):
        seal_local_repo_realizations(
            {"repo": root},
            lock_relative_paths={"repo": ["uv.lock"]},
            expected_lock_digests={"repo": {"uv.lock": expected}},
            snapshot_parent=tmp_path / "other-snapshots",
        )
