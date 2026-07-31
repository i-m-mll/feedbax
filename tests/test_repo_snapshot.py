from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import feedbax
from feedbax.orchestration.repo_snapshot import (
    REPO_SNAPSHOT_CACHE_DIR_ENV,
    RepoSnapshotCacheFault,
    RepoSnapshotError,
    RepoSnapshotManifest,
    default_repo_snapshot_cache_dir,
    restore_repo_snapshots,
    seal_repo_snapshot,
    verify_repo_snapshot,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "tracked repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "snapshot@example.invalid")
    _git(root, "config", "user.name", "Snapshot Test")
    (root / ".gitignore").write_text("*.secret\nignored-dir/\n", encoding="utf-8")
    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    (root / "space name.txt").write_text("space\n", encoding="utf-8")
    _git(root, "add", ".gitignore", "tracked.txt", "space name.txt")
    _git(root, "commit", "-m", "fixture")
    return root


def test_snapshot_contains_only_tracked_working_tree_bytes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "tracked.txt").write_text("modified\n", encoding="utf-8")
    (root / "ignored.secret").write_text("never ship\n", encoding="utf-8")
    (root / "ignored-dir").mkdir()
    (root / "ignored-dir" / "novel-cache.bin").write_bytes(b"never ship")

    snapshot = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")

    assert (snapshot.staging_root / "tracked.txt").read_text(encoding="utf-8") == "modified\n"
    assert (snapshot.staging_root / "space name.txt").read_text(encoding="utf-8") == "space\n"
    assert not (snapshot.staging_root / "ignored.secret").exists()
    assert not (snapshot.staging_root / "ignored-dir").exists()
    assert snapshot.record.dirty
    assert snapshot.record.file_count == 3
    assert snapshot.staging_root.stat().st_mode & 0o222 == 0
    assert snapshot.staging_root.joinpath("tracked.txt").stat().st_mode & 0o222 == 0


def test_snapshot_is_immutable_after_seal_and_distinguishes_dirty_bytes(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    (root / "tracked.txt").write_text("dirty one\n", encoding="utf-8")
    first = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")
    (root / "tracked.txt").write_text("dirty two\n", encoding="utf-8")
    second = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")

    assert (first.staging_root / "tracked.txt").read_text(encoding="utf-8") == "dirty one\n"
    assert first.record.commit == second.record.commit
    assert first.record.dirty is second.record.dirty is True
    assert first.record.content_sha256 != second.record.content_sha256


def test_snapshot_omits_tracked_deletion_and_preserves_symlink(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "target.txt").write_text("target bytes\n", encoding="utf-8")
    (root / "tracked-link").symlink_to("target.txt")
    (root / "deleted.txt").write_text("delete me\n", encoding="utf-8")
    _git(root, "add", "target.txt", "tracked-link", "deleted.txt")
    _git(root, "commit", "-m", "link and deletion fixture")
    (root / "deleted.txt").unlink()

    snapshot = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")

    link = snapshot.staging_root / "tracked-link"
    assert link.is_symlink()
    assert os.readlink(link) == "target.txt"
    assert not (snapshot.staging_root / "deleted.txt").exists()

    (root / "tracked-link").unlink()
    (root / "tracked-link").write_text("now regular\n", encoding="utf-8")
    converted = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")
    assert converted.staging_root.joinpath("tracked-link").is_file()
    assert not converted.staging_root.joinpath("tracked-link").is_symlink()


def test_snapshot_preserves_unstaged_working_tree_type_and_executable_edits(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    tracked = root / "tracked.txt"
    tracked.chmod(0o755)
    executable = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")
    assert executable.record.dirty
    assert executable.staging_root.joinpath("tracked.txt").stat().st_mode & 0o111

    tracked.unlink()
    tracked.symlink_to("space name.txt")
    symlink = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")
    assert symlink.staging_root.joinpath("tracked.txt").is_symlink()
    assert os.readlink(symlink.staging_root / "tracked.txt") == "space name.txt"

    tracked.unlink()
    tracked.write_text("regular again\n", encoding="utf-8")
    regular = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")
    assert regular.staging_root.joinpath("tracked.txt").is_file()
    assert not regular.staging_root.joinpath("tracked.txt").is_symlink()


def test_restored_snapshot_rejects_staging_byte_mutation(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    snapshot_parent = tmp_path / "snapshots"
    snapshot = seal_repo_snapshot("repo", root, snapshot_parent=snapshot_parent)
    snapshot.staging_root.chmod(0o755)
    tracked = snapshot.staging_root / "tracked.txt"
    tracked.chmod(0o644)
    tracked.write_text("tampered\n", encoding="utf-8")
    manifest = RepoSnapshotManifest(repos={"repo": snapshot.record})

    with pytest.raises(RepoSnapshotCacheFault, match="cache entry is damaged"):
        restore_repo_snapshots(
            {"repo": root},
            manifest,
            snapshot_parent=snapshot_parent,
        )


def test_transferred_content_mismatch_is_not_reported_as_a_cache_fault(tmp_path: Path) -> None:
    """A transfer destination that disagrees with its record is a real content mismatch."""
    root = _repo(tmp_path)
    snapshot = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")
    destination = tmp_path / "transferred"
    shutil.copytree(snapshot.staging_root, destination)
    subprocess.run(["chmod", "-R", "u+w", destination], check=True)
    (destination / "tracked.txt").write_text("different bytes\n", encoding="utf-8")

    with pytest.raises(RepoSnapshotError) as mismatch:
        verify_repo_snapshot(
            destination,
            content_sha256=snapshot.record.content_sha256,
            file_count=snapshot.record.file_count,
        )

    assert "digest mismatch" in str(mismatch.value)
    assert not isinstance(mismatch.value, RepoSnapshotCacheFault)


def test_snapshot_fails_closed_for_non_top_level_and_gitlink(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    nested = root / "nested"
    nested.mkdir()
    with pytest.raises(RepoSnapshotError, match="must equal the Git top level"):
        seal_repo_snapshot("repo", nested, snapshot_parent=tmp_path / "snapshots")

    commit = _git(root, "rev-parse", "HEAD")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{commit},vendor/sub")
    with pytest.raises(RepoSnapshotError, match="gitlink/submodule"):
        seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")


def test_wholesale_local_rsync_deletes_stale_ungoverned_file(tmp_path: Path) -> None:
    rsync = shutil.which("rsync")
    assert rsync is not None, "rsync is required for the governed-transfer contract test"
    root = _repo(tmp_path)
    (root / "ignored.secret").write_text("local secret\n", encoding="utf-8")
    snapshot = seal_repo_snapshot("repo", root, snapshot_parent=tmp_path / "snapshots")
    remote = tmp_path / "remote set"
    remote.mkdir()
    (remote / "ignored.secret").write_text("stale remote secret\n", encoding="utf-8")
    (remote / "space name.txt").write_text("stale mode\n", encoding="utf-8")
    (remote / "space name.txt").chmod(0o755)

    subprocess.run(
        [
            rsync,
            "-a",
            "--delete",
            f"{snapshot.staging_root}/",
            f"{remote}/",
        ],
        check=True,
    )
    subprocess.run(["chmod", "-R", "u+w", remote], check=True)
    assert not (remote / "ignored.secret").exists()
    assert not remote.joinpath("space name.txt").stat().st_mode & 0o111

    (root / "tracked.txt").unlink()
    after_deletion = seal_repo_snapshot(
        "repo", root, snapshot_parent=tmp_path / "snapshots"
    )
    subprocess.run(
        [
            rsync,
            "-a",
            "--delete",
            f"{after_deletion.staging_root}/",
            f"{remote}/",
        ],
        check=True,
    )
    subprocess.run(["chmod", "-R", "u+w", remote], check=True)

    assert not (remote / "tracked.txt").exists()
    assert (remote / "space name.txt").read_text(encoding="utf-8") == "space\n"


# --- sealed-cache location, damage repair, and concurrent sealers ------------------------

_SEALER_COUNT = 6

# One sealer process. Every sealer announces itself in the gate directory and spins until
# all of them have, so the processes enter seal_repo_snapshot within microseconds of each
# other rather than merely overlapping somewhere.
_SEALER_SCRIPT = """
import json
import os
import sys
import time
import traceback
from pathlib import Path

from feedbax.orchestration.repo_snapshot import seal_repo_snapshot

index, source, parent, gate, result = sys.argv[1:6]
expected = int(os.environ["FEEDBAX_TEST_SEALER_COUNT"])
gate_path = Path(gate)
(gate_path / f"ready-{index}").write_text("", encoding="utf-8")
deadline = time.monotonic() + 120.0
while len(list(gate_path.glob("ready-*"))) < expected:
    if time.monotonic() > deadline:
        raise TimeoutError("concurrent sealers failed to rendezvous")

try:
    record = seal_repo_snapshot("repo", source, snapshot_parent=parent).record
    payload = {"ok": True, "content_sha256": record.content_sha256,
               "file_count": record.file_count}
except BaseException:
    payload = {"ok": False, "error": traceback.format_exc()}
Path(result).write_text(json.dumps(payload), encoding="utf-8")
"""


def _bulky_repo(root: Path) -> Path:
    """Build a repo whose seal takes long enough for concurrent sealers to interleave."""
    root.mkdir(parents=True)
    _git(root, "init")
    _git(root, "config", "user.email", "concurrent@example.invalid")
    _git(root, "config", "user.name", "Concurrent Test")
    for index in range(48):
        package = root / f"package {index % 6}"
        package.mkdir(exist_ok=True)
        module = package / f"module_{index}.py"
        module.write_text(f"VALUE = {index}\n" + "# padding\n" * 64, encoding="utf-8")
        if index % 8 == 0:
            module.chmod(0o755)
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "concurrent fixture")
    return root


def _run_concurrent_sealers(tmp_path: Path, sources: list[Path], parent: Path) -> list[dict]:
    """Seal identical content from separate OS processes that start together."""
    workspace = tmp_path / "concurrent"
    gate = workspace / "gate"
    gate.mkdir(parents=True)
    script = workspace / "sealer.py"
    script.write_text(_SEALER_SCRIPT, encoding="utf-8")
    # The shared virtualenv may have Feedbax installed from a different checkout, so pin
    # the subprocesses to the same package tree this test process imported.
    package_root = Path(feedbax.__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "FEEDBAX_TEST_SEALER_COUNT": str(len(sources)),
        "PYTHONPATH": os.pathsep.join(
            [str(package_root), *([os.environ["PYTHONPATH"]] if "PYTHONPATH" in os.environ else [])]
        ),
    }
    results = [workspace / f"result-{index}.json" for index in range(len(sources))]
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(index),
                str(source),
                str(parent),
                str(gate),
                str(results[index]),
            ],
            env=environment,
        )
        for index, source in enumerate(sources)
    ]
    for process in processes:
        assert process.wait(timeout=180) == 0, "a concurrent sealer process crashed"
    return [json.loads(path.read_text(encoding="utf-8")) for path in results]


def _reap_sealed_entry(staging_root: Path, *, keep: int = 0) -> None:
    """Damage a sealed entry the way an operating-system temporary-file reaper does.

    The reaper runs as root, so the sealed read-only modes do not stop it; it unlinks
    files and leaves the read-only entry directories in place.
    """
    subprocess.run(["chmod", "-R", "u+w", staging_root], check=True)
    files = sorted(path for path in staging_root.rglob("*") if path.is_file())
    for path in files[keep:]:
        path.unlink()
    subprocess.run(["chmod", "-R", "a-w", staging_root], check=True)


def test_concurrent_sealers_of_identical_content_all_succeed(tmp_path: Path) -> None:
    template = _bulky_repo(tmp_path / "template")
    sources = []
    for index in range(_SEALER_COUNT):
        source = tmp_path / f"checkout-{index}"
        shutil.copytree(template, source, symlinks=True)
        sources.append(source)
    parent = tmp_path / "shared-cache"

    results = _run_concurrent_sealers(tmp_path, sources, parent)

    failures = [result["error"] for result in results if not result["ok"]]
    assert not failures, "concurrent sealers of identical content must all succeed:\n" + (
        "\n".join(failures)
    )
    digests = {result["content_sha256"] for result in results}
    assert len(digests) == 1
    name_key = next(path for path in parent.iterdir() if path.is_dir())
    published = [path for path in name_key.iterdir() if path.name == digests.copy().pop()]
    assert len(published) == 1
    assert published[0].stat().st_mode & 0o222 == 0
    assert sorted(path.name for path in published[0].rglob("*.py")) == sorted(
        path.name for path in template.rglob("*.py")
    )


def test_concurrent_sealers_repair_a_reaped_cache_entry(tmp_path: Path) -> None:
    template = _bulky_repo(tmp_path / "template")
    sources = []
    for index in range(_SEALER_COUNT):
        source = tmp_path / f"checkout-{index}"
        shutil.copytree(template, source, symlinks=True)
        sources.append(source)
    parent = tmp_path / "shared-cache"
    seeded = seal_repo_snapshot("repo", sources[0], snapshot_parent=parent)
    _reap_sealed_entry(seeded.staging_root)
    assert not any(path.is_file() for path in seeded.staging_root.rglob("*"))

    results = _run_concurrent_sealers(tmp_path, sources, parent)

    failures = [result["error"] for result in results if not result["ok"]]
    assert not failures, "a reaped cache entry must not fail concurrent later runs:\n" + (
        "\n".join(failures)
    )
    assert {result["content_sha256"] for result in results} == {
        seeded.record.content_sha256
    }
    restored = seeded.staging_root
    assert restored.stat().st_mode & 0o222 == 0
    verify_repo_snapshot(
        restored,
        content_sha256=seeded.record.content_sha256,
        file_count=seeded.record.file_count,
    )


@pytest.mark.parametrize("keep", [0, 1])
def test_damaged_cache_entry_is_repaired_by_a_later_run(tmp_path: Path, keep: int) -> None:
    root = _repo(tmp_path)
    parent = tmp_path / "snapshots"
    first = seal_repo_snapshot("repo", root, snapshot_parent=parent)
    _reap_sealed_entry(first.staging_root, keep=keep)

    second = seal_repo_snapshot("repo", root, snapshot_parent=parent)

    assert second.staging_root == first.staging_root
    assert second.record == first.record
    assert second.staging_root.joinpath("tracked.txt").read_text(encoding="utf-8") == "committed\n"
    assert second.staging_root.stat().st_mode & 0o222 == 0
    verify_repo_snapshot(
        second.staging_root,
        content_sha256=second.record.content_sha256,
        file_count=second.record.file_count,
    )
    assert not list(second.staging_root.parent.glob(".damaged-*"))


def test_default_cache_dir_is_per_checkout_and_overridable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REPO_SNAPSHOT_CACHE_DIR_ENV, str(tmp_path / "explicit cache"))
    assert default_repo_snapshot_cache_dir() == tmp_path / "explicit cache"

    monkeypatch.delenv(REPO_SNAPSHOT_CACHE_DIR_ENV)
    default = default_repo_snapshot_cache_dir()
    package_root = Path(feedbax.__file__).resolve().parents[1]
    common_dir = Path(
        subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=package_root,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout.strip()
    ).resolve()

    assert default == common_dir / "feedbax_repo_snapshots"
    system_temporary_root = Path(tempfile.gettempdir()).resolve()
    assert system_temporary_root not in default.parents
    assert default != system_temporary_root / "feedbax-repo-snapshots"
