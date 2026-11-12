"""Verify project store identity and ref visibility across process boundaries."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from feedbax.contracts.artifact_custody import (
    ArtifactBlobIntegrityError,
    ArtifactBlobReferenceError,
)
from feedbax.contracts.dataset import DatasetProductRef
from feedbax.contracts.project_io import load_text_project
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
    open_project_store,
)
from feedbax.persistence.dataset import open_stored_dataset, publish_dataset


def test_authored_store_retains_identity_refs_and_the_same_object_after_move(tmp_path, monkeypatch):
    root = tmp_path / "authored"
    root.mkdir()
    (root / "feedbax.project.json").write_text(
        '{"schema_version":"feedbax.project.v1","project":"reaching","store":"local/store"}'
    )
    legacy = ImmutableArtifactBlobProvider(root / "local/store")
    dataset = Dataset.from_arrays(
        {"position": Variable(np.array([[1, 2], [3, 4]]), ("trial", "sample"))}
    )
    binding = publish_dataset(legacy, dataset)
    object_before = legacy.object_path(binding.object_ref).stat()
    monkeypatch.chdir(tmp_path)

    store = open_project_store(load_text_project(root))
    identity = store.store_id
    assert json.loads((store.root / "store.json").read_bytes()) == {"id": identity}
    assert store.compare_and_swap_ref(
        "fixtures/dataset", expected=None, replacement=binding.model_dump(mode="json")
    )
    relocated = root.rename(tmp_path / "relocated")

    reopened = open_project_store(load_text_project(relocated))
    assert reopened.store_id == identity
    assert ImmutableArtifactBlobProvider(tmp_path / "other").store_id != identity
    assert reopened.root == relocated.resolve() / "local/store"
    restored = DatasetProductRef.model_validate(reopened.read_ref("fixtures/dataset"))
    assert restored == binding
    path = reopened.object_path(restored.object_ref)
    assert path.parent.parent.parent == reopened.root / "artifacts"
    assert path.stat().st_ino == object_before.st_ino
    with open_stored_dataset(reopened, restored) as dataset:
        np.testing.assert_array_equal(dataset.data("position", (1, slice(None))), [3, 4])


def test_ref_compares_canonical_values_and_refuses_corruption(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path)
    name = "runs/evaluate/evaluate:" + "a" * 64
    old = {"attempt": 2, "accepted": True}
    new = {"attempt": 3}
    assert store.read_ref(name) is None
    assert not store.compare_and_swap_ref(name, expected=old, replacement=new)
    assert store.compare_and_swap_ref(name, expected=None, replacement=old)
    assert not store.compare_and_swap_ref(
        name, expected={"attempt": 2, "accepted": 1}, replacement=new
    )
    assert store.compare_and_swap_ref(
        name, expected={"accepted": True, "attempt": 2}, replacement=new
    )
    assert not store.compare_and_swap_ref(name, expected=old, replacement=None)
    assert store.compare_and_swap_ref(name, expected=new, replacement=None)
    assert store.read_ref(name) is None

    assert store.compare_and_swap_ref(name, expected=None, replacement=old)
    ref_path = tmp_path / "refs" / name
    corrupt = b'{"attempt":2,"attempt":3}'
    ref_path.write_bytes(corrupt)
    with pytest.raises(ArtifactBlobIntegrityError, match="duplicate JSON"):
        store.compare_and_swap_ref(name, expected=None, replacement=new)
    assert ref_path.read_bytes() == corrupt


@pytest.mark.parametrize("name", ["/plan", "../store", "runs//plan", ".temporary", "Pins/alias"])
def test_ref_refuses_ambiguous_or_uncontained_names(tmp_path, name):
    store = ImmutableArtifactBlobProvider(tmp_path)
    with pytest.raises(ArtifactBlobReferenceError, match="ref name"):
        store.compare_and_swap_ref(name, expected=None, replacement={})
    assert not (tmp_path / "refs").exists()


def test_a_store_with_refs_refuses_a_missing_or_corrupt_id(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path)
    assert store.compare_and_swap_ref("plan", expected=None, replacement={})
    identity = tmp_path / "store.json"
    identity.write_text('{"id":"incorrect"}')
    with pytest.raises(ArtifactBlobIntegrityError, match="invalid store identity"):
        _ = store.store_id
    identity.unlink()
    with pytest.raises(ArtifactBlobIntegrityError, match="store identity is missing"):
        store.compare_and_swap_ref("plan", expected={}, replacement={"attempt": 2})
    assert store.read_ref("plan") == {}


def test_competing_processes_mint_one_identity_and_only_one_ref_wins(tmp_path):
    code = """
import json
import sys
from pathlib import Path
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

store = ImmutableArtifactBlobProvider(Path(sys.argv[1]))
sys.stdin.read()
identity = store.store_id
won = store.compare_and_swap_ref("plan", expected=None, replacement={"owner": sys.argv[2]})
print(json.dumps([identity, won]))
"""

    def contender(owner: str) -> tuple[str, bool]:
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path), owner],
            input="\n",
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        identity, won = json.loads(result.stdout)
        return identity, won

    owners = ("left", "right")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(contender, owners))
    store = ImmutableArtifactBlobProvider(tmp_path)
    assert {identity for identity, _ in results} == {store.store_id}
    winners = [owner for owner, (_, won) in zip(owners, results, strict=True) if won]
    assert len(winners) == 1
    assert store.read_ref("plan") == {"owner": winners[0]}


@pytest.mark.parametrize("cut", ["before_rename", "after_rename"])
def test_interrupted_ref_writer_leaves_one_complete_value_and_releases_lock(tmp_path, cut):
    store = ImmutableArtifactBlobProvider(tmp_path)
    old = {"attempt": 1, "status": "ready"}
    new = {"attempt": 2, "status": "ready"}
    assert store.compare_and_swap_ref("plan", expected=None, replacement=old)

    code = """
import fcntl
import os
import sys
from pathlib import Path
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

replace = os.replace
root = Path(sys.argv[1])
cut = sys.argv[2]

def interrupted_replace(source, destination):
    if destination != root / "refs" / "plan":
        raise AssertionError(destination)
    with (root / ".content-store.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise AssertionError("ref writer did not hold the store lock")
    if cut == "after_rename":
        replace(source, destination)
    os._exit(91)

os.replace = interrupted_replace
store = ImmutableArtifactBlobProvider(root)
store.compare_and_swap_ref(
    "plan", expected={"attempt": 1, "status": "ready"},
    replacement={"attempt": 2, "status": "ready"},
)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(store.root), cut],
        capture_output=True,
        timeout=10,
    )
    assert child.returncode == 91, child.stderr.decode()
    visible = old if cut == "before_rename" else new
    assert store.read_ref("plan") == visible
    assert store.compare_and_swap_ref("plan", expected=visible, replacement={"attempt": 3})
    assert store.read_ref("plan") == {"attempt": 3}
