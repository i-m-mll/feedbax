"""Build a verifiable Git identity into Feedbax wheels."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


SCHEMA_VERSION = "feedbax.distribution_provenance.v1"
PROVENANCE_PATH = "feedbax/_distribution_provenance.json"
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "LC_ALL": "C",
    "PATH": os.defpath,
}


def _git(root: Path, *args: str, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
        env=_GIT_ENVIRONMENT,
        text=text,
    )
    return result.stdout


def _object_id(kind: str, data: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).hexdigest()


def _tree_entries(data: bytes) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    offset = 0
    while offset < len(data):
        mode_end = data.index(b" ", offset)
        name_end = data.index(b"\0", mode_end + 1)
        oid_end = name_end + 21
        if oid_end > len(data):
            raise RuntimeError("Feedbax build provenance encountered a malformed Git tree")
        mode = data[offset:mode_end].decode("ascii")
        name = data[mode_end + 1 : name_end].decode("utf-8")
        oid = data[name_end + 1 : oid_end].hex()
        entries.append((mode, name, oid))
        offset = oid_end
    return entries


def _tree_object(root: Path, oid: str) -> bytes:
    data = _git(root, "cat-file", "tree", oid)
    assert isinstance(data, bytes)
    if _object_id("tree", data) != oid:
        raise RuntimeError(f"Git returned tree bytes that do not match object {oid}")
    return data


def _collect_package_trees(root: Path, root_tree_oid: str) -> dict[str, str]:
    root_tree = _tree_object(root, root_tree_oid)
    package_matches = [
        oid
        for mode, name, oid in _tree_entries(root_tree)
        if mode == "40000" and name == "feedbax"
    ]
    if len(package_matches) != 1:
        raise RuntimeError("the build commit must contain exactly one feedbax package tree")

    encoded = {root_tree_oid: base64.b64encode(root_tree).decode("ascii")}
    pending = [package_matches[0]]
    while pending:
        oid = pending.pop()
        if oid in encoded:
            continue
        data = _tree_object(root, oid)
        encoded[oid] = base64.b64encode(data).decode("ascii")
        pending.extend(
            child_oid
            for mode, _name, child_oid in _tree_entries(data)
            if mode == "40000"
        )
    return encoded


def build_provenance(root: Path) -> bytes:
    """Return canonical versioned provenance for a clean Git checkout."""
    try:
        revision = str(_git(root, "rev-parse", "--verify", "HEAD^{commit}", text=True)).strip()
        dirty = str(
            _git(root, "status", "--porcelain=v1", "--untracked-files=normal", text=True)
        ).strip()
        commit_object = _git(root, "cat-file", "commit", revision)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Feedbax wheels must be built from a Git checkout with a resolvable HEAD"
        ) from exc
    if dirty:
        raise RuntimeError(
            "Feedbax wheels must be built from a clean source checkout; "
            "commit or remove every tracked and untracked change before building: "
            f"{dirty}"
        )
    assert isinstance(commit_object, bytes)
    if _object_id("commit", commit_object) != revision:
        raise RuntimeError("the build checkout returned unverifiable Git commit bytes")
    first_line = commit_object.splitlines()[0].decode("ascii")
    if not first_line.startswith("tree "):
        raise RuntimeError("the build commit does not declare a Git tree")
    root_tree_oid = first_line.removeprefix("tree ")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "revision": revision,
        "commit_object": base64.b64encode(commit_object).decode("ascii"),
        "tree_objects": _collect_package_trees(root, root_tree_oid),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


class CustomBuildHook(BuildHookInterface):
    """Add commit- and tree-authenticated provenance to wheel contents."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        provenance = build_provenance(Path(self.root))
        artifact = Path(self.root) / PROVENANCE_PATH
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(provenance)
        self._artifact = artifact

    def finalize(
        self, version: str, build_data: dict[str, object], artifact_path: str
    ) -> None:
        self._artifact.unlink(missing_ok=True)
