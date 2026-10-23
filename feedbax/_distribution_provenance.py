"""Versioned, stdlib-only distribution provenance generation and verification."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path


SCHEMA_VERSION = "feedbax.distribution_provenance.v1"
PROVENANCE_FILENAME = "_distribution_provenance.json"
PROVENANCE_PATH = f"feedbax/{PROVENANCE_FILENAME}"
_GIT_OBJECT_ID_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "LC_ALL": "C",
    "PATH": os.defpath,
}


class DistributionProvenanceError(ValueError):
    """Distribution provenance is absent, malformed, or unverifiable."""


def _git(
    root: Path,
    args: Sequence[str],
    *,
    check: bool,
    text: bool,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=check,
        env=_GIT_ENVIRONMENT,
        text=text,
    )


def _object_id(kind: str, data: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).hexdigest()


def _parse_tree(data: bytes) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    offset = 0
    try:
        while offset < len(data):
            mode_end = data.index(b" ", offset)
            name_end = data.index(b"\0", mode_end + 1)
            oid_end = name_end + 21
            if oid_end > len(data):
                raise ValueError
            mode = data[offset:mode_end].decode("ascii")
            name = data[mode_end + 1 : name_end].decode("utf-8")
            entries.append((mode, name, data[name_end + 1 : oid_end].hex()))
            offset = oid_end
    except (UnicodeDecodeError, ValueError) as exc:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance contains a malformed Git tree"
        ) from exc
    return entries


def _tree_object(root: Path, oid: str) -> bytes:
    try:
        result = _git(root, ["cat-file", "tree", oid], check=True, text=False)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DistributionProvenanceError(
            f"cannot read Git tree object {oid} while building Feedbax provenance"
        ) from exc
    data = result.stdout
    assert isinstance(data, bytes)
    if _object_id("tree", data) != oid:
        raise DistributionProvenanceError(
            f"Git returned tree bytes that do not match object {oid}"
        )
    return data


def _collect_package_trees(root: Path, root_tree_oid: str) -> dict[str, str]:
    root_tree = _tree_object(root, root_tree_oid)
    package_matches = [
        oid
        for mode, name, oid in _parse_tree(root_tree)
        if mode == "40000" and name == "feedbax"
    ]
    if len(package_matches) != 1:
        raise DistributionProvenanceError(
            "the build commit must contain exactly one feedbax package tree"
        )

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
            for mode, _name, child_oid in _parse_tree(data)
            if mode == "40000"
        )
    return encoded


def _exact_git_root(root: Path) -> bool:
    try:
        result = _git(root, ["rev-parse", "--show-toplevel"], check=False, text=True)
    except OSError:
        return False
    return result.returncode == 0 and Path(str(result.stdout).strip()).resolve() == root.resolve()


def build_provenance_from_git(root: Path) -> bytes:
    """Generate canonical provenance from one exact, clean Git checkout."""
    if not _exact_git_root(root):
        raise DistributionProvenanceError(
            "Feedbax provenance generation requires the exact root of a Git checkout"
        )
    try:
        revision_result = _git(
            root, ["rev-parse", "--verify", "HEAD^{commit}"], check=True, text=True
        )
        dirty_result = _git(
            root,
            ["status", "--porcelain=v1", "--untracked-files=normal"],
            check=True,
            text=True,
        )
        commit_result = _git(
            root,
            ["cat-file", "commit", str(revision_result.stdout).strip()],
            check=True,
            text=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DistributionProvenanceError(
            "Feedbax wheels and sdists must be built from a Git checkout with "
            "a resolvable HEAD"
        ) from exc
    revision = str(revision_result.stdout).strip()
    dirty = str(dirty_result.stdout).strip()
    if dirty:
        raise DistributionProvenanceError(
            "Feedbax wheels and sdists must be built from a clean source checkout; "
            "commit or remove every tracked and untracked change before building: "
            f"{dirty}"
        )
    commit_object = commit_result.stdout
    assert isinstance(commit_object, bytes)
    if not _GIT_OBJECT_ID_RE.fullmatch(revision) or _object_id("commit", commit_object) != revision:
        raise DistributionProvenanceError(
            "the build checkout returned unverifiable Git commit bytes"
        )
    try:
        first_line = commit_object.splitlines()[0].decode("ascii")
    except (IndexError, UnicodeDecodeError) as exc:
        raise DistributionProvenanceError("the build commit object is malformed") from exc
    if not first_line.startswith("tree "):
        raise DistributionProvenanceError("the build commit does not declare a Git tree")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "revision": revision,
        "commit_object": base64.b64encode(commit_object).decode("ascii"),
        "tree_objects": _collect_package_trees(root, first_line.removeprefix("tree ")),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def verify_provenance_bytes(package_root: Path, encoded_payload: bytes) -> str:
    """Verify provenance structure, Git identity, and every installed package byte."""
    try:
        payload = json.loads(encoded_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance is unreadable or malformed"
        ) from exc
    expected_keys = {"schema_version", "revision", "commit_object", "tree_objects"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance has an unsupported structure"
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance schema is unsupported: "
            f"observed={payload['schema_version']!r} expected={SCHEMA_VERSION!r}"
        )
    revision = payload["revision"]
    if not isinstance(revision, str) or not _GIT_OBJECT_ID_RE.fullmatch(revision):
        raise DistributionProvenanceError(
            "Feedbax distribution provenance has a malformed revision"
        )
    try:
        commit_object = base64.b64decode(payload["commit_object"], validate=True)
    except (TypeError, ValueError) as exc:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance has a malformed commit object"
        ) from exc
    if _object_id("commit", commit_object) != revision:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance commit identity is unverifiable"
        )
    try:
        commit_tree_line = commit_object.splitlines()[0].decode("ascii")
    except (IndexError, UnicodeDecodeError) as exc:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance commit object is malformed"
        ) from exc
    if not commit_tree_line.startswith("tree "):
        raise DistributionProvenanceError(
            "Feedbax distribution provenance commit object has no tree"
        )
    tree_payload = payload["tree_objects"]
    if not isinstance(tree_payload, dict) or not tree_payload:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance has no Git tree objects"
        )
    trees: dict[str, bytes] = {}
    try:
        for oid, encoded in tree_payload.items():
            if not isinstance(oid, str) or not _GIT_OBJECT_ID_RE.fullmatch(oid):
                raise ValueError
            data = base64.b64decode(encoded, validate=True)
            if _object_id("tree", data) != oid:
                raise ValueError
            trees[oid] = data
    except (TypeError, ValueError) as exc:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance contains an unverifiable Git tree"
        ) from exc

    root_oid = commit_tree_line.removeprefix("tree ")
    root_tree = trees.get(root_oid)
    if root_tree is None:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance omits the commit's root tree"
        )
    package_entries = [
        oid
        for mode, name, oid in _parse_tree(root_tree)
        if mode == "40000" and name == "feedbax"
    ]
    if len(package_entries) != 1:
        raise DistributionProvenanceError(
            "Feedbax distribution provenance does not identify one package tree"
        )

    visited = {root_oid}
    expected_files: set[Path] = set()

    def verify_tree(oid: str, relative: Path) -> None:
        data = trees.get(oid)
        if data is None:
            raise DistributionProvenanceError(
                "Feedbax distribution provenance omits a package tree"
            )
        visited.add(oid)
        for mode, name, child_oid in _parse_tree(data):
            child = relative / name
            if mode == "40000":
                verify_tree(child_oid, child)
                continue
            if mode not in {"100644", "100755"}:
                raise DistributionProvenanceError(
                    "Feedbax distribution provenance contains an unsupported package "
                    f"entry mode: path={child} mode={mode}"
                )
            installed = package_root / child
            try:
                contents = installed.read_bytes()
            except OSError as exc:
                raise DistributionProvenanceError(
                    f"Feedbax distribution is missing committed file {child}"
                ) from exc
            if _object_id("blob", contents) != child_oid:
                raise DistributionProvenanceError(
                    f"Feedbax distribution file does not match commit: {child}"
                )
            expected_files.add(child)

    verify_tree(package_entries[0], Path())
    if visited != set(trees):
        raise DistributionProvenanceError(
            "Feedbax distribution provenance contains conflicting Git trees"
        )
    actual_files = {
        path.relative_to(package_root)
        for path in package_root.rglob("*")
        if path.is_file()
        and path.name != PROVENANCE_FILENAME
        and "__pycache__" not in path.parts
    }
    unexpected = sorted(actual_files - expected_files)
    if unexpected:
        raise DistributionProvenanceError(
            "Feedbax distribution contains files outside its commit identity: "
            + ", ".join(map(str, unexpected))
        )
    return revision


def load_and_verify_provenance(package_root: Path) -> tuple[bytes, str]:
    """Read and verify a provenance-bearing Feedbax package directory."""
    provenance_path = package_root / PROVENANCE_FILENAME
    try:
        encoded = provenance_path.read_bytes()
    except OSError as exc:
        raise DistributionProvenanceError(
            f"Feedbax distribution provenance is missing: {provenance_path}"
        ) from exc
    return encoded, verify_provenance_bytes(package_root, encoded)


def provenance_for_distribution_build(root: Path) -> bytes:
    """Generate from clean Git or verify and carry forward an sdist identity."""
    if _exact_git_root(root):
        return build_provenance_from_git(root)
    encoded, _revision = load_and_verify_provenance(root / "feedbax")
    return encoded
