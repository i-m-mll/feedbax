"""Pure-stdlib decoder for content-addressed resolved-semantics snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping


SNAPSHOT_SCHEMA_ID = "feedbax.spec.training_run_resolved_semantics"
SNAPSHOT_SCHEMA_VERSION = f"{SNAPSHOT_SCHEMA_ID}.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def decode_resolved_snapshot(snapshot: Mapping[str, Any]) -> Any:
    """Reconstruct a JSON tree from a snapshot root hash and node table."""
    if set(snapshot) != {"schema_id", "schema_version", "root_hash", "nodes"}:
        raise ValueError("resolved-semantics snapshot has missing or extra fields")
    if snapshot.get("schema_id") != SNAPSHOT_SCHEMA_ID:
        raise ValueError("unsupported resolved-semantics snapshot schema_id")
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported resolved-semantics snapshot schema_version")
    nodes = snapshot.get("nodes")
    root_hash = snapshot.get("root_hash")
    if not isinstance(nodes, Mapping) or not isinstance(root_hash, str):
        raise ValueError("snapshot requires an object node table and string root_hash")
    if not _SHA256_RE.fullmatch(root_hash):
        raise ValueError("snapshot root_hash must be lowercase 64-hex sha256")
    if not all(isinstance(key, str) and _SHA256_RE.fullmatch(key) for key in nodes):
        raise ValueError("snapshot node keys must be lowercase 64-hex sha256")
    active: set[str] = set()
    decoded: dict[str, Any] = {}

    def decode(digest: str) -> Any:
        if digest in decoded:
            return decoded[digest]
        if digest in active:
            raise ValueError(f"cycle in resolved-semantics node table at {digest}")
        raw = nodes.get(digest)
        if not isinstance(raw, Mapping):
            raise ValueError(f"missing resolved-semantics node {digest}")
        active.add(digest)
        try:
            node_type = raw.get("type")
            if node_type == "scalar":
                if set(raw) != {"type", "value"}:
                    raise ValueError(f"invalid scalar node fields at {digest}")
                value = raw["value"]
                if value is not None and not isinstance(value, (str, bool, int, float)):
                    raise ValueError(f"invalid scalar primitive at {digest}")
                result = value
            if node_type == "array":
                if set(raw) != {"type", "children"}:
                    raise ValueError(f"invalid array node fields at {digest}")
                children = raw.get("children")
                if not isinstance(children, list) or not all(
                    isinstance(child, str) and _SHA256_RE.fullmatch(child)
                    for child in children
                ):
                    raise ValueError(f"invalid array node {digest}")
                result = [decode(child) for child in children]
            elif node_type == "object":
                if set(raw) != {"type", "children"}:
                    raise ValueError(f"invalid object node fields at {digest}")
                children = raw.get("children")
                if not isinstance(children, Mapping) or not all(
                    isinstance(key, str)
                    and isinstance(child, str)
                    and _SHA256_RE.fullmatch(child)
                    for key, child in children.items()
                ):
                    raise ValueError(f"invalid object node {digest}")
                result = {key: decode(child) for key, child in children.items()}
            elif node_type != "scalar":
                raise ValueError(f"unsupported resolved-semantics node type {node_type!r}")
            if hashlib.sha256(_canonical_bytes(raw)).hexdigest() != digest:
                raise ValueError(f"resolved-semantics node hash mismatch at {digest}")
            decoded[digest] = result
            return result
        finally:
            active.remove(digest)

    result = decode(root_hash)
    for digest in nodes:
        decode(digest)
    return result


def _canonical_bytes(value: Any) -> bytes:
    """Mirror the snapshot's documented standalone numeric encoding rule."""
    return json.dumps(
        _normalize_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalize_json(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("resolved-semantics nodes require finite numbers")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("resolved-semantics object keys must all be strings")
        return {key: _normalize_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_json(child) for child in value]
    raise TypeError(f"node is not JSON-compatible: {type(value).__name__}")
