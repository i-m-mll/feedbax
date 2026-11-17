from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import pytest

from feedbax.contracts.spec_storage import (
    build_resolved_semantics_snapshot,
    training_spec_canonical_bytes,
)


def test_round_trip_numeric_canonicalization_and_structural_sharing() -> None:
    left = {"x": 1e-5, "nested": {"value": 2}, "zero": -0.0}
    right = {"nested": {"value": 2}, "x": 0.00001, "zero": 0.0}

    assert training_spec_canonical_bytes(left) == training_spec_canonical_bytes(right)
    assert training_spec_canonical_bytes({"large": 2**80}) == f'{{"large":{2**80}}}'.encode()
    assert b"e-" in training_spec_canonical_bytes({"small": 1e-100})
    assert training_spec_canonical_bytes({"integer": 2}) != training_spec_canonical_bytes(
        {"integer": 2.0}
    )
    with pytest.raises(TypeError, match="keys must all be strings"):
        training_spec_canonical_bytes({1: "integer", "1": "string"})
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            training_spec_canonical_bytes({"value": value})
    snapshot = build_resolved_semantics_snapshot({"a": left["nested"], "b": right["nested"]})
    object_nodes = [node for node in snapshot["nodes"].values() if node["type"] == "object"]
    assert len(object_nodes) == 2  # shared nested object plus root


def test_snapshot_decodes_with_pure_stdlib_decoder_alone() -> None:
    expected = {"rows": [{"x": 1.25}, {"x": 1.25}]}
    snapshot = build_resolved_semantics_snapshot(expected)
    decoder_path = (
        Path(__file__).parents[1] / "feedbax" / "contracts" / "resolved_snapshot_decoder.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "standalone_snapshot_decoder", decoder_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    assert module.decode_resolved_snapshot(snapshot) == expected

    root = snapshot["root_hash"]
    tampered = json.loads(json.dumps(snapshot))
    tampered["nodes"][root]["extra"] = True
    with pytest.raises(ValueError, match="extra fields|node fields"):
        module.decode_resolved_snapshot(tampered)


def test_decoder_rejects_invalid_variants_missing_refs_cycles_and_hash_mismatch() -> None:
    from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot

    snapshot = build_resolved_semantics_snapshot({"value": 1})
    scalar_hash = next(
        digest for digest, node in snapshot["nodes"].items() if node["type"] == "scalar"
    )
    missing_value = json.loads(json.dumps(snapshot))
    missing_value["nodes"][scalar_hash].pop("value")
    with pytest.raises(ValueError, match="scalar node fields"):
        decode_resolved_snapshot(missing_value)

    missing_ref = json.loads(json.dumps(snapshot))
    root = missing_ref["root_hash"]
    missing_ref["nodes"][root]["children"]["value"] = "f" * 64
    with pytest.raises(ValueError, match="missing resolved-semantics node"):
        decode_resolved_snapshot(missing_ref)

    cycle_hash = "a" * 64
    cyclic = {
        "schema_id": snapshot["schema_id"],
        "schema_version": snapshot["schema_version"],
        "root_hash": cycle_hash,
        "nodes": {cycle_hash: {"type": "array", "children": [cycle_hash]}},
    }
    with pytest.raises(ValueError, match="cycle"):
        decode_resolved_snapshot(cyclic)

    tampered = json.loads(json.dumps(snapshot))
    tampered["nodes"][scalar_hash]["value"] = 2
    with pytest.raises(ValueError, match="hash mismatch"):
        decode_resolved_snapshot(tampered)
