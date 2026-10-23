from feedbax.contracts.intent_diff import detect_near_duplicate_lanes, layered_semantic_diff
from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    CompositionDelta,
    CompositionNode,
    InlineIntentParent,
    authored_envelope_hash,
    flatten_composition,
)
from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.training.run_matrix import resolve_base_payload_with_attribution


def _delta(layer: str, path: str, value: object, *, acknowledge: bool = False) -> CompositionDelta:
    return CompositionDelta(
        layer_id=layer,
        patches=[{"op": "replace", "path": path, "value": value}],
        acknowledges_ancestor_paths=[path] if acknowledge else [],
    )


def test_three_layer_attribution_and_layered_diff_views() -> None:
    root = CompositionNode(
        name="root",
        parent=InlineIntentParent(
            payload={"schema_id": "method.a", "schema_version": "method.a.v1", "lr": 0.1, "width": 8},
            schema_id="method.a",
            schema_version="method.a.v1",
        ),
        deltas=[_delta("root", "lr", 0.2)],
    )
    middle = CompositionNode(
        name="middle",
        parent=AuthoredIntentParent(ref="root.json", content_hash=authored_envelope_hash(root)),
        deltas=[_delta("middle", "width", 16)],
    )
    child = CompositionNode(
        name="child",
        parent=AuthoredIntentParent(ref="middle.json", content_hash=authored_envelope_hash(middle)),
        deltas=[_delta("child", "lr", 0.3, acknowledge=True)],
    )
    nodes = {"root.json": root, "middle.json": middle}
    resolve = lambda parent: nodes[parent.ref]
    root_flat = flatten_composition(root, resolve)
    middle_flat = flatten_composition(middle, resolve)
    child_flat = flatten_composition(child, resolve)

    assert child_flat.attribution == {"lr": "child", "width": "middle"}
    assert [item.path for item in layered_semantic_diff(middle_flat, child_flat)] == ["lr"]
    assert {item.path for item in layered_semantic_diff(root_flat, child_flat)} == {"lr", "width"}
    assert layered_semantic_diff(child_flat, child_flat) == []


def test_near_duplicate_detection_has_positive_and_negative_cases() -> None:
    lanes = {
        "a": {"lr": 0.1, "width": 8},
        "near": {"lr": 0.2, "width": 8},
        "far": {"lr": 0.3, "width": 16},
    }
    pairs = detect_near_duplicate_lanes(lanes, max_differences=1)
    assert [(left, right) for left, right, _ in pairs] == [("a", "near")]


def test_materializer_resolution_exposes_attribution(tmp_path) -> None:
    spec = TrainingRunMatrixSpec.model_validate({
        "name": "attributed",
        "base": {"kind": "inline", "inline": {
            "schema_id": "method.a", "schema_version": "method.a.v1", "lr": 0.1,
        }},
        "deltas": [{
            "layer_id": "tuning",
            "patches": [{"op": "replace", "path": "lr", "value": 0.2}],
        }],
        "rows": [{"row_id": "row"}],
    })
    payload, attribution = resolve_base_payload_with_attribution(spec, repo_root=tmp_path)
    assert payload["lr"] == 0.2
    assert attribution == {"lr": "tuning"}
