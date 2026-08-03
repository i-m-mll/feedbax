import pytest

from feedbax.contracts.intent_diff import detect_near_duplicate_lanes, layered_semantic_diff
from feedbax.contracts.lineage import LineageDag, LineageEvent, LineageParentRef
from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    CompositionDelta,
    CompositionNode,
    ExecutionDependencyLayer,
    ForkFromSelectedCheckpoint,
    InlineIntentParent,
    LineageGraftDependency,
    authored_envelope_hash,
    composed_intent_hash,
    flatten_composition,
)
from feedbax.contracts.run_matrix import apply_composition_deltas


def _hash(char: str) -> str:
    return char * 64


def _replace(layer: str, path: str, value: object, **kwargs: object) -> CompositionDelta:
    return CompositionDelta(
        layer_id=layer,
        patches=[{"op": "replace", "path": path, "value": value}],
        **kwargs,
    )


def test_synthetic_six_wave_program_has_real_schema_boundary() -> None:
    root = CompositionNode(
        name="root",
        parent=InlineIntentParent(
            payload={"schema_id": "method.a", "schema_version": "method.a.v1", "lr": 0.1},
            schema_id="method.a",
            schema_version="method.a.v1",
        ),
    )
    root_hash = authored_envelope_hash(root)
    lanes: dict[str, CompositionNode] = {}
    dependencies: dict[str, ExecutionDependencyLayer] = {}
    for wave in range(1, 7):
        deltas = [_replace(f"wave-{wave}", "lr", wave / 100)]
        if wave == 6:
            deltas.append(
                CompositionDelta(
                    layer_id="wave-6-schema",
                    schema_id="method.b",
                    schema_version="method.b.v2",
                    patches=[
                        {"op": "replace", "path": "schema_id", "value": "method.b"},
                        {"op": "replace", "path": "schema_version", "value": "method.b.v2"},
                    ],
                )
            )
        lanes[f"wave-{wave}"] = CompositionNode(
            name=f"wave-{wave}",
            parent=AuthoredIntentParent(ref="program/root.json", content_hash=root_hash),
            deltas=deltas,
        )
        dependencies[f"wave-{wave}"] = ExecutionDependencyLayer(
            dependencies=[ForkFromSelectedCheckpoint(
                source_execution_hash=_hash("b"), source_row_id="shared-source",
                checkpoint_transaction_id="checkpoint-shared", checkpoint_root_hash=_hash("a"),
            )]
        )

    flattened = {name: flatten_composition(lane, lambda _parent: root) for name, lane in lanes.items()}
    assert flattened["wave-6"].payload["schema_id"] == "method.b"
    assert flattened["wave-6"].payload["schema_version"] == "method.b.v2"
    assert flattened["wave-3"].attribution["lr"] == "wave-3"
    assert layered_semantic_diff(flattened["wave-1"], flattened["wave-2"])[0].path == "lr"
    assert detect_near_duplicate_lanes(flattened, max_differences=1)
    assert all(layer.dependencies[0].checkpoint_root_hash == _hash("a") for layer in dependencies.values())

    original = LineageEvent(event_kind="execution", execution_hash=_hash("c"), parents=[LineageParentRef(execution_hash=_hash("d"))])
    dag = LineageDag([original])
    correction = LineageEvent(
        event_kind="graft_correction", execution_hash=_hash("c"),
        parents=[LineageParentRef(execution_hash=_hash("e"))],
        original_event_hash=original.content_hash,
        correction_mode="supersedes_for_interpretation",
    )
    correction_hash = dag.append(correction)
    dependencies["wave-4"] = ExecutionDependencyLayer(dependencies=[
        *dependencies["wave-4"].dependencies,
        LineageGraftDependency(lineage_event_hash=correction_hash, interpretation="supersedes_for_interpretation"),
    ])
    assert dag.interpreted_parents(_hash("c"))[0].execution_hash == _hash("e")


def test_schema_identity_change_without_boundary_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"/deltas/cross.*method.a.*method.b.*without"):
        apply_composition_deltas(
            {"schema_id": "method.a", "schema_version": "method.a.v1"},
            [CompositionDelta(layer_id="cross", patches=[
                {"op": "replace", "path": "schema_id", "value": "method.b"},
                {"op": "replace", "path": "schema_version", "value": "method.b.v2"},
            ])],
        )


def test_declared_schema_boundary_must_match_flattened_identity() -> None:
    with pytest.raises(ValueError, match=r"/deltas/cross.*declares schema boundary"):
        apply_composition_deltas(
            {"schema_id": "method.a", "schema_version": "method.a.v1"},
            [CompositionDelta(layer_id="cross", schema_id="method.b", schema_version="method.b.v2")],
        )


def test_nested_identity_bearing_subtree_change_requires_boundary() -> None:
    payload = {
        "schema_id": "method.a",
        "schema_version": "method.a.v1",
        "worker": {"schema_id": "worker.a", "schema_version": "worker.a.v1"},
    }
    with pytest.raises(ValueError, match=r"/deltas/nested.*worker.a.*worker.b.*without"):
        apply_composition_deltas(payload, [CompositionDelta(layer_id="nested", patches=[{
            "op": "replace",
            "path": "worker",
            "value": {"schema_id": "worker.b", "schema_version": "worker.b.v1"},
        }])])


@pytest.mark.parametrize(
    ("payload", "delta", "message"),
    [
        ({"x": 1}, CompositionDelta(layer_id="child", patches=[{"op": "add", "path": "x", "value": 2}]), r"/deltas/child.*already exists"),
        ({}, CompositionDelta(layer_id="child", patches=[{"op": "replace", "path": "x", "value": 2}]), r"/deltas/child.*missing"),
    ],
)
def test_patch_precondition_failures_name_layer(payload: dict[str, object], delta: CompositionDelta, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apply_composition_deltas(payload, [delta])


def test_repeated_dash_appends_in_one_layer_do_not_overlap_each_other() -> None:
    resolved, attribution, written = apply_composition_deltas(
        {"items": [0]},
        [
            CompositionDelta(
                layer_id="child",
                patches=[
                    {"op": "add", "path": "items.-", "value": 1},
                    {"op": "add", "path": "items.-", "value": 2},
                ],
            )
        ],
    )

    assert resolved == {"items": [0, 1, 2]}
    assert attribution == {"items.-": "child"}
    assert written == {"items.-"}


def test_dash_append_does_not_hide_a_later_parent_path_overlap() -> None:
    with pytest.raises(ValueError, match=r"/deltas/child/items.*overlaps"):
        apply_composition_deltas(
            {"items": [0]},
            [
                CompositionDelta(
                    layer_id="child",
                    patches=[
                        {"op": "add", "path": "items.-", "value": 1},
                        {"op": "replace", "path": "items", "value": []},
                    ],
                )
            ],
        )


def test_unacknowledged_ancestor_override_names_layer() -> None:
    with pytest.raises(ValueError, match=r"/deltas/child/x.*without explicit acknowledgement"):
        apply_composition_deltas(
            {"x": 1}, [_replace("child", "x", 2)], ancestor_written_paths={"x"}
        )


def test_rebase_vs_squash_changes_envelope_not_composed_intent() -> None:
    inline = InlineIntentParent(
        payload={"schema_id": "method.a", "schema_version": "method.a.v1", "x": 0, "y": 0},
        schema_id="method.a", schema_version="method.a.v1",
    )
    parent = CompositionNode(name="parent", parent=inline, deltas=[_replace("x-layer", "x", 1)])
    deep = CompositionNode(
        name="deep",
        parent=AuthoredIntentParent(ref="parent.json", content_hash=authored_envelope_hash(parent)),
        deltas=[_replace("y-layer", "y", 2)],
    )
    squashed = CompositionNode(
        name="squashed", parent=inline,
        deltas=[_replace("x-layer", "x", 1), _replace("y-layer", "y", 2)],
    )
    deep_flat = flatten_composition(deep, lambda _parent: parent)
    squashed_flat = flatten_composition(squashed, lambda _parent: parent)
    assert authored_envelope_hash(deep) != authored_envelope_hash(squashed)
    assert composed_intent_hash(deep_flat) == composed_intent_hash(squashed_flat)
