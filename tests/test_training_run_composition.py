import pytest

from feedbax.contracts.run_matrix import MatrixCompositionDelta, apply_composition_deltas


def _replace(layer: str, path: str, value: object, **kwargs: object) -> MatrixCompositionDelta:
    return MatrixCompositionDelta(
        layer_id=layer,
        patches=[{"op": "replace", "path": path, "value": value}],
        **kwargs,
    )


def test_schema_identity_change_without_boundary_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"/deltas/cross.*method.a.*method.b.*without"):
        apply_composition_deltas(
            {"schema_id": "method.a", "schema_version": "method.a.v1"},
            [MatrixCompositionDelta(layer_id="cross", patches=[
                {"op": "replace", "path": "schema_id", "value": "method.b"},
                {"op": "replace", "path": "schema_version", "value": "method.b.v2"},
            ])],
        )


def test_declared_schema_boundary_must_match_flattened_identity() -> None:
    with pytest.raises(ValueError, match=r"/deltas/cross.*declares schema boundary"):
        apply_composition_deltas(
            {"schema_id": "method.a", "schema_version": "method.a.v1"},
            [MatrixCompositionDelta(layer_id="cross", schema_id="method.b", schema_version="method.b.v2")],
        )


def test_nested_identity_bearing_subtree_change_requires_boundary() -> None:
    payload = {
        "schema_id": "method.a",
        "schema_version": "method.a.v1",
        "worker": {"schema_id": "worker.a", "schema_version": "worker.a.v1"},
    }
    with pytest.raises(ValueError, match=r"/deltas/nested.*worker.a.*worker.b.*without"):
        apply_composition_deltas(payload, [MatrixCompositionDelta(layer_id="nested", patches=[{
            "op": "replace",
            "path": "worker",
            "value": {"schema_id": "worker.b", "schema_version": "worker.b.v1"},
        }])])


@pytest.mark.parametrize(
    ("payload", "delta", "message"),
    [
        ({"x": 1}, MatrixCompositionDelta(layer_id="child", patches=[{"op": "add", "path": "x", "value": 2}]), r"/deltas/child.*already exists"),
        ({}, MatrixCompositionDelta(layer_id="child", patches=[{"op": "replace", "path": "x", "value": 2}]), r"/deltas/child.*missing"),
    ],
)
def test_patch_precondition_failures_name_layer(payload: dict[str, object], delta: MatrixCompositionDelta, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apply_composition_deltas(payload, [delta])


def test_repeated_dash_appends_in_one_layer_do_not_overlap_each_other() -> None:
    resolved, attribution, written = apply_composition_deltas(
        {"items": [0]},
        [
            MatrixCompositionDelta(
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
                MatrixCompositionDelta(
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
