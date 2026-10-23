from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
from pydantic import TypeAdapter, ValidationError

from feedbax.analysis.bundles import ManifestPredicate, _equals_all, _get_path
from feedbax.contracts.expressions import (
    MAX_EXPR_DEPTH,
    PATH_EXPRESSION_SCHEMA_ID,
    PATH_EXPRESSION_SCHEMA_VERSION,
    AllOf,
    AnyOf,
    Coerce,
    Coalesce,
    Compare,
    ContextItem,
    Expr,
    ExpressionContext,
    ExpressionItemMissing,
    ExpressionPathMissing,
    ExpressionSelectAmbiguous,
    ExpressionTypeError,
    Filter,
    MapObjectList,
    NamedPredicateRef,
    NamedPredicateUnresolved,
    Not,
    Select,
    ValueExpr,
    ValueQuery,
    canonical_expression_json,
    evaluate_expr,
    evaluate_query,
    expression_hash,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry


HASH_CORPUS_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "path_expressions"
    / "pre_extension_hash_corpus.json"
)


def _ctx(payload: object, *, kind: str = "manifest") -> ExpressionContext:
    return ExpressionContext(items={"manifest": ContextItem(kind=kind, payload=payload)})


def _hash_corpus_expressions() -> dict[str, object]:
    select = Select(
        where=Compare(
            item="entry",
            path="factor",
            op="approx_eq",
            value=1.05,
            tolerance=1e-12,
        )
    )
    score_gt = Compare(item="manifest", path="metrics.score", op="gt", value=0.8)
    loss_ok = Compare(item="manifest", path="metrics.loss", op="lt", value=0.3)
    group_ok = Compare(item="manifest", path="metadata.group", op="eq", value="control")
    return {
        "compare_exists": Compare(item="manifest", path="metadata.group", op="exists"),
        "compare_eq": Compare(
            item="manifest",
            path="metadata.group",
            op="eq",
            value="control",
        ),
        "compare_ne": Compare(
            item="manifest",
            path="metadata.group",
            op="ne",
            value="treated",
        ),
        "compare_approx_eq": Compare(
            item="manifest",
            path="metrics.score",
            op="approx_eq",
            value=0.125,
            tolerance=1e-9,
        ),
        "compare_lt": Compare(item="manifest", path="metrics.loss", op="lt", value=0.2),
        "compare_le": Compare(item="manifest", path="metrics.loss", op="le", value=0.2),
        "compare_gt": Compare(item="manifest", path="metrics.score", op="gt", value=0.1),
        "compare_ge": Compare(item="manifest", path="metrics.score", op="ge", value=0.1),
        "compare_in": Compare(
            item="manifest",
            path="metadata.group",
            op="in",
            value=["control", "treated"],
        ),
        "compare_contains": Compare(
            item="manifest",
            path="tags",
            op="contains",
            value="stable",
        ),
        "compare_has_type": Compare(
            item="manifest",
            op="has_type",
            value="evaluation_manifest",
        ),
        "nested_all_any_not": AllOf(
            exprs=[score_gt, AnyOf(exprs=[loss_ok, Not(expr=group_ok)])]
        ),
        "named_predicate": NamedPredicateRef(
            predicate_id="feedbax.test.predicate",
            params={"z": 2, "a": 1},
        ),
        "select_factor": select,
        "value_query_empty_path": ValueQuery(item="manifest"),
        "value_query_nested_path": ValueQuery(item="manifest", path="metadata.group"),
        "value_query_select": ValueQuery(
            item="manifest",
            path="frontier",
            select=select,
        ),
        "value_query_coerce_float_scale": ValueQuery(
            item="manifest",
            path="percent",
            coerce=Coerce(to="float", scale=0.01),
        ),
        "value_query_coerce_int": ValueQuery(
            item="manifest",
            path="count",
            coerce=Coerce(to="int"),
        ),
        "value_query_coerce_str": ValueQuery(
            item="manifest",
            path="label",
            coerce=Coerce(to="str"),
        ),
        "value_query_coerce_bool": ValueQuery(
            item="manifest",
            path="enabled",
            coerce=Coerce(to="bool"),
        ),
        "value_query_select_coerce": ValueQuery(
            item="manifest",
            path="frontier",
            select=select,
            coerce=Coerce(to="float", scale=2.0),
        ),
    }


def test_path_ref_matches_manifest_get_path_for_dict_and_attribute_walks() -> None:
    payload = {
        "metadata": {"group": "control"},
        "object": SimpleNamespace(nested={"score": 7}),
    }
    ctx = _ctx(payload)

    for path in ("metadata.group", "object.nested.score"):
        assert evaluate_query(ValueQuery(item="manifest", path=path), ctx) == _get_path(
            payload, path
        )


@pytest.mark.parametrize(
    ("op", "path", "value", "expected"),
    [
        ("eq", "scalar", 3, True),
        ("ne", "scalar", 4, True),
        ("approx_eq", "scalar", 3.0000000001, True),
        ("lt", "scalar", 4, True),
        ("le", "scalar", 3, True),
        ("gt", "scalar", 2, True),
        ("ge", "scalar", 3, True),
        ("in", "color", ["red", "green"], True),
        ("contains", "tags", "fast", True),
    ],
)
def test_compare_ops_cover_equality_ordering_and_membership(
    op: str,
    path: str,
    value: object,
    expected: bool,
) -> None:
    payload = {"scalar": 3, "color": "red", "tags": ["fast", "stable"]}
    kwargs = {"tolerance": 1e-9} if op == "approx_eq" else {}

    assert evaluate_expr(
        Compare(item="manifest", path=path, op=op, value=value, **kwargs),
        _ctx(payload),
    ) is expected


def test_compare_validation_requires_explicit_tolerance_only_for_approx_eq() -> None:
    with pytest.raises(ValidationError):
        Compare(item="manifest", path="scalar", op="approx_eq", value=3)

    with pytest.raises(ValidationError):
        Compare(item="manifest", path="scalar", op="eq", value=3, tolerance=1e-9)

    with pytest.raises(ValidationError):
        Compare(item="manifest", path="scalar", op="exists", value=True)

    Compare(item="manifest", path="scalar", op="approx_eq", value=3, tolerance=0.0)


def test_combinators_support_nested_de_morgan_equivalence() -> None:
    ctx = _ctx({"a": True, "b": False})
    a = Compare(item="manifest", path="a", op="eq", value=True)
    b = Compare(item="manifest", path="b", op="eq", value=True)

    not_all = Not(expr=AllOf(exprs=[a, b]))
    any_not = AnyOf(exprs=[Not(expr=a), Not(expr=b)])

    assert evaluate_expr(not_all, ctx) is True
    assert evaluate_expr(not_all, ctx) == evaluate_expr(any_not, ctx)


def test_value_query_selects_exactly_one_entry_with_tolerance_predicate() -> None:
    ctx = _ctx(
        {
            "frontier": [
                {"factor": 1.0, "score": 10},
                {"factor": 1.05, "score": 12},
            ]
        }
    )
    query = ValueQuery(
        item="manifest",
        path="frontier",
        select=Select(
            where=Compare(
                item="entry",
                path="factor",
                op="approx_eq",
                value=1.05,
                tolerance=1e-12,
            )
        ),
    )

    assert evaluate_query(query, ctx) == {"factor": 1.05, "score": 12}


@pytest.mark.parametrize(
    ("frontier", "expected_error"),
    [
        ([{"factor": 1.0}, {"factor": 1.1}], ExpressionPathMissing),
        ([{"factor": 1.05}, {"factor": 1.0500000000001}], ExpressionSelectAmbiguous),
    ],
)
def test_value_query_select_raises_for_zero_or_multiple_matches(
    frontier: list[dict[str, float]],
    expected_error: type[Exception],
) -> None:
    query = ValueQuery(
        item="manifest",
        path="frontier",
        select=Select(
            where=Compare(
                item="entry",
                path="factor",
                op="approx_eq",
                value=1.05,
                tolerance=1e-9,
            )
        ),
    )

    with pytest.raises(expected_error):
        evaluate_query(query, _ctx({"frontier": frontier}))


def test_value_query_default_only_handles_absence_class_misses() -> None:
    select = Select(where=Compare(item="entry", path="id", op="eq", value="missing"))

    assert (
        evaluate_query(
            ValueQuery(item="manifest", path="missing.path", default=None),
            _ctx({"present": 1}),
        )
        is None
    )
    assert evaluate_query(
        ValueQuery(item="manifest", path="frontier", select=select, default={"id": "fallback"}),
        _ctx({"frontier": [{"id": "other"}]}),
    ) == {"id": "fallback"}
    assert evaluate_query(
        ValueQuery(
            item="manifest",
            path="missing",
            coerce=Coerce(to="float"),
            default="not-float",
        ),
        _ctx({}),
    ) == "not-float"

    with pytest.raises(ExpressionItemMissing):
        evaluate_query(ValueQuery(item="unknown", path="x", default=0), _ctx({}))
    with pytest.raises(ExpressionSelectAmbiguous):
        evaluate_query(
            ValueQuery(
                item="manifest",
                path="frontier",
                select=Select(where=Compare(item="entry", path="id", op="exists")),
                default={"id": "fallback"},
            ),
            _ctx({"frontier": [{"id": 1}, {"id": 2}]}),
        )
    with pytest.raises(ExpressionTypeError):
        evaluate_query(
            ValueQuery(
                item="manifest",
                path="value",
                coerce=Coerce(to="float"),
                default=0.0,
            ),
            _ctx({"value": "not-float"}),
        )
    with pytest.raises(ExpressionPathMissing):
        evaluate_query(
            ValueQuery(
                item="manifest",
                path="frontier",
                select=Select(where=Compare(item="entry", path="missing", op="eq", value=1)),
                default={"id": "fallback"},
            ),
            _ctx({"frontier": [{"id": 1}]}),
        )


def test_coalesce_returns_first_hit_and_only_falls_back_on_absence_misses() -> None:
    ctx = _ctx({"primary": {}, "secondary": {"value": 3}, "bad": "not-float"})

    assert evaluate_query(
        Coalesce(
            queries=[
                ValueQuery(item="manifest", path="primary.value"),
                ValueQuery(item="manifest", path="secondary.value"),
            ]
        ),
        ctx,
    ) == 3
    assert (
        evaluate_query(
            Coalesce(
                queries=[ValueQuery(item="manifest", path="primary.value")],
                default=None,
            ),
            ctx,
        )
        is None
    )

    with pytest.raises(ExpressionPathMissing, match="primary.value"):
        evaluate_query(
            Coalesce(
                queries=[
                    ValueQuery(item="manifest", path="primary.value"),
                    ValueQuery(item="manifest", path="secondary.missing"),
                ]
            ),
            ctx,
        )
    with pytest.raises(ExpressionTypeError):
        evaluate_query(
            Coalesce(
                queries=[
                    ValueQuery(
                        item="manifest",
                        path="bad",
                        coerce=Coerce(to="float"),
                    ),
                    ValueQuery(item="manifest", path="secondary.value"),
                ]
            ),
            ctx,
        )
    with pytest.raises(ExpressionPathMissing, match="missing"):
        evaluate_query(
            Coalesce(
                queries=[
                    ValueQuery(
                        item="manifest",
                        path="frontier",
                        select=Select(
                            where=Compare(item="entry", path="missing", op="eq", value=1)
                        ),
                    ),
                    ValueQuery(item="manifest", path="secondary.value"),
                ]
            ),
            _ctx({"frontier": [{"id": 1}], "secondary": {"value": 3}}),
        )


def test_filter_returns_zero_or_more_matches_and_maps_coercion_per_entry() -> None:
    ctx = _ctx(
        {
            "values": [
                {"kind": "keep", "amount": "1.5"},
                {"kind": "drop", "amount": "2.5"},
                {"kind": "keep", "amount": "3.5"},
            ]
        }
    )
    filtered = ValueQuery(
        item="manifest",
        path="values",
        filter=Filter(where=Compare(item="entry", path="kind", op="eq", value="keep")),
    )
    amounts = ValueQuery(
        item="manifest",
        path="values",
        filter=Filter(where=Compare(item="entry", path="kind", op="eq", value="keep")),
        coerce=Coerce(to="str"),
    )

    assert evaluate_query(filtered, ctx) == [
        {"kind": "keep", "amount": "1.5"},
        {"kind": "keep", "amount": "3.5"},
    ]
    assert evaluate_query(
        ValueQuery(
            item="manifest",
            path="values",
            filter=Filter(where=Compare(item="entry", path="kind", op="eq", value="none")),
            default=["not-used"],
        ),
        ctx,
    ) == []
    assert evaluate_query(amounts, ctx) == [
        "{'kind': 'keep', 'amount': '1.5'}",
        "{'kind': 'keep', 'amount': '3.5'}",
    ]
    assert evaluate_query(
        ValueQuery(
            item="manifest",
            path="missing",
            filter=Filter(where=Compare(item="entry", path="kind", op="eq", value="keep")),
            default=["fallback"],
        ),
        ctx,
    ) == ["fallback"]

    with pytest.raises(ExpressionTypeError):
        evaluate_query(
            ValueQuery(
                item="manifest",
                path="metadata",
                filter=Filter(where=Compare(item="entry", path="kind", op="eq", value="keep")),
            ),
            _ctx({"metadata": {"kind": "keep"}}),
        )
    with pytest.raises(ValidationError):
        ValueQuery(
            item="manifest",
            path="values",
            select=Select(where=Compare(item="entry", path="kind", op="eq", value="keep")),
            filter=Filter(where=Compare(item="entry", path="kind", op="eq", value="keep")),
        )


def test_value_query_scalar_coercion_and_scaling() -> None:
    query = ValueQuery(
        item="manifest",
        path="percent",
        coerce=Coerce(to="float", scale=0.01),
    )

    assert evaluate_query(query, _ctx({"percent": "12.5"})) == 0.125


def test_absence_tolerant_ops_and_missing_path_tri_state() -> None:
    ctx = _ctx({"present": 1}, kind="evaluation_manifest")

    assert evaluate_expr(Compare(item="missing", path="x", op="exists"), ctx) is False
    assert evaluate_expr(Compare(item="manifest", path="missing", op="exists"), ctx) is False
    assert evaluate_expr(Compare(item="missing", op="has_type", value="anything"), ctx) is False
    assert (
        evaluate_expr(Compare(item="manifest", op="has_type", value="evaluation_manifest"), ctx)
        is True
    )
    assert evaluate_expr(
        Not(expr=Compare(item="manifest", path="missing", op="exists")),
        ctx,
    ) is True

    with pytest.raises(ExpressionPathMissing):
        evaluate_expr(Compare(item="manifest", path="missing", op="eq", value=1), ctx)
    with pytest.raises(ExpressionItemMissing):
        evaluate_expr(Compare(item="missing", path="x", op="eq", value=1), ctx)


def test_named_predicate_resolution_and_purity_contract_docstring() -> None:
    def high_score(ctx: ExpressionContext, *, threshold: int) -> bool:
        return ctx.items["manifest"].payload["score"] > threshold

    expr = NamedPredicateRef(
        predicate_id="feedbax.test.high_score",
        params={"threshold": 10},
    )

    assert evaluate_expr(
        expr,
        _ctx({"score": 12}),
        named_predicates={"feedbax.test.high_score": high_score},
    )
    assert "pure and deterministic" in (NamedPredicateRef.__doc__ or "")


def test_negative_canaries_for_typed_failures() -> None:
    ctx = _ctx({"value": {"nested": 1}, "frontier": [], "text": "abc"})

    with pytest.raises(ExpressionItemMissing):
        evaluate_expr(Compare(item="unknown", path="value", op="eq", value=1), ctx)
    with pytest.raises(ExpressionPathMissing):
        evaluate_expr(Compare(item="manifest", path="missing", op="eq", value=1), ctx)
    with pytest.raises(ExpressionPathMissing):
        evaluate_query(
            ValueQuery(
                item="manifest",
                path="frontier",
                select=Select(where=Compare(item="entry", path="x", op="exists")),
            ),
            ctx,
        )
    with pytest.raises(ExpressionTypeError):
        evaluate_expr(Compare(item="manifest", path="value", op="lt", value=1), ctx)
    with pytest.raises(ExpressionTypeError):
        evaluate_query(ValueQuery(item="manifest", path="text", coerce=Coerce(to="float")), ctx)
    with pytest.raises(NamedPredicateUnresolved):
        evaluate_expr(NamedPredicateRef(predicate_id="missing.predicate"), ctx)


def test_startswith_and_endswith_are_strict_string_comparisons() -> None:
    ctx = _ctx({"label": "feedbax-run", "count": 3})

    assert evaluate_expr(
        Compare(item="manifest", path="label", op="startswith", value="feed"),
        ctx,
    )
    assert evaluate_expr(
        Compare(item="manifest", path="label", op="endswith", value="run"),
        ctx,
    )
    with pytest.raises(ExpressionTypeError):
        evaluate_expr(Compare(item="manifest", path="count", op="startswith", value="3"), ctx)
    with pytest.raises(ExpressionTypeError):
        evaluate_expr(Compare(item="manifest", path="label", op="endswith", value=3), ctx)


def test_depth_guard_allows_depth_limit_and_rejects_pathological_nesting() -> None:
    expr = Compare(item="manifest", path="x", op="eq", value=True)
    for _ in range(MAX_EXPR_DEPTH - 1):
        expr = Not(expr=expr)

    evaluate_expr(expr, _ctx({"x": False}))

    with pytest.raises(ValidationError):
        Not(expr=expr)


def test_depth_guard_covers_filter_and_coalesce_query_children() -> None:
    expr = Compare(item="entry", path="x", op="eq", value=True)
    for _ in range(MAX_EXPR_DEPTH - 1):
        expr = Not(expr=expr)

    with pytest.raises(ValidationError):
        ValueQuery(item="manifest", path="entries", filter=Filter(where=expr))
    with pytest.raises(ValidationError):
        Coalesce(
            queries=[
                ValueQuery(
                    item="manifest",
                    path="entries",
                    select=Select(where=expr),
                )
            ]
        )


def test_expression_hash_uses_sorted_canonical_json_excluding_none() -> None:
    left = NamedPredicateRef(predicate_id="feedbax.test.predicate", params={"b": 2, "a": 1})
    right = NamedPredicateRef(predicate_id="feedbax.test.predicate", params={"a": 1, "b": 2})

    assert canonical_expression_json(left) == (
        '{"kind":"named","params":{"a":1,"b":2},'
        '"predicate_id":"feedbax.test.predicate"}'
    )
    assert expression_hash(left) == expression_hash(right)
    assert "tolerance" not in canonical_expression_json(
        Compare(item="manifest", path="x", op="eq", value=1)
    )


def test_pre_extension_expression_hash_corpus_is_byte_stable() -> None:
    corpus = json.loads(HASH_CORPUS_PATH.read_text(encoding="utf-8"))
    expressions = _hash_corpus_expressions()

    assert {entry["name"] for entry in corpus} == set(expressions)
    for entry in corpus:
        expr = expressions[entry["name"]]
        assert canonical_expression_json(expr) == entry["canonical_json"]
        assert expression_hash(expr) == entry["expression_hash"]


def test_discriminated_expr_schema_validates_from_plain_payloads() -> None:
    expr = TypeAdapter(Expr).validate_python(
        {
            "kind": "any",
            "exprs": [
                {"kind": "compare", "item": "manifest", "path": "x", "op": "eq", "value": 1},
                {"kind": "not", "expr": {"kind": "compare", "item": "manifest", "op": "exists"}},
            ],
        }
    )

    assert evaluate_expr(expr, _ctx({"x": 1})) is True


def test_structural_value_expr_union_validates_value_query_and_coalesce_payloads() -> None:
    adapter = TypeAdapter(ValueExpr)

    value_query = adapter.validate_python({"item": "manifest", "path": "x"})
    coalesce = adapter.validate_python(
        {
            "queries": [
                {"item": "manifest", "path": "missing"},
                {"item": "fallback", "path": "x"},
            ],
            "default": None,
        }
    )

    assert isinstance(value_query, ValueQuery)
    assert isinstance(coalesce, Coalesce)
    with pytest.raises(ValidationError):
        adapter.validate_python({"item": "manifest", "queries": []})


def test_map_object_list_parses_hashes_and_preserves_order_duplicates_and_numbers() -> None:
    adapter = TypeAdapter(ValueExpr)
    payload = {
        "kind": "map_object_list",
        "items": {"item": "manifest", "path": "values"},
        "template": {"record": {"fixed": 7}, "enabled": True},
        "item_output_path": "record.value",
    }
    query = adapter.validate_python(payload)
    assert isinstance(query, MapObjectList)
    assert canonical_expression_json(query) == (
        '{"item_output_path":"record.value","items":{"item":"manifest",'
        '"path":"values"},"kind":"map_object_list","template":{"enabled":true,'
        '"record":{"fixed":7}}}'
    )
    assert expression_hash(query) == expression_hash(
        adapter.validate_python(dict(reversed(payload.items())))
    )

    source = [1, 1, 2.5, -0.0, {"nested": [3]}]
    result = evaluate_query(query, _ctx({"values": source}))
    assert result == [
        {"record": {"fixed": 7, "value": 1}, "enabled": True},
        {"record": {"fixed": 7, "value": 1}, "enabled": True},
        {"record": {"fixed": 7, "value": 2.5}, "enabled": True},
        {"record": {"fixed": 7, "value": -0.0}, "enabled": True},
        {"record": {"fixed": 7, "value": {"nested": [3]}}, "enabled": True},
    ]
    result[0]["record"]["fixed"] = 99
    result[-1]["record"]["value"]["nested"].append(4)
    assert result[1]["record"]["fixed"] == 7
    assert source[-1] == {"nested": [3]}


def test_map_object_list_fails_closed_for_missing_or_non_list_items() -> None:
    query = MapObjectList(
        items=ValueQuery(item="manifest", path="values"),
        template={"record": {}},
        item_output_path="record.value",
    )

    with pytest.raises(ExpressionPathMissing):
        evaluate_query(query, _ctx({}))
    with pytest.raises(ExpressionItemMissing):
        evaluate_query(
            MapObjectList(
                items=ValueQuery(item="missing", path="values"),
                template={"record": {}},
                item_output_path="record.value",
            ),
            _ctx({"values": []}),
        )
    with pytest.raises(ExpressionTypeError, match="requires a list"):
        evaluate_query(query, _ctx({"values": {"not": "a list"}}))
    with pytest.raises(ExpressionTypeError, match="must return a JSON list"):
        evaluate_query(query, _ctx({"values": [float("nan")]}))
    with pytest.raises(ExpressionTypeError, match="must return a JSON list"):
        evaluate_query(query, _ctx({"values": [SimpleNamespace(value=1)]}))


@pytest.mark.parametrize(
    ("template", "path", "message"),
    [
        ({}, "", "not dotted-path-like"),
        ({}, ".value", "not dotted-path-like"),
        ({}, "record.value", "missing segment"),
        ({"record": 1}, "record.value", "terminate in an object"),
        ({"record": {"value": None}}, "record.value", "collides"),
        ({"record": {"value": 1}}, "record.value", "collides"),
        ({"record": {}, "bad": float("nan")}, "record.value", "closed JSON value"),
    ],
)
def test_map_object_list_rejects_invalid_or_colliding_output_paths(
    template: object,
    path: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        MapObjectList(
            items=ValueQuery(item="manifest", path="values"),
            template=template,
            item_output_path=path,
        )


def test_map_object_list_depth_includes_its_items_query() -> None:
    expr = Compare(item="entry", path="x", op="eq", value=True)
    for _ in range(MAX_EXPR_DEPTH - 3):
        expr = Not(expr=expr)
    items = ValueQuery(item="manifest", path="values", filter=Filter(where=expr))

    with pytest.raises(ValidationError, match="exceeds max supported depth"):
        MapObjectList(items=items, template={}, item_output_path="value")


def test_manifest_predicate_expressibility_without_bundle_behavior_change() -> None:
    predicate = ManifestPredicate(
        metadata_equals={"group.name": "control"},
        params_equals={"model.width": 32},
    )
    metadata = {"group": {"name": "control"}}
    params = {"model": {"width": 32}}
    expr = AllOf(
        exprs=[
            *[
                Compare(item="metadata", path=path, op="eq", value=value)
                for path, value in predicate.metadata_equals.items()
            ],
            *[
                Compare(item="params", path=path, op="eq", value=value)
                for path, value in predicate.params_equals.items()
            ],
        ]
    )
    ctx = ExpressionContext(
        items={
            "metadata": ContextItem(kind="metadata", payload=metadata),
            "params": ContextItem(kind="params", payload=params),
        }
    )

    assert evaluate_expr(expr, ctx) is True
    assert evaluate_expr(expr, ctx) == (
        _equals_all(metadata, predicate.metadata_equals)
        and _equals_all(params, predicate.params_equals)
    )


def test_path_expression_schema_registry_family_and_old_version_rejection() -> None:
    family = default_spec_registry.resolve("PathExpression")

    assert family.identity == PATH_EXPRESSION_SCHEMA_ID
    assert family.current_version == PATH_EXPRESSION_SCHEMA_VERSION
    result = default_spec_registry.migrate(
        "PathExpression",
        {"schema_version": PATH_EXPRESSION_SCHEMA_VERSION},
    )
    assert result.schema_id == PATH_EXPRESSION_SCHEMA_ID
    assert not result.migrated

    with pytest.raises(UnsupportedSpecVersion):
        default_spec_registry.migrate(
            "PathExpression",
            {"schema_version": "feedbax.spec.path_expression.v0"},
        )


class _Knot(NamedTuple):
    mean: float
    upper: float


def test_value_query_indexes_into_lists_and_tuples_by_numeric_segment() -> None:
    ctx = _ctx({"items": [10, 20, 30], "pair": (9, 8)})
    assert evaluate_query(ValueQuery(item="manifest", path="items.1"), ctx) == 20
    assert evaluate_query(ValueQuery(item="manifest", path="pair.0"), ctx) == 9


def test_value_query_walks_nested_list_of_dicts_knot_series_shape() -> None:
    # The rlrmp2 per-knot diagnostics artifact shape: a JSON list of per-knot
    # dicts whose numeric fields are per-batch series.
    payload = {
        "knot_series": {
            "post_update_lambda": [
                {"mean": [1.0, 2.0], "upper": [1.5, 2.5]},
                {"mean": [3.0, 4.0], "upper": [3.5, 4.5]},
            ]
        }
    }
    ctx = _ctx(payload)
    assert evaluate_query(
        ValueQuery(item="manifest", path="knot_series.post_update_lambda.0.mean"), ctx
    ) == [1.0, 2.0]
    assert evaluate_query(
        ValueQuery(item="manifest", path="knot_series.post_update_lambda.1.mean.1"), ctx
    ) == 4.0


def test_value_query_sequence_index_fails_closed_out_of_range_and_non_numeric() -> None:
    ctx = _ctx({"items": [10, 20, 30]})
    with pytest.raises(ExpressionPathMissing, match="missing segment '5'"):
        evaluate_query(ValueQuery(item="manifest", path="items.5"), ctx)
    with pytest.raises(ExpressionPathMissing, match="missing segment 'name'"):
        evaluate_query(ValueQuery(item="manifest", path="items.name"), ctx)


def test_value_query_namedtuple_keeps_attribute_access_not_positional() -> None:
    ctx = _ctx({"knot": _Knot(mean=7.0, upper=9.0)})
    # NamedTuples resolve by field name, not by positional index.
    assert evaluate_query(ValueQuery(item="manifest", path="knot.mean"), ctx) == 7.0
    with pytest.raises(ExpressionPathMissing, match="missing segment '0'"):
        evaluate_query(ValueQuery(item="manifest", path="knot.0"), ctx)
