from __future__ import annotations

from types import SimpleNamespace

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
    Compare,
    ContextItem,
    Expr,
    ExpressionContext,
    ExpressionItemMissing,
    ExpressionPathMissing,
    ExpressionSelectAmbiguous,
    ExpressionTypeError,
    NamedPredicateRef,
    NamedPredicateUnresolved,
    Not,
    Select,
    ValueQuery,
    canonical_expression_json,
    evaluate_expr,
    evaluate_query,
    expression_hash,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry


def _ctx(payload: object, *, kind: str = "manifest") -> ExpressionContext:
    return ExpressionContext(items={"manifest": ContextItem(kind=kind, payload=payload)})


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
    "frontier",
    [
        [{"factor": 1.0}, {"factor": 1.1}],
        [{"factor": 1.05}, {"factor": 1.0500000000001}],
    ],
)
def test_value_query_select_raises_for_zero_or_multiple_matches(
    frontier: list[dict[str, float]],
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

    with pytest.raises(ExpressionSelectAmbiguous):
        evaluate_query(query, _ctx({"frontier": frontier}))


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
    with pytest.raises(ExpressionSelectAmbiguous):
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


def test_depth_guard_allows_depth_limit_and_rejects_pathological_nesting() -> None:
    expr = Compare(item="manifest", path="x", op="eq", value=True)
    for _ in range(MAX_EXPR_DEPTH - 1):
        expr = Not(expr=expr)

    evaluate_expr(expr, _ctx({"x": False}))

    with pytest.raises(ValidationError):
        Not(expr=expr)


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
