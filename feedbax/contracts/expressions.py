"""Versioned path-expression contracts and evaluator.

The expression algebra is deliberately small: field paths, leaf comparisons,
boolean combinators, exact-cardinality list selection, zero-or-more list
filtering, scalar coercion, ordered value coalescing, and a registered
named-predicate escape hatch. Named predicates are contractually pure and
deterministic; hosts are responsible for registering reviewed callable
components and for mapping typed evaluator failures to host-specific statuses.

``Compare(op="exists")`` checks whether a payload path exists inside the bound
expression context. It does not check whether a source file exists on disk;
filesystem existence is handled by extraction ``SourceBinding`` loading.

Additive changelog, 2026-07-05: v1 accepts optional value-query defaults,
``Coalesce`` value expressions, ``Filter`` list subsets, and string
``startswith``/``endswith`` comparisons. These additions preserve canonical
JSON and hash bytes for pre-existing expressions because all new fields are
None-default optional fields or new node types.

Additive changelog, 2026-08-03: v1 accepts ``MapObjectList`` value expressions.
The new arm has its own discriminator, so pre-existing canonical JSON and hash
bytes remain unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import Field, JsonValue, model_validator

from feedbax.contracts.manifest import StrictModel


PATH_EXPRESSION_SCHEMA_ID = "feedbax.spec.path_expression"
PATH_EXPRESSION_SCHEMA_VERSION = "feedbax.spec.path_expression.v1"
MAX_EXPR_DEPTH = 32

PathRef: TypeAlias = str
CompareOp: TypeAlias = Literal[
    "exists",
    "eq",
    "ne",
    "approx_eq",
    "lt",
    "le",
    "gt",
    "ge",
    "in",
    "contains",
    "startswith",
    "endswith",
    "has_type",
]
CoerceTarget: TypeAlias = Literal["float", "int", "str", "bool"]


class ExpressionEvaluationError(ValueError):
    """Base class for typed path-expression evaluation failures."""


class ExpressionItemMissing(ExpressionEvaluationError):
    """Raised when an expression references an unbound context item."""


class ExpressionPathMissing(ExpressionEvaluationError):
    """Raised when a non-absence-tolerant expression path is missing."""


class _ExpressionAbsenceMiss(ExpressionPathMissing):
    """Internal absence-class miss that defaults and Coalesce may handle."""


class ExpressionSelectAmbiguous(ExpressionEvaluationError):
    """Raised when exact-cardinality list selection matches multiple entries."""


class ExpressionTypeError(ExpressionEvaluationError):
    """Raised when an operation receives values of incompatible types."""


class NamedPredicateUnresolved(ExpressionEvaluationError):
    """Raised when a named predicate cannot be resolved to a callable."""


class NamedPredicate(Protocol):
    """Pure named predicate callable resolved by host component registries."""

    def __call__(self, ctx: "ExpressionContext", **params: Any) -> bool:
        """Return whether ``ctx`` satisfies the predicate."""


NamedPredicateResolver: TypeAlias = (
    Mapping[str, NamedPredicate] | Callable[[str], NamedPredicate | None]
)


class ContextItem(StrictModel):
    """Host-bound item available to expression evaluation."""

    kind: str
    payload: Any


class ExpressionContext(StrictModel):
    """Host-agnostic expression bindings."""

    items: dict[str, ContextItem] = Field(default_factory=dict)


class Compare(StrictModel):
    """One leaf judgment about one context value."""

    kind: Literal["compare"] = "compare"
    item: str
    path: PathRef = ""
    op: CompareOp
    value: Any | None = None
    tolerance: float | None = None

    @model_validator(mode="after")
    def _validate_compare_fields(self) -> "Compare":
        if self.op == "approx_eq":
            if self.tolerance is None:
                raise ValueError("Compare tolerance is required when op='approx_eq'")
        elif self.tolerance is not None:
            raise ValueError("Compare tolerance is only allowed when op='approx_eq'")

        if self.op != "exists" and "value" not in self.model_fields_set:
            raise ValueError(f"Compare value is required when op={self.op!r}")
        if self.op == "exists" and "value" in self.model_fields_set:
            raise ValueError("Compare value is not allowed when op='exists'")
        return self


class NamedPredicateRef(StrictModel):
    """Reference to a registered pure predicate component.

    ``predicate_id`` names a reviewed host/component-registry entry implementing
    ``__call__(ctx, **params) -> bool``. The callable is pure and deterministic
    by contract: no I/O, clock reads, randomness, or mutation of host state.
    """

    kind: Literal["named"] = "named"
    predicate_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class AllOf(StrictModel):
    """Boolean conjunction over non-empty child expressions."""

    kind: Literal["all"] = "all"
    exprs: list[Expr] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_depth(self) -> "AllOf":
        _check_depth(self)
        return self


class AnyOf(StrictModel):
    """Boolean disjunction over non-empty child expressions."""

    kind: Literal["any"] = "any"
    exprs: list[Expr] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_depth(self) -> "AnyOf":
        _check_depth(self)
        return self


class Not(StrictModel):
    """Boolean negation of a child expression."""

    kind: Literal["not"] = "not"
    expr: Expr

    @model_validator(mode="after")
    def _validate_depth(self) -> "Not":
        _check_depth(self)
        return self


Expr: TypeAlias = Annotated[
    Compare | NamedPredicateRef | AllOf | AnyOf | Not,
    Field(discriminator="kind"),
]


class Select(StrictModel):
    """Exact-cardinality list search using the reserved ``entry`` alias.

    A zero-match ``Select`` result is an absence-class miss. ``ValueQuery``
    defaults and ``Coalesce`` may handle that miss; multiple matches always
    raise ``ExpressionSelectAmbiguous``.
    """

    where: Expr

    @model_validator(mode="after")
    def _validate_depth(self) -> "Select":
        _check_depth(self.where)
        return self


class Filter(StrictModel):
    """Zero-or-more list search using the reserved ``entry`` alias.

    Unlike ``Select``, an empty ``Filter`` result is a real value (``[]``), not
    an absence-class miss.
    """

    where: Expr

    @model_validator(mode="after")
    def _validate_depth(self) -> "Filter":
        _check_depth(self.where)
        return self


class Coerce(StrictModel):
    """Scalar coercion and optional unit scaling."""

    to: CoerceTarget | None = None
    scale: float | None = None


class ValueQuery(StrictModel):
    """Read one value from a bound context item, optionally selecting/filtering it."""

    item: str
    path: PathRef = ""
    select: Select | None = None
    filter: Filter | None = None
    coerce: Coerce | None = None
    default: Any | None = None

    @model_validator(mode="after")
    def _validate_selection_fields(self) -> "ValueQuery":
        if self.select is not None and self.filter is not None:
            raise ValueError("ValueQuery select and filter are mutually exclusive")
        _check_value_query_depth(self)
        return self


class Coalesce(StrictModel):
    """Ordered fallback over absence-class value-query misses.

    Inner ``ValueQuery.default`` values are allowed and make later queries
    unreachable when that query misses; that is valid but usually a spec smell.
    """

    queries: list[ValueQuery] = Field(min_length=1)
    default: Any | None = None

    @model_validator(mode="after")
    def _validate_depth(self) -> "Coalesce":
        for query in self.queries:
            _check_value_query_depth(query)
        return self


class MapObjectList(StrictModel):
    """Project an ordered queried list into deep-copied JSON objects.

    ``item_output_path`` must name a missing leaf beneath existing object-valued
    parents in ``template``. This makes the one authored write explicit and
    refuses both implicit object synthesis and replacement of template content.
    """

    kind: Literal["map_object_list"] = "map_object_list"
    items: ValueQuery
    template: JsonValue
    item_output_path: PathRef

    @model_validator(mode="after")
    def _validate_projection(self) -> "MapObjectList":
        try:
            json.dumps(self.template, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("MapObjectList template must be a closed JSON value") from exc
        _validate_map_output_path(self.template, self.item_output_path)
        _check_depth(self)
        return self


ValueExpr: TypeAlias = ValueQuery | Coalesce | MapObjectList


AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()
Select.model_rebuild()
Filter.model_rebuild()
ValueQuery.model_rebuild()
Coalesce.model_rebuild()
MapObjectList.model_rebuild()


def expression_hash(expr: Expr | ValueExpr | Select | Filter) -> str:
    """Return a stable SHA-256 digest for a path expression or value query."""
    canonical = canonical_expression_json(expr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_expression_json(expr: Expr | ValueExpr | Select | Filter) -> str:
    """Return canonical JSON for hashing and deterministic review diffs."""
    payload = expr.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def evaluate_expr(
    expr: Expr,
    ctx: ExpressionContext,
    *,
    named_predicates: NamedPredicateResolver | None = None,
) -> bool:
    """Evaluate a boolean expression against a host-bound context.

    Args:
        expr: Expression AST to evaluate.
        ctx: Host-agnostic item bindings.
        named_predicates: Optional resolver for registered, pure predicate
            components referenced by ``NamedPredicateRef`` leaves.

    Returns:
        The deterministic boolean result.

    Raises:
        ExpressionItemMissing: A non-absence-tolerant item alias is missing.
        ExpressionPathMissing: A non-absence-tolerant path is missing.
        ExpressionTypeError: The operation is incompatible with operand types.
        NamedPredicateUnresolved: A named predicate cannot be resolved.
    """
    _check_depth(expr)
    if isinstance(expr, Compare):
        return _evaluate_compare(expr, ctx)
    if isinstance(expr, NamedPredicateRef):
        predicate = _resolve_named_predicate(expr.predicate_id, named_predicates)
        result = predicate(ctx, **expr.params)
        if not isinstance(result, bool):
            raise ExpressionTypeError(
                f"named predicate {expr.predicate_id!r} returned {type(result).__name__}, "
                "expected bool"
            )
        return result
    if isinstance(expr, AllOf):
        return all(
            evaluate_expr(child, ctx, named_predicates=named_predicates)
            for child in expr.exprs
        )
    if isinstance(expr, AnyOf):
        return any(
            evaluate_expr(child, ctx, named_predicates=named_predicates)
            for child in expr.exprs
        )
    if isinstance(expr, Not):
        return not evaluate_expr(expr.expr, ctx, named_predicates=named_predicates)
    raise ExpressionTypeError(f"unsupported expression type: {type(expr).__name__}")


def evaluate_query(
    query: ValueExpr,
    ctx: ExpressionContext,
    *,
    named_predicates: NamedPredicateResolver | None = None,
) -> Any:
    """Evaluate a value query against a host-bound context."""
    if isinstance(query, ValueQuery):
        return _evaluate_value_query(query, ctx, named_predicates=named_predicates)
    if isinstance(query, Coalesce):
        return _evaluate_coalesce(query, ctx, named_predicates=named_predicates)
    if isinstance(query, MapObjectList):
        return _evaluate_map_object_list(query, ctx, named_predicates=named_predicates)
    raise ExpressionTypeError(f"unsupported value query type: {type(query).__name__}")


def _evaluate_value_query(
    query: ValueQuery,
    ctx: ExpressionContext,
    *,
    named_predicates: NamedPredicateResolver | None,
) -> Any:
    try:
        value = _resolve_path(ctx, query.item, query.path)
    except ExpressionPathMissing as exc:
        if "default" in query.model_fields_set:
            return query.default
        raise _ExpressionAbsenceMiss(str(exc)) from exc

    if query.select is not None:
        try:
            value = _select_one(value, query.select, ctx, named_predicates=named_predicates)
        except _ExpressionAbsenceMiss:
            if "default" in query.model_fields_set:
                return query.default
            raise
    if query.filter is not None:
        value = _filter_many(value, query.filter, ctx, named_predicates=named_predicates)

    if query.coerce is not None:
        if query.filter is not None:
            value = [_coerce(entry, query.coerce) for entry in value]
        else:
            value = _coerce(value, query.coerce)
    return value


def _evaluate_coalesce(
    query: Coalesce,
    ctx: ExpressionContext,
    *,
    named_predicates: NamedPredicateResolver | None,
) -> Any:
    misses: list[str] = []
    for child in query.queries:
        try:
            return _evaluate_value_query(child, ctx, named_predicates=named_predicates)
        except _ExpressionAbsenceMiss as exc:
            misses.append(f"{child.item}:{child.path or '<root>'} ({exc})")
    if "default" in query.model_fields_set:
        return query.default
    attempted = "; ".join(misses)
    raise ExpressionPathMissing(f"Coalesce found no available query value; attempted {attempted}")


def _evaluate_map_object_list(
    query: MapObjectList,
    ctx: ExpressionContext,
    *,
    named_predicates: NamedPredicateResolver | None,
) -> list[JsonValue]:
    items = _evaluate_value_query(query.items, ctx, named_predicates=named_predicates)
    if not isinstance(items, list):
        raise ExpressionTypeError(
            "MapObjectList items query requires a list at the query terminus, "
            f"got {type(items).__name__}"
        )
    try:
        json.dumps(items, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExpressionTypeError("MapObjectList items query must return a JSON list") from exc

    projected: list[JsonValue] = []
    for item in items:
        record = deepcopy(query.template)
        _write_map_item(record, query.item_output_path, deepcopy(item))
        projected.append(record)
    return projected


def _check_depth(expr: Expr | Select | Filter | ValueQuery | Coalesce | MapObjectList) -> None:
    depth = _expr_depth(expr)
    if depth > MAX_EXPR_DEPTH:
        raise ValueError(
            f"path expression depth {depth} exceeds max supported depth {MAX_EXPR_DEPTH}"
        )


def _expr_depth(expr: Expr | Select | Filter | ValueQuery | Coalesce | MapObjectList) -> int:
    if isinstance(expr, (Compare, NamedPredicateRef)):
        return 1
    if isinstance(expr, Not):
        return 1 + _expr_depth(expr.expr)
    if isinstance(expr, (AllOf, AnyOf)):
        return 1 + max(_expr_depth(child) for child in expr.exprs)
    if isinstance(expr, (Select, Filter)):
        return 1 + _expr_depth(expr.where)
    if isinstance(expr, ValueQuery):
        child_depths = [
            _expr_depth(child)
            for child in (expr.select, expr.filter)
            if child is not None
        ]
        return 1 + (max(child_depths) if child_depths else 0)
    if isinstance(expr, Coalesce):
        return 1 + max(_expr_depth(query) for query in expr.queries)
    if isinstance(expr, MapObjectList):
        return 1 + _expr_depth(expr.items)
    return 1


def _check_value_query_depth(query: ValueQuery) -> None:
    _check_depth(query)


def _validate_map_output_path(template: JsonValue, path: PathRef) -> None:
    if not path.strip() or any(not part for part in path.split(".")):
        raise ValueError(f"MapObjectList item_output_path is not dotted-path-like: {path!r}")
    current: JsonValue = template
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ValueError(
                "MapObjectList item_output_path must traverse existing object fields: "
                f"{path!r}; missing segment {part!r}"
            )
        current = current[part]
    if not isinstance(current, dict):
        raise ValueError(f"MapObjectList item_output_path must terminate in an object: {path!r}")
    if parts[-1] in current:
        raise ValueError(f"MapObjectList item_output_path collides with template content: {path!r}")


def _write_map_item(template: JsonValue, path: PathRef, item: JsonValue) -> None:
    current = template
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ExpressionTypeError(
                f"MapObjectList cannot write invalid item_output_path {path!r}"
            )
        current = current[part]
    if not isinstance(current, dict) or parts[-1] in current:
        raise ExpressionTypeError(f"MapObjectList cannot write colliding item_output_path {path!r}")
    current[parts[-1]] = item


def _evaluate_compare(expr: Compare, ctx: ExpressionContext) -> bool:
    if expr.op == "has_type":
        item = ctx.items.get(expr.item)
        return item is not None and item.kind == expr.value
    if expr.op == "exists":
        if expr.item not in ctx.items:
            return False
        try:
            _resolve_path(ctx, expr.item, expr.path)
        except ExpressionPathMissing:
            return False
        return True

    actual = _resolve_path(ctx, expr.item, expr.path)
    expected = expr.value
    try:
        if expr.op == "eq":
            return actual == expected
        if expr.op == "ne":
            return actual != expected
        if expr.op == "approx_eq":
            return _approx_eq(actual, expected, expr.tolerance)
        if expr.op == "lt":
            return actual < expected
        if expr.op == "le":
            return actual <= expected
        if expr.op == "gt":
            return actual > expected
        if expr.op == "ge":
            return actual >= expected
        if expr.op == "in":
            return actual in expected
        if expr.op == "contains":
            return expected in actual
        if expr.op in {"startswith", "endswith"}:
            if not isinstance(actual, str) or not isinstance(expected, str):
                raise TypeError
            return (
                actual.startswith(expected)
                if expr.op == "startswith"
                else actual.endswith(expected)
            )
    except TypeError as exc:
        raise ExpressionTypeError(
            f"Compare op {expr.op!r} cannot compare {type(actual).__name__} "
            f"with {type(expected).__name__}"
        ) from exc
    raise ExpressionTypeError(f"unsupported compare op: {expr.op!r}")


def _approx_eq(actual: Any, expected: Any, tolerance: float | None) -> bool:
    if tolerance is None:
        raise ExpressionTypeError("approx_eq requires an explicit tolerance")
    if tolerance < 0:
        raise ExpressionTypeError("approx_eq tolerance must be non-negative")
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError) as exc:
        raise ExpressionTypeError(
            "approx_eq requires operands coercible to float: "
            f"actual={type(actual).__name__}, expected={type(expected).__name__}"
        ) from exc
    return math.fabs(actual_float - expected_float) <= tolerance


def _resolve_path(ctx: ExpressionContext, item_name: str, path: PathRef) -> Any:
    try:
        item = ctx.items[item_name]
    except KeyError as exc:
        raise ExpressionItemMissing(f"context item {item_name!r} is not bound") from exc
    return _get_path(item.payload, path, item=item_name)


def _get_path(value: Any, path: PathRef, *, item: str) -> Any:
    if path == "":
        return value
    current = value
    for part in path.split("."):
        try:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, (list, tuple)) and not hasattr(current, "_fields"):
                # Positional access into a plain sequence: only a base-10 integer
                # literal indexes it. A non-numeric segment or an out-of-range
                # index fails closed. NamedTuples (which carry ``_fields``) keep
                # attribute-by-name access below.
                if not (part.isascii() and part.isdigit()):
                    raise KeyError(part)
                current = current[int(part)]
            else:
                current = getattr(current, part)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ExpressionPathMissing(
                f"context item {item!r} has no path {path!r}; missing segment {part!r}"
            ) from exc
    return current


def _select_one(
    value: Any,
    select: Select,
    ctx: ExpressionContext,
    *,
    named_predicates: NamedPredicateResolver | None,
) -> Any:
    if not isinstance(value, list):
        raise ExpressionTypeError(
            f"Select requires a list at the query terminus, got {type(value).__name__}"
        )

    matches = []
    for entry in value:
        entry_ctx = ExpressionContext(
            items={**ctx.items, "entry": ContextItem(kind="entry", payload=entry)}
        )
        if evaluate_expr(select.where, entry_ctx, named_predicates=named_predicates):
            matches.append(entry)

    if len(matches) != 1:
        if not matches:
            raise _ExpressionAbsenceMiss("Select expected exactly one match, found 0")
        raise ExpressionSelectAmbiguous(
            f"Select expected exactly one match, found {len(matches)}"
        )
    return matches[0]


def _filter_many(
    value: Any,
    filter_: Filter,
    ctx: ExpressionContext,
    *,
    named_predicates: NamedPredicateResolver | None,
) -> list[Any]:
    if not isinstance(value, list):
        raise ExpressionTypeError(
            f"Filter requires a list at the query terminus, got {type(value).__name__}"
        )

    matches = []
    for entry in value:
        entry_ctx = ExpressionContext(
            items={**ctx.items, "entry": ContextItem(kind="entry", payload=entry)}
        )
        if evaluate_expr(filter_.where, entry_ctx, named_predicates=named_predicates):
            matches.append(entry)
    return matches


def _coerce(value: Any, coerce: Coerce) -> Any:
    result = value
    try:
        if coerce.to == "float":
            result = float(result)
        elif coerce.to == "int":
            result = int(result)
        elif coerce.to == "str":
            result = str(result)
        elif coerce.to == "bool":
            result = _coerce_bool(result)
    except (TypeError, ValueError) as exc:
        raise ExpressionTypeError(
            f"cannot coerce {type(value).__name__} to {coerce.to!r}"
        ) from exc

    if coerce.scale is not None:
        try:
            result = result * coerce.scale
        except TypeError as exc:
            raise ExpressionTypeError(
                f"cannot apply scale to {type(result).__name__}"
            ) from exc
    return result


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"cannot coerce {value!r} to bool")


def _resolve_named_predicate(
    predicate_id: str,
    named_predicates: NamedPredicateResolver | None,
) -> NamedPredicate:
    if named_predicates is None:
        raise NamedPredicateUnresolved(
            f"named predicate {predicate_id!r} is not resolvable without a registry"
        )

    predicate: NamedPredicate | None
    if isinstance(named_predicates, Mapping):
        predicate = named_predicates.get(predicate_id)
    else:
        predicate = named_predicates(predicate_id)

    if predicate is None or not callable(predicate):
        raise NamedPredicateUnresolved(f"named predicate {predicate_id!r} is not registered")
    return predicate


__all__ = [
    "PATH_EXPRESSION_SCHEMA_ID",
    "PATH_EXPRESSION_SCHEMA_VERSION",
    "MAX_EXPR_DEPTH",
    "AllOf",
    "AnyOf",
    "Coerce",
    "Coalesce",
    "Compare",
    "CompareOp",
    "ContextItem",
    "Expr",
    "ExpressionContext",
    "ExpressionEvaluationError",
    "ExpressionItemMissing",
    "ExpressionPathMissing",
    "ExpressionSelectAmbiguous",
    "ExpressionTypeError",
    "MapObjectList",
    "NamedPredicate",
    "NamedPredicateRef",
    "NamedPredicateResolver",
    "NamedPredicateUnresolved",
    "Not",
    "PathRef",
    "Filter",
    "Select",
    "ValueExpr",
    "ValueQuery",
    "canonical_expression_json",
    "evaluate_expr",
    "evaluate_query",
    "expression_hash",
]
