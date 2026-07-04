"""Versioned path-expression contracts and evaluator.

The expression algebra is deliberately small: field paths, leaf comparisons,
boolean combinators, exact-cardinality list selection, scalar coercion, and a
registered named-predicate escape hatch. Named predicates are contractually pure
and deterministic; hosts are responsible for registering reviewed callable
components and for mapping typed evaluator failures to host-specific statuses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import math
from typing import Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import Field, model_validator

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
    "has_type",
]
CoerceTarget: TypeAlias = Literal["float", "int", "str", "bool"]


class ExpressionEvaluationError(ValueError):
    """Base class for typed path-expression evaluation failures."""


class ExpressionItemMissing(ExpressionEvaluationError):
    """Raised when an expression references an unbound context item."""


class ExpressionPathMissing(ExpressionEvaluationError):
    """Raised when a non-absence-tolerant expression path is missing."""


class ExpressionSelectAmbiguous(ExpressionEvaluationError):
    """Raised when list selection matches zero or multiple entries."""


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
    """Exact-cardinality list search using the reserved ``entry`` alias."""

    where: Expr

    @model_validator(mode="after")
    def _validate_depth(self) -> "Select":
        _check_depth(self.where)
        return self


class Coerce(StrictModel):
    """Scalar coercion and optional unit scaling."""

    to: CoerceTarget | None = None
    scale: float | None = None


class ValueQuery(StrictModel):
    """Read one value from a bound context item, optionally selecting/coercing it."""

    item: str
    path: PathRef = ""
    select: Select | None = None
    coerce: Coerce | None = None


AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()
Select.model_rebuild()


def expression_hash(expr: Expr | ValueQuery) -> str:
    """Return a stable SHA-256 digest for a path expression or value query."""
    canonical = canonical_expression_json(expr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_expression_json(expr: Expr | ValueQuery) -> str:
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
    query: ValueQuery,
    ctx: ExpressionContext,
    *,
    named_predicates: NamedPredicateResolver | None = None,
) -> Any:
    """Evaluate a value query against a host-bound context."""
    value = _resolve_path(ctx, query.item, query.path)
    if query.select is not None:
        value = _select_one(value, query.select, ctx, named_predicates=named_predicates)
    if query.coerce is not None:
        value = _coerce(value, query.coerce)
    return value


def _check_depth(expr: Expr) -> None:
    depth = _expr_depth(expr)
    if depth > MAX_EXPR_DEPTH:
        raise ValueError(
            f"path expression depth {depth} exceeds max supported depth {MAX_EXPR_DEPTH}"
        )


def _expr_depth(expr: Expr) -> int:
    if isinstance(expr, (Compare, NamedPredicateRef)):
        return 1
    if isinstance(expr, Not):
        return 1 + _expr_depth(expr.expr)
    if isinstance(expr, (AllOf, AnyOf)):
        return 1 + max(_expr_depth(child) for child in expr.exprs)
    return 1


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
            else:
                current = getattr(current, part)
        except (AttributeError, KeyError, TypeError) as exc:
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
        raise ExpressionSelectAmbiguous(
            f"Select expected exactly one match, found {len(matches)}"
        )
    return matches[0]


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
    "NamedPredicate",
    "NamedPredicateRef",
    "NamedPredicateResolver",
    "NamedPredicateUnresolved",
    "Not",
    "PathRef",
    "Select",
    "ValueQuery",
    "canonical_expression_json",
    "evaluate_expr",
    "evaluate_query",
    "expression_hash",
]
