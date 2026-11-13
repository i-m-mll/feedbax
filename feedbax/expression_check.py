"""Check stored expressions before constructing executable closures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    DIMENSIONLESS,
    RESERVED_NAMES,
    CallValue,
    Dimensions,
    DrawValue,
    Expression,
    Level,
    LiteralValue,
    NameValue,
    NodePath,
    Problem,
    ProblemCode,
    Site,
    TextSpan,
    ValueBase,
    ValueType,
    children,
    scope_problems,
)
from feedbax.expression_registry import FunctionClass, FunctionRegistry, OperandHint, SignatureError


@dataclass(frozen=True)
class TypedNode:
    type: ValueType
    level: Level


@dataclass(frozen=True)
class Checked:
    root: TypedNode
    nodes: Mapping[NodePath, TypedNode]


def references(tree: Expression) -> frozenset[str]:
    pending = [tree]
    names = set()
    while pending:
        node = pending.pop()
        if isinstance(node, NameValue):
            names.add(node.reference)
        pending.extend(child for _, child in children(node))
    return frozenset(names)


def dependency_order(values: Mapping[str, Expression]) -> tuple[str, ...] | tuple[Problem, ...]:
    """Return definitions in dependency order or a problem on the closing edge."""
    dependencies = {key: references(value) & values.keys() for key, value in values.items()}
    result = []
    active = []
    done = set()

    def visit(key: str) -> Problem | None:
        if key in done:
            return None
        if key in active:
            cycle = (*active[active.index(key) :], key)
            return Problem(
                code=ProblemCode.CYCLE,
                reason=" → ".join(cycle),
                path=("variables", active[-1], "value"),
            )
        active.append(key)
        for dependency in sorted(dependencies[key]):
            if problem := visit(dependency):
                return problem
        active.pop()
        done.add(key)
        result.append(key)
        return None

    for key in values:
        if problem := visit(key):
            return (problem,)
    return tuple(result)


def _known_dimension(
    node: Expression,
    scope: Mapping[str, TypedNode],
    functions: FunctionRegistry,
) -> Dimensions | None:
    if isinstance(node, LiteralValue) and node.unit in UNITS:
        return UNITS[node.unit].dimensions
    if isinstance(node, NameValue) and node.reference in scope:
        return scope[node.reference].type.dimensions
    if isinstance(node, (CallValue, DrawValue)):
        key = node.function if isinstance(node, CallValue) else node.family
        function = functions.resolve(key)
        if function is not None and function.result_dimensions is not None:
            if function.info.min_args <= len(node.args) and (
                function.info.max_args is None or len(node.args) <= function.info.max_args
            ):
                return function.result_dimensions(
                    tuple(_known_dimension(child, scope, functions) for child in node.args)
                )
    return None


def check(
    tree: Expression,
    site: Site,
    functions: FunctionRegistry,
    *,
    path: NodePath = (),
    spans: Mapping[NodePath, TextSpan] | None = None,
    missing_functions: frozenset[str] = frozenset(),
) -> Checked | tuple[Problem, ...]:
    """Return path-addressed Problems for bad authored values."""
    if problems := scope_problems(site, path):
        return problems
    scope = {value.id: TypedNode(value.type, value.level) for value in site.names}
    by_name = {value.name: value for value in site.names}
    problems: list[Problem] = []
    checked: dict[NodePath, TypedNode] = {}

    def problem(node: Expression, code: ProblemCode, reason: str, at: NodePath) -> None:
        problems.append(
            Problem(
                code=code,
                reason=reason,
                path=at,
                span=spans.get(at[len(path) :]) if spans else None,
            )
        )

    def visit(
        node: Expression,
        at: NodePath,
        dimension: Dimensions | None = None,
        requested: ValueBase | None = None,
    ) -> TypedNode | None:
        if isinstance(node, LiteralValue):
            base = {
                bool: ValueBase.BOOLEAN,
                int: ValueBase.INTEGER,
                float: ValueBase.REAL,
            }[type(node.value)]
            if node.unit is not None and (node.unit not in UNITS or base == ValueBase.BOOLEAN):
                problem(node, ProblemCode.UNIT_MISMATCH, f"Invalid literal unit {node.unit!r}.", at)
                return None
            if requested == ValueBase.REAL and base == ValueBase.INTEGER:
                base = ValueBase.REAL
            unit = (
                UNITS[node.unit].dimensions
                if node.unit is not None
                else dimension
                if dimension is not None and base != ValueBase.BOOLEAN
                else DIMENSIONLESS
            )
            value = TypedNode(ValueType(base=base, dimensions=unit), Level.RUN)
        elif isinstance(node, NameValue):
            value = scope.get(node.reference)
            if value is None:
                code = (
                    ProblemCode.OUT_OF_SCOPE
                    if node.reference in RESERVED_NAMES
                    else ProblemCode.UNKNOWN_NAME
                )
                problem(node, code, f"{node.reference!r} is not in scope here.", at)
                return None
        else:
            key = node.function if isinstance(node, CallValue) else node.family
            function = functions.resolve(key)
            if function is None:
                code = (
                    ProblemCode.PLUGIN_MISSING
                    if key in missing_functions or "." in key
                    else ProblemCode.UNKNOWN_FUNCTION
                )
                problem(node, code, f"Expression function {key!r} is unavailable.", at)
                return None
            info = function.info
            if site.structural and info.random:
                problem(
                    node,
                    ProblemCode.STRUCTURE,
                    "A structural field takes Value, Sweep, Formula.",
                    at,
                )
                return None
            if isinstance(node, DrawValue) != (info.display_class == FunctionClass.DRAW):
                problem(node, ProblemCode.STRUCTURE, f"{key!r} has the wrong expression form.", at)
                return None
            kwargs = node.kwargs if isinstance(node, CallValue) else {}
            if (
                not info.min_args <= len(node.args)
                or info.max_args is not None
                and len(node.args) > info.max_args
            ):
                problem(node, ProblemCode.ARITY, f"Expected {info.signature}.", at)
                return None
            if unknown := kwargs.keys() - set(info.keywords):
                problem(
                    node, ProblemCode.ARITY, f"Unknown arguments {sorted(unknown)} for {key!r}.", at
                )
                return None
            implicit = tuple(
                name for name in info.implicit_names if name != "T" or "over" not in kwargs
            )
            if absent := set(implicit) - by_name.keys():
                problem(
                    node,
                    ProblemCode.OUT_OF_SCOPE,
                    f"{key!r} requires {', '.join(sorted(absent))} here.",
                    at,
                )
                return None
            for name in implicit:
                expected = info.implicit_names[name]
                if not expected.accepts(by_name[name].type):
                    problem(
                        node,
                        ProblemCode.TYPE_MISMATCH,
                        f"{key!r} requires {name} as {expected}.",
                        at,
                    )
                    return None
            known = tuple(_known_dimension(arg, scope, functions) for arg in node.args)
            hints, named_hints = (
                function.argument_context(OperandHint(dimension, requested), known, tuple(kwargs))
                if function.argument_context is not None
                else ((OperandHint(dimension, requested),) * len(node.args), {})
            )
            positional = [
                visit(arg, (*at, "args", i), hint.dimensions, hint.base)
                for i, (arg, hint) in enumerate(zip(node.args, hints, strict=True))
            ]
            named = {
                name: visit(child, (*at, "kwargs", name), hint.dimensions, hint.base)
                for name, child in kwargs.items()
                for hint in (named_hints.get(name, OperandHint()),)
            }
            if any(item is None for item in (*positional, *named.values())):
                return None
            arguments = tuple(item for item in positional if item is not None)
            keywords = {name: item for name, item in named.items() if item is not None}
            inferred = function.infer(
                tuple(item.type for item in arguments),
                {name: item.type for name, item in keywords.items()},
            )
            if isinstance(inferred, SignatureError):
                problem(node, inferred.code, inferred.reason, at)
                return None
            level = max(
                (
                    info.level,
                    *(item.level for item in arguments),
                    *(item.level for item in keywords.values()),
                    *(by_name[name].level for name in implicit),
                )
            )
            if isinstance(node, DrawValue):
                limit = Level.TRIAL if node.per == "trial" else Level.RUN
                if level > limit:
                    problem(
                        node,
                        ProblemCode.LEVEL_TOO_FINE,
                        f"A per-{node.per} draw has a finer-level input.",
                        at,
                    )
                    return None
                level = limit
            value = TypedNode(inferred, level)
        checked[at[len(path) :]] = value
        return value

    root = visit(tree, path, site.expected.dimensions, site.expected.base)
    if root is not None:
        if not site.expected.accepts_shape_and_base(root.type):
            problem(
                tree,
                ProblemCode.TYPE_MISMATCH,
                f"Expected {site.expected.base} {site.expected.shape}.",
                path,
            )
        elif root.type.dimensions != site.expected.dimensions:
            problem(
                tree,
                ProblemCode.UNIT_MISMATCH,
                "Expression dimensions differ from the site's dimensions.",
                path,
            )
        if root.level > site.finest:
            problem(
                tree,
                ProblemCode.LEVEL_TOO_FINE,
                f"This site accepts at most {site.finest.name.lower()} values.",
                path,
            )
    if problems:
        return tuple(problems)
    assert root is not None
    return Checked(root, MappingProxyType(checked))
