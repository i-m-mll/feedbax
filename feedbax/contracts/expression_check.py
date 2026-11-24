"""Check stored expressions before constructing executable closures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_text import resolve_expression
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
    UnresolvedValue,
    ValueBase,
    ValueType,
    walk,
    scope_problems,
)
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.expression_registry import (
    Function,
    FunctionClass,
    FunctionRegistry,
    OperandHint,
    SignatureError,
)


@dataclass(frozen=True)
class TypedNode:
    type: ValueType
    level: Level
    implicit_item: bool = False


@dataclass(frozen=True)
class Checked:
    tree: Expression
    root: TypedNode
    nodes: Mapping[NodePath, TypedNode]
    calls: Mapping[NodePath, tuple[Function, CallValue]]


def references(tree: Expression) -> frozenset[str]:
    return frozenset(node.reference for _, node in walk(tree) if isinstance(node, NameValue))


def dependency_order(
    values: Mapping[str, Expression], *, counts: Mapping[str, Expression] = MappingProxyType({}),
) -> tuple[tuple[str, ...], tuple[Problem, ...]]:
    """The definitions in dependency order, and a Problem on each definition in a cycle, which the order leaves out."""
    dependencies = {key: references(value) & values.keys() for key, value in values.items()}
    for key, count in counts.items():
        dependencies[key] |= references(count) & values.keys()
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    components: list[list[str]] = []

    def connect(key: str) -> None:
        index[key] = low[key] = len(index)
        stack.append(key)
        for dependency in sorted(dependencies[key]):
            if dependency not in index:
                connect(dependency)
                low[key] = min(low[key], low[dependency])
            elif dependency in stack:
                low[key] = min(low[key], index[dependency])
        if low[key] == index[key]:
            split = stack.index(key)
            components.append(stack[split:])
            del stack[split:]

    for key in values:
        if key not in index:
            connect(key)

    def loop(start: str, members: frozenset[str]) -> tuple[str, ...]:
        paths = {start: (start,)}
        pending = [start]
        for key in pending:
            for dependency in sorted(dependencies[key] & members):
                if dependency == start:
                    return (*paths[key], start)
                if dependency not in paths:
                    paths[dependency] = (*paths[key], dependency)
                    pending.append(dependency)
        raise AssertionError(f"{start!r} is in no cycle of its component")

    order = []
    problems = []
    for component in components:
        (key, *others) = component
        if not others and key not in dependencies[key]:
            order.append(key)
            continue
        members = frozenset(component)
        problems.extend(
            Problem(
                code=ProblemCode.CYCLE, reason=" → ".join(loop(member, members)), path=("variables", member, "value"),
            )
            for member in sorted(members, key=list(values).index)
        )
    return tuple(order), tuple(problems)


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
                if function.bind is not None and isinstance(node, CallValue):
                    bound = function.bind(function, node)
                    if isinstance(bound, SignatureError):
                        return None
                    function, node = bound
                assert function.result_dimensions is not None
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
    if isinstance(tree, UnresolvedValue):
        resolved = resolve_expression(tree, site, functions)
        if isinstance(resolved, tuple):
            return tuple(problem.model_copy(update={"path": (*path, *problem.path)}) for problem in resolved)
        tree, spans = resolved.value, resolved.spans
    scope = {value.id: TypedNode(value.type, value.level) for value in site.names}
    by_name = {value.name: value for value in site.names}
    lists = {value.id: value.list_count for value in site.names if value.list_count is not None}
    problems: list[Problem] = []
    checked: dict[NodePath, TypedNode] = {}
    calls: dict[NodePath, tuple[Function, CallValue]] = {}

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
        *, indexed: bool = False,
    ) -> TypedNode | None:
        if isinstance(node, UnresolvedValue):
            problem(node, ProblemCode.PARSE, "The expression text has not resolved.", at)
            return None
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
                problem(node, code, f"{_reference_label(node.reference)!r} is not in scope here.", at)
                return None
            if count := lists.get(node.reference):
                if indexed:
                    value = TypedNode(value.type.model_copy(update={"shape": (count, *value.type.shape)}), value.level)
                elif site.list_count is None:
                    problem(node, ProblemCode.TYPE_MISMATCH, "Choose an item from this list.", at)
                    return None
                elif site.list_count != count:
                    problem(node, ProblemCode.LIST_COUNT, "The lists must have the same count.", at)
                    return None
                else:
                    value = TypedNode(value.type, value.level, implicit_item=True)
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
            if isinstance(node, CallValue) and function.bind is not None:
                bound = function.bind(function, node)
                if isinstance(bound, SignatureError):
                    problem(node, bound.code, bound.reason, (*at, *bound.path))
                    return None
                function, node = bound
                calls[at[len(path) :]] = bound
                kwargs = node.kwargs
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
            if key == "index" and isinstance(source := node.args[0], NameValue) and source.reference in lists:
                count = lists[source.reference]
                item = node.args[1]
                if isinstance(item, NameValue) and item.reference == "i" and site.repeat_count is not None:
                    if site.repeat_count != count:
                        problem(node, ProblemCode.LIST_COUNT, "The list and repeat must have the same count.", at)
                        return None
                elif isinstance(item, LiteralValue) and type(item.value) is int and item.unit is None:
                    if not 0 <= item.value < count:
                        problem(item, ProblemCode.LIST_COUNT, f"The list has {count} items.", (*at, "args", 1))
                        return None
                elif not isinstance(item, NameValue) or item.reference != "i":
                    problem(item, ProblemCode.TYPE_MISMATCH, "A list item is an integer literal or i.", (*at, "args", 1))
                    return None
            positional = [
                visit(arg, (*at, "args", i), hint.dimensions, hint.base, indexed=key == "index" and i == 0)
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
    return Checked(tree, root, MappingProxyType(checked), MappingProxyType(calls))


def _reference_label(reference: str) -> str:
    """A reference is a plain name or the canonical text of a Model address."""
    return ModelAddress.from_text(reference).label() if reference.startswith("{") else reference
