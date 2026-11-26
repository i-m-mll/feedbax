"""The typed field entry point consumed by authoring surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from pydantic import Field

from feedbax.contracts.expression_values import (
    CallValue,
    DrawValue,
    Expression,
    FieldValue,
    Level,
    LiteralValue,
    NameValue,
    NodePath,
    Problem,
    ProblemCode,
    Site,
    SweepList,
    SequenceValue,
    SweepSample,
    UnresolvedValue,
    ValueBase,
    ValueRecord,
    scope_problems,
    walk,
)
from feedbax.contracts.expression_check import Checked, _checker, check
from feedbax.contracts.expression_registry import FunctionClass, FunctionInfo, FunctionRegistry
from feedbax.contracts.expression_sequence import sequence_definition
from feedbax.contracts.expression_text import ParsedField, parse


class FieldKind(StrEnum):
    VALUE = "value"
    SWEEP = "sweep"
    DRAW = "draw"
    FORMULA = "formula"
    SHAPE = "shape"
    LINK = "link"


class ExpressionPartSite(ValueRecord):
    path: tuple[str | int, ...]
    """The expression root or an argument, addressed within the field value."""
    functions: tuple[str, ...]
    names: tuple[str, ...]


class SiteProjection(ValueRecord):
    site: Site
    kinds: tuple[FieldKind, ...]
    functions: tuple[FunctionInfo, ...]
    parts: tuple[ExpressionPartSite, ...]
    """The allowed function and name IDs at each expression path."""
    name_paths: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Display levels for scoped names that belong to a tree, indexed by opaque ID."""


def project_site(
    site: Site,
    functions: FunctionRegistry,
    value: Expression | None = None,
    *,
    name_paths: Mapping[str, tuple[str, ...]] = MappingProxyType({}),
    defer_parts: bool = False,
) -> SiteProjection:
    """Derive the editor's menu and completion from the same declarations as check."""
    offered = _offered_functions(site, functions)
    kinds = [FieldKind.VALUE, FieldKind.SWEEP, FieldKind.FORMULA]
    if any(entry.display_class == FunctionClass.DRAW for entry in offered):
        kinds.append(FieldKind.DRAW)
    if site.finest == Level.STEP and any(
        entry.display_class == FunctionClass.SHAPE for entry in offered
    ):
        kinds.append(FieldKind.SHAPE)
    if any(item.level <= site.finest and site.expected.accepts(item.type) for item in site.names):
        kinds.append(FieldKind.LINK)
    parts = ()
    if not defer_parts:
        parts = (
            (_blank_part(site, offered),) if value is None
            else _expression_parts(value, site, functions, offered)
        )
    return SiteProjection(
        site=site,
        kinds=tuple(kinds),
        functions=offered,
        parts=parts,
        name_paths={name.id: name_paths[name.id] for name in site.names if name.id in name_paths},
    )


def _offered_functions(site: Site, functions: FunctionRegistry) -> tuple[FunctionInfo, ...]:
    scope = {name.name for name in site.names}
    return tuple(
        entry
        for entry in functions.definitions()
        if entry.level <= site.finest
        and set(entry.implicit_names) - {"T"} <= scope
        and not (site.structural and entry.random)
    )


def _blank_part(site: Site, offered: tuple[FunctionInfo, ...]) -> ExpressionPartSite:
    return ExpressionPartSite(
        path=(),
        functions=tuple(entry.id for entry in offered),
        names=tuple(name.id for name in site.names if name.level <= site.finest),
    )


def project_expression_part(
    site: Site, functions: FunctionRegistry, value: Expression | None, part: NodePath,
) -> ExpressionPartSite | None:
    """Project only the opened part, with the same eligibility as an eager site."""
    offered = _offered_functions(site, functions)
    if value is None:
        return None if part else _blank_part(site, offered)
    return next(iter(_expression_parts(value, site, functions, offered, selected=part)), None)


def _replace_part(tree: Expression, path: NodePath, value: Expression) -> Expression:
    match path:
        case ():
            return value
        case ("args", int(index), *rest) if isinstance(tree, CallValue | DrawValue):
            args = list(tree.args)
            args[index] = _replace_part(args[index], tuple(rest), value)
            return tree.model_copy(update={"args": tuple(args)})
        case ("kwargs", str(name), *rest) if isinstance(tree, CallValue):
            child = _replace_part(tree.kwargs[name], tuple(rest), value)
            return tree.model_copy(update={"kwargs": tree.kwargs | {name: child}})
    raise AssertionError(f"{path} is not a path in this expression.")


def _with_function(node: Expression, info: FunctionInfo, finest: Level) -> Expression | None:
    args = node.args if isinstance(node, CallValue | DrawValue) else (node,)
    kwargs = node.kwargs if isinstance(node, CallValue) else {}
    if len(args) < info.min_args or info.max_args is not None and len(args) > info.max_args:
        return None
    if kwargs.keys() - set(info.keywords) or info.display_class == FunctionClass.SWEEP_RULE:
        return None
    if info.display_class == FunctionClass.DRAW:
        if kwargs:
            return None
        per = node.per if isinstance(node, DrawValue) else ("run" if finest == Level.RUN else "trial")
        return DrawValue(family=info.id, args=args, per=per)
    return CallValue(function=info.id, args=args, kwargs=kwargs)


def _expression_parts(
    tree: Expression, site: Site, functions: FunctionRegistry, offered: tuple[FunctionInfo, ...],
    *, selected: NodePath | None = None,
) -> tuple[ExpressionPartSite, ...]:
    """Check a replacement with its enclosing expression so sibling constraints still apply."""
    nodes = tuple(walk(tree, include_sequence_bounds=False))
    by_path = dict(nodes)
    if selected is not None and selected not in by_path:
        return ()
    invalid_scope = scope_problems(site, ())
    recheck, local, insensitive = _checker(site, functions, reusable=frozenset(id(node) for _, node in nodes))
    original = (
        check(tree, site, functions) if isinstance(tree, UnresolvedValue) else invalid_scope or recheck(tree)
    )
    names = tuple(
        (name.id, NameValue(reference=name.id), (name.type, name.level, name.list_count))
        for name in site.names if name.level <= site.finest
    )

    def fits(path: NodePath, replacement: Expression) -> bool:
        after = replacement
        if isinstance(original, Checked) and not isinstance(tree, UnresolvedValue):
            # A changed child hint can leave its parent's output unchanged. Lift
            # the edit one edge at a time, stopping when the outer context agrees.
            for size in range(len(path), 0, -2):
                if (decision := local(by_path[path[:size]], after, path[:size])) is not None:
                    return decision
                after = _replace_part(by_path[path[: size - 2]], path[size - 2 : size], after)
        else:
            after = _replace_part(tree, path, replacement)
        candidate = invalid_scope or recheck(after, collect=False)
        if not candidate:
            return True
        if isinstance(original, Checked):
            return False
        return all(problem in original and problem.path[:len(path)] != path for problem in candidate)

    def fitted_names(path: NodePath) -> tuple[str, ...]:
        equivalent = {}
        accepted = []
        group_names = isinstance(original, Checked) and insensitive(path)
        for key, replacement, signature in names:
            group = signature if group_names else key
            if group not in equivalent:
                equivalent[group] = fits(path, replacement)
            if equivalent[group]:
                accepted.append(key)
        return tuple(accepted)

    return tuple(
        ExpressionPartSite(
            path=path,
            functions=tuple(
                info.id
                for info in offered
                if (replacement := _with_function(node, info, site.finest)) is not None
                and fits(path, replacement)
            ),
            names=fitted_names(path),
        )
        for path, node in (nodes if selected is None else ((selected, by_path[selected]),))
    )


def _literal_vector(value: Expression) -> bool:
    return isinstance(value, LiteralValue | SequenceValue) or (
        isinstance(value, CallValue)
        and value.function == "vector"
        and not value.kwargs
        and bool(value.args)
        and all(_literal_vector(child) for child in value.args)
    )


def field_kind(value: FieldValue, functions: FunctionRegistry) -> FieldKind:
    if isinstance(value, (SweepList, SequenceValue, SweepSample)):
        return FieldKind.SWEEP
    if _literal_vector(value):
        return FieldKind.VALUE
    if isinstance(value, DrawValue):
        return FieldKind.DRAW
    if isinstance(value, NameValue):
        return FieldKind.LINK
    if isinstance(value, CallValue) and (function := functions.resolve(value.function)):
        if function.info.display_class == FunctionClass.SHAPE:
            return FieldKind.SHAPE
    return FieldKind.FORMULA


def read_field(
    text: str, site: Site, functions: FunctionRegistry
) -> ParsedField | tuple[Problem, ...]:
    """Parse and check a field without materializing its sweep."""
    parsed = parse(text, site, functions)
    if not isinstance(parsed, ParsedField):
        return parsed
    value = parsed.value
    checks = [((), value, site)]
    if isinstance(value, SweepList):
        checks = [(("members", i), item, site) for i, item in enumerate(value.members)]
    elif isinstance(value, SequenceValue):
        if site.expected.shape or site.expected.base == ValueBase.BOOLEAN:
            return (
                Problem(
                    code=ProblemCode.TYPE_MISMATCH, reason="A range requires a numeric scalar site."
                ),
            )
        try:
            sequence_definition(value)
        except ValueError as error:
            return (Problem(code=ProblemCode.STRUCTURE, reason=str(error), span=parsed.spans.get(())),)
        run_site = site.model_copy(update={"finest": Level.RUN})
        checks = [(("first",), value.first, run_site), (("last",), value.last, run_site)]
    elif isinstance(value, SweepSample):
        run_site = site.model_copy(update={"finest": Level.RUN, "structural": False})
        checks = [(("draw",), value.draw.model_copy(update={"per": "run"}), run_site)]
    problems = []
    for path, tree, check_site in checks:
        result = check(
            tree,
            check_site,
            functions,
            path=path,
            spans={
                node_path[len(path) :]: span
                for node_path, span in parsed.spans.items()
                if node_path[: len(path)] == path
            },
        )
        if not isinstance(result, Checked):
            problems.extend(result)
    return tuple(problems) if problems else parsed
