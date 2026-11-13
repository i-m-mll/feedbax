"""The typed field entry point consumed by authoring surfaces."""

from __future__ import annotations

from enum import StrEnum

from feedbax.contracts.expression_values import (
    CallValue,
    DrawValue,
    Expression,
    FieldValue,
    Level,
    LiteralValue,
    NameValue,
    Problem,
    ProblemCode,
    Site,
    SweepList,
    SweepRange,
    SweepSample,
    ValueBase,
    ValueRecord,
)
from feedbax.expression_check import Checked, check
from feedbax.expression_registry import FunctionClass, FunctionInfo, FunctionRegistry
from feedbax.expression_text import ParsedField, parse


class FieldKind(StrEnum):
    VALUE = "value"
    SWEEP = "sweep"
    DRAW = "draw"
    FORMULA = "formula"
    SHAPE = "shape"
    LINK = "link"


class SiteProjection(ValueRecord):
    site: Site
    kinds: tuple[FieldKind, ...]
    functions: tuple[FunctionInfo, ...]


def project_site(site: Site, functions: FunctionRegistry) -> SiteProjection:
    """Derive the editor's menu and completion from the same declarations as check."""
    scope = {name.name for name in site.names}
    offered = tuple(
        entry
        for entry in functions.definitions()
        if entry.level <= site.finest
        and set(entry.implicit_names) - {"T"} <= scope
        and not (site.structural and entry.random)
    )
    kinds = [FieldKind.VALUE, FieldKind.SWEEP, FieldKind.FORMULA]
    if any(entry.display_class == FunctionClass.DRAW for entry in offered):
        kinds.append(FieldKind.DRAW)
    if site.finest == Level.STEP and any(
        entry.display_class == FunctionClass.SHAPE for entry in offered
    ):
        kinds.append(FieldKind.SHAPE)
    if any(item.level <= site.finest and site.expected.accepts(item.type) for item in site.names):
        kinds.append(FieldKind.LINK)
    return SiteProjection(site=site, kinds=tuple(kinds), functions=offered)


def _literal_vector(value: Expression) -> bool:
    return isinstance(value, LiteralValue) or (
        isinstance(value, CallValue)
        and value.function == "vector"
        and not value.kwargs
        and bool(value.args)
        and all(_literal_vector(child) for child in value.args)
    )


def field_kind(value: FieldValue, functions: FunctionRegistry) -> FieldKind:
    if isinstance(value, (SweepList, SweepRange, SweepSample)):
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
    elif isinstance(value, SweepRange):
        if site.expected.shape or site.expected.base == ValueBase.BOOLEAN:
            return (
                Problem(
                    code=ProblemCode.TYPE_MISMATCH, reason="A range requires a numeric scalar site."
                ),
            )
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
