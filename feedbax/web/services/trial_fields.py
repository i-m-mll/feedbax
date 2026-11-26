"""The fields of a step's Task and Objective documents, addressed by the ids of the members they lie in."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TypeGuard

from pydantic import BaseModel

from feedbax.contracts.expression_check import infer_constant_type
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    CallValue, DrawValue, Expression, Level, LiteralValue, NameValue, Problem, ProblemCode, ScopedValue, SequenceValue, Site, SweepList,
    SweepSample, UnresolvedValue, ValueBase, ValueType,
)
from feedbax.contracts.expression_site import read_field
from feedbax.contracts.expression_text import ParsedField, print_stored_expression
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.member_family import ExpandedFamilyMember, FamilyProblem, NamedIndexValue, expand_family
from feedbax.contracts.matrix_core import prepare_stored_patch
from feedbax.contracts.objective_document import (
    STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument, StoredObjectiveDocument, objective_weight_title,
)
from feedbax.contracts.project import Payload, SetProjectValue, Step
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_storage import stored_expression_site
from feedbax.contracts.studio_field import FieldKindEdit, FieldNameEdit, FieldTextEdit
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.contracts.task_document import STORED_TASK_DOCUMENT_SCHEMA_ID
from feedbax.contracts.time_units import TimeUse, TimeValue, time_steps
from feedbax.contracts.window import window_offset_expression
from feedbax.studio.authoring_sweeps import FieldSweeps, SweepTarget
from feedbax.studio.numeric_fields import NumericField, prepare_numeric_field
from feedbax.studio.selection import StoredEdit
from feedbax.web.services.project_folder import ProjectFolderInvalid

type FieldPath = tuple[str, ...]
"""A path into a document's payload that names a member of a list of identified items by its id."""

_EXPRESSIONS = (LiteralValue, NameValue, CallValue, DrawValue, SequenceValue, UnresolvedValue)


def member(value: object, part: str) -> object:
    """The child ``part`` of a document value: a field, an identified member by its id, or an item by its index."""
    if isinstance(value, tuple | list):
        found = next((item for item in value if _id(item) == part), None)
        if found is not None or not part.isdigit():
            return found
        return value[int(part)] if int(part) < len(value) else None
    if isinstance(value, BaseModel):
        return getattr(value, part) if part in type(value).model_fields else None
    if isinstance(value, dict):
        return value.get(part)
    return None


def _id(item: object) -> object:
    return item.get("id") if isinstance(item, dict) else getattr(item, "id", None)


def at(value: object, path: FieldPath) -> object:
    for part in path:
        value = member(value, part)
        if value is None:
            return None
    return value


def site_expression(document: BaseModel, path: FieldPath) -> Expression | None:
    """The expression a document holds at ``path``, if it holds one."""
    value = at(document, path)
    if isinstance(value, TimeValue):
        return window_offset_expression(value)
    return value if isinstance(value, _EXPRESSIONS) else None


def with_family_indices(site: Site, indices: tuple[ScopedValue, ...]) -> Site:
    """Put the family indices in the member's lexical scope."""
    if not indices:
        return site
    names = {index.id for index in indices}
    return site.model_copy(update={"names": (*(
        value.model_copy(update={"name": value.id}) if value.name in names else value
        for value in site.names if value.id not in names
    ), *indices)})


def family_indices(
    member: ExpandedFamilyMember, functions: FunctionRegistry, path: FieldPath,
) -> tuple[tuple[ScopedValue, ...], tuple[Problem, ...]]:
    names, problems = [], []
    for key, value in member.indices.items():
        kind = infer_constant_type(value, functions, path=(*path, "over", key))
        if isinstance(kind, tuple):
            problems.extend(kind)
        else:
            names.append(ScopedValue(id=key, name=key, type=kind, level=Level.RUN))
    return tuple(names), tuple(problems)


def family_problem(error: FamilyProblem, path: FieldPath) -> Problem:
    return Problem(code=ProblemCode.STRUCTURE, path=(*path, *error.path), reason=family_reason(str(error), error.coordinates))


def family_reason(reason: str, coordinates: Mapping[str, str]) -> str:
    if not coordinates:
        return reason
    values = ", ".join(f"{name}={value}" for name, value in coordinates.items())
    return f"{reason} (family {values})"


def is_family_rule(path: FieldPath) -> bool:
    return len(path) == 5 and path[:2] == ("objective", "terms") and path[3] == "over"


@dataclass(frozen=True)
class ObjectiveEditor:
    objective: ObjectiveDocument
    names: dict[str, tuple[ScopedValue, ...]]
    coordinates: dict[str, dict[str, str]]
    rules: dict[FieldPath, tuple[Site, Expression]]
    problems: tuple[Problem, ...]


def objective_editor(stored: StoredObjectiveDocument, functions: FunctionRegistry) -> ObjectiveEditor:
    """Prepare one member per authored term, with separate rule and expression sites."""
    terms, names, coordinates, rules, problems = [], {}, {}, {}, []
    for key, term in stored.terms.items():
        path = ("objective", "terms", key)
        try:
            member = next(expand_family(key, term, first=True, label_placeholders=False))
        except FamilyProblem as error:
            problems.append(family_problem(error, path))
            terms.append(term.lower(key).model_copy(update={"enabled": False}))
        else:
            terms.append(member.value.lower(key))
            coordinates[key] = dict(member.coordinates)
            names[key], found = family_indices(member, functions, path)
            problems.extend(found)
        types = {index.id: index.type for index in names.get(key, ())}
        for name, rule in (term.over or {}).items():
            rule_path = (*path, "over", name)
            if isinstance(rule, SequenceValue | UnresolvedValue):
                expected = types.get(name, ValueType(base=ValueBase.REAL))
                rules[rule_path] = Site(expected=expected, finest=Level.RUN, structural=True), rule
                continue
            for item, value in rule.items():
                item_path = (*rule_path, item)
                if isinstance(value, NamedIndexValue):
                    value = value.value
                    item_path = (*item_path, "value")
                actual = infer_constant_type(value, functions, path=item_path)
                if isinstance(actual, tuple):
                    problems.extend(actual)
                    actual = types.get(name, ValueType(base=ValueBase.REAL))
                rules[item_path] = Site(expected=actual, finest=Level.RUN, structural=True), value
    return ObjectiveEditor(ObjectiveDocument(id=stored.id, terms=tuple(terms)), names, coordinates, rules, tuple(problems))


def is_window_offset(role: str, path: FieldPath) -> bool:
    """Whether a path names the whole offset of an Objective's Window span."""
    return (
        role == "objective" and len(path) == 7 and path[0] == "terms"
        and path[2:4] == ("window", "spans") and path[4].isdigit()
        and path[5] in {"start", "end", "at"} and path[6] == "offset"
    )


def index_pointer(raw: object, path: FieldPath) -> str | None:
    """The JSON pointer into ``raw`` of ``path``, each identified member by its index; None when it is not there."""
    parts, value = [], raw
    for part in path:
        if isinstance(value, list):
            index = next((i for i, item in enumerate(value) if _id(item) == part), None)
            if index is None and part.isdigit() and int(part) < len(value):
                index = int(part)
            if index is None:
                return None
            parts.append(str(index))
            value = value[index]
        elif isinstance(value, dict) and part in value:
            parts.append(part)
            value = value[part]
        else:
            return None
    return pointer(tuple(parts))


def numeric_nodes(expression: Expression | None, functions: FunctionRegistry) -> Iterator[tuple[FieldPath, str]]:
    """The paths in ``expression`` of the numbers a sweep may set, with a label for each: the expression itself when it
    is a number, and each number given to a draw."""
    if _numeric(expression):
        yield (), ""
        return
    if isinstance(expression, CallValue):
        for index, arg in enumerate(expression.args):
            for path, label in numeric_nodes(arg, functions):
                yield ("args", str(index), *path), label
    elif isinstance(expression, DrawValue):
        declared = functions.resolve(expression.family)
        names = [name for _, name in getattr(declared, "parameters", ()) or ()]
        for index, arg in enumerate(expression.args):
            if _numeric(arg):
                yield ("args", str(index)), names[index] if index < len(names) else str(index + 1)


def _numeric(node: object) -> TypeGuard[LiteralValue]:
    return isinstance(node, LiteralValue) and type(node.value) in (int, float) and (
        node.unit is None or node.unit == "st" or node.unit in UNITS and UNITS[node.unit].scale == 1
    )


@dataclass(frozen=True)
class TrialFieldSweep:
    """A number in a document the step reads, as a field the step sweeps at ``role`` and ``path``."""

    role: str
    path: FieldPath
    field: NumericField
    axis: FieldSweeps
    stored: bool
    """The document stores the number, rather than leaving its field to a default."""
    write: FieldPath
    """The owning scalar or text field, including when the literal is nested in its expression."""


def trial_field_sweep(
    step: Step, payload: Payload, parsed: BaseModel, role: str, path: FieldPath, site: Site, label: str,
) -> TrialFieldSweep | None:
    """The number at ``path`` of a validated sweep source, when the step can sweep it.

    ``payload`` retains the authored key presence and ``site`` describes the enclosing expression.
    """
    node = site_expression(parsed, path)
    if not _numeric(node):
        return None
    write = next(path[:end] for end in range(1, len(path) + 1) if site_expression(parsed, path[:end]) is not None)
    time_use = trial_time_use(role, path) if node.unit in {"s", "st"} else None
    def validate(value):
        if time_use == "offset" and node.unit == "st":
            return time_steps(TimeValue(value=value, unit="st"), 1)
        return value

    field = NumericField(
        key=(role, *path), label=label, value=node.value,
        site=Site(
            expected=ValueType(base=ValueBase.REAL, dimensions=site.expected.dimensions), finest=Level.RUN,
            structural=True,
        ),
        stored=True, validate=validate, unit=node.unit, time_use=time_use,
    )
    patch = ".".join(("trial_documents", role, *path, "value"))
    target = SweepTarget(node.value, lambda _value: [], patch)
    weight = objective_weight_title(payload, path) if role == "objective" else None
    named = path[1:3] if weight is not None else _names(parsed, path) or (role,)
    return TrialFieldSweep(
        role, path, field,
        FieldSweeps(
            step, target, weight or label, named, integer=node.unit == "st" and time_use in {"length", "offset"},
            derived_label=weight is not None,
        ),
        index_pointer(payload, write) is not None, write,
    )


def trial_time_use(role: str, path: FieldPath) -> TimeUse:
    if role == "task" and path[:1] == ("epochs",) and path[2:4] in (("end", "length"), ("end", "limit")):
        return "length"
    if is_window_offset(role, path[:7]):
        return "offset"
    return "constant"


def _names(document: BaseModel, path: FieldPath) -> tuple[str, ...]:
    """The words that name the field at ``path``: each member's name where it has one, and each field's own except a
    list's."""
    names, value = [], document
    for part in path:
        parent, value = value, member(value, part)
        if isinstance(parent, dict) and isinstance(value, BaseModel):
            names.append(getattr(value, "label", None) or part)
        elif _id(value) == part:
            names.append(getattr(value, "name", None) or part)
        elif not part.isdigit() and not isinstance(value, tuple | list):
            names.append(part)
    return tuple(names)


def routes_to_sweep(
    edit: FieldTextEdit | FieldKindEdit | FieldNameEdit, swept: TrialFieldSweep, site: Site, functions: FunctionRegistry,
) -> bool:
    """The edit is the step's sweep of the field: it edits a sweep, makes one, or ends one."""
    if edit.kind != "text" or edit.value_id is not None or swept.axis.covering:
        return True
    parsed = read_field(edit.text, site, functions)
    return isinstance(parsed, ParsedField) and isinstance(parsed.value, SweepList | SequenceValue | SweepSample)


def trial_field_sweep_edit(
    edit: FieldTextEdit | FieldKindEdit | FieldNameEdit, swept: TrialFieldSweep, step_id: str, raw: dict,
    functions: FunctionRegistry, write_document: Callable[[list[SetProjectValue]], StoredEdit],
) -> StoredEdit | tuple[Problem, ...]:
    """The files that store an edit of the field: the step's sweep, and the value the document keeps; a value with no
    sweep is the document's alone.

    A sweep sets a number the document stores, so a field left to its default is given it explicitly. ``raw`` is the
    document's payload as stored.
    """
    prepared = prepare_numeric_field(swept.field, swept.axis, edit, functions)
    result = prepared.result
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        stored = StoredEdit(
            [StudioProjectFilePatch(address=AuthoredFileAddress(kind="step", id=step_id), patches=list(result))],
            reached=frozenset({step_id}),
        )
        kept = next((swept.field.literal(patch.value) for patch in reversed(prepared.keep) if patch.op != "remove"), None)
    else:
        stored = StoredEdit([])
        kept = result if isinstance(result, LiteralValue) else swept.field.literal(result)
    if kept is None and swept.stored:
        return stored
    if kept is None:
        kept = swept.field.literal(swept.field.value)
    path = index_pointer(raw, swept.write)
    parent = index_pointer(raw, swept.write[:-1])
    if parent is None:
        raise ProjectFolderInvalid("The swept field's document has changed.")
    value = print_stored_expression(kept)
    if swept.write != swept.path:
        schema_id = STORED_TASK_DOCUMENT_SCHEMA_ID if swept.role == "task" else STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
        node = OverridePatch(path=".".join(swept.path), op="replace", value=value)
        physical, expression = prepare_stored_patch(raw, node, expression_site=partial(stored_expression_site, schema_id))
        assert expression is not None
        value = physical.value
    edit = SetProjectValue(op="replace" if path else "add", path=path or parent + pointer(swept.write[-1:]), value=value)
    return write_document([edit]) + stored
