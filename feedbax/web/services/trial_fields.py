"""The fields of a step's Task and Objective documents, addressed by the ids of the members they lie in."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeGuard

from pydantic import BaseModel

from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    CallValue, DrawValue, Expression, Level, LiteralValue, NameValue, Problem, Site, SweepList, SweepRange,
    SweepSample, UnresolvedValue, ValueBase, ValueType,
)
from feedbax.contracts.expression_site import read_field
from feedbax.contracts.expression_text import ParsedField
from feedbax.contracts.project import BasedDocument, Project, RootDocument, SetProjectValue, json_pointer_parts
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.studio_field import FieldKindEdit, FieldNameEdit, FieldTextEdit
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.contracts.time_units import TimeUse, TimeValue, time_steps
from feedbax.contracts.window import WindowPosition, store_window_offset, window_offset_expression
from feedbax.studio.authoring_sweeps import FieldSweeps, SweepTarget
from feedbax.studio.numeric_fields import NumericField, prepare_numeric_field
from feedbax.studio.selection import StoredEdit
from feedbax.web.services.project_folder import ProjectFolderInvalid

type FieldPath = tuple[str, ...]
"""A path into a document's payload that names a member of a list of identified items by its id."""

_EXPRESSIONS = (LiteralValue, NameValue, CallValue, DrawValue, UnresolvedValue)


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


def is_window_offset(role: str, path: FieldPath) -> bool:
    """Whether a path names the whole offset of an Objective's Window span."""
    return (
        role == "objective" and len(path) == 7 and path[0] == "terms"
        and path[2:4] == ("window", "spans") and path[4].isdigit()
        and path[5] in {"start", "end", "at"} and path[6] == "offset"
    )


def held_expression_file(
    project: Project, origin: ValueSource, held: UnresolvedValue,
) -> tuple[AuthoredFileAddress, str] | None:
    """Locate the whole held value in the file that contributed its text."""
    parts = json_pointer_parts(origin.path)
    if parts[-1:] != ("text",):
        return None
    parts = parts[:-1]
    if origin.kind == "document":
        document = project.documents.get(origin.name)
        if isinstance(document, RootDocument):
            location = ("payload", *parts)
        elif isinstance(document, BasedDocument):
            ancestor = next((
                (path, parent) for path in document.differences
                if parts[:len(parent := json_pointer_parts(path))] == parent
            ), None)
            if ancestor is None:
                return None
            path, parent = ancestor
            location = ("differences", path, "value", *parts[len(parent):])
        else:
            return None
        address = AuthoredFileAddress(kind="document", id=origin.name)
        raw = document.model_dump(mode="json", by_alias=True, exclude_unset=True)
    elif origin.kind == "variant":
        variant = project.sweep.rows.get(origin.name) if project.sweep else None
        if variant is None:
            return None
        patch = next((
            (path, parent) for path in variant.patches
            if parts[:len(parent := tuple(path.split(".")))] == parent
        ), None)
        if patch is None:
            return None
        path, parent = patch
        location = ("patches", path, "value", *parts[len(parent):])
        address = AuthoredFileAddress(kind="variant", id=origin.name)
        raw = variant.model_dump(mode="json", by_alias=True, exclude_unset=True)
    elif origin.step in project.steps and origin.kind != "upstream":
        location = parts
        address = AuthoredFileAddress(kind="step", id=origin.step)
        raw = project.steps[origin.step].model_dump(mode="json", by_alias=True, exclude_unset=True)
    else:
        return None
    current = raw
    for part in location:
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return (address, pointer(location)) if current == held.model_dump(mode="json") else None


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
    """Where the number's literal is written in the document when it is stored."""


def trial_field_sweep(
    project: Project, step_id: str, document: str | None, role: str, path: FieldPath, site: Site, label: str,
    parse: Callable[[dict], BaseModel],
) -> TrialFieldSweep | None:
    """The number at ``path`` of ``document``, read by the step in ``role``, when it is one the step can sweep.

    ``site`` is the site of the expression the number lies in; ``parse`` reads the document's payload.
    """
    entry = compose_document(project.documents.get, document) if document is not None else None
    if entry is None or isinstance(entry, ProjectDiagnostic) or not isinstance(entry.payload, dict):
        return None
    parsed = parse(entry.payload)
    node = site_expression(parsed, path)
    if node is None and is_window_offset(role, path) and isinstance(at(parsed, path[:-1]), WindowPosition):
        node = window_offset_expression(None)
    if not _numeric(node):
        return None
    time_use = trial_time_use(role, path) if node.unit in {"s", "st"} else None
    step = project.steps[step_id]
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
    named = _names(parsed, path) or (role,)
    return TrialFieldSweep(
        role, path, field,
        FieldSweeps(step, target, label, named, integer=node.unit == "st" and time_use in {"length", "offset"}),
        at(entry.payload, path) is not None, path,
    )


def trial_time_use(role: str, path: FieldPath) -> TimeUse:
    if role == "task" and path[:1] == ("epochs",) and path[2:4] == ("end", "length"):
        return "length"
    if is_window_offset(role, path[:7]):
        return "offset"
    return "constant"


def _names(document: BaseModel, path: FieldPath) -> tuple[str, ...]:
    """The words that name the field at ``path``: each member's name where it has one, and each field's own except a
    list's."""
    names, value = [], document
    for part in path:
        value = member(value, part)
        if _id(value) == part:
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
    return isinstance(parsed, ParsedField) and isinstance(parsed.value, SweepList | SweepRange | SweepSample)


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
        kept = [patch.value for patch in prepared.keep if patch.op != "remove"]
    else:
        stored = StoredEdit([])
        kept = [result.value if isinstance(result, LiteralValue) else result]
    edits = []
    if not swept.stored:
        parent = index_pointer(raw, swept.write[:-1])
        if parent is None:
            raise ProjectFolderInvalid("The swept field's document has changed.")
        literal = swept.field.literal(swept.field.value)
        value = store_window_offset(literal) if is_window_offset(swept.role, swept.write) else literal
        path = index_pointer(raw, swept.write)
        edits.append(SetProjectValue(
            op="replace" if path else "add", path=path or parent + pointer(swept.write[-1:]),
            value=value.model_dump(mode="json", exclude_none=True),
        ))
    if kept:
        written = index_pointer(raw, swept.write) or edits[0].path
        edits.append(SetProjectValue(op="replace", path=written + "/value", value=kept[-1]))
    return write_document(edits) + stored if edits else stored
