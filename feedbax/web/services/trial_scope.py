"""Store an edit of a step's definition by the scope of the runs it is made for."""

from collections.abc import Callable, Collection
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

from pydantic import TypeAdapter, ValidationError

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_override_patches
from feedbax.contracts.model_edit import (
    EvaluateAttachmentAxis, EvaluateAttachmentRows, EvaluateAttachmentStep, EvaluateAttachmentVariant,
)
from feedbax.contracts.project import (
    DocumentEdit, InlineBase, Project, RemoveProjectValue, SetProjectValue, apply_document_edits, document_edit_ops,
)
from feedbax.contracts.project_inputs import SweepExceptionTerms
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.project_resolution import StepTemplate, compose_step, document_differences, document_write_target
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import AxisCondition, NamedAxis, SetValue, RemoveValue, _compose_writes
from feedbax.contracts.run_matrix import MatrixCompositionDelta
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.project_members import generated_name
from feedbax.studio.selection import TrainSelection, merged_files
from feedbax.web.services.project_folder import ProjectFolderInvalid


_SWEEP_WRITE = TypeAdapter(SetValue | RemoveValue)


@dataclass(frozen=True)
class SweepValuesScope:
    """Runs selected as values of the step's sweeps, other than one named value of one sweep."""

    axes: dict[str, tuple[str, ...]]


type RunScope = (
    EvaluateAttachmentStep | EvaluateAttachmentAxis | SweepValuesScope | EvaluateAttachmentRows | EvaluateAttachmentVariant
)


def selection_scope(project, plan, key, rows=(), axes=None) -> RunScope:
    """Where an edit made for the selected runs is stored.

    No rows is the step, or the variant outside the primary one. Rows that are exactly the runs of a set of values per
    sweep are those values: one named value's patches, or an exception over the values. Other rows are an exception over
    those rows.
    """
    axes = axes or {}
    if not rows:
        if axes:
            raise ProjectFolderInvalid("Sweep values were given without their runs.")
        if key.variant != plan.variants[0]:
            return EvaluateAttachmentVariant(kind="variant", variant=key.variant)
        return EvaluateAttachmentStep(kind="step")
    if key.variant != plan.variants[0]:
        raise ProjectFolderInvalid("A selection of runs is stored in the primary variant only.")
    available = {row.key.row: row for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, key.step)}
    if not set(rows) <= available.keys() or len(set(rows)) != len(rows):
        raise ProjectFolderInvalid("The selected rows no longer belong to this step.")
    covered = {
        row_id for row_id, row in available.items()
        if all(axis in row.coordinates and row.coordinates[axis].value_id in values for axis, values in axes.items())
    }
    if not axes or set(rows) != covered:
        return EvaluateAttachmentRows(kind="rows", rows=tuple(rows))
    if len(axes) == 1:
        (axis, values), = axes.items()
        sweep = project.steps[key.step].sweep
        named = sweep.axes.get(axis) if sweep else None
        if len(values) == 1 and isinstance(named, NamedAxis) and named.values.get(values[0]) is not None:
            return EvaluateAttachmentAxis(kind="axis_value", axis=axis, value=values[0])
    return SweepValuesScope({axis: tuple(values) for axis, values in axes.items()})


def file_value(address, raw, parts, value):
    parent = raw
    for part in parts[:-1]:
        parent = parent[part]
    leaf = parts[-1]
    return StudioProjectFilePatch(address=address, patches=[
        SetProjectValue(op="replace" if leaf in parent else "add", path=pointer(parts), value=value),
    ])


def selected_runs(plan, key, scope, *, lit_steps=()):
    rows = [row for row in plan.rows if row.key.step == key.step]
    if isinstance(scope, EvaluateAttachmentVariant):
        selected = {row.key for row in rows if row.key.variant == scope.variant}
    elif isinstance(scope, EvaluateAttachmentAxis):
        selected = {
            row.key for row in rows
            if (coordinate := row.coordinates.get(scope.axis)) is not None and coordinate.value_id == scope.value
        }
    elif isinstance(scope, EvaluateAttachmentRows):
        selected = {row.key for row in rows if row.key.row in scope.rows}
        if {item.row for item in selected} != set(scope.rows):
            raise ProjectFolderInvalid("The selection contains an unknown run row.")
    else:
        selected = {
            row.key for row in plan.rows
            if row.key.step in (lit_steps or (key.step,))
        }
    if key not in selected or any(item.variant != key.variant for item in selected):
        raise ProjectFolderInvalid("The edit must address runs in the selected variant only.")
    return selected


def _store_patches(previous, writes):
    authored = {path: _SWEEP_WRITE.validate_python(value) for path, value in previous.items() if value is not None}
    changes = {}
    for patch in writes:
        changes[patch.path] = (
            RemoveValue(op="remove") if patch.op == "remove" else SetValue(op=patch.op, value=patch.value)
        )
    composed = _compose_writes(authored, changes)
    return {path: value for path, value in previous.items() if value is None and path not in changes} | {
        path: write.model_dump(mode="json", exclude_unset=True) for path, write in composed.items()
    }


def variant_file(project, variant, writes):
    row = project.sweep.rows.get(variant) if project.sweep else None
    if row is None:
        raise ProjectFolderInvalid("The variant needs an authored row.")
    raw = row.model_dump(mode="json", by_alias=True, exclude_unset=True)
    return file_value(AuthoredFileAddress(kind="variant", id=variant), raw, ("patches",),
                      _store_patches(raw.get("patches", {}), writes))


def sweep_value_file(address, raw, axis, value, writes):
    parts = ("sweep", "axes", axis, "values", value)
    authored = raw["sweep"]["axes"][axis]["values"][value]
    return file_value(address, raw, (*parts, "patches"), _store_patches(authored.get("patches", {}), writes))


def row_exception_file(address, raw, rows, writes):
    return _exception_file(address, raw, {"rows": sorted(set(rows))}, writes)


def values_exception_file(address, raw, axes, writes):
    return _exception_file(address, raw, {"axes": {axis: sorted(set(values)) for axis, values in sorted(axes.items())}}, writes)


def _canonical_where(where) -> bytes | None:
    try:
        terms = SweepExceptionTerms.model_validate(where)
    except ValidationError:
        return None
    selector = terms.model_dump(mode="json", exclude_defaults=True)
    if "rows" in selector:
        selector["rows"] = sorted(set(selector["rows"]))
    if "axes" in selector:
        selector["axes"] = {axis: sorted(set(values)) for axis, values in selector["axes"].items()}
    return canonical_json_v2_bytes(selector)


def _exception_file(address, raw, where, writes):
    """Add ``writes`` to the exception whose selector is ``where``, or to a new one named by :func:`generated_name`."""
    sweep = raw.get("sweep") or {}
    exceptions = sweep.get("exceptions", {})
    selector = _canonical_where(where)
    name = next(
        (name for name, exception in exceptions.items()
         if exception is not None and _canonical_where(exception.get("where")) == selector),
        None,
    )
    if name is None:
        name, prior = generated_name("exception", exceptions.__contains__), {"where": where}
    else:
        prior = exceptions[name]
        if prior.get("exclude"):
            raise ProjectFolderInvalid("The selected exception has an incompatible definition.")
    updated = prior | {"patches": _store_patches(prior.get("patches", {}), writes)}
    return file_value(address, raw, ("sweep",), sweep | {"exceptions": exceptions | {name: updated}})


type DocumentUsers = Callable[[str], Collection[str]]
"""The steps of the primary variant whose runs read a document."""


def _overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _covers(outer: str, inner: str) -> bool:
    return inner == outer or inner.startswith(outer + "/")


def _step_difference(project: Project, step: str, document: str, local: list[DocumentEdit]) -> StudioProjectFilePatch:
    authored = project.steps[step]
    differences = authored.model_dump(mode="json")["document_overrides"]
    if not local:
        differences.pop(document, None)
    else:
        differences[document] = [edit.model_dump(mode="json") for edit in local]
    return StudioProjectFilePatch(
        address=AuthoredFileAddress(kind="step", id=step),
        patches=[SetProjectValue(
            op="replace" if "document_overrides" in authored.model_fields_set else "add",
            path="/document_overrides", value=differences,
        )],
    )


def document_files(
    project: Project, document: str, edits: list[DocumentEdit], lit: Collection[str], users: Collection[str],
) -> list[StudioProjectFilePatch]:
    """Store ``edits`` of ``document``'s payload for the steps ``lit``: in the document when they are all the steps that
    read it, otherwise as a difference from it on each of them.

    Stored in the document, an edit takes the place of the differences at the same path that the lit steps apply, on
    the steps that store them: a step's own, or those it inherits from its base chain.
    """
    entry = project.documents[document]
    shared = set(lit) == set(users)
    files = []
    if shared:
        raw = deepcopy(entry.payload)
        apply_document_edits(raw, edits)
        files.append(file_value(AuthoredFileAddress(kind="document", id=document),
                                {"payload": entry.payload}, ("payload",), raw))
        storing = {
            difference.step for step in lit for difference in document_differences(project, step).get(document, ())
        }
    else:
        storing = set(lit)
    for step in sorted(storing | set(lit)):
        previous = document_edit_ops(project.steps[step].document_overrides.get(document, {}))
        if shared:
            local = []
            for edit in previous:
                affected = [requested for requested in edits if _overlap(edit.path, requested.path)]
                if not affected:
                    local.append(edit)
                elif any(requested.path != edit.path or edit.op != "replace" for requested in affected):
                    raise ProjectFolderInvalid(f"{step} has an overlapping document difference.")
        else:
            local = [*(edit for edit in previous if not any(_covers(requested.path, edit.path) for requested in edits)),
                     *edits]
        if local != previous:
            files.append(_step_difference(project, step, document, local))
    return files


def _document_edit(patch: OverridePatch, document_path: str, payload) -> DocumentEdit:
    parts = tuple(document_path.split("."))
    if patch.op == "remove":
        return RemoveProjectValue(op="remove", path=pointer(parts))
    parent = payload
    for part in parts[:-1]:
        parent = parent.get(part) if isinstance(parent, dict) else None
    exists = isinstance(parent, dict) and parts[-1] in parent
    return SetProjectValue(op="replace" if exists else "add", path=pointer(parts), value=patch.value)


def _authored_payload(project: Project, step_id: str):
    authored = project.steps[step_id]
    if isinstance(authored.base, InlineBase):
        return authored.base.inline
    template = compose_step(project, step_id)
    return template.authored_payload() if isinstance(template, StepTemplate) else None


def scoped_file(
    project, step_id, scope: RunScope, writes, *, lit: Collection[str] = (), users: DocumentUsers | None = None,
) -> list[StudioProjectFilePatch]:
    """The files that store ``writes``, paths in the step's definition, for the runs ``scope`` selects.

    A write whose path passes through a document reference changes that document. For the whole step it is stored by
    :func:`document_files` for the steps ``lit`` (the step alone when none are named) among the document's ``users``.
    For a sweep value or exception it is applied to the selected runs' copy of the document, and for a variant it is the
    variant's write of the document.
    """
    authored = project.steps[step_id]
    raw = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
    address = AuthoredFileAddress(kind="step", id=step_id)
    payload = _authored_payload(project, step_id)
    through: dict[str, list[tuple[OverridePatch, str]]] = {}
    direct = []
    for patch in writes:
        target = document_write_target(payload, patch.path)
        if target is None:
            direct.append(patch)
        else:
            through.setdefault(target[0], []).append((patch, target[1]))
    if isinstance(scope, EvaluateAttachmentVariant):
        writes = [
            OverridePatch(path=f"steps.{step_id}.base.inline.{patch.path}", op=patch.op, **(
                {"value": patch.value} if patch.op != "remove" else {}
            ))
            for patch in direct
        ] + [
            patch.model_copy(update={"path": f"documents.{document}.payload.{path}"})
            for document, patches in through.items() for patch, path in patches
        ]
        return [variant_file(project, scope.variant, writes)]
    if isinstance(scope, EvaluateAttachmentAxis | SweepValuesScope | EvaluateAttachmentRows):
        return [_selection_file(address, raw, authored, scope, writes)]
    files = [_step_file(address, raw, authored, direct)] if direct else []
    if through:
        lit = set(lit) or {step_id}
        if users is None:
            raise ProjectFolderInvalid("This edit changes a document and needs the steps that read it.")
        for document, patches in through.items():
            readers = set(users(document))
            if step_id not in lit or not lit <= readers:
                raise ProjectFolderInvalid("The lit steps no longer match this document.")
            payload = project.documents[document].payload
            if lit != readers:
                payload = deepcopy(payload)
                apply_document_edits(payload, [
                    difference.edit for difference in document_differences(project, step_id).get(document, ())
                ])
            edits = [_document_edit(patch, path, payload) for patch, path in patches]
            files.extend(document_files(project, document, edits, lit, readers))
    return merged_files(files)


def _selection_file(address, raw, authored, scope, writes) -> StudioProjectFilePatch:
    if isinstance(scope, EvaluateAttachmentAxis):
        axis = authored.sweep.axes.get(scope.axis) if authored.sweep else None
        if not isinstance(axis, NamedAxis) or axis.values.get(scope.value) is None:
            raise ProjectFolderInvalid("The group needs a named sweep value authored on this step.")
        return sweep_value_file(address, raw, scope.axis, scope.value, writes)
    if isinstance(scope, SweepValuesScope):
        return values_exception_file(address, raw, scope.axes, writes)
    return row_exception_file(address, raw, scope.rows, writes)


def _step_file(address, raw, authored, writes) -> StudioProjectFilePatch:
    if isinstance(authored.base, InlineBase) and not authored.deltas:
        updated = apply_override_patches(authored.base.inline, writes)
        return file_value(address, raw, ("base", "inline"), updated)
    # Compose a local difference without modifying the inherited definition.
    layer = MatrixCompositionDelta(
        layer_id="trial-authoring", patches=writes, acknowledges_ancestor_paths=[patch.path for patch in writes],
    )
    return file_value(address, raw, ("deltas",), [*raw.get("deltas", []), layer.model_dump(mode="json", exclude_unset=True)])


def train_selection(
    folders, project_id, revision, project, step_id, run, rows=(), axes=None, lit_steps=(),
) -> TrainSelection:
    """The runs a Train edit names, read from the plan of ``revision``, and the write that stores an edit for them.

    Without rows the selection is every run of the step and of the other ``lit_steps`` in the named run's variant.
    ``lit_steps`` are the steps an edit of a document the step reads reaches; none is the step alone.
    """
    def users(document: str) -> frozenset[str]:
        plan = folders.read_plan(project_id, revision=revision).plan
        return plan.document_users(plan.variants[0], document)

    store = partial(scoped_file, project, step_id, lit=tuple(lit_steps), users=users)
    if run is None:
        if rows or axes:
            raise ProjectFolderInvalid("Rows were given without the run they belong to.")
        return TrainSelection((), None, None, False, partial(store, EvaluateAttachmentStep(kind="step")))
    key = RunKey(run.variant, run.step, run.row)
    if key.step != step_id:
        raise ProjectFolderInvalid(f"Run {key.row!r} belongs to step {key.step!r}, not {step_id!r}.")
    plan = folders.read_plan(project_id, revision=revision, source_runs=(key,)).plan
    scope = selection_scope(project, plan, key, tuple(rows), axes)
    lit = {step_id, *lit_steps}
    shown = tuple(RunKey(key.variant, step_id, row) for row in rows) or tuple(sorted(
        (row.key for row in plan.rows if row.key.variant == key.variant and row.key.step in lit),
        key=lambda item: item.step != step_id,
    ))
    if set(shown) != {key}:
        plan = folders.read_plan(project_id, revision=revision, source_runs=tuple(dict.fromkeys((key, *shown)))).plan
    definitions = plan.selected_definitions
    payloads = tuple(definitions[item].payload for item in shown if item in definitions)
    scoped = not isinstance(scope, EvaluateAttachmentStep)
    named = definitions.get(key)
    values = (
        AxisCondition(axes={scope.axis: [scope.value]}) if isinstance(scope, EvaluateAttachmentAxis)
        else AxisCondition(axes={axis: sorted(set(ids)) for axis, ids in sorted(scope.axes.items())})
        if isinstance(scope, SweepValuesScope) else None
    )
    return TrainSelection(
        payloads, next((row for row in plan.rows if row.key == key), None),
        named.payload if isinstance(scope, SweepValuesScope | EvaluateAttachmentAxis | EvaluateAttachmentRows)
        and named is not None else None, scoped,
        partial(store, scope), values,
    )
