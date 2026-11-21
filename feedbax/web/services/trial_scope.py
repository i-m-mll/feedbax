"""Store an edit of a step's definition by the scope of the runs it is made for."""

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any

from pydantic import TypeAdapter, ValidationError

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_override_patches
from feedbax.contracts.model_edit import (
    EvaluateAttachmentAxis, EvaluateAttachmentRows, EvaluateAttachmentStep, EvaluateAttachmentVariant,
)
from feedbax.contracts.project import (
    BasedDocument, Differences, Document, DocumentBase, DocumentEdit, DocumentReference, InlineBase, Payload, Project,
    DifferenceNotApplied, RemoveDocumentValue, RemoveProjectValue, SetDocumentValue, SetProjectValue, StepBase,
    apply_differences,
    apply_document_edit, based_on, diff,
    document_edit_value, json_pointer_parts, same_json,
)
from feedbax.contracts.project_inputs import SweepExceptionTerms
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.project_resolution import (
    ProjectDiagnostic, StepTemplate, compose_document, compose_step, document_write_target,
)
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import (
    AxisCondition, NamedAxis, SetAxis, SetValue, RemoveValue, _compose_writes,
)
from feedbax.contracts.run_matrix import MatrixCompositionDelta
from feedbax.contracts.studio_project import AuthoredFileAddress, CreateNamedDocument, StudioProjectFilePatch
from feedbax.project_members import generated_name
from feedbax.project_folder import _free_member_name, document_file
from feedbax.studio.selection import StoredEdit, TrainSelection, merged_files
from feedbax.studio.step_differences import normalized_patches
from feedbax.web.services.project_folder import ProjectFolderInvalid, preview_authored_files


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
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    leaf = parts[-1]
    return StudioProjectFilePatch(address=address, patches=[
        SetProjectValue(op="replace" if leaf in parent else "add", path=pointer(parts), value=value),
    ])


def selected_runs(plan, key, scope, *, chosen_steps=()):
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
            if row.key.variant == key.variant and row.key.step in (chosen_steps or (key.step,))
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


def base_descendants(project: Project, steps: Collection[str]) -> frozenset[str]:
    """The steps based, through ``base: {step}`` at any depth, on one of ``steps``; they read everything stored there."""
    found = set(steps)
    while added := {
        name for name, step in project.steps.items()
        if name not in found and isinstance(step.base, StepBase) and step.base.step in found
    }:
        found |= added
    return frozenset(found - set(steps))


def reached_steps(project: Project, chosen: Collection[str]) -> frozenset[str]:
    """The steps an edit for the ``chosen`` steps reaches: those and the steps based on them."""
    return frozenset(chosen) | base_descendants(project, chosen)


def refuse_unreached(plan, candidate, reach: Collection[RunKey]) -> None:
    """Refuse a candidate that changes the request of a run outside ``reach``, the runs of the reached steps.

    A run fed by a reached run's product may change its request with it, provided its own definition does not change.
    """
    previous = {run.key: run for run in plan.runs}
    current = {run.key: run for run in candidate.runs}
    changed = {
        key for key in previous.keys() | current.keys()
        if getattr(previous.get(key), "request_id", None) != getattr(current.get(key), "request_id", None)
    }
    affected = changed & set(reach)
    for graph in (plan, candidate):
        for run in graph.runs:
            if run.key in changed and any(binding.producer in affected for binding in run.inputs.values()):
                other = previous.get(run.key) if graph is candidate else current.get(run.key)
                if other is not None:
                    before = decode_request(plan.requests[previous[run.key].request_id], plan.documents.__getitem__)
                    after = decode_request(candidate.requests[current[run.key].request_id], candidate.documents.__getitem__)
                    if before.spec != after.spec:
                        continue
                affected.add(run.key)
    if changed - affected:
        raise ProjectFolderInvalid("The proposed edit reaches runs outside the selected scope.")


def reached_runs(
    plan, variants: Collection[str], selected: Collection[RunKey], steps: Collection[str],
) -> frozenset[RunKey]:
    """The runs an edit for ``selected`` may change: those and every run of the reached ``steps`` in ``variants``."""
    return frozenset(selected) | {row.key for row in plan.rows if row.key.variant in variants and row.key.step in steps}


type DocumentUsers = Callable[[str], Collection[str]]
"""The steps of the primary variant whose runs read a document."""


def preview_stored(project: Project, stored: StoredEdit) -> Project:
    """``project`` with the documents ``stored`` creates and its patches."""
    if stored.created:
        project = project.model_copy(update={
            "documents": project.documents | {created.id: created.document for created in stored.created},
        })
    return preview_authored_files(project, stored.files)


def _composed(project: Project, name: str) -> Payload:
    entry = compose_document(project.documents.get, name)
    if isinstance(entry, ProjectDiagnostic):
        raise ProjectFolderInvalid(entry.detail)
    if not isinstance(entry.payload, dict):
        raise ProjectFolderInvalid(f"{name} is not an object.")
    return entry.payload


def _edited(content: Payload, edits: Sequence[DocumentEdit], name: str) -> Payload:
    """``content`` with ``edits``: a value set or removed at each path, whatever the document held there."""
    result = deepcopy(content)
    for edit in edits:
        parts = json_pointer_parts(edit.path)
        parent: Any = result
        for part in parts[:-1]:
            parent = parent.get(part) if isinstance(parent, dict) else None
        if isinstance(parent, dict):
            if isinstance(edit, RemoveProjectValue) and parts[-1] not in parent:
                continue
            if isinstance(edit, SetProjectValue):
                edit = edit.model_copy(update={"op": "replace" if parts[-1] in parent else "add"})
        try:
            apply_document_edit(result, edit, document_edit_value(edit))
        except ValueError as error:
            raise ProjectFolderInvalid(f"The edit does not apply to {name}: {error}") from error
    return result


def lit_steps(documents: Mapping[str, Document], readers: DocumentUsers, name: str) -> frozenset[str]:
    """The steps an edit of ``name`` would change: its readers and those of the documents based on it."""
    lit: set[str] = set()
    pending, seen = [name], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        lit.update(readers(current))
        pending.extend(based_on(documents, current))
    return frozenset(lit)


@dataclass(frozen=True)
class _DocumentStores:
    files: list[StudioProjectFilePatch]
    created: list[CreateNamedDocument]
    pointing: dict[str, dict[str, str]]
    """Per chosen step, the documents it reads that it reads a new document in place of."""
    written: dict[str, tuple[Payload, Payload]]
    """Each document that holds the edit in place, with its composed content before and after."""


def _document_stores(
    project: Project, shown: str, edits: Sequence[DocumentEdit], chosen: frozenset[str] | None, readers: DocumentUsers,
    taken: set[str],
) -> _DocumentStores:
    """Where ``edits`` of ``shown`` are stored for the ``chosen`` steps, every step when None.

    Each document of ``shown``'s family, from ``shown`` down, holds the edit when every step whose content that would
    change is chosen, and is otherwise left as it was, with a new document based on it holding the edit for its chosen
    readers.
    """
    documents = project.documents
    if chosen is not None and (not chosen or not chosen <= lit_steps(documents, readers, shown)):
        raise ProjectFolderInvalid(f"A chosen step does not read {shown} or a document based on it.")
    stores = _DocumentStores([], [], {}, {})
    before: dict[str, Payload] = {}
    after: dict[str, Payload] = {}

    def old(name: str) -> Payload:
        if name not in before:
            before[name] = _composed(project, name)
        return before[name]

    def changed(name: str, content: Payload | None, seen: tuple[str, ...]) -> frozenset[str]:
        """The steps whose content changes when ``name`` composes to ``content``, its children keeping their
        differences; None is content their differences no longer apply to."""
        if name in seen:
            return frozenset()
        steps = set(readers(name)) if content is None or not same_json(content, old(name)) else set()
        for child in based_on(documents, name):
            try:
                inherited = None if content is None else apply_differences(content, documents[child].differences)
            except DifferenceNotApplied:
                inherited = None
            steps |= changed(child, inherited, (*seen, name))
        return frozenset(steps)

    def visit(name: str, seen: tuple[str, ...]) -> None:
        document = documents[name]
        inherits = isinstance(document, BasedDocument) and document.base.document in after
        if name in seen or chosen is not None and not inherits and not lit_steps(documents, readers, name) & chosen:
            return
        content = _edited(old(name), edits, name)
        if chosen is None or changed(name, content, ()) <= chosen:
            if not same_json(content, old(name)):
                after[name] = content
                stores.written[name] = (old(name), content)
            base = None
            if isinstance(document, BasedDocument):
                base = after.get(document.base.document) or old(document.base.document)
            file = document_file(name, document, content, base)
            if file is not None:
                stores.files.append(file)
        elif direct := frozenset(readers(name)) & chosen:
            variant = _document_name(project, name, taken)
            taken.add(variant)
            stores.created.append(CreateNamedDocument(kind="document", id=variant, document=BasedDocument(
                schema=document.schema_id, base=DocumentBase(document=name), differences=diff(old(name), content),
            )))
            for step in direct:
                stores.pointing.setdefault(step, {})[name] = variant
        for child in based_on(documents, name):
            visit(child, (*seen, name))

    visit(shown, ())
    return stores


def _document_name(project: Project, name: str, taken: Collection[str]) -> str:
    data = project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    data["documents"] |= {other: {} for other in taken if other not in data["documents"]}
    return _free_member_name(data, "documents", name)


def _references(payload: Any, at: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], DocumentReference]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("document"), str):
            try:
                yield at, DocumentReference.model_validate(payload)
                return
            except ValidationError:
                pass
        for key, value in payload.items():
            yield from _references(value, (*at, key))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _references(value, (*at, str(index)))


def _step_references(project: Project, step: str) -> dict[tuple[str, ...], DocumentReference]:
    payload = authored_payload(project, step)
    return dict(_references(payload)) if payload is not None else {}


def _step_chain(project: Project, step: str) -> int:
    depth, seen = 0, {step}
    while isinstance(base := project.steps[step].base, StepBase) and base.step in project.steps and base.step not in seen:
        step = base.step
        seen.add(step)
        depth += 1
    return depth


def _slot_write(payload: Any, slot: str, document: str) -> OverridePatch:
    """The write that gives a step whose payload is ``payload`` a reference of its own to ``document`` in ``slot``."""
    reference = {"document": document}
    if isinstance(payload, dict) and isinstance(payload.get("trial_documents"), dict):
        return OverridePatch(path=f"trial_documents.{slot}", op="add", value=reference)
    return OverridePatch(path="trial_documents", op="add", value={slot: reference})


_VARIANT_LAYER = "variant"


class _Store:
    """Where an edit keeps its changes: the project's files in the default variant, ``variant`` None, and otherwise the
    variant's row. ``view`` is the project the edit reads, the variant's with its row applied, kept up to date with
    what has been stored."""

    def __init__(self, project: Project, variant: str | None, view: Project):
        self.project, self.variant, self.view = project, variant, view
        self.files: list[StudioProjectFilePatch] = []
        self.created: list[CreateNamedDocument] = []
        self.row: list[OverridePatch] = []
        self.layers: dict[str, list[dict[str, Any]]] = {}
        """Per step outside its file's own definition, the layer the row appends to it; none drops the row's layer."""

    def documents(self, stores: _DocumentStores) -> None:
        """Store the documents ``stores`` writes and creates: in their files, or as the row's
        ``documents.<name>.payload.<path>`` writes."""
        self.created.extend(stores.created)
        if self.variant is None:
            self.files.extend(stores.files)
        else:
            self.row.extend(
                OverridePatch(path=".".join(("documents", name, "payload", *json_pointer_parts(path))), op="remove")
                if isinstance(record, RemoveDocumentValue) else OverridePatch(
                    path=".".join(("documents", name, "payload", *json_pointer_parts(path))),
                    op="replace" if _holds(before, json_pointer_parts(path)) else "add", value=record.value,
                )
                for name, (before, after) in stores.written.items() for path, record in diff(before, after).items()
            )
        self.view = preview_stored(self.view, StoredEdit(
            stores.files if self.variant is None else [], tuple(stores.created),
        ))

    def base(self, step: str, document: str) -> None:
        """Make ``step``, whose definition is a document, read ``document`` in its place."""
        current = self.view.steps[step]
        if self.variant is None:
            raw = current.model_dump(mode="json", by_alias=True, exclude_unset=True)
            file = file_value(AuthoredFileAddress(kind="step", id=step), raw, ("base", "document"), document)
            self.files.append(file)
            self.view = preview_authored_files(self.view, [file])
            return
        self.row.append(OverridePatch(path=f"steps.{step}.base.document", op="replace", value=document))
        self.view = self.view.model_copy(update={"steps": self.view.steps | {
            step: current.model_copy(update={"base": current.base.model_copy(update={"document": document})}),
        }})

    def step(self, step: str, writes: Sequence[OverridePatch]) -> None:
        """Store ``writes``, paths in ``step``'s payload: in its file, or in the row. The row writes the step's own
        definition when its file holds one alone, and otherwise a layer it appends to the step's deltas."""
        current = self.view.steps[step]
        if self.variant is None:
            file = _step_file(self.view, step, AuthoredFileAddress(kind="step", id=step),
                              current.model_dump(mode="json", by_alias=True, exclude_unset=True), current, writes)
            if file is not None:
                self.files.append(file)
                self.view = preview_authored_files(self.view, [file])
            return
        authored = self.project.steps[step]
        if isinstance(authored.base, InlineBase) and not authored.deltas:
            self.row.extend(patch.model_copy(update={"path": f"steps.{step}.base.inline.{patch.path}"}) for patch in writes)
            updated = current.model_copy(update={"base": InlineBase(inline=apply_override_patches(current.base.inline, writes))})
        else:
            count = len(authored.deltas)
            prior = [patch for layer in current.deltas[count:] for patch in layer.patches]
            bare = current.model_copy(update={"deltas": current.deltas[:count]})
            before = authored_payload(self.view.model_copy(update={"steps": self.view.steps | {step: bare}}), step)
            patches = normalized_patches(before, [*prior, *writes])
            layers = [MatrixCompositionDelta(
                layer_id=_VARIANT_LAYER, patches=patches, acknowledges_ancestor_paths=[patch.path for patch in patches],
            )] if patches else []
            self.layers[step] = [layer.model_dump(mode="json", exclude_unset=True) for layer in layers]
            updated = bare.model_copy(update={"deltas": [*bare.deltas, *layers]})
        self.view = self.view.model_copy(update={"steps": self.view.steps | {step: updated}})

    def point(self, pointing: Mapping[str, Mapping[str, str]], named: Mapping[str, tuple[str, str]]) -> None:
        """Point each step of ``pointing`` at its new documents, visiting steps parents first.

        A step's desired reference is its reference before the edit, mapped for the pointed steps; a ``named`` step,
        which reads its document through its subject, is given a reference of its own in its slot. A reference is
        written only where the composed one differs.
        """
        if not pointing:
            return
        before = self.view
        moved = {old for mapping in pointing.values() for old in mapping}
        desired: dict[str, dict[tuple[str, ...], DocumentReference]] = {}
        for step in before.steps:
            references = _step_references(before, step)
            if any(reference.document in moved for reference in references.values()):
                mapping = pointing.get(step, {})
                desired[step] = {
                    path: reference.model_copy(update={"document": mapping.get(reference.document, reference.document)})
                    for path, reference in references.items() if reference.document in moved
                }
        slots = {
            step: (slot, pointing[step][document]) for step, (slot, document) in named.items()
            if document in pointing.get(step, {})
        }
        bases = {
            step: mapping[base.document] for step, mapping in pointing.items()
            if isinstance(base := before.steps[step].base, DocumentReference) and base.document in mapping
        }
        for step, document in bases.items():
            self.base(step, document)
        if missing := set(pointing) - desired.keys() - slots.keys() - bases.keys():
            raise ProjectFolderInvalid(f"{', '.join(sorted(missing))} reads the document only through its sweep.")
        for step in sorted(desired.keys() | slots.keys(), key=lambda name: (_step_chain(before, name), name)):
            composed = _step_references(self.view, step)
            writes = [
                OverridePatch(path=".".join(path), op="replace" if path in composed else "add",
                              value=reference.model_dump(mode="json", exclude_none=True))
                for path, reference in desired.get(step, {}).items() if composed.get(path) != reference
            ]
            if step in slots:
                writes.append(_slot_write(authored_payload(self.view, step), *slots[step]))
            if writes:
                self.step(step, writes)

    def stored(self, reached: frozenset[str]) -> StoredEdit:
        if self.variant is None:
            return StoredEdit(merged_files(self.files), tuple(self.created), reached)
        return StoredEdit(_row_file(self.project, self.variant, self.row, self.layers), tuple(self.created), reached)


def _row_file(
    project: Project, variant: str, writes: Sequence[OverridePatch], layers: Mapping[str, list[dict[str, Any]]],
) -> list[StudioProjectFilePatch]:
    """The variant's row with ``writes``, which compose with its writes, and ``layers``, which replace its layer of
    each step. A write that leaves a value as the default variant has it is dropped from the row."""
    row = project.sweep.rows.get(variant) if project.sweep else None
    if row is None:
        raise ProjectFolderInvalid("The variant needs an authored row.")
    appended = {f"steps.{step}.deltas.-": layer for step, layer in layers.items()}
    if not writes and not appended:
        return []
    raw = row.model_dump(mode="json", by_alias=True, exclude_unset=True)
    previous = raw.get("patches", {})
    patches = _store_patches(previous, [
        *writes, *(OverridePatch(path=path, op="add", value=layer[0]) for path, layer in appended.items() if layer),
    ])
    touched = {write.path for write in writes}
    kept = {
        path: value for path, value in patches.items()
        if not (path in appended and not appended[path])
        and not (path in touched and value is not None and _as_default(project, path, value))
    }
    if kept == previous:
        return []
    return [file_value(AuthoredFileAddress(kind="variant", id=variant), raw, ("patches",), kept)]


def _as_default(project: Project, path: str, write: Any) -> bool:
    """Whether the row's ``write`` at ``path`` leaves the value the default variant has there."""
    parts = path.split(".")
    if parts[0] == "documents" and len(parts) > 3 and parts[2] == "payload" and parts[1] in project.documents:
        content: Any = _composed(project, parts[1])
    elif parts[0] == "steps" and parts[2:4] == ["base", "inline"] and parts[1] in project.steps and isinstance(
        project.steps[parts[1]].base, InlineBase,
    ):
        content = project.steps[parts[1]].base.inline
    else:
        return False
    rest = parts[4:] if parts[0] == "steps" else parts[3:]
    present = _holds(content, rest)
    if write.get("op") == "remove":
        return not present
    if not present:
        return False
    for part in rest:
        content = content[int(part)] if isinstance(content, list) else content[part]
    return same_json(content, write.get("value"))


def variant_view(plan, project: Project, variant: str | None) -> Project:
    """The project an edit in ``variant`` reads: ``project`` in the default variant, None, else the variant's."""
    if variant is None:
        return project
    view = plan.variant_projects.get(variant)
    if view is None:
        raise ProjectFolderInvalid(f"The {variant} variant's row does not apply; fix it before editing in it.")
    return view


def document_write(
    project: Project, shown: str, edits: Sequence[DocumentEdit], chosen: Collection[str], readers: DocumentUsers,
    variant: str | None = None, named: Mapping[str, tuple[str, str]] = {}, view: Project | None = None,
) -> StoredEdit:
    """Store ``edits`` of the document ``shown`` for the ``chosen`` steps, which read it or a document based on it.

    ``readers`` are the steps of ``variant`` that read a document, None being the default variant, and ``view`` that
    variant's project. After it every chosen step has the edited value in that variant and no other step's definition
    in any variant has changed.
    """
    chosen = frozenset(chosen)
    if variant is not None and view is None:
        raise ValueError("An edit in a variant reads the variant's project.")
    store = _Store(project, variant, view or project)
    stores = _document_stores(store.view, shown, edits, chosen, readers, set())
    store.documents(stores)
    store.point(stores.pointing, named)
    return store.stored(chosen)


def content_edits(before: Payload, after: Payload) -> list[DocumentEdit]:
    """The edits that turn a document's composed content ``before`` into ``after``."""
    return [
        RemoveProjectValue(op="remove", path=path) if isinstance(record, RemoveDocumentValue)
        else SetProjectValue(op="replace", path=path, value=record.value)
        for path, record in diff(before, after).items()
    ]


def changed_from_base(project: Project, document: str | None) -> list[str]:
    """The pointers, at every depth, whose value in ``document`` differs from its base's, when it has a base.

    A list is compared item by item, as the edits that address its items do, though a difference stores it whole.
    """
    entry = project.documents.get(document) if document is not None else None
    if not isinstance(entry, BasedDocument):
        return []
    composed = [compose_document(project.documents.get, name) for name in (entry.base.document, document)]
    if any(isinstance(item, ProjectDiagnostic) for item in composed):
        return []
    return sorted(set(_differing(composed[0].payload, composed[1].payload)) - {""})


def _differing(before, after, at: str = ""):
    if before == after:
        return
    yield at
    if isinstance(after, dict):
        for key, item in after.items():
            inherited = before.get(key, _ABSENT) if isinstance(before, dict) else _ABSENT
            yield from _differing(inherited, item, f"{at}/{str(key).replace('~', '~0').replace('/', '~1')}")
    elif isinstance(after, list):
        for index, item in enumerate(after):
            inherited = before[index] if isinstance(before, list) and index < len(before) else _ABSENT
            yield from _differing(inherited, item, f"{at}/{index}")


def base_value_edits(project: Project, document: str, paths: Sequence[str]) -> list[DocumentEdit]:
    """The edits that give each of ``paths`` in ``document`` the value of its base, removing what the base lacks."""
    entry = project.documents[document]
    if not isinstance(entry, BasedDocument):
        raise ProjectFolderInvalid(f"{document} is not based on another document.")
    base = _composed(project, entry.base.document)
    edits: list[DocumentEdit] = []
    for path in paths:
        value: Any = base
        for part in json_pointer_parts(path):
            if isinstance(value, dict):
                value = value.get(part, _ABSENT)
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                value = _ABSENT
        edits.append(RemoveProjectValue(op="remove", path=path) if value is _ABSENT
                     else SetProjectValue(op="replace", path=path, value=value))
    return edits


_ABSENT = object()


def evaluation_document_write(
    project: Project, plan, shown: str, edits: Sequence[DocumentEdit], chosen: Collection[str], slot: str,
    variant: str | None = None,
) -> StoredEdit:
    """:func:`document_write` of ``shown`` for the ``chosen`` steps of ``variant``, among which an Evaluate step may
    read it through its subject at ``trial_documents.<slot>``.

    Such a step follows its subject when the subject is chosen. Otherwise it is a reader the edit can narrow to, given a
    reference of its own to the document it is pointed at.
    """
    variant_id = variant or plan.variants[0]
    variant = None if variant_id == plan.variants[0] else variant_id
    subjects = {
        run.key.step: subject.producer.step for run in plan.runs
        if run.key.variant == variant_id and run.kind == "evaluate"
        and (subject := run.inputs.get("subject")) is not None and subject.producer is not None
    }
    chosen = set(chosen)
    named: dict[str, tuple[str, str]] = {}
    for row in plan.rows:
        if row.key.variant == variant_id and row.key.step in chosen and shown in row.followed_documents:
            if subjects.get(row.key.step) in chosen:
                chosen.discard(row.key.step)
            else:
                named[row.key.step] = (slot, shown)

    def readers(name: str) -> frozenset[str]:
        return plan.document_users(variant_id, name) | (frozenset(named) if name == shown else frozenset())

    return document_write(project, shown, edits, chosen, readers, variant, named, variant_view(plan, project, variant))


def _holds(payload: Any, parts: Sequence[str]) -> bool:
    for part in parts:
        if isinstance(payload, dict) and part in payload:
            payload = payload[part]
        elif isinstance(payload, list) and part.isdigit() and int(part) < len(payload):
            payload = payload[int(part)]
        else:
            return False
    return True


def _in_form_of(before: Payload, after: Payload) -> Payload:
    """``after``, a normalized Model, spelled as ``before`` wherever the two mean the same Model."""
    from feedbax.contracts.model_document import ModelDocument

    result = deepcopy(before)
    for path, record in diff(ModelDocument.model_validate(before).model_dump(mode="json"), after).items():
        target, source = result, after
        parts = json_pointer_parts(path)
        for index, part in enumerate(parts):
            if index == len(parts) - 1 or not isinstance(target.get(part), dict):
                if isinstance(record, RemoveDocumentValue) and index == len(parts) - 1:
                    target.pop(part, None)
                elif part in source:
                    target[part] = deepcopy(source[part])
                break
            target, source = target[part], source[part]
    return result


def model_document_store(folders, project_id: str, project: Project, body, before: Payload, after: Payload) -> StoredEdit:
    """Store a Model edit of ``body.document`` by the document rule, after the request's step files."""
    head = StoredEdit(list(body.files), reached=frozenset(file.address.id for file in body.files))
    project = preview_authored_files(project, body.files)
    edits = content_edits(before, _in_form_of(before, after))
    if body.step is None:
        return head.then(_document_stores(project, body.document, edits, None, lambda name: (), set()).files)
    plan = folders.read_plan(project_id, revision=body.revision).plan
    return head + evaluation_document_write(
        project, plan, body.document, edits, {body.step, *body.chosen_steps}, "model", body.variant,
    )


def _document_edit(patch: OverridePatch, document_path: str, payload) -> DocumentEdit:
    parts = tuple(document_path.split("."))
    if patch.op == "remove":
        return RemoveProjectValue(op="remove", path=pointer(parts))
    parent = payload
    for part in parts[:-1]:
        parent = parent.get(part) if isinstance(parent, dict) else None
    exists = isinstance(parent, dict) and parts[-1] in parent
    return SetProjectValue(op="replace" if exists else "add", path=pointer(parts), value=patch.value)


def authored_payload(project: Project, step_id: str):
    authored = project.steps[step_id]
    if isinstance(authored.base, InlineBase):
        return authored.base.inline
    template = compose_step(project, step_id)
    return template.authored_payload() if isinstance(template, StepTemplate) else None


def scoped_file(
    project, step_id, scope: RunScope, writes, *, chosen: Collection[str] = (), users: DocumentUsers | None = None,
    view: Project | None = None,
) -> StoredEdit:
    """What stores ``writes``, paths in the step's definition, for the runs ``scope`` selects.

    A write whose path passes through a document reference changes that document. For the whole step, in the default
    variant or in another one, whose project is ``view``, it is stored by the document rule for the ``chosen`` steps
    (the step alone when none are named) among the steps of that variant that read it or a document based on it, given
    by ``users``. For a sweep value or exception it is applied to the selected runs' copy of the document.
    """
    authored = project.steps[step_id]
    if isinstance(scope, EvaluateAttachmentAxis | SweepValuesScope | EvaluateAttachmentRows):
        raw = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
        address = AuthoredFileAddress(kind="step", id=step_id)
        return StoredEdit([_selection_file(address, raw, authored, scope, writes)], reached=frozenset({step_id}))
    variant = scope.variant if isinstance(scope, EvaluateAttachmentVariant) else None
    if variant is not None and view is None:
        raise ValueError("An edit in a variant reads the variant's project.")
    store = _Store(project, variant, view or project)
    payload = authored_payload(store.view, step_id)
    through: dict[str, list[tuple[OverridePatch, str]]] = {}
    direct = []
    for patch in writes:
        target = document_write_target(payload, patch.path)
        if target is None:
            direct.append(patch)
        else:
            through.setdefault(target[0], []).append((patch, target[1]))
    reached = reached_steps(project, {step_id}) if direct else frozenset()
    if direct:
        store.step(step_id, direct)
    if through:
        chosen = frozenset(chosen) or frozenset({step_id})
        if users is None:
            raise ProjectFolderInvalid("This edit changes a document and needs the steps that read it.")
        if step_id not in chosen:
            raise ProjectFolderInvalid("The chosen steps no longer match this document.")
        reached |= chosen
        taken: set[str] = set()
        pointing: dict[str, dict[str, str]] = {}
        for document, patches in through.items():
            shown = _composed(store.view, document)
            edits = [_document_edit(patch, path, shown) for patch, path in patches]
            found = _document_stores(store.view, document, edits, chosen, users, taken)
            store.documents(found)
            for step, mapping in found.pointing.items():
                pointing.setdefault(step, {}).update(mapping)
        store.point(pointing, {})
    return store.stored(reached)


def _selection_file(address, raw, authored, scope, writes) -> StudioProjectFilePatch:
    if isinstance(scope, EvaluateAttachmentAxis):
        axis = authored.sweep.axes.get(scope.axis) if authored.sweep else None
        if not isinstance(axis, NamedAxis) or axis.values.get(scope.value) is None:
            raise ProjectFolderInvalid("The group needs a named sweep value authored on this step.")
        return sweep_value_file(address, raw, scope.axis, scope.value, writes)
    if isinstance(scope, SweepValuesScope):
        return values_exception_file(address, raw, scope.axes, writes)
    return row_exception_file(address, raw, scope.rows, writes)


_STUDIO_LAYER = "trial-authoring"


def _step_file(project, step_id, address, raw, authored, writes) -> StudioProjectFilePatch | None:
    """The step's file with ``writes`` stored: in its own definition, or else in Studio's delta layer.

    Studio's layer is the trailing run of its layers, kept as one: one patch per path, none that writes the value the
    step has without it, and no layer when none is left.
    """
    if isinstance(authored.base, InlineBase) and not authored.deltas:
        updated = apply_override_patches(authored.base.inline, writes)
        return file_value(address, raw, ("base", "inline"), updated)
    kept = len(authored.deltas)
    while kept and authored.deltas[kept - 1].layer_id == _STUDIO_LAYER:
        kept -= 1
    prior = authored.deltas[:kept]
    stored = [patch for layer in authored.deltas[kept:] for patch in layer.patches]
    before = authored_payload(project.model_copy(update={"steps": project.steps | {
        step_id: authored.model_copy(update={"deltas": prior}),
    }}), step_id)
    patches = normalized_patches(before, [*stored, *writes])
    deltas = [layer.model_dump(mode="json", exclude_unset=True) for layer in prior]
    if patches:
        deltas.append(MatrixCompositionDelta(
            layer_id=_STUDIO_LAYER, patches=patches, acknowledges_ancestor_paths=[patch.path for patch in patches],
        ).model_dump(mode="json", exclude_unset=True))
    if deltas:
        return file_value(address, raw, ("deltas",), deltas)
    if "deltas" in raw:
        return StudioProjectFilePatch(address=address, patches=[RemoveProjectValue(op="remove", path="/deltas")])
    return None


def train_selection(
    folders, project_id, revision, project, step_id, run, rows=(), axes=None, chosen_steps=(),
) -> TrainSelection:
    """The runs a Train edit names, read from the plan of ``revision``, and the write that stores an edit for them.

    ``chosen_steps`` are the steps the user named for an edit of a document the step reads; none is the step alone.
    Without rows the selection is every run of the chosen steps in the named run's variant. An edit of a document
    reaches the chosen steps, and one of the step's own values the steps based on it; the write refuses one that
    changes a request of any other step.
    """
    chosen = {step_id, *chosen_steps}

    def checked(scope, selected, writes):
        variant = isinstance(scope, EvaluateAttachmentVariant)

        def users(document: str) -> frozenset[str]:
            plan = folders.read_plan(project_id, revision=revision).plan
            return plan.document_users(scope.variant if variant else plan.variants[0], document)

        settled = folders.read_plan(project_id, revision=revision)
        view = variant_view(settled.plan, project, scope.variant if variant else None)
        stored = scoped_file(project, step_id, scope, writes, chosen=tuple(chosen_steps), users=users, view=view)
        if stored.files or stored.created:
            candidate = compile_project(
                preview_stored(project, stored), readers=settled.readers,
                operations=folders.operations, components=folders.components, functions=folders.functions,
                python_tasks=folders.python_tasks, figure_templates=folders.figure_templates,
            )
            refuse_unreached(settled.plan, candidate, selected or reached_runs(
                settled.plan, {scope.variant} if variant else settled.plan.variants, (), stored.reached,
            ))
        return stored

    if run is None:
        if rows or axes:
            raise ProjectFolderInvalid("Rows were given without the run they belong to.")
        return TrainSelection((), None, None, False, partial(checked, EvaluateAttachmentStep(kind="step"), frozenset()))
    key = RunKey(run.variant, run.step, run.row)
    if key.step != step_id:
        raise ProjectFolderInvalid(f"Run {key.row!r} belongs to step {key.step!r}, not {step_id!r}.")
    plan = folders.read_plan(project_id, revision=revision, source_runs=(key,)).plan
    scope = selection_scope(project, plan, key, tuple(rows), axes)
    shown = tuple(RunKey(key.variant, step_id, row) for row in rows) or tuple(sorted(
        (row.key for row in plan.rows if row.key.variant == key.variant and row.key.step in chosen),
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
        partial(checked, scope, _scoped_runs(project, plan, key, scope, chosen)), values,
        tuple(row.coordinates for row in plan.rows if row.key in shown and row.key.step == step_id),
    )


def _scoped_runs(project, plan, key: RunKey, scope: RunScope, chosen: Collection[str]) -> frozenset[RunKey]:
    """The runs an edit for ``scope`` may change: the step's selected runs and those of the other steps it reaches.

    For the whole step, in a variant or not, the write says which steps it reaches, and none are named here.
    """
    if isinstance(scope, EvaluateAttachmentStep | EvaluateAttachmentVariant):
        return frozenset()
    variants = set(plan.variants)
    reached = reached_steps(project, chosen)
    picked = {
        row.key for row in plan.rows if row.key.variant in variants and row.key.step == key.step and (
            row.key.row in scope.rows if isinstance(scope, EvaluateAttachmentRows)
            else all(
                (coordinate := row.coordinates.get(axis)) is not None and coordinate.value_id in values
                for axis, values in ({scope.axis: (scope.value,)} if isinstance(scope, EvaluateAttachmentAxis)
                                     else scope.axes).items()
            )
        )
    }
    return reached_runs(plan, variants, picked, reached - {key.step})


@dataclass(frozen=True)
class SelectedValue:
    """The selected runs as one value of a named axis of the step: an inline axis's value, or a set's entry."""

    axis: str
    value: str
    set: str | None
    """The Set document the axis reads, when it reads one."""


def selected_value(project: Project, step: str, scope: RunScope) -> SelectedValue | None:
    """The value of a named axis whose runs are the runs ``scope`` selects, when they are exactly one."""
    sweep = project.steps[step].sweep
    if isinstance(scope, EvaluateAttachmentAxis):
        return SelectedValue(scope.axis, scope.value, None)
    if isinstance(scope, SweepValuesScope) and len(scope.axes) == 1 and sweep is not None:
        (axis, values), = scope.axes.items()
        named = sweep.axes.get(axis)
        if len(values) == 1 and isinstance(named, SetAxis):
            return SelectedValue(axis, values[0], named.set)
    return None


def set_readers(project: Project, name: str) -> frozenset[str]:
    """The steps whose sweep has an axis reading the Set document ``name``, and the steps based on them."""
    direct = {
        step for step, authored in project.steps.items() if authored.sweep is not None and any(
            isinstance(axis, SetAxis) and axis.set == name for axis in authored.sweep.axes.values()
        )
    }
    return frozenset(direct) | base_descendants(project, direct)


@dataclass(frozen=True)
class RowModel:
    """The Model the first selected run reads as ``document``, and where an edit for the selection is stored."""

    content: Payload
    scope: RunScope
    follows: bool
    """The run reads the Model through its subject, so the step is given a reference of its own to it."""


def row_model(folders, project_id: str, project: Project, body) -> RowModel:
    """The Model a Model edit for ``body.rows`` of ``body.step`` is applied to: the first row's copy of the document."""
    plan = folders.read_plan(project_id, revision=body.revision).plan
    key = RunKey(plan.variants[0], body.step, body.rows[0])
    scope = selection_scope(project, plan, key, tuple(body.rows), body.axes or None)
    selected = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
    definition = selected.selected_definitions.get(key)
    row = next((row for row in selected.rows if row.key == key), None)
    from feedbax.project_model_selection import selected_model

    content = selected_model(definition, body.document) if definition is not None else None
    if isinstance(content, dict):
        return RowModel(content, scope, False)
    if row is not None and body.document in row.followed_documents:
        named = selected_value(project, body.step, scope)
        content = _composed(project, body.document)
        if named is not None:
            content = _value_model(project, body.step, named, body.document)
        return RowModel(content, scope, True)
    raise ProjectFolderInvalid(f"The selected runs do not read {body.document}.")


def _value_differences(project: Project, step: str, named: SelectedValue, document: str) -> Differences:
    """The differences the selected value makes to ``document``."""
    if named.set is not None:
        from feedbax.contracts.set_document import read_set

        try:
            entry = read_set(project.documents.get, named.set).entries.get(named.value)
        except ValueError as error:
            raise ProjectFolderInvalid(str(error)) from error
        if entry is None:
            raise ProjectFolderInvalid(f"{named.set} has no entry {named.value}.")
        return dict(entry.differences)
    axis = project.steps[step].sweep.axes.get(named.axis)
    value = axis.values.get(named.value) if isinstance(axis, NamedAxis) else None
    if value is None:
        raise ProjectFolderInvalid("The selected value is not a named value authored on this step.")
    prefix = f"documents.{document}.payload."
    return {
        pointer(tuple(path.removeprefix(prefix).split("."))): (
            SetDocumentValue(value=write.value) if isinstance(write, SetValue) else RemoveDocumentValue(op="remove")
        )
        for path, write in value.patches.items() if write is not None and path.startswith(prefix)
    }


def _value_model(project: Project, step: str, named: SelectedValue, document: str) -> Payload:
    try:
        return apply_differences(_composed(project, document), _value_differences(project, step, named, document))
    except DifferenceNotApplied as error:
        raise ProjectFolderInvalid(f"The selected value does not apply to {document}: {error}") from error


def _payload_target(document: str, path: str) -> str:
    parts = json_pointer_parts(path)
    if any(not part or "." in part for part in parts):
        raise ProjectFolderInvalid(f"{path} has a key a sweep patch cannot name.")
    return ".".join(("documents", document, "payload", *parts))


def _value_patches(base: Payload, document: str, differences: Differences) -> dict[str, Any]:
    """``differences`` of ``document``'s composed content ``base`` as sweep patches into its payload."""
    return {
        _payload_target(document, path): (
            {"op": "replace" if _holds(base, json_pointer_parts(path)) else "add", "value": record.value}
            if isinstance(record, SetDocumentValue) else {"op": "remove"}
        )
        for path, record in differences.items()
    }


def scoped_model_store(project: Project, body, row: RowModel, after: Payload) -> StoredEdit:
    """Store a Model edit of the selected runs' copy of ``body.document``, ``row.content`` before it and ``after``.

    On one value of a named axis the value holds its differences from the document: on its patches when the axis is
    inline, on the set's entry when it reads a set, which is then an edit of the set for the chosen steps. Any other
    selection holds the change on the sweep's exception for it.
    """
    step = body.step
    document = body.document
    delta = diff(row.content, _in_form_of(row.content, after))
    address = AuthoredFileAddress(kind="step", id=step)
    authored = project.steps[step]
    raw = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
    stored = StoredEdit([], reached=frozenset({step}))
    named = selected_value(project, step, row.scope)
    if named is None:
        writes = [
            OverridePatch(path=_payload_target(document, path), op="remove") if isinstance(record, RemoveDocumentValue)
            else OverridePatch(
                path=_payload_target(document, path), value=record.value,
                op="replace" if _holds(row.content, json_pointer_parts(path)) else "add",
            )
            for path, record in delta.items()
        ]
        if writes:
            stored = stored.then([_selection_file(address, raw, authored, row.scope, writes)])
    else:
        base = _composed(project, document)
        try:
            edited = apply_differences(_value_model(project, step, named, document), delta)
        except DifferenceNotApplied as error:
            raise ProjectFolderInvalid(f"The edit does not apply to the selected value: {error}") from error
        differences = diff(base, edited)
        if named.set is None:
            value = raw["sweep"]["axes"][named.axis]["values"][named.value]
            prefix = f"documents.{document}.payload."
            patches = {path: write for path, write in value.get("patches", {}).items() if not path.startswith(prefix)}
            stored = stored.then([file_value(
                address, raw, ("sweep", "axes", named.axis, "values", named.value, "patches"),
                patches | _value_patches(base, document, differences),
            )])
        else:
            stored = stored + set_entry_write(project, named.set, named.value, differences, {step, *body.chosen_steps})
    if row.follows:
        stored = stored.then([_step_file(
            project, step, address, raw, authored, [_slot_write(authored_payload(project, step), "model", document)],
        )])
    return stored


def set_entry_write(
    project: Project, name: str, entry: str, differences: Differences, chosen: Collection[str],
) -> StoredEdit:
    """Give the set ``name``'s ``entry`` ``differences`` for the ``chosen`` steps among those that read it.

    The set holds the edit when every step it would change is chosen; otherwise a set based on it holds it, and the
    chosen steps' axes read that set.
    """
    content = _composed(project, name)
    entries = content.get("entries", {})
    if entry not in entries:
        raise ProjectFolderInvalid(f"{name} has no entry {entry}.")
    updated = entries[entry] | {"differences": {
        path: value.model_dump(mode="json", exclude_unset=True) for path, value in differences.items()
    }}
    edits = content_edits(content, content | {"entries": entries | {entry: updated}})
    return set_document_write(project, name, edits, chosen)


def set_document_write(project: Project, name: str, edits: Sequence[DocumentEdit], chosen: Collection[str]) -> StoredEdit:
    """Store ``edits`` of the Set document ``name`` for the ``chosen`` steps by the document rule; a step pointed at a
    new set has its axis read it."""
    chosen = frozenset(chosen)
    stores = _document_stores(project, name, edits, chosen, lambda document: set_readers(project, document), set())
    files = list(stores.files)
    view = preview_stored(project, StoredEdit(files, tuple(stores.created)))
    for step, mapping in stores.pointing.items():
        authored = view.steps[step]
        raw = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
        axes = {
            axis: mapping[read.set] for axis, read in (authored.sweep.axes.items() if authored.sweep else ())
            if isinstance(read, SetAxis) and read.set in mapping
        }
        if not axes:
            raise ProjectFolderInvalid(f"{step} reads the set through the step it is based on.")
        address = AuthoredFileAddress(kind="step", id=step)
        files.extend(file_value(address, raw, ("sweep", "axes", axis, "set"), new) for axis, new in axes.items())
    return StoredEdit(merged_files(files), tuple(stores.created), chosen | set_readers(project, name))
