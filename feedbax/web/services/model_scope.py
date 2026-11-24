"""Select the authored store of a Model command."""

from collections.abc import Collection
from dataclasses import replace

from pydantic import JsonValue

from feedbax.contracts.model_edit import ListedRuns, ModelEditAsked, ModelEditRequest, SetField
from feedbax.contracts.project import Project, SetDocumentValue
from feedbax.contracts.project_resolution import ProjectDiagnostic, RunKey, compose_document
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.result_address import family_documents
from feedbax.contracts.set_document import read_set
from feedbax.contracts.studio_scope import StoreKind
from feedbax.planner.project_plan import ProjectPlan
from feedbax.studio.selection import StoredEdit
from feedbax.web.services.authored_scope import choose_authored_store, edited_authored_rows, preview_plan
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService, ProjectRevisionConflict
from feedbax.web.services.store_choice import EditStore, all_readers_store
from feedbax.web.services.trial_scope import (
    RowModel, model_document_store, preview_stored, scoped_model_store, selected_value, set_entry_write,
    set_readers, source_value_file,
)


def _observed_model(plan: ProjectPlan, project: Project, key: RunKey, document: str) -> object:
    row = plan.row(key)
    if row is None:
        return None
    definition = plan.selected_definitions.get(key)
    if definition is not None:
        managed = definition.payload.get("trial_documents")
        if isinstance(managed, dict) and managed.get("model") is not None:
            return managed["model"]
        subject = definition.input_documents.get("subject")
        if subject is not None:
            return subject.payload
    if document in row.followed_documents:
        subject = next((run.inputs.get("subject") for run in plan.runs if run.key == key), None)
        if subject is not None and subject.producer is not None and subject.producer != key:
            return _observed_model(plan, project, subject.producer, document)
    if document not in row.document_uses:
        return row.problems
    composed = compose_document(plan.variant_projects.get(key.variant, project).documents.get, document)
    return (composed.detail if isinstance(composed, ProjectDiagnostic) else composed.payload, row.problems)


def _selected(plan: ProjectPlan, body: ModelEditRequest, variant: str) -> frozenset[RunKey]:
    if body.rows:
        selected = frozenset(RunKey(variant, body.step, row) for row in body.rows)
        if any(plan.row(item) is None for item in selected):
            raise ProjectFolderInvalid("The selected Model runs changed.")
        return selected
    steps = {body.step, *body.chosen_steps}
    return frozenset(row.key for row in plan.rows if row.key.variant == variant and row.key.step in steps)


def _read_field(model: object, parts: tuple[str, ...]) -> JsonValue:
    value = model
    for part in parts:
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _overridden_fields(
    project: Project, plan: ProjectPlan, body: ModelEditRequest, selected: frozenset[RunKey], operation: SetField,
) -> tuple[frozenset[RunKey], list[tuple[StoreKind, str, str, StoredEdit]]]:
    parts = operation.address.stored_path()
    path = pointer(parts)
    field = f"/payload/trial_documents/model{path}"
    remaining = set(selected)
    sources: dict[ValueSource, JsonValue] = {}
    for key in sorted(selected, key=lambda item: item.row):
        definition = plan.selected_definitions.get(key)
        origin = definition.sources.get(field) if definition is not None else None
        if origin is not None and (
            origin.kind in ("sweep_value", "exception", "variant")
            or origin.kind == "document" and origin.axis is not None and origin.value is not None
        ):
            remaining.discard(key)
            sources[origin] = _read_field(_observed_model(plan, project, key, body.document), parts)
    alternatives = []
    for source, before in sources.items():
        if source.kind == "document":
            entry = read_set(project.documents.get, source.name).entries[source.value]
            stored = set_entry_write(
                project, source.name, source.value,
                entry.differences | {path: SetDocumentValue(value=operation.value)}, set_readers(project, source.name),
            )
            alternatives.append(("set_entry", "Set Entry", f"{source.name}:{source.value}", stored))
        elif (file := source_value_file(project, source, before, operation.value)) is not None:
            kind, label = {
                "sweep_value": ("sweep_value", "Sweep Value"), "exception": ("row_exception", "Row Exception"),
                "variant": ("variant", "Variant"),
            }[source.kind]
            alternatives.append((kind, label, f"{source.name}:{source.path}", StoredEdit([file])))
    return frozenset(remaining), alternatives


def model_edit_store(
    folders: ProjectFolderService, project_id: str, project: Project, body: ModelEditRequest,
    before: dict, after: dict, row: RowModel | None = None,
) -> StoredEdit | ModelEditAsked:
    if body.step is None:
        if body.store_choice is not None:
            raise ProjectRevisionConflict("project", body.revision)
        return model_document_store(folders, project_id, project, body, before, after)
    settled = folders.read_plan(project_id, revision=body.revision)
    plan = settled.plan
    selected = _selected(plan, body, body.variant or plan.variants[0])
    if not selected and (body.rows or body.step not in project.steps or body.variant is not None):
        raise ProjectFolderInvalid("The selected Model runs changed.")
    families = {
        variant: frozenset(family_documents(view.documents, body.document))
        for variant, view in plan.variant_projects.items()
        if body.document in view.documents
    }
    observed = frozenset(
        item.key for item in plan.rows
        if item.document_uses & families.get(item.key.variant, {body.document}) or item.key in selected
        or any(file.address.id == item.key.step for file in body.files)
    )
    observed |= frozenset(
        binding.producer for run in plan.runs if run.key in observed
        and (binding := run.inputs.get("subject")) is not None and binding.producer is not None
    )
    tracked = folders.read_plan(project_id, revision=body.revision, source_runs=observed).plan
    values = {key: _observed_model(tracked, project, key, body.document) for key in observed}

    def reach(stored: StoredEdit) -> frozenset[RunKey]:
        if not (stored.files or stored.created):
            return frozenset()
        candidate = preview_stored(project, stored)
        projection = preview_plan(folders, settled, candidate, source_runs=observed)
        authored = edited_authored_rows(tracked, projection)
        return frozenset(
            key for key, value in values.items()
            if key in authored and value != _observed_model(projection, candidate, key, body.document)
        )

    def only_readers(steps: Collection[str]) -> ModelEditRequest:
        return body.model_copy(update={"chosen_steps": [step for step in sorted(steps) if step != body.step]})

    if row is None:
        broad = (
            model_document_store(folders, project_id, project, body.model_copy(update={
                "step": None, "variant": None, "chosen_steps": [],
            }), before, after)
            if body.variant is None
            else model_document_store(folders, project_id, project, only_readers({
                step for step in plan.document_users(body.variant, body.document)
            }), before, after)
        )
        narrow = model_document_store(folders, project_id, project, body, before, after)
        reached = reach(broad)
        preferred = (
            all_readers_store(body.document, reached, broad) if body.variant is None
            else EditStore("variant", "Variant", body.document, reached, broad)
        )
        narrowed = EditStore("selected_readers", "Selected Readers", body.document, reach(narrow), narrow)
        if narrow != broad and not narrow.created and narrowed.runs and narrowed.runs <= selected:
            preferred, options = narrowed, []
        else:
            options = [narrowed] if narrow != broad else []
    else:
        named = selected_value(project, body.step, row.scope)
        if named is not None:
            if named.set is not None:
                broad = scoped_model_store(project, only_readers(set_readers(project, named.set)), row, after)
                narrow = scoped_model_store(project, body, row, after)
                preferred = EditStore("set_entry", "Set Entry", named.set, reach(broad), broad)
                options = [EditStore("selected_readers", "Selected Readers", named.set, reach(narrow), narrow)] if narrow != broad else []
            else:
                stored = scoped_model_store(project, body, row, after)
                preferred = EditStore("sweep_value", "Sweep Value", f"{body.step}:{named.axis}:{named.value}", reach(stored), stored)
                options = []
        else:
            remaining = selected
            alternatives = []
            if isinstance(body.command.operation, SetField) and isinstance(row.scope, ListedRuns):
                remaining, alternatives = _overridden_fields(project, tracked, body, selected, body.command.operation)
            recipient = (
                replace(row, scope=ListedRuns(kind="rows", rows=tuple(sorted(key.row for key in remaining))))
                if remaining and remaining != selected else row
            )
            stored = scoped_model_store(project, body, recipient, after) if remaining else StoredEdit([])
            preferred = EditStore("other_runs", "Other Runs", body.step, reach(stored), stored)
            options = [EditStore(kind, label, source, reach(edit), edit) for kind, label, source, edit in alternatives]
            if not preferred.runs and len(options) == 1 and options[0].runs == selected:
                preferred, options = options[0], []
    if not selected and not preferred.runs and body.store_choice is None:
        return preferred.value
    chosen = choose_authored_store(
        body.revision, body.model_dump(mode="json", exclude={"store_choice"}), selected, preferred, options, body.store_choice,
    )
    return ModelEditAsked(status="asked", store_choices=chosen) if isinstance(chosen, list) else chosen.value
