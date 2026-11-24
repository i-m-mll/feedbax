"""Select the authored store of a Model command."""

from collections.abc import Collection

from feedbax.contracts.model_edit import ModelEditAsked, ModelEditRequest
from feedbax.contracts.project import Project
from feedbax.contracts.project_resolution import ProjectDiagnostic, RunKey, compose_document
from feedbax.planner.project_plan import ProjectPlan
from feedbax.studio.selection import StoredEdit
from feedbax.web.services.authored_scope import choose_authored_store, preview_plan
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService
from feedbax.web.services.store_choice import EditStore
from feedbax.web.services.trial_scope import (
    RowModel, model_document_store, preview_stored, scoped_model_store, selected_value, set_readers,
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


def model_edit_store(
    folders: ProjectFolderService, project_id: str, project: Project, body: ModelEditRequest,
    before: dict, after: dict, row: RowModel | None = None,
) -> StoredEdit | ModelEditAsked:
    if body.step is None:
        return model_document_store(folders, project_id, project, body, before, after)
    settled = folders.read_plan(project_id, revision=body.revision)
    plan = settled.plan
    selected = _selected(plan, body, body.variant or plan.variants[0])
    if not selected:
        raise ProjectFolderInvalid("The selected Model runs changed.")
    observed = frozenset(
        item.key for item in plan.rows
        if body.document in item.document_uses or item.key in selected
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
        return frozenset(
            key for key, value in values.items()
            if value != _observed_model(projection, candidate, key, body.document)
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
        kind, label = ("all_readers", "All Readers") if body.variant is None else ("variant", "Variant")
        preferred = EditStore(kind, label, body.document, reach(broad), broad)
        options = [EditStore("selected_readers", "Selected Readers", body.document, reach(narrow), narrow)] if narrow != broad else []
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
            stored = scoped_model_store(project, body, row, after)
            preferred = EditStore("row_exception", "Row Exception", body.step, reach(stored), stored)
            options = []
    chosen = choose_authored_store(
        body.revision, body.model_dump(mode="json", exclude={"store_choice"}), selected, preferred, options, body.store_choice,
    )
    return ModelEditAsked(status="asked", store_choices=chosen) if isinstance(chosen, list) else chosen.value
