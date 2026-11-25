"""Resolve revision-bound authored stores for prepared and direct edits."""

from collections.abc import Collection, Mapping, Sequence
from itertools import chain

from pydantic import JsonValue

from feedbax.contracts.project import Project, base_chain
from feedbax.contracts.project_resolution import RunKey
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID
from feedbax.contracts.studio_authoring import StudioRunSelection, StudioTrainAuthoringRequest
from feedbax.contracts.studio_field import TrainFieldRequest
from feedbax.contracts.studio_scope import StoreKind, StudioStoreChoice
from feedbax.planner.project_plan import ProjectPlan, compile_project
from feedbax.project_folder import SettledProjectPlan
from feedbax.studio.selection import StoredEdit
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService, ProjectRevisionConflict
from feedbax.web.services.store_choice import EditStore, choose_store


def choose_authored_store[T](
    revision: str,
    intent: JsonValue,
    selection: Collection[RunKey],
    preferred: EditStore[T],
    alternatives: Sequence[EditStore[T]],
    choice: str | None,
) -> EditStore[T] | list[StudioStoreChoice]:
    result = choose_store(revision, intent, selection, preferred, alternatives, choice=choice)
    if result == "stale":
        raise ProjectRevisionConflict("project", revision)
    if result == "unavailable":
        raise ProjectFolderInvalid("The selection has no writable authored source.")
    return result


def preview_plan(
    folders: ProjectFolderService,
    settled: SettledProjectPlan,
    project: Project,
    *,
    source_runs: Collection[RunKey] = (),
    capture_sources: bool = False,
) -> ProjectPlan:
    return compile_project(
        project, readers=settled.readers, source_runs=source_runs, capture_sources=capture_sources,
        operations=folders.operations, components=folders.components, functions=folders.functions,
        python_tasks=folders.python_tasks, figure_templates=folders.figure_templates,
        training_methods=folders.training_methods,
        synthesis_methods=folders.synthesis_methods,
    )


def selected_edit_runs(plan: ProjectPlan, step: str, scope: StudioRunSelection | None) -> frozenset[RunKey]:
    variant = plan.variants[0]
    if scope is not None and scope.run is not None:
        key = RunKey(scope.run.variant, scope.run.step, scope.run.row)
        if key.step != step or plan.row(key) is None:
            raise ProjectFolderInvalid("The selected runs changed.")
        variant = key.variant
    if scope is not None and scope.rows:
        runs = frozenset(RunKey(variant, step, row) for row in scope.rows)
        if any(plan.row(key) is None for key in runs):
            raise ProjectFolderInvalid("The selected runs changed.")
        return runs
    steps = {step, *(scope.chosen_steps if scope is not None else ())}
    return frozenset(row.key for row in plan.rows if row.key.variant == variant and row.key.step in steps)


def _changed_names[T](before: Mapping[str, T], after: Mapping[str, T]) -> set[str]:
    return {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}


def edited_authored_rows(before: ProjectPlan, after: ProjectPlan) -> frozenset[RunKey]:
    """The rows whose authored step or directly read documents are touched, across effective variants."""
    from feedbax.web.services.trial_scope import reached_steps, set_readers

    empty = Project()
    sources: dict[str, tuple[frozenset[str], set[str]]] = {}
    for variant in before.variant_projects.keys() | after.variant_projects.keys():
        old = before.variant_projects.get(variant, empty)
        new = after.variant_projects.get(variant, empty)
        steps = _changed_names(old.steps, new.steps)
        documents = _changed_names(old.documents, new.documents)
        reached = reached_steps(old, steps) | reached_steps(new, steps)
        if documents:
            documents = documents | {
                name for view in (old, new) for name in view.documents
                if documents.intersection(base_chain(view.documents, name))
            }
            for view in (old, new):
                for name in documents:
                    if (entry := view.documents.get(name)) is not None and entry.schema_id == SET_DOCUMENT_SCHEMA_ID:
                        reached |= set_readers(view, name)
        sources[variant] = reached, documents
    rows = set()
    for row in chain(before.rows, after.rows):
        steps, documents = sources[row.key.variant]
        if row.key.step in steps or documents.intersection(row.document_uses - row.followed_documents):
            rows.add(row.key)
    return frozenset(rows)


def prepared_reach(
    folders: ProjectFolderService, settled: SettledProjectPlan, project: Project,
    stored: StoredEdit, observed: frozenset[RunKey], before: ProjectPlan,
    *, restrict_new_to: frozenset[RunKey] | None = None,
) -> frozenset[RunKey]:
    from feedbax.web.services.trial_scope import preview_stored

    if not stored.files and not stored.created:
        return frozenset()
    candidate = preview_plan(folders, settled, preview_stored(project, stored), source_runs=observed)
    introduced = {(row.key.variant, row.key.step) for row in candidate.rows if before.row(row.key) is None}
    restricted = restrict_new_to or frozenset()
    scoped = {(key.variant, key.step) for key in restricted}

    def values(plan: ProjectPlan, key: RunKey):
        row = plan.row(key)
        if row is None:
            return None
        definition = plan.selected_definitions.get(key)
        fields = plan.variant_projects[key.variant].steps[key.step].set_fields
        content = (definition.payload, definition.input_documents, definition.task_additions) if definition else None
        # An existing run can acquire or lose the display name Run 1 when copies change.
        return row.coordinates, row.problems, row.resolved, content, row.train, fields

    return frozenset(
        key for key in observed & edited_authored_rows(before, candidate)
        if (key.variant, key.step) in introduced and (
            (key.variant, key.step) not in scoped or key in restricted
        )
        or values(before, key) != values(candidate, key)
    )


def choose_prepared_edit(
    folders: ProjectFolderService, project_id: str, revision: str, project: Project, selected: frozenset[RunKey],
    intent: JsonValue, choice: str | None,
    preferred: tuple[StoreKind, str, str, StoredEdit],
    alternatives: Sequence[tuple[StoreKind, str, str, StoredEdit]] = (),
    *, reader_steps: Collection[str] = (), copies_selection: bool = False,
    retain_equivalent: bool = False,
) -> StoredEdit | list[StudioStoreChoice]:
    """Resolve the stores by their effect on runs. Authored conversions can retain an equivalent plan."""
    from feedbax.web.services.trial_scope import reached_steps

    if not selected:
        if choice is not None:
            raise ProjectRevisionConflict("project", revision)
        return preferred[3]

    settled = folders.read_plan(project_id, revision=revision)
    files = [file for _, _, _, writes in (preferred, *alternatives) for file in writes.files]
    documents = {file.address.id for file in files if file.address.kind == "document"}
    steps = reached_steps(project, {file.address.id for file in files if file.address.kind == "step"})
    variants = {file.address.id for file in files if file.address.kind == "variant"}
    if variants and project.sweep is not None:
        while children := {
            name for name, row in project.sweep.rows.items()
            if name not in variants and row is not None and row.source in variants
        }:
            variants |= children
    observed = selected | {
        row.key for row in settled.plan.rows
        if row.key.step in steps or row.key.step in reader_steps or row.key.variant in variants or documents & row.document_uses
    }
    before = folders.read_plan(project_id, revision=revision, source_runs=observed).plan
    options = []
    for kind, label, source, stored in (preferred, *alternatives):
        reach = prepared_reach(
            folders, settled, project, stored, observed, before,
            restrict_new_to=selected if copies_selection else None,
        )
        options.append(EditStore(kind, label, source, reach, stored))
    if choice is None and not any(option.runs for option in options):
        return options[0].value if retain_equivalent else StoredEdit([])
    answer = choose_authored_store(revision, intent, selected, options[0], options[1:], choice)
    return answer if isinstance(answer, list) else answer.value


def train_edit_store(
    folders: ProjectFolderService, project_id: str, project: Project, step: str,
    body: StudioTrainAuthoringRequest | TrainFieldRequest, stored: StoredEdit,
    *, new_rows_for_selection: bool = False,
) -> StoredEdit | list[StudioStoreChoice]:
    if not stored.files and not stored.created and body.store_choice is None:
        return stored
    plan = folders.read_plan(project_id, revision=body.revision).plan
    selected = selected_edit_runs(plan, step, body)
    if body.run is not None and body.run.variant != plan.variants[0]:
        kind, label = "variant", "Variant"
    elif len(body.axes) == 1 and sum(map(len, body.axes.values())) == 1:
        kind, label = "sweep_value", "Sweep Value"
    elif body.rows:
        kind, label = "row_exception", "Selected Runs"
    else:
        kind, label = "whole_step", "Whole Step"
    edit = body.edit if isinstance(body, TrainFieldRequest) else body.class_sweep.edit if body.class_sweep else None
    return choose_prepared_edit(
        folders, project_id, body.revision, project, selected,
        body.model_dump(mode="json", exclude={"store_choice"}), body.store_choice, (kind, label, step, stored),
        copies_selection=new_rows_for_selection,
        retain_equivalent=edit is not None and edit.kind in {"to_sweep", "to_list", "to_value"},
    )
