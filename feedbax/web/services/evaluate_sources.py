"""Read the recorded upstream Train for a selected Evaluate definition, and the Model the Evaluate evaluates."""

from __future__ import annotations

from dataclasses import dataclass

from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS, MODEL_DIFFERENCE_BATCHES, AttachmentDifference
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_edit import (
    AxisValueRuns, EvaluateModelRequest, ListedRuns, EvaluateModelState,
    VariantRuns, StudioEvaluateValueSource,
)
from feedbax.contracts.project import DifferenceNotApplied, Differences, Project, apply_differences, diff, nearest_shared_base
from feedbax.planner.project_plan import EVALUATION_MODEL, ProjectPlan, RunKey, SelectedRunDefinition
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.project_folder import ProjectFolderInvalid
from feedbax.project_model_selection import selected_model
from feedbax.web.services.axis_values import axis_values


def evaluate_subject(definition: SelectedRunDefinition | None) -> tuple[RunKey, ValueSource] | None:
    source = definition.sources.get("/inputs/subject") if definition is not None else None
    if source is None or source.kind != "upstream" or not (source.variant and source.step and source.row):
        return None
    return RunKey(source.variant, source.step, source.row), source


def evaluate_value_source(source: ValueSource, *, evaluate_step: str | None = None) -> StudioEvaluateValueSource:
    if evaluate_step is not None:
        return StudioEvaluateValueSource(kind="evaluate", step=source.step or evaluate_step)
    return StudioEvaluateValueSource(kind="trained", step=source.step or source.name)


@dataclass(frozen=True)
class EvaluateReading:
    state: EvaluateModelState
    differences: Differences
    """What the read Model document changes on the trained model: its differences from ``shared``."""
    shared: str | None
    """The nearest document the read Model document shares with the subject's, when they differ."""


def read_evaluate_model(
    folders, project_id, body: EvaluateModelRequest, components=None, *, selected: ProjectPlan | None = None,
    model_document: str | None = None,
) -> EvaluateReading:
    """The Model the selected Evaluate run evaluates: the trained model with the run's Model differences applied."""
    snapshot = folders.read(project_id, revision=body.revision)
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    if selected is None:
        selected = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
    definition = selected.selected_definitions.get(key)
    editable = _editable(selected, key, body)
    effective = MODEL_DIFFERENCE_BATCHES.validate_python(
        definition.payload.get(MODEL_ATTACHMENTS, {}) if definition else {},
    )
    differences = tuple(record for batch in effective.values() for record in batch)
    resolved = None
    source = None
    field_sources = {}
    changes: Differences = {}
    shared = None
    if upstream := evaluate_subject(definition):
        producer, provenance = upstream
        source = evaluate_value_source(provenance)
        train_plan = folders.read_plan(project_id, revision=body.revision, source_runs=(producer,)).plan
        train = train_plan.selected_definitions.get(producer)
        row = next((row for row in selected.rows if row.key == key), None)
        if train is not None and row is not None:
            document = train.document_ids.get(EVALUATION_MODEL)
            read = row.document_ids.get(EVALUATION_MODEL)
            if model_document is not None and read != model_document:
                raise ProjectFolderInvalid("The selected run does not use this Model.")
            if document and read and (model := selected_model(train, document)) is not None:
                own = selected_model(definition, read)
                shared, changes = _document_changes(snapshot.project, document, read, own)
                try:
                    model = apply_differences(model, changes)
                except DifferenceNotApplied as error:
                    raise ProjectFolderInvalid(f"{read} does not apply to the trained model: {error}") from error
                resolved = resolve_effective_model(
                    ModelDocument.model_validate(model), components or folders.components,
                    differences, functions=folders.functions,
                )
                for batch, records in effective.items():
                    for index, record in enumerate(records):
                        site = ("payload", MODEL_ATTACHMENTS, batch, str(index))
                        origin = definition.sources.get(pointer((*site, "value"))) or definition.sources.get(
                            pointer((*site, "op")),
                        )
                        if origin is not None:
                            field_sources[record.address.to_text()] = evaluate_value_source(origin, evaluate_step=key.step)
    return EvaluateReading(EvaluateModelState(
        revision=body.revision,
        editable=editable,
        effective=tuple(record for record in differences if isinstance(record, AttachmentDifference)),
        resolved_model=resolved,
        source=source,
        field_sources=field_sources,
        changed_from=shared,
    ), changes, shared)


def _document_changes(
    project: Project, subject: str, read: str, own: object = None,
) -> tuple[str | None, Differences]:
    """The nearest document the read one shares with the subject's, and the run's copy of it, ``own`` when the run
    reads one, differs from it by."""
    if read == subject and own is None:
        return None, {}
    shared = nearest_shared_base(project.documents, subject, read)
    if shared is None:
        raise ProjectFolderInvalid(f"{read} shares no document with {subject}.")
    composed = [compose_document(project.documents.get, name) for name in (shared, read)]
    if bad := next((entry for entry in composed if isinstance(entry, ProjectDiagnostic)), None):
        raise ProjectFolderInvalid(bad.detail)
    changes = diff(composed[0].payload, own if isinstance(own, dict) else composed[1].payload)
    return (shared if changes or read != subject else None), changes


def _editable(plan: ProjectPlan, key: RunKey, body: EvaluateModelRequest) -> bool:
    scope = body.scope
    rows = {row.key: row for row in plan.rows if row.key.step == key.step}
    editable = True
    if isinstance(scope, AxisValueRuns):
        ids = axis_values(rows.values())
        selected = {item for item in rows if ids.get((item, scope.axis)) == scope.value}
    elif isinstance(scope, ListedRuns):
        editable = all(item.variant == key.variant for item in rows)
        selected = {item for item in rows if item.variant == key.variant and item.row in scope.rows}
        if {item.row for item in selected} != set(scope.rows):
            raise ProjectFolderInvalid("The selection contains an unknown Evaluate row.")
    elif isinstance(scope, VariantRuns):
        selected = {item for item in rows if item.variant == scope.variant}
    else:
        selected = set(rows)
    if key not in selected:
        raise ProjectFolderInvalid("The selected run is outside the selection.")
    return editable
