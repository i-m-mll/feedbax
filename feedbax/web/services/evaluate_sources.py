"""Read the recorded upstream Train for a selected Evaluate definition, and the Model the Evaluate evaluates."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from feedbax.application.stored_regions import compose_document
from feedbax.compiler.model_attachment import lower_model_difference_values, resolve_effective_model
from feedbax.compiler.model_document import ModelCompilationProblem
from feedbax.compiler.model_edit import model_ungroup_eligibility
from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS, MODEL_DIFFERENCE_BATCHES, AttachmentDifference
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.stored_model import StoredModelDocument, authored_model_payload, model_keyed_collection
from feedbax.contracts.model_edit import (
    AxisValueRuns, EvaluateModelRequest, ListedRuns, EvaluateModelState,
    VariantRuns, StudioEvaluateValueSource, StudioStepNameReference,
)
from feedbax.contracts.project import (
    DifferenceNotApplied, Differences, LoadedModelDocument, Project, apply_differences, diff, nearest_shared_base,
)
from feedbax.planner.project_plan import EVALUATION_MODEL, ProjectPlan, RunKey, SelectedRunDefinition
from feedbax.contracts.project_io import project_declaration_name
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolutionCode
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
    return StudioEvaluateValueSource(
        kind="evaluate" if evaluate_step is not None else "trained",
        reference=StudioStepNameReference(kind="step", key=source.step or evaluate_step or source.name),
    )


@dataclass(frozen=True)
class EvaluateReading:
    state: EvaluateModelState
    differences: Differences
    """What the read Model document changes on the trained model: its differences from ``shared``."""
    shared: str | None
    """The nearest document the read Model document shares with the subject's, when they differ."""
    model: ModelDocument | None
    """The effective Model in the compiler's form."""


def read_evaluate_model(
    folders, project_id, body: EvaluateModelRequest, components=None, *, selected: ProjectPlan | None = None,
    model_document: str | None = None,
) -> EvaluateReading:
    """The Model the selected Evaluate run evaluates: the trained model with the run's Model differences applied."""
    snapshot = folders.read(project_id, revision=body.revision)
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    step = snapshot.project.steps.get(key.step)
    if step is None or step.kind != "evaluate":
        raise ProjectFolderInvalid("Choose an Evaluate step.")
    if selected is None:
        selected = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
    if components is None:
        components = folders.components
    definition = selected.selected_definitions.get(key)
    problems = [
        problem for problem in selected.diagnostics
        if problem.step == key.step and problem.variant in (None, key.variant) and problem.row in (None, key.row)
    ]
    editable = _editable(selected, key, body, has_problem=bool(problems))
    effective = MODEL_DIFFERENCE_BATCHES.validate_python(
        definition.payload.get(MODEL_ATTACHMENTS, {}) if definition else {},
    )
    differences = tuple(record for batch in effective.values() for record in batch)
    resolved = None
    source = None
    field_sources = {}
    changes: Differences = {}
    shared = None

    def located(problem: ProjectDiagnostic) -> ProjectDiagnostic:
        return replace(problem, step=key.step, row=key.row, variant=key.variant)

    if definition is not None and definition.pinned_model is not None:
        loaded = LoadedModelDocument.model_validate(definition.input_documents["subject"].payload)
        subject = loaded.model.document
        read = definition.document_ids.get(EVALUATION_MODEL, subject)
        if model_document is not None and read != model_document:
            raise ProjectFolderInvalid("The selected run does not use this Model.")
        authored = selected_model(definition, read)
        changed = _document_changes(snapshot.project, subject, read, authored)
        if isinstance(changed, ProjectDiagnostic):
            problems.append(located(changed))
        else:
            shared, changes = changed
            try:
                pinned = _read_changed_model(authored_model_payload(definition.pinned_model, components), changes, components)
                resolved = resolve_effective_model(
                    pinned, components, lower_model_difference_values(pinned, components, differences),
                    functions=folders.functions,
                )
            except (DifferenceNotApplied, ModelCompilationProblem) as error:
                problems.append(located(ProjectDiagnostic(
                    ResolutionCode.INVALID_DOCUMENT, EVALUATION_MODEL, str(error), document=read,
                    model_address=error.address if isinstance(error, ModelCompilationProblem) else None,
                )))
        editable &= not loaded.model.unresolved and loaded.model.pointer is None and read in snapshot.project.documents
    elif upstream := evaluate_subject(definition):
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
                changes_from_shared = _document_changes(snapshot.project, document, read, own)
                if isinstance(changes_from_shared, ProjectDiagnostic):
                    if not any(problem.path == EVALUATION_MODEL for problem in problems):
                        problems.append(located(changes_from_shared))
                else:
                    shared, changes = changes_from_shared
                    try:
                        base = _read_changed_model(model, changes, components)
                        resolved = resolve_effective_model(
                            base, components, lower_model_difference_values(base, components, differences),
                            functions=folders.functions,
                        )
                    except (DifferenceNotApplied, ModelCompilationProblem) as error:
                        problems.append(located(ProjectDiagnostic(
                            ResolutionCode.INVALID_DOCUMENT, EVALUATION_MODEL, str(error), document=read,
                            model_address=error.address if isinstance(error, ModelCompilationProblem) else None,
                        )))
    if definition is not None:
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
        resolved_model=StoredModelDocument.from_lowered(resolved, components) if resolved is not None else None,
        ungroup=model_ungroup_eligibility(resolved) if resolved is not None else (),
        source=source,
        field_sources=field_sources,
        changed_from=shared,
        problems=[
            StudioProjectWindowMapProblem.from_diagnostic(problem, declaration=project_declaration_name(project_id))
            for problem in problems
        ],
    ), changes, shared, resolved)


def _read_changed_model(payload, changes: Differences, components=None) -> ModelDocument:
    authored = apply_differences(payload, changes, keyed_collection=model_keyed_collection)
    return StoredModelDocument.model_validate(authored).lower(components)


def _document_changes(
    project: Project, subject: str, read: str, own: object = None,
) -> tuple[str | None, Differences] | ProjectDiagnostic:
    """The nearest document the read one shares with the subject's, and the run's copy of it, ``own`` when the run
    reads one, differs from it by."""
    if read == subject and own is None:
        return None, {}
    shared = nearest_shared_base(project.documents, subject, read)
    if shared is None:
        return ProjectDiagnostic(
            ResolutionCode.BAD_REFERENCE, EVALUATION_MODEL, f"{read} shares no document with {subject}.", document=read,
        )
    composed = [compose_document(project.documents.get, name) for name in (shared, read)]
    if bad := next((entry for entry in composed if isinstance(entry, ProjectDiagnostic)), None):
        return replace(bad, path=EVALUATION_MODEL, document=read)
    changes = diff(
        composed[0].payload, own if isinstance(own, dict) else composed[1].payload,
        keyed_collection=model_keyed_collection,
    )
    return (shared if changes or read != subject else None), changes


def _editable(plan: ProjectPlan, key: RunKey, body: EvaluateModelRequest, *, has_problem: bool) -> bool:
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
        if not selected and has_problem:
            return False
    if key not in selected:
        raise ProjectFolderInvalid("The selected run is outside the selection.")
    return editable
