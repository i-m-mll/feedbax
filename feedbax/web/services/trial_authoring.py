"""Project and prepare edits for one saved trial pairing."""


from copy import deepcopy
from dataclasses import dataclass, field, replace

from feedbax.execution.trial_resolution import TrialPair, pair_trial_source
from feedbax.contracts.expression_values import (
    Level, LiteralValue, Problem, ProblemCode, ScopedValue, Site, SweepList, SweepRange, SweepSample,
    ValueBase, ValueType,
)
from feedbax.tasks.registry import resolve_python_task_instance
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import stored_parts
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import VariantRuns
from feedbax.contracts.project import (
    SetProjectValue, apply_document_edits, based_on, json_pointer_parts,
)
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.planner.project_plan import EVALUATION_TASK, RunKey, compile_project, own_task_variation
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec, EvaluationRequestSpec
from feedbax.contracts.rollout import ROLLOUT_TYPE, authored_rollout_settings
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_authoring import StudioSweptParameterMissing
from feedbax.contracts.studio_project import AuthoredFileAddress
from feedbax.studio.authoring import training_document_choice
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.contracts.studio_trials import (
    StudioTaskChoice, StudioTaskType, StudioTrialAuthoring, StudioTrialAuthoringRequest, StudioTrialBindingEdit,
    StudioTrialDocumentEdit, StudioTrialDocumentReset, StudioTrialEpoch, StudioTrialExpressionEdit, StudioTrialField,
    StudioTrialInput,
    StudioTrialOutput, StudioTrialReset, StudioTrialSelectDocument, StudioTrialStepScope,
)
from feedbax.contracts.task_document import (
    PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID, PythonTaskSelection, TaskDocument,
)
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument, ObjectiveTerm
from feedbax.contracts.training import TrialDocuments
from feedbax.contracts.expression_check import references
from feedbax.contracts.expression_site import FieldKind, project_site, read_field
from feedbax.contracts.expression_text import ParsedField, parse, print_field
from feedbax.studio.value_structure import value_structure
from feedbax.objectives.document import ObjectiveExecutor, ObjectiveUse, compile_objective
from feedbax.application.bundle import ApplicationRegistryBundle
from feedbax.web.services.evaluate_sources import evaluate_subject, evaluate_value_source
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService
from feedbax.studio.selection import StoredEdit
from feedbax.studio.authoring_sweeps import FieldSweeps, SweepTarget
from feedbax.studio.numeric_fields import NumericField, prepare_numeric_field, project_numeric_field
from feedbax.contracts.studio_project import StudioProjectFilePatch
from feedbax.web.services.trial_scope import (
    authored_payload, base_value_edits, changed_from_base, document_write, evaluation_document_write, lit_steps, preview_stored, reached_runs, refuse_unreached, scoped_file,
    selected_runs, variant_view,
)


@dataclass(frozen=True)
class _Projection:
    pair: TrialPair | None
    variables: tuple[ScopedValue, ...]
    signals: tuple[ScopedValue, ...]
    epochs: tuple[StudioTrialEpoch, ...]
    fields: dict[tuple[str | int, ...], Site]
    problems: tuple[Problem, ...]
    task_origins: dict[tuple[str, ...], str] = field(default_factory=dict)


def _project(
    documents, bundle, *, show_objective=True, graph_override=None,
    objective_use: ObjectiveUse = ObjectiveUse.TRAIN,
) -> _Projection:
    task = documents.task_source
    fields = {}
    if isinstance(task, PythonTaskSelection):
        resolved = resolve_python_task_instance(task, bundle.python_tasks)
        if isinstance(resolved, tuple):
            return _Projection(None, (), (), (), {}, resolved)
        definition = resolved.definition
        template = resolved.template()
        signals = tuple(ScopedValue(id=name, name=name, type=kind, level=Level.STEP)
                        for name, kind in template.signals.items())
        epochs = ()
        variables = ()
        fields.update({("python_task", "params", name): site
                       for name, site in definition.parameter_sites.items()})
    else:
        signals = tuple(ScopedValue(id=signal.id, name=signal.name, type=signal.type, level=Level.STEP)
                        for signal in task.signals)
        variables = ()
        epochs = tuple(StudioTrialEpoch(id=epoch.id, name=epoch.name, steps=None) for epoch in task.epochs)
    def pair(task):
        sites = {}
        paired = pair_trial_source(
            documents.model, task, bundle.components, bundle.expression_functions,
            python_tasks=bundle.python_tasks, input_overrides=documents.input_overrides,
            task_sites=sites, graph_override=graph_override,
        )
        return paired, sites

    paired, task_sites = pair(task)
    if isinstance(paired, tuple) and isinstance(task, TaskDocument):
        # The other documents stay authorable against the links that do resolve.
        unresolved = {problem.path for problem in paired}
        linked = tuple(link for link in task.links if ("links", link.model_field) not in unresolved)
        if len(linked) < len(task.links):
            partial, partial_sites = pair(task.model_copy(update={"links": linked}))
            if not isinstance(partial, tuple):
                paired, task_sites = replace(partial, problems=(*paired, *partial.problems)), partial_sites
    fields.update({("task", *path): site for path, site in task_sites.items()})
    if isinstance(paired, tuple):
        if show_objective:
            fields.update(_unpaired_objective_sites(documents, signals))
        return _Projection(None, variables, signals, epochs, fields, paired)
    source = paired.task
    variables, signals = source.interface.variables, source.interface.signals
    names = {epoch.id: epoch.name for epoch in epochs}
    epochs = tuple(
        StudioTrialEpoch(id=epoch, name=names.get(epoch, epoch), steps=steps)
        for epoch, steps in zip(source.epochs, source.step_counts, strict=True)
    )
    objective_sites = {}
    objective = compile_objective(
        documents.objective, (*paired.model.outputs, *variables, *signals), bundle.expression_functions,
        task=source, sites=objective_sites if show_objective else None, use=objective_use,
    )
    fields.update({("objective", *path): site for path, site in objective_sites.items()})
    problems = paired.problems
    if not isinstance(objective, ObjectiveExecutor):
        problems += tuple(problem.model_copy(update={"path": ("objective", *problem.path)})
                          for problem in objective)
    return _Projection(paired, variables, signals, epochs, fields, problems)


def _unpaired_objective_sites(documents, signals) -> dict[tuple[str | int, ...], Site]:
    """Each term's Measure and Weight over the names the documents declare, for a Model and Task that do not pair.

    A commit through one of these sites is parsed, not checked: the declared names have no resolved types."""
    def declared(value_type) -> ValueType:
        shape = value_type.shape if value_type is not None and value_type.shape else ()
        return ValueType(
            base=ValueBase.REAL, shape=tuple(size for size in shape if type(size) is int and size > 0),
        )

    names = (
        *(ScopedValue(id=_output_address(port.id).to_text(), name=port.name, type=declared(port.type), level=Level.STEP)
          for port in documents.model.ports.outputs),
        *((ScopedValue(id="time", name="time", type=ValueType(base=ValueBase.REAL), level=Level.STEP),)
          if isinstance(documents.task_source, TaskDocument) else ()),
        *(ScopedValue(id=variable.id, name=variable.name, type=variable.type, level=Level.STEP)
          for variable in (documents.task.variables if documents.task is not None else ())),
        *signals,
    )
    weight_names = (*names, *(
        ScopedValue(id=name, name=name, type=ValueType(base=ValueBase.REAL), level=Level.STEP)
        for name in ("t", "T", "u")
    ))
    sites: dict[tuple[str | int, ...], Site] = {}
    for term in documents.objective.terms:
        path = ("objective", "terms", term.id)
        sites[(*path, "measure")] = Site(expected=term.measure_type, names=names, finest=Level.STEP)
        sites[(*path, "weight")] = Site(expected=ValueType(base=ValueBase.REAL), names=weight_names, finest=Level.STEP)
    return sites


def _output_address(port: str) -> ModelAddress:
    return ModelAddress.model_validate({
        "composites": [],
        "element": {"kind": "port", "end": {"owner": "outputs", "port": port}},
    })


def _input_address(port: str) -> ModelAddress:
    return ModelAddress.model_validate({
        "composites": [],
        "element": {"kind": "port", "end": {"owner": "inputs", "port": port}},
    })


def _input_rows(documents, projection) -> list[StudioTrialInput]:
    paired = projection.pair
    resolved = {port.id: port.type for port in paired.model.inputs} if paired else {}
    effective = dict(paired.input_bindings) if paired else {}
    return [
        StudioTrialInput(
            address=address, name=port.name, type=resolved.get(address.to_text()),
            required=port.default is None, signal=effective.get(port.name),
            explicit=port.name in documents.input_overrides,
        )
        for port in documents.model.ports.inputs
        for address in (_input_address(port.id),)
    ]


def _selected_task(project, document) -> TaskDocument | PythonTaskSelection | None:
    entry = compose_document(project.documents.get, document)
    if isinstance(entry, ProjectDiagnostic):
        return None
    if entry.schema_id == TASK_DOCUMENT_SCHEMA_ID:
        return TaskDocument.model_validate(entry.payload)
    if entry.schema_id == PYTHON_TASK_SCHEMA_ID:
        return PythonTaskSelection.model_validate(entry.payload)
    return None


def _with_task(documents, task):
    if isinstance(task, PythonTaskSelection):
        return documents.model_copy(update={"task": None, "python_task": task})
    return documents.model_copy(update={"task": task, "python_task": None})


def _expression_edit(change, documents, projection, bundle):
    site = projection.fields.get(("objective", "terms", change.term, change.field))
    index = next((i for i, term in enumerate(documents.objective.terms) if term.id == change.term), None)
    if site is None or index is None:
        raise ProjectFolderInvalid("The selected objective field is unavailable.")
    if change.edit.kind != "text" or change.edit.value_id is not None:
        return (Problem(code=ProblemCode.STRUCTURE, reason="This objective field takes one expression."),)
    text = change.edit.text
    parsed = (
        read_field(text, site, bundle.expression_functions) if projection.pair is not None
        else parse(text, site, bundle.expression_functions)
    )
    if not isinstance(parsed, ParsedField):
        return parsed
    if isinstance(parsed.value, SweepList | SweepRange | SweepSample):
        return (Problem(code=ProblemCode.STRUCTURE, reason="This objective field takes one expression."),)
    return StudioTrialDocumentEdit(kind="objective", edits=[SetProjectValue(
        op="replace" if change.field in documents.objective.terms[index].model_fields_set else "add",
        path=pointer(("terms", str(index), change.field)),
        value=parsed.value.model_dump(mode="json"),
    )])


@dataclass(frozen=True)
class _WeightSweep:
    """A term's numeric Weight as a field the Train step sweeps, at the term's id."""

    index: int
    field: NumericField
    axis: FieldSweeps
    stored: bool
    """The document stores the Weight, rather than leaving it to the default."""


def _weight_sweep(project, definition, step_id, term_id) -> _WeightSweep | None:
    """The Weight of ``term_id`` in the Objective document the step reads, when it is a number the step can sweep."""
    name = _whole_document(definition, "payload.trial_documents.objective")
    entry = compose_document(project.documents.get, name) if name is not None else None
    if entry is None or isinstance(entry, ProjectDiagnostic):
        return None
    objective = ObjectiveDocument.model_validate(entry.payload)
    index, term = next(((i, term) for i, term in enumerate(objective.terms) if term.id == term_id), (None, None))
    if term is None or not isinstance(term.weight, LiteralValue) or type(term.weight.value) not in (int, float):
        return None
    stored = "weight" in entry.payload["terms"][index]
    step = project.steps[step_id]
    label = "Weight"
    path = f"trial_documents.objective.terms.{term.id}.weight.value"
    field = NumericField(
        key=("objective", "terms", term.id, "weight"), label=label, value=term.weight.value,
        site=Site(expected=ValueType(base=ValueBase.REAL), finest=Level.RUN, structural=True),
        stored=True, validate=lambda value: value, unit=term.weight.unit,
    )
    name = (term.name,) if term.name else ("term", str(index + 1))
    target = SweepTarget(term.weight.value, lambda _value: [], path)
    return _WeightSweep(index, field, FieldSweeps(step, target, label, (*name, "weight"), integer=False), stored)


def _routes_to_sweep(change, swept: _WeightSweep, site: Site, functions) -> bool:
    """The edit is the step's sweep of the Weight: it edits a sweep, makes one, or ends one."""
    edit = change.edit
    if edit.kind != "text" or edit.value_id is not None or swept.axis.covering:
        return True
    parsed = read_field(edit.text, site, functions)
    return isinstance(parsed, ParsedField) and isinstance(parsed.value, SweepList | SweepRange | SweepSample)


def _weight_sweep_edit(
    change, swept: _WeightSweep, step_id, functions, write_document,
) -> StoredEdit | tuple[Problem, ...]:
    """The files that store a sweep edit of a term's Weight: the step's sweep, and the value it is reduced to.

    A sweep sets the number of a Weight the document stores, so a term that leaves its Weight to the default is
    given it explicitly.
    """
    prepared = prepare_numeric_field(swept.field, swept.axis, change.edit, functions)
    result = prepared.result
    if isinstance(result, tuple):
        return result
    if not isinstance(result, list):
        raise ProjectFolderInvalid("The Weight's sweep has no value to keep.")
    stored = StoredEdit(
        [StudioProjectFilePatch(address=AuthoredFileAddress(kind="step", id=step_id), patches=result)],
        reached=frozenset({step_id}),
    )
    weight = ("terms", str(swept.index), "weight")
    kept = [patch.value for patch in prepared.keep if patch.op != "remove"]
    edits = [] if swept.stored else [SetProjectValue(
        op="add", path=pointer(weight), value=LiteralValue(value=swept.field.value, unit=swept.field.unit)
        .model_dump(mode="json", exclude_none=True),
    )]
    if kept:
        edits.append(SetProjectValue(op="replace", path=pointer((*weight, "value")), value=kept[-1]))
    return write_document(edits) + stored if edits else stored


def _task_field(task):
    return "python_task" if isinstance(task, PythonTaskSelection) else "task"


def _edit_value(value, change):
    raw = deepcopy(value)
    apply_document_edits(raw, change.edits)
    return raw


def _changed_value(documents, definition, kind):
    field = _task_field(documents.task_source) if kind == "task" else "objective"
    return field, definition.payload["trial_documents"][field]


def _override(path, op, value=None):
    return OverridePatch(path=path, op=op, **({"value": value} if op != "remove" else {}))


def _edit_patches(prefix, raw, change):
    patches = []
    for edit in change.edits:
        parts = stored_parts(raw, json_pointer_parts(edit.path))
        if any(not part or "." in part for part in parts):
            raise ProjectFolderInvalid("This scoped edit needs path-safe field names.")
        path = ".".join((prefix, *parts))
        patches.append(_override(path, edit.op, edit.value if isinstance(edit, SetProjectValue) else None))
    return patches


def _value_writes(prefix, raw, change):
    # Ordered table edits can shift the meaning of every subsequent index.
    if change.kind == "objective" and any(
        edit.op != "replace" and len(json_pointer_parts(edit.path)) == 2 for edit in change.edits
    ):
        edited = _edit_value(raw, change)
        if all(json_pointer_parts(edit.path)[:1] == ("terms",) for edit in change.edits):
            return [_override(f"{prefix}.terms", "replace" if "terms" in raw else "add", edited["terms"])]
    return _edit_patches(prefix, raw, change)


def _authored_task_entry(raw):
    if not isinstance(raw, (dict, list)):
        return None
    entries = raw.items() if isinstance(raw, dict) else enumerate(raw)
    names = [name for name, edit in entries if isinstance(edit, dict) and edit.get("path") == "task"]
    return names[-1] if names else None


def _task_selection_writes(definition, value):
    raw = definition.payload["params"]["variation"]
    entry = _authored_task_entry(raw)
    writes = []
    if isinstance(raw, dict):
        writes.extend(
            _override(f"params.variation.{name}", "remove")
            for name, edit in raw.items() if edit["path"].startswith("task.")
        )
        target = entry or "task"
        if entry is None:
            while target in raw:
                target += "-selection"
    elif entry is None or any(edit["path"].startswith("task.") for edit in raw[entry + 1:]):
        target = len(raw)
    else:
        target = entry
    writes.append(_override(
        f"params.variation.{target}", "replace" if entry == target else "add",
        {"op": "replace", "path": "task", "value": value},
    ))
    return writes


def _train_writes(change, project, definition, documents, projection):
    raw = definition.payload["trial_documents"]
    if isinstance(change, StudioTrialSelectDocument) and change.role != "task":
        _chosen_document(project, change.document, change.role)
        return [_override(f"trial_documents.{change.role}", "replace" if change.role in raw else "add",
                          {"document": change.document})]
    if isinstance(change, StudioTrialSelectDocument):
        task = _chosen_task(project, change.document)
        field = _task_field(task)
        if field == "python_task":
            return [
                _override("trial_documents.task", "replace" if "task" in raw else "add", None),
                _override("trial_documents.python_task", "replace" if "python_task" in raw else "add",
                          {"document": change.document}),
            ]
        writes = [_override("trial_documents.task", "replace" if "task" in raw else "add",
                            {"document": change.document})]
        if "python_task" in raw:
            writes.append(_override("trial_documents.python_task", "remove"))
        return writes
    if isinstance(change, StudioTrialBindingEdit):
        row = next((row for row in _input_rows(documents, projection) if row.address == change.input), None)
        if row is None:
            raise ProjectFolderInvalid("The address is not a declared Model input for this row.")
        overrides = dict(documents.input_overrides)
        if change.signal is None or change.signal == row.name:
            overrides.pop(row.name, None)
        else:
            overrides[row.name] = change.signal
        present = "input_overrides" in raw
        if present:
            path = f"trial_documents.input_overrides.{row.name}"
            if "." in row.name:
                raise ProjectFolderInvalid("This input needs a path-safe name for a scoped edit.")
            if row.name in overrides:
                op = "replace" if row.explicit else "add"
                return [_override(path, op, overrides[row.name])]
            if row.explicit:
                return [_override(path, "remove")]
            return []
        if overrides:
            return [_override("trial_documents.input_overrides", "add", overrides)]
        return []
    field, value = _changed_value(documents, definition, change.kind)
    return _value_writes(f"trial_documents.{field}", value, change)


_DOCUMENT_SCHEMAS = {"model": MODEL_DOCUMENT_SCHEMA_ID, "objective": OBJECTIVE_DOCUMENT_SCHEMA_ID}


def _chosen_document(project, document, role):
    entry = project.documents.get(document)
    if entry is None or entry.schema_id != _DOCUMENT_SCHEMAS[role]:
        raise ProjectFolderInvalid(f"{document} is not a{'n' if role == 'objective' else ''} {role.title()} document.")


def _chosen_task(project, document):
    task = _selected_task(project, document)
    if task is None:
        raise ProjectFolderInvalid(f"{document} is not a Task document.")
    return task


def _evaluation_task_path(definition):
    entry = _authored_task_entry(definition.payload.get("params", {}).get("variation"))
    return f"payload.params.variation.{entry}.value" if entry is not None else EVALUATION_TASK


def _evaluate_documents(plan, key, definition, bundle):
    upstream = evaluate_subject(definition)
    subject = next((row for row in plan.runs if upstream and row.key == upstream[0]), None)
    raw = plan.requests.get(subject.request_id) if subject else None
    if raw is None:
        raise ProjectFolderInvalid("The selected Evaluate has no planned Train subject.")
    request = decode_request(raw, plan.documents.__getitem__).spec
    if not isinstance(request, DocumentTrainingRequestSpec):
        raise ProjectFolderInvalid("The selected Evaluate has no managed Train pairing.")
    documents = request.trial_documents
    projection = _project(documents, bundle, show_objective=False)
    if projection.pair is None or projection.problems:
        return documents, projection, None
    try:
        if definition.payload.get("evaluation_type") != ROLLOUT_TYPE:
            raise ValueError("Task authoring requires a saved rollout Evaluate.")
        raw_params = definition.payload.get("params")
        own = (definition.payload.get("trial_documents") or {}).get("task")
        if own is not None and isinstance(raw_params, dict) and (
            variation := own_task_variation(documents.task_source, own, raw_params.get("variation", []))
        ) is not None:
            raw_params = raw_params | {"variation": variation}
        params = authored_rollout_settings(raw_params)
        variant = resolve_evaluation_variant(
            projection.pair.lowered.graph, documents.task_source, params.variation,
            components=bundle.components,
        )
        graph, task = variant.graph, variant.task
        run = next((row for row in plan.runs if row.key == key), None)
        if run is not None:
            effective = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
            if isinstance(effective, EvaluationRequestSpec):
                effective_params = authored_rollout_settings(effective.params)
                if effective_params.variation != params.variation:
                    lowered = resolve_evaluation_variant(
                        projection.pair.lowered.graph, documents.task_source, effective_params.variation,
                        components=bundle.components,
                    )
                    graph, task = lowered.graph, lowered.task
    except ValueError as error:
        return documents, _Projection(
            projection.pair, projection.variables, projection.signals, projection.epochs,
            projection.fields, (*projection.problems, Problem(
                code=ProblemCode.STRUCTURE, path=("variation",), reason=str(error),
            )),
        ), None
    documents = _with_task(documents, task)
    paired = _project(documents, bundle, show_objective=False, graph_override=graph, objective_use=ObjectiveUse.INHERITED)
    return documents, replace(paired, task_origins=variant.edit_origins), graph


def _pairing(plan, key, bundle):
    definition = plan.selected_definitions.get(key)
    if definition is None:
        raise ProjectFolderInvalid("The selected run has no resolved trial definition.")
    step = plan.steps[key.variant, key.step]
    if step.kind == "train":
        documents = TrialDocuments.model_validate(definition.payload.get("trial_documents"))
        return documents, _project(documents, bundle), None
    if step.kind == "evaluate":
        return _evaluate_documents(plan, key, definition, bundle)
    raise ProjectFolderInvalid("The selected run has no authored trial pairing.")


def _prepare_edits(body, folders, project_id, project, settled, key, bundle, documents, projected):
    change, scope = body.change, body.scope
    plan = settled.plan
    definition = plan.selected_definitions[key]
    step = plan.steps[key.variant, key.step]
    expression = change if isinstance(change, StudioTrialExpressionEdit) else None
    if expression is not None:
        if step.kind != "train":
            raise ProjectFolderInvalid("Evaluate authoring edits the task variation.")
        site = projected.fields.get(("objective", "terms", expression.term, expression.field))
        swept = _weight_sweep(project, definition, key.step, expression.term) if (
            expression.field == "weight" and site is not None and projected.pair is not None
        ) else None
        if swept is not None and _routes_to_sweep(expression, swept, site, bundle.expression_functions):
            if scope is not None and not isinstance(scope, StudioTrialStepScope):
                raise ProjectFolderInvalid("A Weight has one sweep for every run of the step.")
            readers = lambda name: plan.document_users(key.variant, name)  # noqa: E731
            shown = _whole_document(definition, "payload.trial_documents.objective")
            made = _weight_sweep_edit(
                expression, swept, key.step, bundle.expression_functions,
                lambda edits: document_write(project, shown, edits, {key.step}, readers),
            )
            return (StoredEdit([]), made) if isinstance(made, tuple) else (made, ())
        change = _expression_edit(expression, documents, projected, bundle)
        if not isinstance(change, StudioTrialDocumentEdit):
            return [], change
    document = None
    if step.kind == "train" and isinstance(change, StudioTrialDocumentEdit | StudioTrialDocumentReset):
        field, _ = _changed_value(
            documents, definition, change.document if isinstance(change, StudioTrialDocumentReset) else change.kind,
        )
        document = _whole_document(definition, f"payload.trial_documents.{field}")
    if isinstance(change, StudioTrialDocumentReset):
        restored = document if step.kind == "train" else (
            _whole_document(definition, _evaluation_task_path(definition)) if change.document == "task" else None
        )
        if restored is None:
            raise ProjectFolderInvalid("Reset needs a value the step reads from a document.")
        change = StudioTrialDocumentEdit(kind=change.document, edits=base_value_edits(project, restored, change.paths))

    variant = None if key.variant == plan.variants[0] else key.variant

    def readers(name):
        return plan.document_users(key.variant, name)

    chosen = set(scope.chosen_steps) if isinstance(scope, StudioTrialStepScope) else set()
    users = set(lit_steps(project.documents, readers, document)) if document else set()
    family = _family(project, document) if document else set()
    if scope is None:
        scope = StudioTrialStepScope(kind="step")
        chosen = users
    if isinstance(scope, StudioTrialStepScope):
        if chosen and document is not None and (key.step not in chosen or not chosen <= users):
            raise ProjectFolderInvalid("The chosen steps no longer match this document.")
        chosen = chosen or {key.step}
    selected = selected_runs(plan, key, scope, chosen_steps=chosen)
    if document is not None and isinstance(scope, StudioTrialStepScope):
        path = f"payload.trial_documents.{field}"
        if any(
            row.key in selected and row.document_ids.get(path) not in family and not family & row.document_uses
            for row in plan.rows
        ):
            raise ProjectFolderInvalid("A chosen step no longer uses the selected document.")
    if selected != {key}:
        settled = folders.read_plan(project_id, revision=body.revision, source_runs=selected)
    evaluate_field = (step.kind == "evaluate" and isinstance(change, StudioTrialDocumentEdit) and change.kind == "task")
    whole = isinstance(scope, StudioTrialStepScope)
    stored_scope = VariantRuns(kind="variant", variant=variant) if whole and variant else scope
    view = variant_view(plan, project, variant if whole else None)
    if step.kind == "evaluate":
        if isinstance(change, StudioTrialSelectDocument):
            role = change.role
            if role == "task":
                _chosen_task(project, change.document)
            elif role == "model" and whole:
                _chosen_document(project, change.document, role)
            else:
                raise ProjectFolderInvalid("An Evaluate reads its Objective and Training from its subject.")
            reference = {"document": change.document}
            stored = scoped_file(project, key.step, stored_scope, [
                _override(f"trial_documents.{role}", "replace" if role in slots else "add", reference)
                if isinstance(slots := (authored_payload(view, key.step) or {}).get("trial_documents"), dict)
                else _override("trial_documents", "add", {role: reference}),
            ] if whole else _task_selection_writes(definition, reference), chosen=chosen, users=readers, view=view)
        elif evaluate_field and whole:
            stored = _evaluate_task_write(project, plan, key, scope, change, variant)
            selected = selected_runs(plan, key, scope, chosen_steps=set(stored.reached) | {key.step})
            settled = folders.read_plan(project_id, revision=body.revision, source_runs=selected)
        elif evaluate_field:
            stored = _evaluate_field_files(project, key, scope, change, documents)
        else:
            raise ProjectFolderInvalid("Evaluate authoring edits the task variation.")
    elif document is not None and whole:
        edits = _reordered_terms(project, document, documents.objective, change) or list(change.edits)
        stored = document_write(project, document, edits, chosen, readers, variant, view=view)
    elif isinstance(change, StudioTrialSelectDocument) and change.role == "training":
        writes = training_document_choice(project, key.step, bundle.training_methods, change.document)
        if isinstance(writes, StudioSweptParameterMissing):
            return StoredEdit([]), writes
        stored = scoped_file(project, key.step, stored_scope, writes, chosen=chosen, users=readers, view=view)
    else:
        writes = _train_writes(change, project, definition, documents, projected)
        stored = scoped_file(
            project, key.step, stored_scope, writes, chosen=chosen, users=readers, view=view,
        ) if writes else StoredEdit([])
    if not stored.files and not stored.created:
        return StoredEdit([]), ()
    variants = {key.variant} if variant and whole else set(plan.variants)
    reach = reached_runs(plan, variants, selected, stored.reached - {key.step})
    return _validate_files(
        stored, project, settled, selected, reach, key.step, folders, bundle, change, expression,
        evaluate_field=evaluate_field and not isinstance(scope, StudioTrialStepScope),
    )


def _reordered(terms: list, change) -> list | None:
    """``terms`` in the order the change gives them, when it writes the Objective's whole term list as a reordering of
    them: each run's terms carry the values its sweeps set, so a reorder is of the terms of the document it is stored
    in, or of the run it is checked against, not of the run it was made in."""
    if change.kind != "objective" or len(change.edits) != 1:
        return None
    (edit,) = change.edits
    if not isinstance(edit, SetProjectValue) or edit.path != "/terms" or not isinstance(edit.value, list):
        return None
    order = [term.get("id") for term in edit.value if isinstance(term, dict)]
    by_id = {term["id"]: term for term in terms}
    return [by_id[term] for term in order] if sorted(order) == sorted(by_id) and len(order) == len(edit.value) else None


def _reordered_terms(project, document, shown, change) -> list[SetProjectValue] | None:
    """The document edit of a change that only reorders the terms the run shows."""
    order = _reordered([term.model_dump(mode="json") for term in shown.terms], change)
    if order is None or list(map(ObjectiveTerm.model_validate, order)) != list(
        map(ObjectiveTerm.model_validate, change.edits[0].value),
    ):
        return None
    entry = compose_document(project.documents.get, document)
    if isinstance(entry, ProjectDiagnostic):
        raise ProjectFolderInvalid(entry.detail)
    return [change.edits[0].model_copy(update={"value": _reordered(entry.payload["terms"], change)})]


def _whole_document(definition, path: str) -> str | None:
    """The document the step reads whole at ``path``, its payload."""
    return None if definition.document_pointers.get(path) else definition.document_ids.get(path)


def _family(project, document) -> set[str]:
    """``document`` and the documents based on it, at any depth."""
    found, pending = set(), [document]
    while pending:
        name = pending.pop()
        if name not in found:
            found.add(name)
            pending.extend(based_on(project.documents, name))
    return found




def _evaluate_task_write(project, plan, key, scope: StudioTrialStepScope, change, variant: str | None) -> StoredEdit:
    """An Evaluate step's edit of its Task, stored in the Task document it reads for the chosen steps of its variant.

    Without chosen steps they are the Evaluate steps of the variant that read the document or one based on it.
    """
    row = next(row for row in plan.rows if row.key == key)
    document = row.document_ids.get(EVALUATION_TASK) or row.document_ids.get("payload.trial_documents.python_task")
    if document is None:
        raise ProjectFolderInvalid("The selected Evaluate reads no Task document.")

    def readers(name):
        return plan.document_users(key.variant, name)

    chosen = {key.step, *scope.chosen_steps} if scope.chosen_steps else {key.step} | {
        step for step in lit_steps(project.documents, readers, document)
        if plan.steps[key.variant, step].kind == "evaluate"
    }
    return evaluation_document_write(project, plan, document, list(change.edits), chosen, "task", variant)


def _present_ops(raw, change):
    """``change`` with each set written as a replace where ``raw`` holds its target and as an add where it does not."""
    def holds(parts):
        value = raw
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                return False
        return True

    return change.model_copy(update={"edits": [
        edit.model_copy(update={"op": "replace" if holds(json_pointer_parts(edit.path)) else "add"})
        if isinstance(edit, SetProjectValue) else edit for edit in change.edits
    ]})


def _evaluate_field_files(project, key, scope, change, documents):
    """The step's writes of the selected runs' Task fields, at the Task's paths as a Train writes them."""
    task = documents.task_source
    raw = task.model_dump(mode="json", exclude_unset=True)
    return scoped_file(project, key.step, scope, _edit_patches(
        f"trial_documents.{_task_field(task)}", raw, _present_ops(raw, change),
    ))


def _validate_files(
    stored, project, settled, selected, reach, named, folders, bundle, change, expression, *, evaluate_field,
):
    plan = settled.plan
    choosing = isinstance(change, StudioTrialSelectDocument)
    authored = preview_stored(project, stored)
    compare_names = isinstance(change, StudioTrialDocumentEdit) and change.kind == "objective" and any(
        "name" in json_pointer_parts(edit.path) or len(json_pointer_parts(edit.path)) <= 2 for edit in change.edits
    )
    outside = {row.key for row in plan.rows if row.kind == "train" and row.key not in selected} if compare_names else set()

    def compile_sources(project, source_runs):
        return compile_project(
            project, readers=settled.readers, source_runs=source_runs,
            operations=folders.operations, components=bundle.components,
            functions=bundle.expression_functions, python_tasks=bundle.python_tasks,
            figure_templates=bundle.figure_templates,
        )

    candidate = compile_sources(authored, selected | outside)
    if outside:
        before = compile_sources(project, outside)
        if any(_objective_names(before, key) != _objective_names(candidate, key) for key in outside):
            raise ProjectFolderInvalid("The proposed edit changes an Objective outside the selected scope.")
    previous = {run.key: run for run in plan.runs}
    current = {run.key: run for run in candidate.runs}
    before_diagnostics = plan.diagnostics
    for key in sorted(selected, key=lambda item: (item.variant, item.step, item.row)):
        if key.step != named and key in previous and key in current and (
            previous[key].request_id == current[key].request_id
        ):
            # A reached step that holds its own value where the edit is made keeps it.
            continue
        if key not in candidate.selected_definitions:
            diagnostic = next((
                item for item in candidate.diagnostics if item not in before_diagnostics
                and item.step in (None, key.step) and item.variant in (None, key.variant)
            ), None)
            raise ProjectFolderInvalid(diagnostic.detail if diagnostic else "The edit does not resolve for every selected run.")
        old, before, _ = _pairing(plan, key, bundle)
        new, after, _ = _pairing(candidate, key, bundle)
        if expression is not None:
            _, index, _ = json_pointer_parts(change.edits[0].path)
            if int(index) >= len(old.objective.terms) or old.objective.terms[int(index)].id != expression.term:
                raise ProjectFolderInvalid("The selected runs have different Objective term orders.")
        introduced = tuple(problem for problem in after.problems if problem not in before.problems)
        if introduced and not evaluate_field and not choosing:
            return StoredEdit([]), introduced
        if evaluate_field and introduced and key in current:
            return StoredEdit([]), introduced
        if isinstance(change, StudioTrialDocumentEdit):
            old_value = old.task_source if change.kind == "task" else old.objective
            new_value = new.task_source if change.kind == "task" else new.objective
            raw = old_value.model_dump(mode="json", exclude_unset=True)
            requested = change
            if evaluate_field:
                requested = _present_ops(raw, change)
            if plan.steps[key.variant, key.step].kind == "train":
                _, raw = _changed_value(old, plan.selected_definitions[key], change.kind)
                if (order := _reordered(raw.get("terms", []), change)) is not None:
                    requested = change.model_copy(update={"edits": [change.edits[0].model_copy(update={"value": order})]})
            expected = type(old_value).model_validate(_edit_value(raw, requested))
            if expected != new_value:
                raise ProjectFolderInvalid("A selected value is controlled by a different authored source.")
        elif isinstance(change, StudioTrialSelectDocument):
            step = candidate.steps[key.variant, key.step]
            definition = candidate.selected_definitions[key]
            field = _task_field(new.task_source) if change.role == "task" else change.role
            path = ("payload.settings" if change.role == "training"
                    else f"payload.trial_documents.{field}" if step.kind == "train" or change.role == "model"
                    else _evaluation_task_path(definition))
            if definition.document_ids.get(path) != change.document:
                raise ProjectFolderInvalid(f"A selected {change.role} is controlled by a different authored source.")
        elif isinstance(change, StudioTrialBindingEdit):
            binding = next((row for row in _input_rows(new, after) if row.address == change.input), None)
            if binding is None:
                raise ProjectFolderInvalid("A selected run has no addressed input.")
            expected = dict(old.input_overrides)
            if change.signal in (None, binding.name):
                expected.pop(binding.name, None)
            else:
                expected[binding.name] = change.signal
            if expected != new.input_overrides:
                raise ProjectFolderInvalid("A selected input is controlled by a different authored source.")
        if not evaluate_field and not choosing and key in previous and key not in current:
            diagnostic = next((
                item for item in candidate.diagnostics if item not in before_diagnostics
                and item.step == key.step and item.variant == key.variant and item.row == key.row
            ), None)
            raise ProjectFolderInvalid(diagnostic.detail if diagnostic else "The edit removes a planned request.")
    refuse_unreached(plan, candidate, reach)
    return stored, ()


def _objective_names(plan, key):
    definition = plan.selected_definitions.get(key)
    pairing = definition.payload.get("trial_documents") if definition else None
    objective = pairing.get("objective") if isinstance(pairing, dict) else None
    terms = objective.get("terms", []) if isinstance(objective, dict) else []
    return {term.get("id"): term.get("name") for term in terms if isinstance(term, dict) and term.get("name")}
def _trial_field(path, site: Site, expression, functions, swept: _WeightSweep | None, row) -> StudioTrialField:
    """A field with its text and structure, where its value names only what is in scope.

    A Weight the step can sweep, or sweeps, carries that sweep; no other field takes the Sweep kind.
    """
    shown = expression is not None and references(expression) <= {item.id for item in site.names}
    projection = project_site(site, functions)
    pointed = pointer(tuple(map(str, path)))
    label = str(path[-1]).replace("_", " ").capitalize()
    if swept is not None:
        view = project_numeric_field(
            StudioTrialField, swept.field, swept.axis, functions, row,
            expression.value if isinstance(expression, LiteralValue) else None, path=pointed,
        )
        if view.sweep is not None or view.can_sweep:
            return view.model_copy(update={"site": projection})
    projection = projection.model_copy(update={"kinds": tuple(kind for kind in projection.kinds if kind != FieldKind.SWEEP)})
    return StudioTrialField(
        path=pointed, label=label, site=projection,
        text=print_field(expression, site) if shown else "",
        structure=value_structure(expression, site, functions) if shown else None,
    )


def trial_authoring(
    folders: ProjectFolderService, project_id: str, body: StudioTrialAuthoringRequest,
    bundle: ApplicationRegistryBundle,
) -> StudioTrialAuthoring:
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    snapshot = folders.read(project_id, revision=body.revision)
    settled = folders.read_plan(project_id, revision=snapshot.revision, source_runs=(key,))
    plan = settled.plan
    definition = plan.selected_definitions.get(key)
    if definition is None:
        raise ProjectFolderInvalid("The selected run has no resolved trial definition.")
    step = plan.steps.get((key.variant, key.step))
    if step is None or step.kind not in ("train", "evaluate"):
        raise ProjectFolderInvalid("The selected run has no authored trial pairing.")
    try:
        documents, projected, graph = _pairing(plan, key, bundle)
        stored, refused = StoredEdit([]), ()
        refusal = None
        if body.change is not None:
            stored, refused = _prepare_edits(
                body, folders, project_id, snapshot.project, settled, key, bundle, documents, projected,
            )
            if isinstance(refused, StudioSweptParameterMissing):
                refusal, refused = refused, ()
    except (ValueError, TypeError, KeyError) as error:
        raise ProjectFolderInvalid(str(error)) from error
    paired = projected.pair
    task = documents.task_source
    task_path = (f"payload.trial_documents.{_task_field(task)}" if step.kind == "train"
                 else _evaluation_task_path(definition))
    model_document = definition.document_ids.get("payload.trial_documents.model")
    task_source = None
    task_field_sources = {}
    if step.kind == "evaluate" and (upstream := evaluate_subject(definition)):
        producer = next((row for row in plan.rows if row.key == upstream[0]), None)
        if producer is not None:
            model_document = producer.document_ids.get("payload.trial_documents.model")
        task_source = evaluate_value_source(upstream[1])
        for target, origin_path in projected.task_origins.items():
            if target[0] != "task":
                continue
            origin = definition.sources.get(f"/payload{origin_path}/value") or definition.sources.get(
                f"/payload{origin_path}/op",
            )
            if origin is not None:
                source = evaluate_value_source(origin, evaluate_step=key.step)
                if target == ("task",):
                    task_source = source
                else:
                    task_field_sources[target[-1]] = source
    links = dict(paired.task.interface.links) if paired else {}
    row = next((item for item in plan.rows if item.key == key), None)
    expressions = {
        ("objective", "terms", term.id, field): getattr(term, field)
        for term in documents.objective.terms for field in ("measure", "weight")
    } if step.kind == "train" else {}
    return StudioTrialAuthoring(
        revision=snapshot.revision, run=body.run, model=model_document,
        task_document=definition.document_ids.get(task_path) if task_path else None,
        objective_document=definition.document_ids.get("payload.trial_documents.objective"),
        task=task, objective=documents.objective if step.kind == "train" else None,
        task_changed=changed_from_base(snapshot.project, _whole_document(definition, task_path) if task_path else None),
        objective_changed=changed_from_base(snapshot.project, _whole_document(
            definition, "payload.trial_documents.objective",
        )) if step.kind == "train" else [],
        task_source=task_source, task_field_sources=task_field_sources,
        task_types=[StudioTaskType(type=key, label=entry.label, parameter_schema=entry.parameters.model_json_schema())
                    for key in bundle.python_tasks.keys() if (entry := bundle.python_tasks.resolve(key))],
        variables=list(projected.variables), signals=list(projected.signals), epochs=list(projected.epochs),
        period_seconds=documents.model.step_seconds,
        n_steps=paired.task.n_steps if paired else None,
        tasks=[
            StudioTaskChoice(document=name, problems=list(_project(
                _with_task(documents, other), bundle, show_objective=step.kind == "train", graph_override=graph,
                objective_use=ObjectiveUse.INHERITED if step.kind == "evaluate" else ObjectiveUse.TRAIN,
            ).problems))
            for name in sorted(snapshot.project.documents)
            if (other := _selected_task(snapshot.project, name)) is not None
        ],
        inputs=_input_rows(documents, projected),
        outputs=[StudioTrialOutput(address=ModelAddress.from_text(value.id), value=value)
                 for value in paired.model.outputs] if paired else [],
        resets=[StudioTrialReset(address=field.address, type=field.type, finest=field.finest,
                                variable=links.get(field.address.to_text()))
                for field in paired.model.reset.fields] if paired else [],
        fields=[_trial_field(
            path, site, expressions.get(path), bundle.expression_functions,
            _weight_sweep(snapshot.project, definition, key.step, path[2])
            if step.kind == "train" and path[:2] == ("objective", "terms") and path[-1] == "weight"
            and paired is not None else None,
            row,
        ) for path, site in projected.fields.items()],
        problems=[*projected.problems, *refused], files=stored.files, created=list(stored.created), refusal=refusal,
    )
