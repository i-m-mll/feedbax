"""Project and prepare edits for one saved trial pairing."""


from copy import deepcopy
from dataclasses import dataclass, field, replace

from feedbax.execution.trial_resolution import TrialPair, compile_unpaired_task, pair_trial_source
from feedbax.contracts.expression_values import (
    Level, Problem, ProblemCode, ScopedValue, Site, SweepList, SweepRange, SweepSample, UnresolvedValue,
    ValueBase, ValueType,
)
from feedbax.tasks.document import TrialSource, zero
from feedbax.tasks.registry import resolve_python_task_instance
from feedbax.tasks.structure import restructure
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import stored_parts
from feedbax.contracts.studio_scope import StoreKind
from feedbax.compiler.model_reset import ModelResetField
from feedbax.compiler.model_signals import model_measure_scope
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import AxisValueRuns, ListedRuns, VariantRuns
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS, MODEL_DIFFERENCE_BATCHES
from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.contracts.project import (
    LoadedModelDocument, RemoveProjectValue, SetProjectValue, apply_document_edits, json_pointer_parts, member_name,
)
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolutionCode, compose_document, resolve_step
from feedbax.planner.project_plan import EVALUATION_TASK, RunKey, compile_project, own_task_variation
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import DocumentTrainingConfig, DocumentTrainingRequestSpec, EvaluationRequestSpec
from feedbax.contracts.rollout import ROLLOUT_TYPE, authored_rollout_settings
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.contracts.studio_place import task_start_path
from feedbax.studio.authoring import training_document_choice
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.contracts.studio_trials import (
    StudioTaskChoice, StudioTaskType, StudioTrialAuthoring, StudioTrialAuthoringRequest, StudioTrialBindingEdit,
    StudioTrialDocumentEdit, StudioTrialDocumentReset, StudioTrialEpoch, StudioTrialEpochInstance,
    StudioTrialExpressionEdit, StudioTrialField,
    StudioTrialInput, StudioTrialObjectiveSource,
    StudioTrialOutput, StudioTrialRelate, StudioTrialReset, StudioTrialSelectDocument, StudioTrialStepScope,
    StudioTrialStructureEdit,
)
from feedbax.contracts.task_document import (
    PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID, PythonTaskSelection, Rest, TaskDocument,
)
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument, ObjectiveTerm
from feedbax.contracts.training import TrialDocuments
from feedbax.contracts.trial_bindings import selected_signal_id
from feedbax.contracts.expression_check import references
from feedbax.contracts.expression_site import FieldKind, project_site, read_field
from feedbax.contracts.expression_text import parse, print_field, resolve_expression
from feedbax.studio.value_structure import value_structure
from feedbax.objectives.document import ObjectiveExecutor, ObjectiveUse, compile_objective
from feedbax.application.bundle import ApplicationRegistryBundle
from feedbax.web.services.authored_scope import edited_authored_rows
from feedbax.web.services.evaluate_sources import evaluate_subject, evaluate_value_source
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService
from feedbax.web.services.set_authoring import trial_set_fields
from feedbax.web.services.store_choice import EditStore, all_readers_store, choose_store
from feedbax.studio.selection import StoredEdit
from feedbax.studio.numeric_fields import project_numeric_field
from feedbax.web.services.trial_fields import (
    TrialFieldSweep, at, held_expression_file, index_pointer, numeric_nodes, routes_to_sweep, site_expression, trial_field_sweep,
    trial_field_sweep_edit,
)
from feedbax.web.services.trial_scope import (
    authored_payload, base_value_edits, changed_from_base, document_write, evaluation_document_write, lit_steps, preview_stored, scoped_file,
    reached_steps, selected_runs, source_value_file, variant_view,
)


@dataclass(frozen=True)
class _InitialTrialDocuments:
    model: ModelDocument
    task_source: TaskDocument | PythonTaskSelection
    objective: ObjectiveDocument | None
    input_bindings: dict[str, str | None]

    @property
    def task(self) -> TaskDocument | None:
        return self.task_source if isinstance(self.task_source, TaskDocument) else None


@dataclass(frozen=True)
class _Projection:
    pair: TrialPair | None
    variables: tuple[ScopedValue, ...]
    signals: tuple[ScopedValue, ...]
    epochs: tuple[StudioTrialEpoch, ...]
    fields: dict[tuple[str | int, ...], Site]
    problems: tuple[Problem, ...]
    task_origins: dict[tuple[str, ...], str] = field(default_factory=dict)
    resets: tuple[ModelResetField, ...] = ()
    partial_task: TrialSource | None = None
    epoch_instances: tuple[StudioTrialEpochInstance, ...] = ()
    output_paths: dict[str, tuple[str, ...]] = field(default_factory=dict)


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
    resets = []

    task_sites = {}
    paired = pair_trial_source(
        documents.model, task, bundle.components, bundle.expression_functions,
        python_tasks=bundle.python_tasks, input_bindings=documents.input_bindings,
        task_sites=task_sites, graph_override=graph_override, reset_fields=resets,
    )
    partial = None
    if isinstance(paired, tuple):
        problems = paired
        paired = None
        if all(problem.path[:1] == ("model",) for problem in problems):
            preview = compile_unpaired_task(
                documents.model, task, bundle.expression_functions, python_tasks=bundle.python_tasks, sites=task_sites,
            )
            if isinstance(preview, tuple):
                problems += preview
            else:
                partial = preview
    else:
        problems = paired.problems
    fields.update({("task", *path): site for path, site in task_sites.items()})
    if paired is None and partial is None:
        if show_objective and documents.objective is not None:
            fields.update(_unpaired_objective_sites(documents, signals))
        return _Projection(None, variables, signals, epochs, fields, problems, resets=tuple(resets))
    source = paired.task if paired is not None else partial.task
    variables, signals = source.interface.variables, source.interface.signals
    names = {epoch.id: epoch.name for epoch in epochs}
    epoch_instances = tuple(
        StudioTrialEpochInstance(id=epoch, name=names.get(epoch, epoch), steps=steps, occurrence=occurrence)
        for occurrence, (epoch, steps) in enumerate(zip(source.epochs, source.step_counts, strict=True))
    )
    epochs = tuple(
        StudioTrialEpoch(id=instance.id, name=instance.name, steps=instance.steps)
        for instance in {item.id: item for item in epoch_instances}.values()
    )
    output_paths = {}
    if documents.objective is not None:
        if paired is not None:
            measures = model_measure_scope(documents.model, paired.model.signals)
            outputs, output_paths = measures.names, measures.paths
        else:
            outputs = partial.outputs
        objective_sites = {} if show_objective else None
        objective = compile_objective(
            documents.objective, (*outputs, *variables, *signals), bundle.expression_functions,
            task=source, sites=objective_sites, use=objective_use,
        )
        if objective_sites:
            fields.update({("objective", *path): site for path, site in objective_sites.items()})
        if not isinstance(objective, ObjectiveExecutor):
            problems += tuple(problem.model_copy(update={"path": ("objective", *problem.path)})
                              for problem in objective)
    return _Projection(
        paired, variables, signals, epochs, fields, problems, resets=tuple(resets),
        partial_task=source if paired is None else None, epoch_instances=epoch_instances, output_paths=output_paths,
    )


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
    signals = {signal.name: signal.id for signal in projection.signals}
    return [
        StudioTrialInput(
            id=port.id, address=address, name=port.name, type=resolved.get(address.to_text()),
            required=port.default is None,
            signal_id=selected_signal_id(port.id, port.name, signals, documents.input_bindings),
            explicit=port.id in documents.input_bindings,
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


def _with_task(documents, task, *, input_bindings: dict[str, str | None] | None = None):
    bindings = documents.input_bindings if input_bindings is None else input_bindings
    if isinstance(documents, _InitialTrialDocuments):
        return replace(documents, task_source=task, input_bindings=bindings)
    if isinstance(task, PythonTaskSelection):
        return documents.model_copy(update={"task": None, "python_task": task, "input_bindings": bindings})
    return documents.model_copy(update={"task": task, "python_task": None, "input_bindings": bindings})


def _field_document(documents, role):
    """The document of ``role`` whose fields the view edits as expressions, if the step reads one."""
    if role == "objective":
        return documents.objective
    return documents.task_source if role == "task" and isinstance(documents.task_source, TaskDocument) else None


def _expression_edit(change, documents, projection, bundle):
    role, *rest = json_pointer_parts(change.path)
    site = projection.fields.get((role, *rest))
    document = _field_document(documents, role)
    if site is None or document is None or index_pointer(document.model_dump(mode="json"), tuple(rest[:-1])) is None:
        raise ProjectFolderInvalid("The selected field is unavailable.")
    one = (Problem(code=ProblemCode.STRUCTURE, reason="This field takes one expression.", path=(role, *rest)),)
    if change.edit.kind != "text" or change.edit.value_id is not None:
        return one
    text = change.edit.text
    end = _end_of(document, role, rest)
    raw = document.model_dump(mode="json", exclude_unset=True)
    if end is not None and text.strip() == "rest":
        return StudioTrialDocumentEdit(kind=role, edits=_end_edits(raw, end, "rest", {}))
    parsed = parse(text, site, bundle.expression_functions)
    value = UnresolvedValue(text=text) if isinstance(parsed, tuple) else parsed.value
    if isinstance(value, SweepList | SweepRange | SweepSample):
        return one
    stored_value = value.model_dump(mode="json")
    if end is not None and isinstance(at(document, tuple(rest[:-1])), Rest):
        return StudioTrialDocumentEdit(kind=role, edits=_end_edits(
            raw, end, "length", {"length": stored_value},
        ))
    stored = index_pointer(raw, tuple(rest))
    path = stored or index_pointer(raw, tuple(rest[:-1])) + pointer(tuple(rest[-1:]))
    return StudioTrialDocumentEdit(kind=role, edits=[SetProjectValue(
        op="replace" if stored else "add", path=path, value=stored_value,
    )])


def _end_of(document, role, rest) -> str | None:
    """The pointer of the End whose length ``rest`` addresses in the task document, which ``rest`` makes Rest and a length
    makes a length again; None for any other field."""
    if role != "task" or len(rest) != 4 or rest[0] != "epochs" or tuple(rest[2:]) != ("end", "length"):
        return None
    return index_pointer(document.model_dump(mode="json"), tuple(rest[:-1]))


def _end_edits(raw, end: str, kind: str, fields: dict) -> list:
    """The edits that make the End at ``end`` of ``kind`` with ``fields``, each under the End so that it stays the
    field's edit: its kind, then each field it gains or loses."""
    held = raw
    for part in json_pointer_parts(end):
        held = held[int(part)] if isinstance(held, list) else held[part]
    edits = [SetProjectValue(op="replace" if "kind" in held else "add", path=f"{end}/kind", value=kind)]
    edits += [SetProjectValue(op="replace" if key in held else "add", path=f"{end}/{key}", value=value)
              for key, value in fields.items()]
    edits += [RemoveProjectValue(op="remove", path=f"{end}/{key}") for key in held if key != "kind" and key not in fields]
    return edits


def _field_label(path) -> str:
    return str(path[-1]).replace("_", " ").capitalize()


def _swept_document(plan, key, definition, role) -> str | None:
    """The document whose fields at ``role`` the run reads whole: a Train's own, and an Evaluate's own Task or else its
    subject's."""
    if plan.steps[key.variant, key.step].kind == "train":
        return _whole_document(definition, f"payload.trial_documents.{role}")
    if role != "task":
        return None
    if (own := _whole_document(definition, _evaluation_task_path(definition))) is not None:
        return own
    upstream = evaluate_subject(definition)
    producer = next((row for row in plan.rows if upstream and row.key == upstream[0]), None)
    return producer.document_ids.get("payload.trial_documents.task") if producer is not None else None


def _field_sweep(project, plan, key, definition, path, site, label) -> TrialFieldSweep | None:
    """The number at ``path`` of a document the step reads whole, when the step can sweep it."""
    role, *rest = path
    parse = {"objective": ObjectiveDocument.model_validate, "task": TaskDocument.model_validate}.get(role)
    if parse is None:
        return None
    document = _swept_document(plan, key, definition, role)
    return trial_field_sweep(project, key.step, document, role, tuple(map(str, rest)), site, label, parse)


def _task_field(task):
    return "python_task" if isinstance(task, PythonTaskSelection) else "task"


def _edit_value(value, change):
    raw = deepcopy(value)
    apply_document_edits(raw, change.edits)
    return raw


def _changed_value(documents, definition, kind):
    field = _task_field(documents.task_source) if kind == "task" else "objective"
    if isinstance(documents, _InitialTrialDocuments):
        return field, definition.input_documents[kind].payload
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
        row = next((row for row in _input_rows(documents, projection) if row.id == change.input_id), None)
        if row is None:
            raise ProjectFolderInvalid("The ID is not a declared Model input for this row.")
        if "input_bindings" in raw:
            return [_override(
                f"trial_documents.input_bindings.{row.id}", "replace" if row.explicit else "add", change.signal_id,
            )]
        return [_override("trial_documents.input_bindings", "add", {row.id: change.signal_id})]
    field, value = _changed_value(documents, definition, change.kind)
    return _value_writes(f"trial_documents.{field}", value, change)


_DOCUMENT_SCHEMAS = {"model": MODEL_DOCUMENT_SCHEMA_ID, "objective": OBJECTIVE_DOCUMENT_SCHEMA_ID}


def _chosen_document(project, document, role):
    entry = project.documents.get(document)
    if entry is None or entry.schema_id != _DOCUMENT_SCHEMAS[role]:
        raise ProjectFolderInvalid(
            f"{member_name(project.documents, document)} is not a{'n' if role == 'objective' else ''} {role.title()} document."
        )


def _chosen_task(project, document):
    task = _selected_task(project, document)
    if task is None:
        raise ProjectFolderInvalid(f"{member_name(project.documents, document)} is not a Task document.")
    return task


def _evaluation_task_path(definition):
    if "subject" in definition.input_documents:
        return "inputs.task"
    entry = _authored_task_entry(definition.payload.get("params", {}).get("variation"))
    return f"payload.params.variation.{entry}.value" if entry is not None else EVALUATION_TASK


def _initial_trial_documents(definition, bundle) -> _InitialTrialDocuments:
    subject = definition.input_documents["subject"]
    if subject.schema_id == LOADED_MODEL_SCHEMA_ID:
        model = definition.pinned_model
        if model is None:
            raise ProjectFolderInvalid("The selected Loaded Model is unavailable.")
    else:
        model = ModelDocument.model_validate(subject.payload)
    task = definition.input_documents["task"]
    if task.schema_id == PYTHON_TASK_SCHEMA_ID:
        source = PythonTaskSelection.model_validate(task.payload)
    else:
        source = TaskDocument.model_validate(task.payload)
    objective = definition.input_documents.get("objective")
    differences = MODEL_DIFFERENCE_BATCHES.validate_python(definition.payload.get(MODEL_ATTACHMENTS, {}))
    if differences:
        model = resolve_effective_model(
            model, bundle.components, tuple(item for batch in differences.values() for item in batch),
            functions=bundle.expression_functions,
        )
    params = authored_rollout_settings(definition.payload.get("params"))
    return _InitialTrialDocuments(
        model, source, ObjectiveDocument.model_validate(objective.payload) if objective is not None else None,
        params.input_bindings,
    )


def _evaluate_documents(plan, key, definition, bundle):
    upstream = evaluate_subject(definition)
    authored = None
    if upstream:
        subject = next((row for row in plan.runs if row.key == upstream[0]), None)
        raw = plan.requests.get(subject.request_id) if subject else None
        if raw is None:
            raise ProjectFolderInvalid("The selected Evaluate has no planned Train subject.")
        request = decode_request(raw, plan.documents.__getitem__).spec
        if not isinstance(request, DocumentTrainingRequestSpec):
            raise ProjectFolderInvalid("The selected Evaluate has no managed Train pairing.")
        documents = request.trial_documents
        if subject.train is not None and isinstance(subject.train.task, TaskDocument):
            authored = subject.train
            documents = documents.model_copy(update={"task": authored.task, "input_bindings": authored.input_bindings})
    elif "subject" in definition.input_documents:
        documents = _initial_trial_documents(definition, bundle)
    else:
        raise ProjectFolderInvalid("The selected Evaluate has no authored trial pairing.")
    objective_use = ObjectiveUse.INHERITED if upstream else ObjectiveUse.TRAIN
    projection = _project(documents, bundle, objective_use=objective_use)
    if projection.pair is None:
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
        authored_bindings = params.input_bindings
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
                params = effective_params
    except ValueError as error:
        return documents, _Projection(
            projection.pair, projection.variables, projection.signals, projection.epochs,
            projection.fields, (*projection.problems, Problem(
                code=ProblemCode.STRUCTURE, path=("variation",), reason=str(error),
            )),
        ), None
    documents = _with_task(
        documents, task, input_bindings=documents.input_bindings | authored_bindings | params.input_bindings,
    )
    paired = _project(documents, bundle, graph_override=graph, objective_use=objective_use)
    return documents, replace(paired, task_origins=variant.edit_origins), graph


def _definition(plan, key):
    """The run's definition; a run whose own definition is refused is refused with its problem."""
    definition = plan.selected_definitions.get(key)
    if definition is None:
        problem = next((
            item for item in plan.diagnostics
            if item.step == key.step and item.row == key.row and item.variant in (None, key.variant)
        ), None)
        if problem is not None:
            raise ProjectFolderInvalid(problem.detail)
        raise ProjectFolderInvalid("The selected run no longer belongs to this step.")
    return definition


def _pairing(plan, key, bundle, *, definition=None):
    if definition is None:
        definition = _definition(plan, key)
    step = plan.steps[key.variant, key.step]
    if step.kind == "train":
        documents = TrialDocuments.model_validate(definition.payload.get("trial_documents"))
        run = next((item for item in plan.runs if item.key == key), None)
        if run is not None and run.train is not None and run.train.task != documents.task_source:
            encoded = plan.requests.get(run.request_id)
            if encoded is not None:
                spec = decode_request(encoded, plan.documents.__getitem__).spec
                if isinstance(spec, DocumentTrainingRequestSpec) and isinstance(run.train.task, TaskDocument):
                    documents = spec.trial_documents.model_copy(update={"task": run.train.task})
        return documents, _project(documents, bundle), None
    if step.kind == "evaluate":
        return _evaluate_documents(plan, key, definition, bundle)
    raise ProjectFolderInvalid("The selected run has no authored trial pairing.")


def _prepare_edits(body, folders, project_id, project, settled, key, bundle, definition, documents, projected, *, change):
    scope = body.scope
    plan = settled.plan
    step = plan.steps[key.variant, key.step]
    expression = change if isinstance(change, StudioTrialExpressionEdit) else None
    if expression is not None:
        path = json_pointer_parts(expression.path)
        holder = next((path[:end] for end in range(len(path), 0, -1) if path[:end] in projected.fields), path)
        site = projected.fields.get(holder)
        swept = _field_sweep(project, plan, key, definition, path, site, _field_label(holder)) if (
            site is not None and projected.pair is not None
        ) else None
        if holder != path and swept is None:
            raise ProjectFolderInvalid("The selected field is unavailable.")
        if swept is not None and (holder != path or routes_to_sweep(expression.edit, swept, site, bundle.expression_functions)):
            readers = lambda name: plan.document_users(key.variant, name)  # noqa: E731
            shown = _swept_document(plan, key, definition, swept.role)
            entry = compose_document(project.documents.get, shown)
            if isinstance(entry, ProjectDiagnostic):
                raise ProjectFolderInvalid(entry.detail)
            variant = None if key.variant == plan.variants[0] else key.variant
            made = trial_field_sweep_edit(
                expression.edit, swept, key.step, entry.payload, bundle.expression_functions,
                (lambda edits: document_write(project, shown, edits, {key.step}, readers)) if step.kind == "train"
                else lambda edits: evaluation_document_write(project, plan, shown, edits, {key.step}, "task", variant),
            )
            if isinstance(made, tuple):
                return StoredEdit([]), made, None
            selected = selected_runs(plan, key, scope or StudioTrialStepScope(kind="step"))
            return _validate_files(
                body, project, settled, key, selected, folders, bundle,
                [("whole_step", "Whole Step", key.step, made)],
            )
        change = _expression_edit(expression, documents, projected, bundle)
        if not isinstance(change, StudioTrialDocumentEdit):
            return StoredEdit([]), change, None
    structural = isinstance(change, StudioTrialStructureEdit)
    if structural:
        change = _structure_edit(change, documents, definition)
        if not isinstance(change, StudioTrialDocumentEdit):
            return StoredEdit([]), change, None
    document = None
    document_path = None
    edited_role = (change.document if isinstance(change, StudioTrialDocumentReset)
                   else change.kind if isinstance(change, StudioTrialDocumentEdit) else None)
    if isinstance(documents, _InitialTrialDocuments) and edited_role is not None:
        document_path = f"inputs.{edited_role}"
    elif step.kind == "train" and edited_role is not None:
        field, _ = _changed_value(documents, definition, edited_role)
        document_path = f"payload.trial_documents.{field}"
    if document_path is not None:
        document = _whole_document(definition, document_path)
    if isinstance(change, StudioTrialDocumentReset):
        restored = document if document_path is not None else (
            _whole_document(definition, _evaluation_task_path(definition)) if change.document == "task" else None
        )
        if restored is None:
            raise ProjectFolderInvalid("Reset needs a value the step reads from a document.")
        change = StudioTrialDocumentEdit(kind=change.document, edits=base_value_edits(project, restored, change.paths))

    variant = None if key.variant == plan.variants[0] else key.variant

    def readers(name):
        return plan.document_users(key.variant, name)

    scope = scope or StudioTrialStepScope(kind="step")
    whole = isinstance(scope, StudioTrialStepScope)
    chosen = set(scope.chosen_steps) if whole and scope.chosen_steps else {key.step}
    view = variant_view(plan, project, variant if whole else None)
    users = set(lit_steps(view.documents, readers, document)) if document else set()
    if document and whole and (key.step not in chosen or not chosen <= users):
        raise ProjectFolderInvalid("The chosen steps no longer match this document.")
    selected = selected_runs(plan, key, scope, chosen_steps=chosen)
    evaluate_field = step.kind == "evaluate" and isinstance(change, StudioTrialDocumentEdit) and change.kind == "task"

    def store_for(target, members, *, step_id=key.step, selected_view=view):
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
                if isinstance(target, StudioTrialStepScope | VariantRuns):
                    slots = (authored_payload(selected_view, step_id) or {}).get("trial_documents")
                    writes = [
                        _override(f"trial_documents.{role}", "replace" if role in slots else "add", reference)
                        if isinstance(slots, dict) else _override("trial_documents", "add", {role: reference}),
                    ]
                else:
                    writes = _task_selection_writes(definition, reference)
            elif evaluate_field and isinstance(target, StudioTrialStepScope | VariantRuns):
                return _evaluate_task_write(project, plan, key, members, change, variant)
            elif evaluate_field:
                return _evaluate_field_files(project, key, target, change, documents)
            else:
                raise ProjectFolderInvalid("Evaluate authoring edits the task variation.")
        elif isinstance(change, StudioTrialSelectDocument) and change.role == "training":
            writes = training_document_choice(project, step_id, bundle.training_methods, change.document)
        else:
            writes = _train_writes(change, project, definition, documents, projected)
        return scoped_file(
            project, step_id, target, writes, chosen=members, users=readers, view=selected_view,
        ) if writes else StoredEdit([])

    stored_scope = VariantRuns(kind="variant", variant=variant) if whole and variant else scope
    if document is not None and whole:
        shown = documents.objective if change.kind == "objective" else documents.task_source
        raw = shown.model_dump(mode="json", exclude_unset=True)

        def write_for(members):
            return document_write(
                project, document, change.edits, members, readers, variant, view=view,
                adapt=(lambda target: _objective_edits(raw, target, change)) if change.kind == "objective" else None,
            )

        candidates = [("all_readers", "All Steps", document, write_for(users))]
        if chosen != users:
            candidates.append(("selected_readers", "Only This Step", document, write_for(chosen)))
    else:
        stored = store_for(stored_scope, chosen)
        kind, label = (
            ("sweep_value", "Sweep Value") if isinstance(scope, AxisValueRuns)
            else ("row_exception", "Selected Runs") if isinstance(scope, ListedRuns)
            else ("variant", "Variant") if variant
            else ("whole_step", "Whole Step")
        )
        candidates = [(kind, label, repr(stored_scope), stored)]

    def controlling(plan, missed):
        if step.kind != "train":
            return
        writes = _train_writes(change, project, definition, documents, projected)
        if not writes or any(write.op != "replace" for write in writes):
            return
        for run in sorted(missed, key=lambda item: (item.variant, item.step, item.row)):
            row = plan.selected_definitions.get(run)
            if row is None:
                continue
            origins = []
            for write in writes:
                indexed = index_pointer(row.payload, tuple(write.path.split(".")))
                origin = row.sources.get(f"/payload{indexed}") if indexed else None
                if origin is None:
                    break
                current = at(row.payload, json_pointer_parts(indexed))
                candidate = source_value_file(project, origin, current, write.value)
                if candidate is None:
                    break
                origins.append((origin, candidate))
            if len(origins) != len(writes):
                continue
            source = origins[0][0]
            if any(_source_store(origin) != _source_store(source) for origin, _ in origins[1:]):
                continue
            identity = repr(_source_store(source))
            label: dict[str, tuple[StoreKind, str]] = {
                "sweep_value": ("sweep_value", "Sweep Value"),
                "exception": ("row_exception", "Row Exception"),
                "variant": ("variant", "Variant"),
            }
            kind, name = label[source.kind]
            yield kind, name, identity, StoredEdit([]).then(file for _, file in origins)

    preserve = None
    if isinstance(change, StudioTrialDocumentEdit) and change.kind == "task" and any(
        (parts := json_pointer_parts(edit.path))[:1] == ("signals",)
        and (len(parts) <= 2 or parts[2:] == ("name",))
        for edit in change.edits
    ):
        from feedbax.web.services.trial_bindings import preserve_renamed_bindings

        def preserve(stored):
            return preserve_renamed_bindings(folders, project_id, body.revision, project, stored)

    return _validate_files(
        body, project, settled, key, selected, folders, bundle, candidates,
        preserve=preserve, controlling=controlling if isinstance(change, StudioTrialDocumentEdit | StudioTrialBindingEdit) else None,
        other_runs=lambda rows: store_for(ListedRuns(kind="rows", rows=tuple(sorted(row.row for row in rows))), chosen),
    )


def _source_store(origin: ValueSource):
    return origin.kind, origin.step, origin.name, origin.variant, origin.axis, origin.value


def _structure_edit(change, documents, definition) -> StudioTrialDocumentEdit | tuple[Problem, ...]:
    """The task document edit that makes structural changes in order: each top-level field they change, written whole.
    One refused change refuses them all."""
    task = documents.task_source
    if not isinstance(task, TaskDocument):
        raise ProjectFolderInvalid("A Python task has no structure to change.")
    changed = task
    for each in change.changes:
        changed = restructure(changed, each)
        if isinstance(changed, Problem):
            return (changed,)
    _, raw = _changed_value(documents, definition, "task")
    before, after = task.model_dump(mode="json"), changed.model_dump(mode="json")
    edits = [
        SetProjectValue(op="replace" if field in raw else "add", path=pointer((field,)), value=after.get(field, []))
        for field in dict.fromkeys((*after, *before)) if after.get(field) != before.get(field)
    ]
    if not edits:
        raise ProjectFolderInvalid("The change leaves the task as it is.")
    return StudioTrialDocumentEdit(kind="task", edits=edits)


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


def _objective_edits(shown, target, change):
    """Translate a concrete term pointer through its authored id when term order differs."""
    if (order := _reordered(shown.get("terms", []), change)) is not None and list(
        map(ObjectiveTerm.model_validate, order)
    ) == list(map(ObjectiveTerm.model_validate, change.edits[0].value)):
        ordered = _reordered(target.get("terms", []), change)
        return [] if ordered is None else [change.edits[0].model_copy(update={"value": ordered})]
    edits = []
    for edit in change.edits:
        parts = json_pointer_parts(edit.path)
        if parts[:1] == ("terms",) and len(parts) >= 2 and parts[1].isdigit() and (
            len(parts) > 2 or edit.op != "add"
        ):
            term_id = shown["terms"][int(parts[1])]["id"]
            path = index_pointer(target, ("terms", term_id, *parts[2:-1])) if len(parts) > 2 else (
                index_pointer(target, ("terms", term_id))
            )
            if path is None:
                continue
            if len(parts) > 2:
                path += pointer(parts[-1:])
            edit = edit.model_copy(update={"path": path})
        edits.append(edit)
    return edits


def _whole_document(definition, path: str) -> str | None:
    """The document the step reads whole at ``path``, its payload."""
    return None if definition.document_pointers.get(path) else definition.document_ids.get(path)


def _evaluate_task_write(project, plan, key, chosen, change, variant: str | None) -> StoredEdit:
    """Store an Evaluate's Task edit in a document its chosen steps read."""
    row = plan.row(key)
    document = row.document_ids.get(EVALUATION_TASK) or row.document_ids.get("payload.trial_documents.python_task")
    if document is None:
        raise ProjectFolderInvalid("The selected Evaluate reads no Task document.")
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


def _held_values(authored, definition, documents, projection, kind):
    """Each unresolved field for which this run identifies a writable authored owner."""
    raw = {}
    for path, site in projection.fields.items():
        role, *rest = map(str, path)
        document = _field_document(documents, role)
        if document is None or not isinstance(value := site_expression(document, tuple(rest)), UnresolvedValue):
            continue
        if role not in raw:
            raw[role] = document.model_dump(mode="json", exclude_unset=True)
        indexed = index_pointer(raw[role], tuple(rest))
        if indexed is None:
            continue
        if kind == "train":
            source_path = f"/payload/trial_documents/{role}{indexed}/text"
        else:
            root = _evaluation_task_path(definition)
            source_path = pointer(tuple(root.split("."))) + indexed + "/text"
        source: ValueSource | None = definition.sources.get(source_path)
        if source is not None and (owner := held_expression_file(authored, source, value)) is not None:
            yield owner, value, site


def _resolved_value_files(authored, candidate, pairings, compile_sources, bundle):
    """Store a parsed value in its original file when its readers agree on the references."""
    functions = bundle.expression_functions
    proposals = {}
    for key, (documents, projection) in pairings.items():
        kind = candidate.steps[key.variant, key.step].kind
        for owner, held, site in _held_values(authored, candidate.selected_definitions[key], documents, projection, kind):
            key_and_path = (owner[0].kind, owner[0].id, owner[1])
            parsed = resolve_expression(held, site, functions)
            proposals.setdefault(key_and_path, []).append(None if isinstance(parsed, tuple) else parsed.value)

    accepted = {
        owner: values[0] for owner, values in proposals.items()
        if values[0] is not None and all(value == values[0] for value in values)
    }
    if not accepted:
        return []

    documents = {name for kind, name, _ in accepted if kind == "document"}
    steps = reached_steps(authored, {name for kind, name, _ in accepted if kind == "step"})
    variants = {name for kind, name, _ in accepted if kind == "variant"}
    other_runs = {
        row.key for row in candidate.rows if row.kind == "train" and row.key not in pairings
        and (row.document_uses & documents or row.key.step in steps or row.key.variant in variants)
    }
    if other_runs:
        other = compile_sources(authored, other_runs)
        for key, definition in other.selected_definitions.items():
            shown, projection, _ = _pairing(other, key, bundle)
            for (address, path), held, site in _held_values(authored, definition, shown, projection, "train"):
                owner = (address.kind, address.id, path)
                if owner not in accepted:
                    continue
                parsed = resolve_expression(held, site, functions)
                if isinstance(parsed, tuple) or parsed.value != accepted[owner]:
                    del accepted[owner]

    return [
        StudioProjectFilePatch(address=AuthoredFileAddress(kind=kind, id=name), patches=[
            SetProjectValue(op="replace", path=path, value=value.model_dump(mode="json")),
        ])
        for (kind, name, path), value in accepted.items()
    ]


def _validate_files(
    body, project, settled, key, selected, folders, bundle, candidates,
    *, preserve=None, controlling=None, other_runs=None,
):
    """Calculate each store's reach on the authored definitions before preparing its files."""
    possible = {
        row.key for row in settled.plan.rows if row.kind in {"train", "evaluate"}
    }
    if not selected <= possible:
        raise ProjectFolderInvalid("The selected runs no longer belong to this edit.")

    def compile_sources(project, source_runs):
        return compile_project(
            project, readers=settled.readers, source_runs=source_runs,
            operations=folders.operations, components=bundle.components,
            functions=bundle.expression_functions, python_tasks=bundle.python_tasks,
            figure_templates=bundle.figure_templates, training_methods=bundle.training_methods,
        )

    before = settled.plan
    if not possible <= before.selected_definitions.keys():
        before = folders.read_plan(settled.project_id, revision=body.revision, source_runs=possible).plan

    def preview(kind, label, source, stored):
        if not stored.files and not stored.created:
            return EditStore(kind, label, source, frozenset(), (stored, project, before))
        if preserve is not None:
            stored = preserve(stored)
        authored = preview_stored(project, stored)
        after = compile_sources(authored, possible)
        runs = _changed_runs(before, after)
        value = stored, authored, after
        if kind == "all_readers":
            return all_readers_store(source, runs, value)
        if kind == "selected_readers" and len({run.step for run in runs}) > 1:
            label = "Selected Steps"
        return EditStore(kind, label, source, runs, value)

    preferred, *alternatives = [preview(*candidate) for candidate in candidates]
    missed = selected - preferred.runs
    if controlling is not None and missed:
        sources = {candidate[2]: candidate for candidate in controlling(before, missed)}
        alternatives.extend(preview(*candidate) for candidate in sources.values())
    remaining = preferred.runs & selected
    if (
        other_runs is not None and remaining and preferred.runs != selected
        and not any(store.runs == remaining for store in alternatives)
        and all(run.step == key.step and run.variant == before.variants[0] for run in remaining)
    ):
        if isinstance(narrowed := other_runs(remaining), StoredEdit):
            option = preview("other_runs", "Other Runs" if missed else "Selected Runs", key.step, narrowed)
            if option.runs and option.runs <= remaining and option.runs != preferred.runs:
                alternatives.append(option)
    alternatives = [
        store for store in alternatives if store.runs != preferred.runs or store.value[0] != preferred.value[0]
    ]
    # A whole-step selection covers the rows that the edit creates in those same steps.
    if body.scope is None or isinstance(body.scope, StudioTrialStepScope):
        steps = {(run.variant, run.step) for run in selected}
        selected = selected | {
            run for store in (preferred, *alternatives) for run in store.runs - possible
            if (run.variant, run.step) in steps
        }
    decision = choose_store(
        body.revision, body.model_dump(mode="json", exclude={"store_choice"}), selected,
        preferred, alternatives, choice=body.store_choice,
    )
    if decision == "stale":
        raise ProjectFolderInvalid("The store choice is stale.")
    if decision == "unavailable":
        return StoredEdit([]), (), None
    if isinstance(decision, list):
        return StoredEdit([]), decision, None
    stored, authored, candidate = decision.value
    pairings = {}
    for reached in decision.runs:
        if reached in candidate.selected_definitions:
            documents, projection, _ = _pairing(candidate, reached, bundle)
            pairings[reached] = (documents, projection)
    if files := _resolved_value_files(authored, candidate, pairings, compile_sources, bundle):
        stored = stored.then(files)
        authored = preview_stored(project, stored)
        candidate = compile_sources(authored, possible)
    return stored, (), (authored, candidate)


def _changed_runs(before, after) -> frozenset[RunKey]:
    """Run identities whose own resolved definition or sweep coordinates change."""
    authored = edited_authored_rows(before, after)

    def values(plan):
        return {
            row.key: (
                (definition.payload, definition.input_documents)
                if (definition := plan.selected_definitions.get(row.key)) is not None else None,
                row.document_ids, row.coordinates,
            )
            for row in plan.rows if row.kind in {"train", "evaluate"} and row.key in authored
        }

    old, new = values(before), values(after)
    return frozenset(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))


def _held_problems(path, problems) -> list[Problem]:
    """The problems about the field at ``path``: a task's are addressed within the task, an Objective's with its role."""
    own = (path, path[1:]) if path[0] == "task" else (path,)
    return [problem for problem in problems if any(problem.path[:len(at)] == at for at in own)]


def _field_problems(path, site: Site, text: str, functions, problems) -> list[Problem]:
    """The field's problems, each at the field's path; one about a term of its text carries the term's place there."""
    held = _held_problems(path, problems)
    if not held or not text:
        return [problem.model_copy(update={"path": path}) for problem in held]
    read = read_field(text, site, functions)
    if isinstance(read, tuple):
        termed = read if any(problem.span for problem in read) else held
        return [problem.model_copy(update={"path": path}) for problem in termed]
    within = len(path) - 1 if path[0] == "task" else len(path)
    return [
        problem.model_copy(update={"path": path, "span": problem.span or read.spans.get(problem.path[within:])})
        for problem in held
    ]


def _trial_fields(path, site: Site, expression, functions, sweep, row, problems, name_paths) -> list[StudioTrialField]:
    """The field at ``path`` with its text, its structure where its value names only what is in scope, its problems,
    and each number in it the step can sweep: ``sweep`` gives the sweep of the number at a path, if the step can sweep it.

    A name no longer in scope is printed as its reference. A field that is a number carries its sweep; each number given
    to a draw in it is a field of its own.
    """
    missing = references(expression) - {item.id for item in site.names} if expression is not None else set()
    shown = expression is not None and not missing
    projection = project_site(site, functions, expression, name_paths=name_paths)
    fields = []
    for inner, label in numeric_nodes(expression, functions) if shown else ():
        swept = sweep((*path, *inner), site, label or _field_label(path))
        if swept is None:
            continue
        view = project_numeric_field(
            StudioTrialField, swept.field, swept.axis, functions, row, at(expression, inner).value,
            path=pointer(tuple(map(str, (*path, *inner)))),
        )
        if inner:
            fields.append(view)
        elif view.sweep is not None or view.can_sweep:
            return [view.model_copy(update={
                "site": projection, "problems": _field_problems(path, site, view.text, functions, problems),
            })]
    projection = projection.model_copy(update={"kinds": tuple(kind for kind in projection.kinds if kind != FieldKind.SWEEP)})
    printed = site.model_copy(update={"names": (*site.names, *(
        ScopedValue(id=reference, name=reference, type=ValueType(base=ValueBase.REAL), level=Level.RUN)
        for reference in sorted(missing)
    ))})
    text = print_field(expression, printed) if expression is not None else ""
    marks = _field_problems(path, site, text, functions, problems)
    if not marks and isinstance(expression, UnresolvedValue):
        resolved = resolve_expression(expression, site, functions)
        if isinstance(resolved, tuple):
            marks = [problem.model_copy(update={"path": path}) for problem in resolved]
    return [StudioTrialField(
        path=pointer(tuple(map(str, path))), label=_field_label(path), site=projection, text=text,
        structure=value_structure(expression, site, functions) if shown else None,
        problems=marks,
    ), *fields]


def _batch_size(kind: str, payload) -> int | None:
    """How many trials the step runs at once: a Train's batch size, an Evaluate's sampled Trials; nothing for a value
    that is not one, which its own field reports."""
    if kind == "train":
        config = payload.get("training_config")
        count = (config if isinstance(config, dict) else {}).get(
            "batch_size", DocumentTrainingConfig.model_fields["batch_size"].default,
        )
    else:
        params = payload.get("params")
        rule = params.get("trial_set") if isinstance(params, dict) else None
        count = rule.get("count") if isinstance(rule, dict) and rule.get("kind") == "sample" else None
    return count if isinstance(count, int) and count >= 1 else None


def trial_authoring(
    folders: ProjectFolderService, project_id: str, body: StudioTrialAuthoringRequest,
    bundle: ApplicationRegistryBundle,
) -> StudioTrialAuthoring:
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    snapshot = folders.read(project_id, revision=body.revision)
    settled = folders.read_plan(project_id, revision=snapshot.revision, source_runs=(key,))
    plan = settled.plan
    step = plan.steps.get((key.variant, key.step))
    if step is None or step.kind not in ("train", "evaluate"):
        raise ProjectFolderInvalid("The selected run has no authored trial pairing.")
    try:
        project = snapshot.project
        definition = plan.selected_definitions.get(key)
        change = body.change
        if (
            definition is None and step.kind == "train"
            and isinstance(change, StudioTrialSelectDocument) and change.role == "training"
            and (body.scope is None or isinstance(body.scope, StudioTrialStepScope))
            and any(
                item.code == ResolutionCode.INVALID_SWEEP and item.step == key.step and item.row == key.row
                and item.variant == key.variant for item in plan.diagnostics
            )
        ):
            # The Training document can be replaced from the authored step even when a sweep row no longer resolves.
            base = resolve_step(plan.variant_projects[key.variant], key.step, readers=settled.readers)
            if not isinstance(base, ProjectDiagnostic):
                definition = base
        if definition is None:
            definition = _definition(plan, key)
        documents, projected, graph = _pairing(plan, key, bundle, definition=definition)
        stored, refused = StoredEdit([]), ()
        related = None
        store_choices = None
        if body.store_choice is not None and body.change is None:
            raise ProjectFolderInvalid("A store choice requires the edit it was offered for.")
        if isinstance(change, StudioTrialRelate):
            from feedbax.web.services.trial_relate import related_term

            if projected.pair is None or documents.objective is None:
                raise ProjectFolderInvalid("Relate needs a paired Model, Task, and Objective.")
            term = related_term(change, documents.model, projected.pair, documents.objective, bundle.components)
            raw = documents.objective.model_dump(mode="json", exclude_unset=True)
            value = term.model_dump(mode="json")
            edit = (
                SetProjectValue(op="add", path="/terms/-", value=value)
                if "terms" in raw else SetProjectValue(op="add", path="/terms", value=[value])
            )
            change = StudioTrialDocumentEdit(kind="objective", edits=[edit])
            related = term.id
        if change is not None:
            stored, refused, preview = _prepare_edits(
                body, folders, project_id, snapshot.project, settled, key, bundle, definition, documents, projected,
                change=change,
            )
            if isinstance(refused, list):
                store_choices, refused = refused, ()
            if preview is not None:
                project, plan = preview
                if changed := plan.selected_definitions.get(key):
                    definition = changed
                    documents, projected, graph = _pairing(plan, key, bundle)
                else:
                    problems = tuple(
                        Problem(code=ProblemCode.STRUCTURE, reason=item.detail)
                        for item in plan.diagnostics
                        if item.step == key.step and item.row in (None, key.row)
                        and item.variant in (None, key.variant)
                    )
                    projected = replace(projected, problems=problems)
    except (ValueError, TypeError, KeyError) as error:
        raise ProjectFolderInvalid(str(error)) from error
    paired = projected.pair
    compiled = paired.task if paired is not None else projected.partial_task
    task = documents.task_source
    task_path = (f"payload.trial_documents.{_task_field(task)}" if step.kind == "train"
                 else _evaluation_task_path(definition))
    model_document = definition.document_ids.get("payload.trial_documents.model")
    objective_path = "payload.trial_documents.objective"
    objective_document = definition.document_ids.get(objective_path)
    objective_source = StudioTrialObjectiveSource(kind="own", step=key.step) if documents.objective is not None else None
    if isinstance(documents, _InitialTrialDocuments):
        model_document = definition.document_ids.get("inputs.subject")
        subject = definition.input_documents["subject"]
        if subject.schema_id == LOADED_MODEL_SCHEMA_ID:
            model_document = LoadedModelDocument.model_validate(subject.payload).model.document
        objective_path = "inputs.objective"
        objective_document = definition.document_ids.get(objective_path)
    task_source = None
    task_field_sources = {}
    reset_model = documents.model
    if step.kind == "evaluate" and (upstream := evaluate_subject(definition)):
        producer = next((row for row in plan.rows if row.key == upstream[0]), None)
        if producer is not None:
            model_document = producer.document_ids.get("payload.trial_documents.model")
            objective_document = producer.document_ids.get(objective_path)
        if model_document is not None:
            authored_model = compose_document(project.documents.get, model_document)
            if not isinstance(authored_model, ProjectDiagnostic):
                reset_model = ModelDocument.model_validate(authored_model.payload)
        objective_source = StudioTrialObjectiveSource(kind="train", step=upstream[0].step)
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
    row = next((item for item in plan.rows if item.key == key), None)
    output_paths = {_output_address(port.id).to_text(): (port.name,)
                    for port in documents.model.ports.outputs} | projected.output_paths

    def expression(path):
        document = _field_document(documents, path[0])
        if document is None:
            return None
        found = site_expression(document, tuple(map(str, path[1:])))
        if found is None and path[0] == "task" and path[1] == "signals" and path[3:] == ("absent",):
            signal = next(item for item in document.signals if item.id == path[2])
            return zero(signal.type.base, signal.type.shape)
        return found

    def sweep(path, site, label):
        if paired is None:
            return None
        return _field_sweep(project, plan, key, definition, path, site, label)

    fields = [
        field for path, site in projected.fields.items() if path[1:2] != ("validation_trials",)
        for field in _trial_fields(
            path, site, expression(path), bundle.expression_functions, sweep, row, projected.problems, output_paths,
        )
    ]
    field_problems = [
        problem for field in fields for problem in field.problems
        if not _held_problems(problem.path, projected.problems)
    ]
    return StudioTrialAuthoring(
        revision=snapshot.revision, run=body.run, model=model_document,
        task_document=definition.document_ids.get(task_path) if task_path else None,
        objective_document=objective_document,
        task=task, objective=documents.objective, objective_source=objective_source,
        task_changed=changed_from_base(project, _whole_document(definition, task_path) if task_path else None),
        objective_changed=changed_from_base(project, _whole_document(
            definition, objective_path,
        )) if objective_source is not None and objective_source.kind == "own" else [],
        task_source=task_source, task_field_sources=task_field_sources,
        set_fields=trial_set_fields(snapshot.project, key.step),
        task_types=[StudioTaskType(type=key, label=entry.label, parameter_schema=entry.parameters.model_json_schema())
                    for key in bundle.python_tasks.keys() if (entry := bundle.python_tasks.resolve(key))],
        variables=list(projected.variables), signals=list(projected.signals), epochs=list(projected.epochs),
        epoch_instances=list(projected.epoch_instances),
        period_seconds=documents.model.step_seconds if isinstance(documents.model.step_seconds, float) else None,
        n_steps=compiled.n_steps if compiled else None,
        batch_size=_batch_size(step.kind, definition.payload),
        reads_model=compiled.reads_model if compiled else False,
        tasks=[
            StudioTaskChoice(document=name, problems=list(_project(
                _with_task(documents, other), bundle, show_objective=False, graph_override=graph,
                objective_use=ObjectiveUse.INHERITED if objective_source is not None and objective_source.kind == "train"
                else ObjectiveUse.TRAIN,
            ).problems))
            for name in sorted(project.documents)
            if (other := _selected_task(project, name)) is not None
        ],
        input_bindings=documents.input_bindings, inputs=_input_rows(documents, projected),
        outputs=[StudioTrialOutput(address=ModelAddress.from_text(value.id), value=value)
                 for value in paired.model.outputs] if paired else [],
        resets=[StudioTrialReset(address=field.address, path=task_start_path(reset_model, field.address),
                                 type=field.type, finest=field.finest, required=field.required)
                for field in projected.resets],
        fields=fields, problems=[*projected.problems, *field_problems, *refused],
        related_term=related if stored.files or stored.created else None,
        files=stored.files, created=list(stored.created),
        store_choices=store_choices,
    )
