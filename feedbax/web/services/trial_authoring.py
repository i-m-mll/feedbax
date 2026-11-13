"""Project and prepare edits for one saved trial pairing."""

from copy import deepcopy
from dataclasses import dataclass

from feedbax.compiler.trial_execution import TrialPair, pair_trial_source, validate_python_task
from feedbax.contracts.expression_values import Level, Problem, ProblemCode, ScopedValue, Site, SweepList, SweepRange, SweepSample
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import EvaluateAttachmentVariant
from feedbax.contracts.project import SetProjectValue, apply_document_edits, document_edit_ops, json_pointer_parts
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import ProductRef
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec, EvaluationRequestSpec, TrainingRequestSpec
from feedbax.contracts.rollout import ROLLOUT_TYPE, RolloutParameters
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_project import AuthoredFileAddress
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.contracts.studio_trials import (
    StudioTaskChoice, StudioTaskType, StudioTrialAuthoring, StudioTrialAuthoringRequest, StudioTrialBindingEdit,
    StudioTrialDocumentEdit, StudioTrialExpressionEdit, StudioTrialField, StudioTrialInput,
    StudioTrialOutput, StudioTrialReset, StudioTrialSelectTask, StudioTrialStepScope,
)
from feedbax.contracts.task_document import (
    PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID, Epoch, PythonTaskSelection, TaskDocument,
)
from feedbax.contracts.training import TrialDocuments
from feedbax.expression_check import references
from feedbax.expression_site import project_site, read_field
from feedbax.expression_text import ParsedField, print_field
from feedbax.objectives.document import ObjectiveExecutor, compile_objective
from feedbax.plugins.application import ApplicationRegistryBundle
from feedbax.web.services.document_authoring import _step_difference
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService, preview_authored_files
from feedbax.web.services.trial_scope import file_value, scoped_file, selected_runs, variant_file


@dataclass(frozen=True)
class _Projection:
    pair: TrialPair | None
    variables: tuple[ScopedValue, ...]
    signals: tuple[ScopedValue, ...]
    epochs: tuple[Epoch, ...]
    fields: dict[tuple[str | int, ...], Site]
    problems: tuple[Problem, ...]


def _project(documents, bundle, *, show_objective=True, graph_override=None) -> _Projection:
    task = documents.task_source
    fields = {}
    if isinstance(task, PythonTaskSelection):
        resolved = validate_python_task(task, bundle.python_tasks)
        if isinstance(resolved[0], Problem):
            return _Projection(None, (), (), (), {}, resolved)
        definition, params = resolved
        template = definition.template(params)
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
        epochs = task.epochs
    task_sites = {}
    paired = pair_trial_source(
        documents.model_name, documents.model, task, bundle.components, bundle.expression_functions,
        python_tasks=bundle.python_tasks, input_overrides=documents.input_overrides,
        task_sites=task_sites, graph_override=graph_override,
    )
    fields.update({("task", *path): site for path, site in task_sites.items()})
    if isinstance(paired, tuple):
        return _Projection(None, variables, signals, epochs, fields, paired)
    source = paired.task
    variables, signals = source.interface.variables, source.interface.signals
    if isinstance(task, PythonTaskSelection):
        epochs = tuple(Epoch(id=epoch, name=epoch) for epoch in source.epochs)
    objective_sites = {}
    objective = compile_objective(
        documents.objective, (*paired.model.outputs, *variables, *signals), bundle.expression_functions,
        task=source, sites=objective_sites if show_objective else None,
    )
    fields.update({("objective", *path): site for path, site in objective_sites.items()})
    problems = paired.problems
    if not isinstance(objective, ObjectiveExecutor):
        problems += tuple(problem.model_copy(update={"path": ("objective", *problem.path)})
                          for problem in objective)
    return _Projection(paired, variables, signals, epochs, fields, problems)


def _input_address(model: str, port: str) -> ModelAddress:
    return ModelAddress.model_validate({
        "document": model, "composites": [],
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
        for address in (_input_address(documents.model_name, port.id),)
    ]


def _selected_task(entry) -> TaskDocument | PythonTaskSelection | None:
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
    parsed = read_field(change.text, site, bundle.expression_functions)
    if not isinstance(parsed, ParsedField):
        return parsed
    if isinstance(parsed.value, SweepList | SweepRange | SweepSample):
        return (Problem(code=ProblemCode.STRUCTURE, reason="This objective field takes one expression."),)
    return StudioTrialDocumentEdit(kind="objective", edits=[SetProjectValue(
        op="replace" if change.field in documents.objective.terms[index].model_fields_set else "add",
        path=pointer(("terms", str(index), change.field)),
        value=parsed.value.model_dump(mode="json"),
    )])


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


def _edit_patches(prefix, change):
    patches = []
    for edit in change.edits:
        parts = json_pointer_parts(edit.path)
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
    return _edit_patches(prefix, change)


def _authored_task_entry(raw):
    if not isinstance(raw, (dict, list)):
        return None
    entries = raw.items() if isinstance(raw, dict) else enumerate(raw)
    names = [name for name, edit in entries if isinstance(edit, dict) and edit.get("path") == "task"]
    return names[-1] if names else None


def _variation_write(definition, value):
    raw = definition.payload["params"]["variation"]
    entry = _authored_task_entry(raw)
    if entry is not None:
        target, op = entry, "replace"
    elif isinstance(raw, dict):
        target, op = "task", "add"
        while target in raw:
            target += "-selection"
    else:
        target, op = len(raw), "add"
    return _override(f"params.variation.{target}", op, {"op": "replace", "path": "task", "value": value})


def _train_writes(change, project, definition, documents, projection):
    raw = definition.payload["trial_documents"]
    if isinstance(change, StudioTrialSelectTask):
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


def _chosen_task(project, document):
    entry = project.documents.get(document)
    task = _selected_task(entry) if entry is not None else None
    if task is None:
        raise ProjectFolderInvalid("The selected document is not a Task.")
    return task


def _overlap(left, right):
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _document_files(project, key, change, document, lit, users):
    entry = project.documents[document]
    files = []
    shared = lit == users
    if shared:
        raw = _edit_value(entry.payload, change)
        files.append(file_value(AuthoredFileAddress(kind="document", id=document),
                                {"payload": entry.payload}, ("payload",), raw))
    for step in sorted(lit):
        previous = project.steps[step].document_overrides.get(document, {})
        if shared:
            local = []
            for edit in document_edit_ops(previous):
                affected = [requested for requested in change.edits if _overlap(edit.path, requested.path)]
                if not affected:
                    local.append(edit)
                elif any(requested.path != edit.path or edit.op != "replace" for requested in affected):
                    raise ProjectFolderInvalid(f"{step} has an overlapping document difference.")
        else:
            local = [*document_edit_ops(previous), *change.edits]
        if local != document_edit_ops(previous):
            files.append(_step_difference(project, step, document, local))
    return files


def _evaluation_task_path(definition):
    entry = _authored_task_entry(definition.payload.get("params", {}).get("variation"))
    return f"payload.params.variation.{entry}.value" if entry is not None else None


def _evaluate_documents(plan, key, definition, bundle):
    run = next((row for row in plan.runs if row.key == key), None)
    subject = run.inputs.get("subject") if run else None
    reference = subject.member.reference if subject else None
    raw = plan.requests.get(reference.request_id) if isinstance(reference, ProductRef) else None
    if raw is None:
        raise ProjectFolderInvalid("The selected Evaluate has no planned Train subject.")
    request = decode_request(raw, plan.documents.__getitem__).spec
    if not isinstance(request, (TrainingRequestSpec, DocumentTrainingRequestSpec)) or request.trial_documents is None:
        raise ProjectFolderInvalid("The selected Evaluate has no managed Train pairing.")
    documents = request.trial_documents
    projection = _project(documents, bundle, show_objective=False)
    if projection.pair is None:
        return documents, projection, None
    try:
        request = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        if not isinstance(request, EvaluationRequestSpec) or request.evaluation_type != ROLLOUT_TYPE:
            raise ValueError("Task authoring requires a saved rollout Evaluate.")
        params = RolloutParameters.model_validate(request.params)
        variant = resolve_evaluation_variant(
            projection.pair.lowered.graph, documents.task_source, params.variation,
        )
    except ValueError as error:
        return documents, _Projection(
            projection.pair, projection.variables, projection.signals, projection.epochs,
            projection.fields, (*projection.problems, Problem(
                code=ProblemCode.STRUCTURE, path=("variation",), reason=str(error),
            )),
        ), None
    documents = _with_task(documents, variant.task)
    return documents, _project(documents, bundle, show_objective=False, graph_override=variant.graph), variant.graph


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
        change = _expression_edit(expression, documents, projected, bundle)
        if not isinstance(change, StudioTrialDocumentEdit):
            return [], change
    document = None
    if step.kind == "train" and isinstance(change, StudioTrialDocumentEdit):
        field, _ = _changed_value(documents, definition, change.kind)
        path = f"payload.trial_documents.{field}"
        if not definition.document_pointers.get(path):
            document = definition.document_ids.get(path)
    lit = set(scope.lit_steps) if isinstance(scope, StudioTrialStepScope) else set()
    users = set(plan.document_users(key.variant, document)) if document else set()
    if scope is None and key.variant != plan.variants[0]:
        scope = EvaluateAttachmentVariant(kind="variant", variant=key.variant)
    elif scope is None:
        scope = StudioTrialStepScope(kind="step")
        lit = users
    if isinstance(scope, StudioTrialStepScope):
        if key.variant != plan.variants[0]:
            raise ProjectFolderInvalid("Use a variant scope for this comparison variant.")
        if lit and (key.step not in lit or not lit <= users):
            raise ProjectFolderInvalid("The lit steps no longer match this document.")
        lit = lit or {key.step}
    selected = selected_runs(plan, key, scope, lit_steps=lit)
    if document is not None and isinstance(scope, StudioTrialStepScope):
        path = f"payload.trial_documents.{field}"
        if any(
            row.key in selected and row.document_ids.get(path) != document
            and document not in row.document_uses
            for row in plan.rows
        ):
            raise ProjectFolderInvalid("A lit step no longer uses the selected document.")
    if document is not None and isinstance(scope, EvaluateAttachmentVariant):
        selected = {row.key for row in plan.rows if row.key.variant == key.variant and row.key.step in users}
    if selected != {key}:
        settled = folders.read_plan(project_id, revision=body.revision, source_runs=selected)
    if step.kind == "evaluate":
        if isinstance(change, StudioTrialSelectTask):
            _chosen_task(project, change.document)
            value = {"document": change.document}
        elif isinstance(change, StudioTrialDocumentEdit) and change.kind == "task":
            value = _edit_value(documents.task_source.model_dump(mode="json", exclude_unset=True), change)
        else:
            raise ProjectFolderInvalid("Evaluate authoring edits the task variation.")
        files = [scoped_file(project, key, scope, [_variation_write(definition, value)])]
    elif document is not None and isinstance(scope, StudioTrialStepScope):
        files = _document_files(project, key, change, document, lit, users)
    elif document is not None and isinstance(scope, EvaluateAttachmentVariant):
        _, value = _changed_value(documents, definition, change.kind)
        files = [variant_file(project, key.variant, _value_writes(f"documents.{document}.payload", value, change))]
    else:
        writes = _train_writes(change, project, definition, documents, projected)
        files = [scoped_file(project, key, scope, writes)] if writes else []
    if not files:
        return [], ()
    return _validate_files(files, project, settled, selected, folders, bundle, change, expression)


def _validate_files(files, project, settled, selected, folders, bundle, change, expression):
    plan = settled.plan
    authored = preview_authored_files(project, files)
    compare_names = isinstance(change, StudioTrialDocumentEdit) and change.kind == "objective" and any(
        "name" in json_pointer_parts(edit.path) or len(json_pointer_parts(edit.path)) <= 2 for edit in change.edits
    )
    outside = {row.key for row in plan.rows if row.kind == "train" and row.key not in selected} if compare_names else set()

    def compile_sources(project, source_runs):
        return compile_project(
            project, readers=settled.readers, source_runs=source_runs,
            operations=folders.operations, components=bundle.components,
            functions=bundle.expression_functions, python_tasks=bundle.python_tasks,
        )

    candidate = compile_sources(authored, selected | outside)
    if outside:
        before = compile_sources(project, outside)
        if any(_objective_names(before, key) != _objective_names(candidate, key) for key in outside):
            raise ProjectFolderInvalid("The proposed edit changes an Objective outside the selected scope.")
    previous = {run.key: run for run in plan.runs}
    current = {run.key: run for run in candidate.runs}
    before_diagnostics = set(plan.diagnostics)
    for key in sorted(selected, key=lambda item: (item.variant, item.step, item.row)):
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
        if introduced:
            return [], introduced
        if isinstance(change, StudioTrialDocumentEdit):
            old_value = old.task_source if change.kind == "task" else old.objective
            new_value = new.task_source if change.kind == "task" else new.objective
            raw = old_value.model_dump(mode="json", exclude_unset=True)
            if plan.steps[key.variant, key.step].kind == "train":
                _, raw = _changed_value(old, plan.selected_definitions[key], change.kind)
            expected = type(old_value).model_validate(_edit_value(raw, change))
            if expected != new_value:
                raise ProjectFolderInvalid("A selected value is controlled by a different authored source.")
        elif isinstance(change, StudioTrialSelectTask):
            step = candidate.steps[key.variant, key.step]
            definition = candidate.selected_definitions[key]
            path = (f"payload.trial_documents.{_task_field(new.task_source)}" if step.kind == "train"
                    else _evaluation_task_path(definition))
            if definition.document_ids.get(path) != change.document:
                raise ProjectFolderInvalid("A selected task is controlled by a different authored source.")
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
        if key in previous and key not in current:
            diagnostic = next((
                item for item in candidate.diagnostics if item not in before_diagnostics
                and item.step == key.step and item.variant == key.variant and item.row == key.row
            ), None)
            raise ProjectFolderInvalid(diagnostic.detail if diagnostic else "The edit removes a planned request.")
    _check_other_requests(plan, candidate, previous, current, selected)
    return files, ()


def _objective_names(plan, key):
    definition = plan.selected_definitions.get(key)
    pairing = definition.payload.get("trial_documents") if definition else None
    objective = pairing.get("objective") if isinstance(pairing, dict) else None
    terms = objective.get("terms", []) if isinstance(objective, dict) else []
    return {term.get("id"): term.get("name") for term in terms if isinstance(term, dict) and term.get("name")}
def _check_other_requests(plan, candidate, previous, current, selected):
    # A changed request can propagate along a saved product edge.
    changed = {
        key for key in previous.keys() | current.keys()
        if getattr(previous.get(key), "request_id", None) != getattr(current.get(key), "request_id", None)
    }
    affected = changed & selected
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


def _field_text(expression, site: Site) -> str | None:
    if references(expression) <= {item.id for item in site.names}:
        return print_field(expression, site)
    return None


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
        files, refused = [], ()
        if body.change is not None:
            files, refused = _prepare_edits(
                body, folders, project_id, snapshot.project, settled, key, bundle, documents, projected,
            )
    except (ValueError, TypeError, KeyError) as error:
        raise ProjectFolderInvalid(str(error)) from error
    paired = projected.pair
    task = documents.task_source
    task_path = (f"payload.trial_documents.{_task_field(task)}" if step.kind == "train"
                 else _evaluation_task_path(definition))
    links = dict(paired.task.interface.links) if paired else {}
    expressions = {
        ("objective", "terms", term.id, field): getattr(term, field)
        for term in documents.objective.terms for field in ("measure", "weight")
    } if step.kind == "train" else {}
    return StudioTrialAuthoring(
        revision=snapshot.revision, run=body.run, model=documents.model_name,
        task_document=definition.document_ids.get(task_path) if task_path else None,
        objective_document=definition.document_ids.get("payload.trial_documents.objective"),
        task=task, objective=documents.objective if step.kind == "train" else None,
        task_types=[StudioTaskType(type=key, label=entry.label, parameter_schema=entry.parameters.model_json_schema())
                    for key in bundle.python_tasks.keys() if (entry := bundle.python_tasks.resolve(key))],
        variables=list(projected.variables), signals=list(projected.signals), epochs=list(projected.epochs),
        period_seconds=documents.model.step_seconds,
        n_steps=paired.task.n_steps if paired else None,
        tasks=[
            StudioTaskChoice(document=name, problems=list(_project(
                _with_task(documents, other), bundle, show_objective=step.kind == "train", graph_override=graph,
            ).problems))
            for name, entry in sorted(snapshot.project.documents.items())
            if (other := _selected_task(entry)) is not None
        ],
        inputs=_input_rows(documents, projected),
        outputs=[StudioTrialOutput(address=ModelAddress.from_text(value.id), value=value)
                 for value in paired.model.outputs] if paired else [],
        resets=[StudioTrialReset(address=field.address, type=field.type, finest=field.finest,
                                variable=links.get(field.address.to_text()))
                for field in paired.model.reset.fields] if paired else [],
        fields=[StudioTrialField(path=pointer(tuple(map(str, path))),
                                 projection=project_site(site, bundle.expression_functions),
                                 text=_field_text(expressions[path], site) if path in expressions else None)
                for path, site in projected.fields.items()],
        problems=[*projected.problems, *refused], files=files,
    )
