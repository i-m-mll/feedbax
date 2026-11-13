"""Project and prepare edits for one saved trial pairing."""

from copy import deepcopy
from dataclasses import dataclass

from pydantic import ValidationError

from feedbax.compiler.trial_execution import TrialPair, pair_trial_source, validate_python_task
from feedbax.contracts.expression_values import Level, Problem, ProblemCode, ScopedValue, Site
from feedbax.contracts.matrix_core import apply_value_patch_in_place
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import DocumentReference, RemoveProjectValue, SetProjectValue, json_pointer_parts
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec, TrainingRequestSpec
from feedbax.contracts.rollout import RolloutParameters
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.studio.canonical_evaluate import CannotRun
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.contracts.studio_trials import (
    StudioTaskChoice, StudioTaskType, StudioTrialAuthoring, StudioTrialAuthoringRequest, StudioTrialBindingEdit,
    StudioTrialField, StudioTrialInput, StudioTrialOutput, StudioTrialReset, StudioTrialSelectTask,
)
from feedbax.contracts.task_document import (
    PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID, Epoch, PythonTaskSelection, TaskDocument,
)
from feedbax.contracts.training import TrialDocuments
from feedbax.expression_site import project_site
from feedbax.objectives.document import ObjectiveExecutor, compile_objective
from feedbax.plugins.application import ApplicationRegistryBundle
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService


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
    if not show_objective:
        return _Projection(paired, variables, signals, epochs, fields, paired.problems)
    objective_sites = {}
    objective = compile_objective(
        documents.objective, (*paired.model.outputs, *variables, *signals), bundle.expression_functions,
        task=source, sites=objective_sites,
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


def _replace_field(address, path, value, present) -> StudioProjectFilePatch:
    return StudioProjectFilePatch(address=address, patches=[
        SetProjectValue(op="replace" if present else "add", path=path, value=value),
    ])


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


def _prepare_edits(change, documents, projection, definition, snapshot, key, bundle):
    if change is None:
        return [], ()
    step = snapshot.project.steps[key.step]
    if key.variant != "default" or key.row != "default" or step.sweep is not None:
        raise ProjectFolderInvalid("This edit needs the step's default row.")
    inline = step.model_dump(mode="json", by_alias=True, exclude_unset=True).get("base", {}).get("inline", {})
    raw = inline.get("trial_documents")
    if not isinstance(raw, dict):
        raise ProjectFolderInvalid("This edit needs a local trial pairing on the step.")
    step_address = AuthoredFileAddress(kind="step", id=key.step)
    if isinstance(change, StudioTrialSelectTask):
        entry = snapshot.project.documents.get(change.document)
        task = _selected_task(entry) if entry is not None else None
        if task is None:
            raise ProjectFolderInvalid("The selected document is not a Task.")
        candidate = _with_task(documents, task)
        field = "python_task" if isinstance(task, PythonTaskSelection) else "task"
        data = {name: value for name, value in raw.items() if name not in {"task", "python_task"}}
        data[field] = {"document": change.document}
        files = [_replace_field(step_address, "/base/inline/trial_documents", data, True)]
    elif isinstance(change, StudioTrialBindingEdit):
        row = next((row for row in _input_rows(documents, projection) if row.address == change.input), None)
        if row is None:
            raise ProjectFolderInvalid("The address is not a declared Model input for this row.")
        overrides = dict(documents.input_overrides)
        if change.signal is None or change.signal == row.name:
            overrides.pop(row.name, None)
        else:
            overrides[row.name] = change.signal
        candidate = documents.model_copy(update={"input_overrides": overrides})
        path = "/base/inline/trial_documents/input_overrides"
        if not overrides and "input_overrides" in raw:
            files = [StudioProjectFilePatch(address=step_address, patches=[RemoveProjectValue(op="remove", path=path)])]
        elif overrides:
            files = [_replace_field(step_address, path, overrides, "input_overrides" in raw)]
        else:
            files = []
    else:
        field = ("python_task" if isinstance(documents.task_source, PythonTaskSelection) else "task"
                 ) if change.kind == "task" else "objective"
        name = definition.document_ids.get(f"payload.trial_documents.{field}")
        if name is not None:
            stored = snapshot.project.documents[name].payload
            reference = DocumentReference.model_validate(raw.get(field))
            if (
                reference.document != name or reference.pointer is not None
                or step.document_overrides.get(name)
            ):
                raise ProjectFolderInvalid("Edit this inherited or overridden value at its authored source.")
            address = AuthoredFileAddress(kind="document", id=name)
            path = "/payload"
        elif field in raw and isinstance(raw[field], dict):
            stored = raw[field]
            address = step_address
            path = f"/base/inline/trial_documents/{field}"
        else:
            raise ProjectFolderInvalid("The selected value has no editable authored source.")
        value = deepcopy(stored)
        try:
            for patch in change.edits:
                apply_value_patch_in_place(
                    value, json_pointer_parts(patch.path), patch.op,
                    patch.value if isinstance(patch, SetProjectValue) else None, path=patch.path,
                )
            edited = type(getattr(documents, field)).model_validate(value)
        except (ValueError, TypeError) as error:
            raise ProjectFolderInvalid(str(error)) from error
        candidate = documents.model_copy(update={field: edited})
        files = [_replace_field(address, path, value, True)]
    proposed = _project(candidate, bundle)
    introduced = tuple(problem for problem in proposed.problems if problem not in projection.problems)
    return ([] if introduced else files), introduced


def _evaluate_documents(plan, key, definition, bundle):
    run = next((row for row in plan.runs if row.key == key), None)
    subject = run.inputs.get("subject") if run else None
    raw = plan.requests.get(subject.member.reference.request_id) if subject else None
    if raw is None:
        raise ProjectFolderInvalid("The selected Evaluate has no planned Train subject.")
    request = decode_request(raw, plan.documents.__getitem__).spec
    if not isinstance(request, (TrainingRequestSpec, DocumentTrainingRequestSpec)) or request.trial_documents is None:
        raise ProjectFolderInvalid("The selected Evaluate has no managed Train pairing.")
    documents = request.trial_documents
    projection = _project(documents, bundle, show_objective=False)
    if projection.pair is None:
        return documents, projection
    try:
        params = RolloutParameters.model_validate(definition.payload.get("params"))
        variant = resolve_evaluation_variant(
            projection.pair.lowered.graph, documents.task_source, params.variation,
        )
    except (CannotRun, ValidationError) as error:
        return documents, _Projection(
            projection.pair, projection.variables, projection.signals, projection.epochs,
            projection.fields, (*projection.problems, Problem(
                code=ProblemCode.STRUCTURE, path=("variation",), reason=str(error),
            )),
        )
    documents = _with_task(documents, variant.task)
    return documents, _project(documents, bundle, show_objective=False, graph_override=variant.graph)


def trial_authoring(
    folders: ProjectFolderService, project_id: str, body: StudioTrialAuthoringRequest,
    bundle: ApplicationRegistryBundle,
) -> StudioTrialAuthoring:
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    snapshot = folders.read(project_id, revision=body.revision)
    plan = folders.read_plan(project_id, revision=snapshot.revision, source_runs=(key,)).plan
    definition = plan.selected_definitions.get(key)
    if definition is None:
        raise ProjectFolderInvalid("The selected run has no resolved trial definition.")
    step = plan.steps.get((key.variant, key.step))
    if step is None or step.kind not in ("train", "evaluate"):
        raise ProjectFolderInvalid("The selected run has no authored trial pairing.")
    if step.kind == "train":
        try:
            documents = TrialDocuments.model_validate(definition.payload.get("trial_documents"))
        except ValidationError as error:
            raise ProjectFolderInvalid(str(error)) from error
        projected = _project(documents, bundle)
        files, refused = _prepare_edits(body.change, documents, projected, definition, snapshot, key, bundle)
    else:
        if body.change is not None:
            raise ProjectFolderInvalid("Edit the Evaluate task through its authored variation.")
        documents, projected = _evaluate_documents(plan, key, definition, bundle)
        files, refused = [], ()
    paired = projected.pair
    task = documents.task_source
    source_field = "python_task" if isinstance(task, PythonTaskSelection) else "task"
    links = dict(paired.task.interface.links) if paired else {}
    return StudioTrialAuthoring(
        revision=snapshot.revision, run=body.run, model=documents.model_name,
        task_document=definition.document_ids.get(f"payload.trial_documents.{source_field}"),
        objective_document=definition.document_ids.get("payload.trial_documents.objective"),
        task=task, objective=documents.objective if step.kind == "train" else None,
        task_types=[StudioTaskType(type=key, label=entry.label, parameter_schema=entry.parameters.model_json_schema())
                    for key in bundle.python_tasks.keys() if (entry := bundle.python_tasks.resolve(key))],
        variables=list(projected.variables), signals=list(projected.signals), epochs=list(projected.epochs),
        period_seconds=documents.model.step_seconds,
        n_steps=paired.task.n_steps if paired else None,
        tasks=[
            StudioTaskChoice(document=name, problems=list(_project(
                _with_task(documents, other), bundle, show_objective=step.kind == "train",
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
                                 projection=project_site(site, bundle.expression_functions))
                for path, site in projected.fields.items()],
        problems=[*projected.problems, *refused], files=files,
    )
