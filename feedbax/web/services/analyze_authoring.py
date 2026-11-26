"""Project registered operation controls and store the edits of an Analyze step's analysis document."""

from collections.abc import Callable

from feedbax.analysis.operations import OperationRegistry
from feedbax.builtins.figure_templates import VelocitySummary, VelocitySummarySettings
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.analysis_document import ANALYSIS_SCHEMA, AnalysisDocument, DatasetSource, OperationDataset, StepInput
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.figure_document import FigureTemplateRef, StoredFigureDocument, StoredSpanLayer, validate_figure_bindings
from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument
from feedbax.contracts.operation import (
    DatasetInputPort,
    DatasetPort,
    OperationDefinition,
    WholeDimensions,
    operation_input_port, require_dataset_fields,
    require_dataset_port,
)
from feedbax.contracts.project import (
    DocumentReference, DocumentEdit, LoadedModelDocument, Payload, Project, RemoveProjectValue, SetProjectValue,
    base_chain, json_pointer_parts, member_name,
)
from feedbax.contracts.project_io import project_declaration_name
from feedbax.contracts.project_inputs import StepReference
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolutionCode, compose_document
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_storage import stored_project_value
from feedbax.contracts.semantic_values import model_state_address
from feedbax.contracts.studio_analyze import (
    StudioAnalyzeAddFigureLayer,
    StudioAnalyzeAuthoring,
    StudioAnalyzeEdit,
    StudioAnalyzeFigureValues,
    StudioAnalyzeRemove,
    StudioAnalyzeRequest,
    StudioAnalyzeReset,
    StudioAnalyzeWire,
    StudioFigureColorScaleValues,
    StudioFigureLayerValues,
)
from feedbax.contracts.studio_place import path_parts, task_start
from feedbax.jobs import JobConflict, JobNotFound
from feedbax.planner.project_plan import EVALUATION_MODEL, PlannedAnalyzeRun, PlannedRun, ProjectPlan, RunKey, compile_project
from feedbax.plot.templates import FigureTemplateRegistry
from feedbax.project_members import generated_name
from feedbax.web.services.authored_scope import preview_plan
from feedbax.web.services.project_folder import ProjectFolderInvalid
from feedbax.web.services.project_plan import ProjectPlanRead, ProjectPlanService
from feedbax.studio.selection import StoredEdit
from feedbax.web.services.trial_scope import (
    base_value_edits, changed_from_base, document_write, lit_steps, preview_stored,
)


def _port(
    source: DatasetSource,
    analysis: AnalysisDocument,
    registry: OperationRegistry,
    states: DatasetPort | None,
) -> DatasetPort:
    if isinstance(source, StepInput):
        if source.input != "states" or states is None:
            raise ValueError(f"Unknown Analyze input {source.input!r}.")
        return states
    try:
        operation = analysis.operations[source.element].operation
        return registry.resolve(operation).value.definition.outputs[source.product]
    except KeyError as error:
        raise ValueError(f"Unknown dataset output {source.element}.{source.product}.") from error


def _analysis(project: Project, document: str) -> AnalysisDocument:
    composed = compose_document(project.documents.get, document)
    if isinstance(composed, ProjectDiagnostic):
        raise ValueError(composed.detail)
    return AnalysisDocument.model_validate(composed.payload)


def _catalog_analysis(project: Project, document: str) -> AnalysisDocument | None:
    try:
        return _analysis(project, document)
    except ValueError:
        # The window map already carries the diagnostics of a document that cannot be composed.
        return None


def _figure_layer_values(figure: StoredFigureDocument) -> dict[str, StudioFigureLayerValues]:
    names = set()
    values = {}
    for key, layer in figure.layers.items():
        name = generated_name(layer.mark.capitalize(), names.__contains__, separator=" ")
        names.add(name)
        values[key] = StudioFigureLayerValues(
            name=name, fill_alpha=layer.style.fill_alpha if isinstance(layer, StoredSpanLayer) else None,
        )
    return values


def _figure_values(analysis: AnalysisDocument | None) -> dict[str, StudioAnalyzeFigureValues]:
    if analysis is None:
        return {}
    return {
        key: StudioAnalyzeFigureValues(
            layers=_figure_layer_values(figure),
            color_scales={
                name: StudioFigureColorScaleValues(legend=scale.legend, palette=scale.palette)
                for name, scale in figure.color_scales.items()
            },
        )
        for key, view in analysis.views.items() if isinstance(figure := view.figure, StoredFigureDocument)
    }


def _sources(
    analysis: AnalysisDocument, registry: OperationRegistry, states: DatasetPort | None,
    row: PlannedAnalyzeRun | None, templates: FigureTemplateRegistry,
) -> dict[str, dict[str, list[DatasetSource]]]:
    operations = {}
    ports: list[tuple[DatasetSource, DatasetPort]] = []
    if states is not None:
        ports.append((StepInput(input="states"), states))
    for name, operation in analysis.operations.items():
        try:
            definition = registry.resolve(operation.operation).value.definition
        except KeyError:
            continue
        operations[name] = definition
        if row is not None:
            ports.extend(
                (OperationDataset(element=name, product=product), port)
                for product, port in row.dataset_outputs.get(name, {}).items()
            )

    sources: dict[str, dict[str, list[DatasetSource]]] = {name: {} for name in analysis.operations | analysis.views}
    for name, definition in operations.items():
        for role, expected in definition.inputs.items():
            if not isinstance(expected, DatasetPort):
                continue
            choices = sources[name][role] = []
            for source, offered in ports:
                if isinstance(source, OperationDataset) and source.element == name:
                    continue
                try:
                    require_dataset_port(offered, expected)
                except ValueError:
                    continue
                choices.append(source)

    for name, view in analysis.views.items():
        figure = row.figures.get(name) if row is not None else None
        if figure is None:
            try:
                if isinstance(view.figure, FigureTemplateRef):
                    template = templates.resolve(view.figure.template)
                    figure = template.function(template.settings_model.model_validate(view.figure.settings))
                elif isinstance(view.figure, StoredFigureDocument):
                    figure = view.figure.lower()
            except (KeyError, ValueError):
                continue
        if figure is None:
            continue
        for role in figure.data:
            use = figure.model_copy(update={
                "data": (role,), "layers": tuple(layer for layer in figure.layers if layer.role == role),
            })
            choices = sources[name][role] = []
            for source, offered in ports:
                try:
                    validate_figure_bindings(use, {role: offered})
                except ValueError:
                    continue
                choices.append(source)
    return sources


def _removal(analysis: AnalysisDocument, element: str) -> list[DocumentEdit]:
    """The edits that remove ``element`` and every input bound to one of its outputs."""
    family = "operations" if element in analysis.operations else "views" if element in analysis.views else None
    if family is None:
        raise ValueError(f"Unknown analysis element {element!r}.")
    edits: list[DocumentEdit] = [RemoveProjectValue(op="remove", path=pointer((family, element)))]
    for name, holder in ("operations", analysis.operations), ("views", analysis.views):
        for target, value in holder.items():
            if target != element:
                edits.extend(
                    RemoveProjectValue(op="remove", path=pointer((name, target, "inputs", role)))
                    for role, source in value.inputs.items() if getattr(source, "element", None) == element
                )
    return edits


def _wire_edit(
    analysis: AnalysisDocument, wire: StudioAnalyzeWire, registry: OperationRegistry, states: DatasetPort | None,
):
    family = "operations" if wire.element in analysis.operations else "views"
    source = _port(wire.source, analysis, registry, states)
    if family == "operations":
        operation = analysis.operations[wire.element]
        expected = registry.resolve(operation.operation).value.definition.inputs.get(wire.role)
        if expected is None:
            raise ValueError(f"Unknown operation input {wire.role!r}.")
        if not isinstance(expected, DatasetPort):
            raise ValueError(f"Operation input {wire.role!r} does not accept a dataset.")
        require_dataset_port(source, expected)
    elif wire.element not in analysis.views:
        raise ValueError(f"Unknown figure {wire.element!r}.")
    return family, SetProjectValue(
        op="replace", path=pointer((family, wire.element, "inputs", wire.role)),
        value=wire.source.model_dump(mode="json"),
    )


def _review_wire(
    plan: ProjectPlan, key: RunKey, analysis: AnalysisDocument, family: str, wire: StudioAnalyzeWire,
    states: DatasetPort | None,
) -> None:
    """Refuse a wire the plan of ``proposal`` cannot resolve on ``key``'s run."""
    problems = [
        problem for problem in plan.diagnostics
        if (problem.variant, problem.step, problem.row) == (key.variant, key.step, key.row)
    ]
    path = f"payload.{family}.{wire.element}"
    if family == "views":
        row = next((row for row in plan.rows if row.key == key and isinstance(row, PlannedAnalyzeRun)), None)
        figure = row.figures.get(wire.element) if row is not None else None
        label = member_name(analysis.views, wire.element)
        if figure is None or row is None:
            problem = next((problem for problem in problems if problem.element == wire.element), None)
            raise ValueError(problem.detail if problem is not None else f"Cannot resolve figure {label}.")
        if wire.role not in figure.data:
            raise ValueError(f"{label} has no input {wire.role!r}.")
        inputs = analysis.views[wire.element].inputs | {wire.role: wire.source}
        schemas = {}
        for role, binding in inputs.items():
            if isinstance(binding, StepInput):
                port = states if binding.input == "states" else row.dataset_inputs.get(binding.input)
            else:
                port = row.dataset_outputs.get(binding.element, {}).get(binding.product)
            if port is None:
                raise ValueError(f"No dataset can be read for figure input {role!r}.")
            schemas[role] = port
        validate_figure_bindings(figure, schemas)
    for problem in problems:
        if problem.path in {f"{path}.inputs", f"{path}.inputs.{wire.role}"}:
            raise ValueError(problem.detail)


def _parameter_edits(payload: Payload, edits: list[DocumentEdit], registry: OperationRegistry) -> list[DocumentEdit]:
    """Give the first parameter write a container when its defaults were omitted."""
    operations = payload.get("operations", {})
    if not isinstance(operations, dict):
        return edits
    operations = dict(operations)
    present = {key for key, value in operations.items() if isinstance(value, dict) and "params" in value}
    adapted = []
    for edit in edits:
        match json_pointer_parts(edit.path):
            case ("operations", key):
                if isinstance(edit, SetProjectValue) and isinstance(edit.value, dict):
                    operations[key] = edit.value
                    present.discard(key)
                    if "params" in edit.value:
                        present.add(key)
                elif isinstance(edit, RemoveProjectValue):
                    operations.pop(key, None)
                    present.discard(key)
            case ("operations", key, "params"):
                if isinstance(edit, SetProjectValue):
                    present.add(key)
                elif isinstance(edit, RemoveProjectValue):
                    present.discard(key)
            case ("operations", key, "params", *_):
                operation = operations.get(key)
                if isinstance(edit, SetProjectValue) and key not in present and isinstance(operation, dict):
                    try:
                        registry.resolve(operation["operation"])
                    except KeyError:
                        pass
                    else:
                        adapted.append(SetProjectValue(op="add", path=pointer(("operations", key, "params")), value={}))
                        present.add(key)
        adapted.append(edit)
    return adapted


def analyze_edit(
    project: Project,
    plan: ProjectPlan,
    key: RunKey,
    document: str,
    change: StudioAnalyzeWire | StudioAnalyzeEdit | StudioAnalyzeRemove | StudioAnalyzeReset,
    chosen: frozenset[str],
    registry: OperationRegistry,
    states: DatasetPort | None,
    *,
    compile_candidate: Callable[[Project], ProjectPlan] | None = None,
) -> tuple[StoredEdit, ProjectPlan]:
    """Store ``change`` of ``document``, which ``key``'s step reads, for the ``chosen`` steps by the document rule.

    The chosen steps read ``document`` or a document based on it; when other steps read it too, a document based on
    it holds the change for the chosen ones. A wire is refused when the plan of the stored edit cannot resolve it.
    """
    chosen = chosen or frozenset({key.step})

    def users(name: str) -> frozenset[str]:
        return plan.document_users(key.variant, name)

    if key.step not in chosen or not chosen <= lit_steps(project.documents, users, document):
        raise ValueError("The chosen steps no longer match this document.")
    family = None
    if isinstance(change, StudioAnalyzeWire):
        analysis = _analysis(project, document)
        family, edit = _wire_edit(analysis, change, registry, states)
        edits: list[DocumentEdit] = [edit]
    elif isinstance(change, StudioAnalyzeEdit):
        edits = list(change.edits)
    elif isinstance(change, StudioAnalyzeRemove):
        edits = _removal(_analysis(project, document), change.element)
    else:
        edits = base_value_edits(project, document, change.paths)
    stored = document_write(
        project, document, edits, chosen, users, adapt=lambda payload: _parameter_edits(payload, edits, registry),
    )
    if not (stored.files or stored.created):
        return stored, plan
    candidate = preview_stored(project, stored)
    reviewed = compile_candidate(candidate) if compile_candidate is not None else compile_project(candidate, operations=registry)
    if family is not None:
        _review_wire(reviewed, key, analysis, family, change, states)
    return stored, reviewed


def _add_figure_layer(
    project: Project, document: str, change: StudioAnalyzeAddFigureLayer,
) -> tuple[str, StudioAnalyzeEdit]:
    view = _analysis(project, document).views.get(change.view)
    if view is None or not isinstance(view.figure, StoredFigureDocument):
        raise ValueError("Select a view with an inline Figure.")
    used = set(view.figure.layers)
    for name in project.documents:
        if name != document and document in base_chain(project.documents, name):
            based = _catalog_analysis(project, name)
            child = based.views.get(change.view) if based is not None else None
            if child is not None and isinstance(child.figure, StoredFigureDocument):
                used.update(child.figure.layers)
    key = generated_name(change.layer.mark, used.__contains__)
    edit = SetProjectValue(
        op="add", path=pointer(("views", change.view, "figure", "layers", key)),
        value=stored_project_value(change.layer),
    )
    return key, StudioAnalyzeEdit(kind="edit", edits=[edit])


def _variable_choices(
    definition: OperationDefinition, states: DatasetPort | None, dimensions: list[str],
) -> dict[str, list[str]]:
    """Each parameter's choices: the matching variables of a dataset selector, and the bound label dimensions of a
    parameter that lists the dimensions a port takes whole."""
    choices: dict[str, list[str]] = {}
    for port in definition.inputs.values():
        if isinstance(port, DatasetInputPort) and isinstance(port.takes, WholeDimensions):
            choices[port.takes.parameter] = dimensions
        if states is None or not isinstance(port, DatasetPort) or states.schema_ids[0] not in port.schema_ids:
            continue
        actual_port = operation_input_port(states, port)
        for parameter, required in port.for_dimensions(actual_port.dimensions).selected_variables.items():
            matching = []
            for name, actual in actual_port.variables.items():
                try:
                    require_dataset_fields({name: actual}, {name: required})
                except ValueError:
                    continue
                matching.append(name)
            choices[parameter] = sorted(matching)
    return choices


def _common_fields(
    left: dict[str, FieldDescriptor], right: dict[str, FieldDescriptor],
) -> dict[str, FieldDescriptor]:
    return {
        name: field if field == other else field.model_copy(update={
            "inline_values": field.inline_values if field.inline_values == other.inline_values else None,
            "label": field.label if field.label == other.label else None,
        })
        for name, field in left.items()
        if (other := right.get(name)) is not None
        and field.model_dump(exclude={"inline_values", "label"}) == other.model_dump(exclude={"inline_values", "label"})
    }


def _states_port(
    service: ProjectPlanService, view: ProjectPlanRead, row: PlannedAnalyzeRun | None, key: RunKey, bound: bool,
) -> DatasetPort | None:
    """The dataset port common to the saved states of every Evaluate run the ``states`` input of ``key`` stacks, or
    none when the step has no ``states`` input."""
    if not bound or row is None:
        return None
    bindings = row.inputs.get("states", ())
    planned = row.dataset_inputs.get("states")
    if any(binding.producer_element is not None for binding in bindings):
        return planned
    members = [binding.producer for binding in bindings]
    if not members or any(member is None or member.variant != key.variant for member in members):
        return None
    states = None
    for member in dict.fromkeys(members):
        evaluation = view.evaluations.get(member)
        if evaluation is None or evaluation.result is None:
            return None
        result = service.jobs.read_evaluation_result(
            project=view.settled.project_id, store=view.settled.store_path, job_id=evaluation.result.job_id,
        )
        if result.receipt != evaluation.result.receipt:
            raise ValueError("A selected Evaluate receipt changed.")
        with result.open_states() as dataset:
            descriptor = dataset.descriptor
        schema = descriptor.schema_id
        if schema is None:
            raise ValueError("The saved states have no declared dataset schema.")
        if states is None:
            states = DatasetPort(
                schema_ids=(schema,), variables=descriptor.variables, coordinates=descriptor.coords,
            )
        elif states.schema_ids != (schema,):
            raise ValueError("The selected Evaluate states have different dataset schemas.")
        else:
            states = DatasetPort(
                schema_ids=states.schema_ids,
                variables=_common_fields(states.variables, descriptor.variables),
                coordinates=_common_fields(states.coordinates, descriptor.coords),
            )
    assert states is not None
    return states.model_copy(update={
        "coordinates": planned.coordinates | states.coordinates,
    }) if planned is not None else states


def _state_models(
    project: Project, plan: ProjectPlan, row: PlannedAnalyzeRun, components: ComponentRegistry | None,
) -> list[ModelDocument]:
    members = {binding.producer for binding in row.inputs.get("states", ())}
    documents = dict.fromkeys(
        name for run in plan.runs if run.key in members
        if (name := run.document_ids.get(EVALUATION_MODEL) or run.document_ids.get("inputs.subject")) is not None
    )
    models = []
    for name in documents:
        document = compose_document(project.documents.get, name)
        if not isinstance(document, ProjectDiagnostic) and document.schema_id == LOADED_MODEL_SCHEMA_ID:
            loaded = LoadedModelDocument.model_validate(document.payload)
            document = compose_document(project.documents.get, loaded.model.document)
        if not isinstance(document, ProjectDiagnostic) and document.schema_id == STORED_MODEL_DOCUMENT_SCHEMA_ID:
            models.append(StoredModelDocument.model_validate(document.payload).lower(components))
    return models


def _task_names(plan: ProjectPlan, row: PlannedAnalyzeRun) -> dict[str, str]:
    """The current Task labels common to the Evaluate members whose states the Analyze reads."""
    names: dict[str, set[str | None]] = {}
    members = dict.fromkeys(binding.producer for binding in row.inputs.get("states", ()))
    for key in members:
        if key is None or not isinstance(run := plan.row(key), PlannedRun) or run.dataset_port is None:
            continue
        for field, descriptor in run.dataset_port.variables.items():
            if field.startswith("/v1/task/output/"):
                names.setdefault(field, set()).add(descriptor.label)
    return {
        field: (next(iter(labels)) if len(labels) == 1 else None) or path_parts(field)[-1]
        for field, labels in names.items()
    }


def _variable_names(
    options: dict[str, dict[str, list[str]]], states: DatasetPort | None, models: list[ModelDocument],
    task_names: dict[str, str],
) -> dict[str, str]:
    fields = states.variables if states is not None else {}
    names = {}
    for key in dict.fromkeys(key for selectors in options.values() for choices in selectors.values() for key in choices):
        if key in task_names:
            names[key] = task_names[key]
            continue
        field = fields.get(key)
        if field is not None and key.startswith("/v1/model/state/"):
            address = model_state_address(key)
            labels = {task_start(model, address).name for model in models}
            if len(labels) == 1:
                names[key] = labels.pop()
                continue
        names[key] = (field.label or field.quantity if field is not None else None) or "Signal"
    return names


def analyze_authoring(
    service: ProjectPlanService, project_id: str, body: StudioAnalyzeRequest
) -> StudioAnalyzeAuthoring:
    snapshot = service.folders.read(project_id, revision=body.revision)
    project = snapshot.project
    key = RunKey(body.key.variant, body.key.step, body.key.row)
    step = project.steps.get(key.step)
    if (
        step is None
        or step.kind != "analyze"
        or not isinstance(step.base, DocumentReference)
        or step.base.pointer
        or step.deltas
    ):
        raise ProjectFolderInvalid(f"{member_name(project.steps, key.step)} has no named Analysis document.")
    document = None if step.base.unresolved else project.documents.get(step.base.document)
    if body.change and (document is None or document.schema_id != ANALYSIS_SCHEMA):
        raise ProjectFolderInvalid("The selected step has no saved analysis document.")
    bound = step.inputs.get("states")
    references = bound if isinstance(bound, list) else [] if bound is None else [bound]
    states_available = bool(references) and all(
        isinstance(reference, StepReference) and not reference.variants and not reference.unresolved
        for reference in references
    )
    try:
        settled = service.folders.read_plan(project_id, revision=body.revision)
        plan = settled.plan
        if plan.variants and plan.variants[0] != key.variant:
            raise ValueError(f"The Analyze step is shared by all variants and is edited in {plan.variants[0]!r}, not {key.variant!r}")
        problems = []
        row = next((row for row in plan.rows if row.key == key and isinstance(row, PlannedAnalyzeRun)), None)
        step_problems = [
            problem for problem in plan.diagnostics
            if problem.step == key.step and problem.variant in (None, key.variant) and problem.row in (None, key.row)
        ]
        if row is None and not step_problems:
            raise ValueError(f"Run {key.row!r} no longer belongs to {member_name(project.steps, key.step)}.")
        if bound is not None and not states_available and not any(
            problem.path.startswith("inputs.states") for problem in step_problems
        ):
            problems.append(ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, "inputs.states", "Select Evaluate runs of this variant.", key.step,
                variant=key.variant, row=key.row,
            ))
        states = None
        if row is not None and states_available and row.inputs.get("states"):
            view = service.read(project_id, revision=body.revision)
            states = _states_port(service, view, row, key, states_available)
        registry = service.jobs.registries.operations
        templates = service.jobs.registries.figure_templates
        definitions = list(registry.definitions())
        stored = StoredEdit([])
        saved_document = step.base.document
        change = body.change
        created_layer = None
        if isinstance(change, StudioAnalyzeAddFigureLayer):
            created_layer, change = _add_figure_layer(project, saved_document, change)
        if change:
            stored, plan = analyze_edit(
                project, plan, key, saved_document, change, frozenset(body.chosen_steps), registry, states,
                compile_candidate=lambda candidate: preview_plan(service.folders, settled, candidate),
            )
            if stored.files or stored.created:
                project = preview_stored(project, stored)
                base = project.steps[key.step].base
                assert isinstance(base, DocumentReference)
                saved_document = base.document
                candidate = plan.row(key)
                row = candidate if isinstance(candidate, PlannedAnalyzeRun) else None
        analysis = _catalog_analysis(project, saved_document)
        sources = _sources(analysis, registry, states, row, templates) if analysis is not None else {}
        dimensions = sorted({
            label
            for bindings in row.inputs.values() for binding in bindings for label in binding.member.labels
        }) if row is not None else []
        problems.extend(
            problem for problem in plan.diagnostics
            if (problem.variant, problem.step, problem.row) == (key.variant, key.step, key.row)
            and problem.element is not None
            and path_parts(problem.path)[3:4] in (("params",), ("inputs",))
        )
        variables = {definition.key: _variable_choices(definition, states, dimensions) for definition in definitions}
        models = _state_models(project, plan, row, service.folders.components) if row is not None and states is not None else []
        task_names = _task_names(plan, row) if row is not None and states is not None else {}
    except (JobConflict, JobNotFound, KeyError, ValueError) as error:
        raise ProjectFolderInvalid(str(error)) from error
    return StudioAnalyzeAuthoring(
        revision=snapshot.revision,
        document=saved_document,
        operations=definitions,
        states=states,
        operation_outputs={name: dict(ports) for name, ports in row.dataset_outputs.items()} if row is not None else {},
        sources=sources,
        variables=variables,
        variable_names=_variable_names(variables, states, models, task_names),
        figure=StoredFigureDocument.from_lowered(VelocitySummary(VelocitySummarySettings())),
        figure_values=_figure_values(analysis),
        created_layer=created_layer if stored.files or stored.created else None,
        changed=changed_from_base(project, saved_document),
        files=stored.files,
        created=list(stored.created),
        problems=[
            StudioProjectWindowMapProblem.from_diagnostic(
                problem, declaration=project_declaration_name(settled.configuration.project),
            ) for problem in problems
        ],
    )
