"""Project registered operation controls and store the edits of an Analyze step's analysis document."""

from feedbax.analysis.operations import OperationRegistry
from feedbax.builtins.figure_templates import VelocitySummary, VelocitySummarySettings
from feedbax.contracts.analysis_document import AnalysisDocument, DatasetSource, StepInput
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.figure_document import validate_figure_bindings
from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.operation import (
    DatasetInputPort,
    DatasetPort,
    OperationDefinition,
    WholeDimensions,
    require_dataset_fields,
    require_dataset_port,
)
from feedbax.contracts.project import DocumentReference, DocumentEdit, Project, RemoveProjectValue, SetProjectValue
from feedbax.contracts.project_io import project_declaration_name
from feedbax.contracts.project_inputs import StepReference
from feedbax.planner.project_plan import PlannedAnalyzeRun, RunKey, compile_project
from feedbax.contracts.project_resolution import BaseReaders, ProjectDiagnostic, ResolutionCode, compose_document
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_analyze import (
    StudioAnalyzeAuthoring,
    StudioAnalyzeChange,
    StudioAnalyzeEdit,
    StudioAnalyzeRemove,
    StudioAnalyzeRequest,
    StudioAnalyzeWire,
)
from feedbax.jobs import JobConflict, JobNotFound
from feedbax.plot.templates import FigureTemplateRegistry
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
    proposal: Project, key: RunKey, analysis: AnalysisDocument, family: str, wire: StudioAnalyzeWire,
    registry: OperationRegistry, states: DatasetPort | None, readers: BaseReaders, figure_templates,
) -> None:
    """Refuse a wire the plan of ``proposal`` cannot resolve on ``key``'s run."""
    plan = compile_project(
        proposal, operations=registry, figure_templates=figure_templates, readers=readers,
    )
    problems = [
        problem for problem in plan.diagnostics
        if (problem.variant, problem.step, problem.row) == (key.variant, key.step, key.row)
    ]
    path = f"payload.{family}.{wire.element}"
    if family == "views":
        row = next((row for row in plan.rows if row.key == key and isinstance(row, PlannedAnalyzeRun)), None)
        figure = row.figures.get(wire.element) if row is not None else None
        if figure is None:
            problem = next((problem for problem in problems if problem.element == wire.element), None)
            raise ValueError(problem.detail if problem is not None else f"Cannot resolve figure {wire.element!r}.")
        if wire.role not in figure.data:
            raise ValueError(f"Unknown figure input {wire.element}.{wire.role}.")
        inputs = analysis.views[wire.element].inputs | {wire.role: wire.source}
        validate_figure_bindings(
            figure, {role: _port(binding, analysis, registry, states) for role, binding in inputs.items()},
        )
    for problem in problems:
        if problem.path in {f"{path}.inputs", f"{path}.inputs.{wire.role}"}:
            raise ValueError(problem.detail)


def analyze_edit(
    project: Project,
    plan,
    key: RunKey,
    document: str,
    change: StudioAnalyzeChange,
    chosen: frozenset[str],
    registry: OperationRegistry,
    states: DatasetPort | None,
    *,
    readers: BaseReaders = BaseReaders(),
    figure_templates: FigureTemplateRegistry | None = None,
) -> StoredEdit:
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
    stored = document_write(project, document, edits, chosen, users)
    if family is not None and (stored.files or stored.created):
        _review_wire(
            preview_stored(project, stored), key, analysis, family, change, registry, states, readers, figure_templates,
        )
    return stored


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
        for parameter, required in port.selected_variables.items():
            matching = []
            for name, actual in states.variables.items():
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
        name: field if field == other else field.model_copy(update={"inline_values": None})
        for name, field in left.items()
        if (other := right.get(name)) is not None
        and field.model_dump(exclude={"inline_values"}) == other.model_dump(exclude={"inline_values"})
    }


def _states_port(
    service: ProjectPlanService, view: ProjectPlanRead, row: PlannedAnalyzeRun | None, key: RunKey, bound: bool,
) -> DatasetPort | None:
    """The dataset port common to the saved states of every Evaluate run the ``states`` input of ``key`` stacks, or
    none when the step has no ``states`` input."""
    if not bound or row is None:
        return None
    members = [binding.producer for binding in row.inputs.get("states", ())]
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
    return states


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
        raise ProjectFolderInvalid(f"Analyze step {key.step!r} has no named analysis document")
    document = None if step.base.unresolved else project.documents.get(step.base.document)
    if body.change and (document is None or document.schema_id != "feedbax.analysis_document.v1"):
        raise ProjectFolderInvalid("The selected step has no saved analysis document.")
    bound = step.inputs.get("states")
    references = bound if isinstance(bound, list) else [] if bound is None else [bound]
    states_available = bool(references) and all(
        isinstance(reference, StepReference) and not reference.variants and not reference.unresolved
        for reference in references
    )
    try:
        view = service.read(project_id, revision=body.revision)
        plan = view.settled.plan
        if plan.variants and plan.variants[0] != key.variant:
            raise ValueError(f"The Analyze step is shared by all variants and is edited in {plan.variants[0]!r}, not {key.variant!r}")
        problems = [
            problem for problem in plan.diagnostics
            if problem.step == key.step and problem.variant in (None, key.variant) and problem.row in (None, key.row)
        ]
        row = next((row for row in plan.rows if row.key == key and isinstance(row, PlannedAnalyzeRun)), None)
        if row is None and not problems:
            raise ValueError(f"Run {key.row!r} no longer belongs to step {key.step!r}.")
        if bound is not None and not states_available and not any(
            problem.path.startswith("inputs.states") for problem in problems
        ):
            problems.append(ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, "inputs.states", "Select Evaluate runs of this variant.", key.step,
                variant=key.variant, row=key.row,
            ))
        states = _states_port(service, view, row, key, states_available)
        registry = service.jobs.registries.operations
        definitions = list(registry.definitions())
        dimensions = sorted({
            label
            for bindings in row.inputs.values() for binding in bindings for label in binding.member.labels
        }) if row is not None else []
        stored = analyze_edit(
            project, plan, key, step.base.document, body.change, frozenset(body.chosen_steps), registry, states,
            readers=view.settled.readers, figure_templates=service.jobs.registries.figure_templates,
        ) if body.change else StoredEdit([])
    except (JobConflict, JobNotFound, KeyError, ValueError) as error:
        raise ProjectFolderInvalid(str(error)) from error
    return StudioAnalyzeAuthoring(
        revision=snapshot.revision,
        document=step.base.document,
        operations=definitions,
        states=states,
        variables={
            definition.key: _variable_choices(definition, states, dimensions) for definition in definitions
        },
        figure=VelocitySummary(VelocitySummarySettings()),
        changed=changed_from_base(project, step.base.document),
        files=stored.files,
        created=list(stored.created),
        problems=[
            StudioProjectWindowMapProblem.from_diagnostic(
                problem, declaration=project_declaration_name(view.settled.configuration.project),
            ) for problem in problems
        ],
    )
