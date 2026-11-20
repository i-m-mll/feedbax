"""Project registered operation controls and review saved-folder Analyze connections."""

from feedbax.analysis.operations import OperationRegistry
from feedbax.builtins.figure_templates import VelocitySummary, VelocitySummarySettings
from feedbax.contracts.analysis_document import AnalysisDocument, DatasetSource, StepInput
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.figure_document import validate_figure_bindings
from feedbax.contracts.operation import (
    DatasetPort,
    OperationDefinition,
    require_dataset_fields,
    require_dataset_port,
)
from feedbax.contracts.project import DocumentReference, Project
from feedbax.contracts.project_inputs import StepReference
from feedbax.contracts.project_plan import PlannedAnalyzeRun, RunKey, compile_project
from feedbax.contracts.project_resolution import BaseReaders, ProjectDiagnostic, compose_document
from feedbax.contracts.studio_analyze import (
    StudioAnalyzeAuthoring,
    StudioAnalyzeRequest,
    StudioAnalyzeWire,
)
from feedbax.contracts.studio_project import SetProjectValue, StudioProjectFilePatch
from feedbax.jobs import JobConflict, JobNotFound
from feedbax.plot.templates import FigureTemplateRegistry
from feedbax.web.services.project_folder import ProjectFolderInvalid, preview_authored_files
from feedbax.web.services.project_plan import ProjectPlanRead, ProjectPlanService
from feedbax.web.services.trial_scope import shared_document_write


def _port(
    source: DatasetSource,
    analysis: AnalysisDocument,
    registry: OperationRegistry,
    states: DatasetPort,
) -> DatasetPort:
    if isinstance(source, StepInput):
        if source.input != "states":
            raise ValueError(f"Unknown Analyze input {source.input!r}.")
        return states
    try:
        operation = analysis.operations[source.element].operation
        return registry.resolve(operation).value.definition.outputs[source.product]
    except KeyError as error:
        raise ValueError(f"Unknown dataset output {source.element}.{source.product}.") from error


def review_wire(
    project: Project,
    key: RunKey,
    document: str,
    wire: StudioAnalyzeWire,
    registry: OperationRegistry,
    states: DatasetPort,
    *,
    readers: BaseReaders = BaseReaders(),
    figure_templates: FigureTemplateRegistry | None = None,
) -> list[StudioProjectFilePatch]:
    composed = compose_document(project.documents.get, document)
    if isinstance(composed, ProjectDiagnostic):
        raise ValueError(composed.detail)
    analysis = AnalysisDocument.model_validate(composed.payload)
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

    binding = wire.source.model_dump(mode="json")
    files = shared_document_write(project, document, [SetProjectValue(
        op="replace", path=f"/{family}/{wire.element}/inputs/{wire.role}", value=binding,
    )])
    proposal = preview_authored_files(project, files)
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
    return files


def _variable_choices(definition: OperationDefinition, states: DatasetPort) -> dict[str, list[str]]:
    choices: dict[str, list[str]] = {}
    for port in definition.inputs.values():
        if not isinstance(port, DatasetPort) or states.schema_ids[0] not in port.schema_ids:
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
    service: ProjectPlanService, project_id: str, revision: str, key: RunKey, reference: StepReference,
) -> tuple[DatasetPort, ProjectPlanRead]:
    view = service.read(project_id, revision=revision)
    plan = view.settled.plan
    if plan.variants[:1] != (key.variant,):
        raise ValueError(f"The Analyze step is shared by all variants and is edited in {plan.variants[0]!r}, not {key.variant!r}")
    subset = plan.subsets.get((key.variant, key.step, "states"))
    if subset is None or not subset.members or any(
        (member.variant, member.step) != (key.variant, reference.step) for member in subset.members
    ):
        raise ValueError(f"The Analyze input of variant {key.variant!r} holds no saved Evaluate runs of {reference.step!r}")
    states = None
    for member in subset.members:
        row = view.evaluations.get(member)
        if row is None or row.result is None:
            raise ValueError("A selected Evaluate run has no saved states.")
        result = service.jobs.read_evaluation_result(
            project=project_id, store=view.settled.store_path, job_id=row.result.job_id,
        )
        if result.receipt != row.result.receipt:
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
    return states, view


def analyze_authoring(
    service: ProjectPlanService, project_id: str, body: StudioAnalyzeRequest
) -> StudioAnalyzeAuthoring:
    snapshot = service.folders.read(project_id, revision=body.revision)
    project = snapshot.project
    key = RunKey(body.key.variant, body.key.step, body.key.row)
    step = project.steps.get(key.step)
    if (
        key.row != "default"
        or step is None
        or step.kind != "analyze"
        or step.sweep
        or not isinstance(step.base, DocumentReference)
        or step.base.pointer
        or step.deltas
        or step.document_overrides
    ):
        raise ProjectFolderInvalid(f"Analyze step {key.step!r} is swept or has no named analysis document")
    document = project.documents.get(step.base.document)
    if document is None or document.schema_id != "feedbax.analysis_document.v1":
        raise ProjectFolderInvalid("The selected step has no saved analysis document.")
    reference = step.inputs.get("states")
    if not isinstance(reference, StepReference) or reference.variants:
        raise ProjectFolderInvalid(f"The states input of Analyze step {key.step!r} is not one Evaluate step of this variant")
    try:
        states, view = _states_port(service, project_id, body.revision, key, reference)
        if any(
            planned.key.step != key.step and step.base.document in planned.document_uses
            for planned in view.settled.plan.rows
        ):
            raise ValueError("This analysis document is shared by more than one step.")
        registry = service.jobs.registries.operations
        definitions = list(registry.definitions())
        reviewed = (
            review_wire(
                project, key, step.base.document, body.wire, registry, states,
                readers=view.settled.readers, figure_templates=service.jobs.registries.figure_templates,
            )
            if body.wire
            else []
        )
    except (JobConflict, JobNotFound, KeyError, ValueError) as error:
        raise ProjectFolderInvalid(str(error)) from error
    return StudioAnalyzeAuthoring(
        revision=snapshot.revision,
        document=step.base.document,
        operations=definitions,
        states=states,
        variables={
            definition.key: _variable_choices(definition, states) for definition in definitions
        },
        figure=VelocitySummary(VelocitySummarySettings()),
        files=reviewed,
    )
