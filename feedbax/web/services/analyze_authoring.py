"""Project registered operation controls and review saved-folder Analyze connections."""

from feedbax.analysis.figure_presets import velocity_summary_figure
from feedbax.analysis.operations import OperationRegistry
from feedbax.contracts.analysis_document import AnalysisDocument, DatasetSource, StepDataset
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
from feedbax.contracts.project_resolution import BaseReaders
from feedbax.contracts.studio_analyze import (
    StudioAnalyzeAuthoring,
    StudioAnalyzeRequest,
    StudioAnalyzeWire,
)
from feedbax.contracts.studio_project import (
    AuthoredFileAddress,
    SetProjectValue,
    StudioProjectFilePatch,
)
from feedbax.jobs import JobConflict, JobNotFound
from feedbax.web.services.project_folder import ProjectFolderInvalid
from feedbax.web.services.project_plan import ProjectPlanService


def _port(
    source: DatasetSource,
    analysis: AnalysisDocument,
    registry: OperationRegistry,
    states: DatasetPort,
) -> DatasetPort:
    if isinstance(source, StepDataset):
        if source.input != "states":
            raise ValueError(f"Unknown Analyze input {source.input!r}.")
        return states
    try:
        operation = analysis.operations[source.element].operation
        return registry.resolve(operation).definition.outputs[source.product]
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
) -> StudioProjectFilePatch:
    analysis = AnalysisDocument.model_validate(project.documents[document].payload)
    family = "operations" if wire.element in analysis.operations else "views"
    source = _port(wire.source, analysis, registry, states)
    if family == "operations":
        operation = analysis.operations[wire.element]
        expected = registry.resolve(operation.operation).definition.inputs.get(wire.role)
        if expected is None:
            raise ValueError(f"Unknown operation input {wire.role!r}.")
        require_dataset_port(source, expected)
    elif wire.element not in analysis.views:
        raise ValueError(f"Unknown figure {wire.element!r}.")

    proposal = project.model_copy(deep=True)
    binding = wire.source.model_dump(mode="json")
    bindings = proposal.documents[document].payload[family][wire.element]["inputs"]
    operation = "replace" if wire.role in bindings else "add"
    bindings[wire.role] = binding
    plan = compile_project(proposal, operations=registry, readers=readers)
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
    return StudioProjectFilePatch(
        address=AuthoredFileAddress(kind="document", id=document),
        patches=[
            SetProjectValue(
                op=operation, path=f"/payload/{family}/{wire.element}/inputs/{wire.role}", value=binding
            )
        ],
    )


def _variable_choices(definition: OperationDefinition, states: DatasetPort) -> dict[str, list[str]]:
    choices: dict[str, list[str]] = {}
    for port in definition.inputs.values():
        if states.schema_ids[0] not in port.schema_ids:
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


def analyze_authoring(
    service: ProjectPlanService, project_id: str, body: StudioAnalyzeRequest
) -> StudioAnalyzeAuthoring:
    snapshot = service.folders.read(project_id, revision=body.revision)
    project = snapshot.project
    key = RunKey(body.key.variant, body.key.step, body.key.row)
    step = project.steps.get(key.step)
    if (
        project.sweep
        or key.variant != "default"
        or key.row != "default"
        or step is None
        or step.kind != "analyze"
        or step.sweep
        or not isinstance(step.base, DocumentReference)
        or step.base.pointer
        or step.deltas
        or step.document_overrides
    ):
        raise ProjectFolderInvalid("Select an unswept Analyze step with a named analysis document.")
    document = project.documents.get(step.base.document)
    if document is None or document.schema_id != "feedbax.analysis_document.v1":
        raise ProjectFolderInvalid("The selected step has no saved analysis document.")
    reference = step.inputs.get("states")
    if not isinstance(reference, StepReference) or reference.row is None or reference.variants:
        raise ProjectFolderInvalid("Select one saved Evaluate run as the Analyze input.")
    try:
        view = service.read(project_id, revision=body.revision)
        if any(
            planned.key.step != key.step and step.base.document in planned.document_uses
            for planned in view.settled.plan.rows
        ):
            raise ValueError("This analysis document is shared by more than one step.")
        row = view.evaluations.get(RunKey(key.variant, reference.step, reference.row))
        if row is None or row.result is None:
            raise ValueError("The selected Evaluate run has no saved states.")
        result = service.jobs.read_evaluation_result(
            project=project_id,
            store=view.settled.store_path,
            job_id=row.result.job_id,
        )
        if result.receipt != row.result.receipt:
            raise ValueError("The selected Evaluate receipt changed.")
        with result.open_states() as dataset:
            descriptor = dataset.descriptor
            if descriptor.schema_id is None:
                raise ValueError("The saved states have no declared dataset schema.")
            states = DatasetPort(
                schema_ids=(descriptor.schema_id,),
                variables=descriptor.variables,
                coordinates=descriptor.coords,
            )
        registry = service.jobs.registries.operations
        definitions = list(registry.definitions())
        reviewed = (
            [review_wire(
                project, key, step.base.document, body.wire, registry, states, readers=view.settled.readers,
            )]
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
        figure=velocity_summary_figure(),
        files=reviewed,
    )
