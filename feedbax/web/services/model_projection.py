"""Project the selected run's compiler and task facts onto stored Model addresses."""

from collections.abc import Iterator

from feedbax.compiler.model_document import lower_model_document
from feedbax.compiler.prototypes import prototypes_from_task_bindings
from feedbax.compiler.resolution import resolve_graph_parameters
from feedbax.compiler.trial_execution import resolve_python_task
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.graph import StudioTaskBindingSpec, StudioTaskDataSpec
from feedbax.contracts.model_address import ComponentPortAddress, ModelAddress, PortElement
from feedbax.contracts.model_document import ModelDocument, SignalType
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.model_edit import ModelPortProjection, ModelProjection, ModelProjectionRequest, ModelTaskSignal
from feedbax.contracts.project_plan import RunKey, SelectedRunDefinition
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.tasks.reaching import builtin_python_tasks
from feedbax.expression_check import references
from feedbax.objectives.service import loss_service
from feedbax.studio.schema import _enumerate_graph_ports
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService


def _objective_selectors(value: object) -> Iterator[str]:
    if not isinstance(value, dict):
        return
    if value.get("kind") == "loss_term":
        pending = [value.get("loss")]
        while pending:
            item = pending.pop()
            if not isinstance(item, dict):
                continue
            for name in ("selector", "target_selector"):
                if isinstance(selector := item.get(name), str):
                    yield selector
            if isinstance(children := item.get("children"), dict):
                pending.extend(children.values())
    elif value.get("kind") == "objective_spec" and isinstance(payload := value.get("payload"), dict):
        terms = payload.get("terms")
        if not isinstance(terms, list):
            return
        for term in terms:
            if not isinstance(term, dict):
                continue
            for name in ("selector", "target"):
                selection = term.get(name)
                if isinstance(selection, dict) and isinstance(selector := selection.get("selector"), str):
                    yield selector


def _signal(item: StudioTaskDataSpec) -> ModelTaskSignal:
    shape = item.expected_shape
    if shape and shape[0] == "time":
        shape = shape[1:]
    return ModelTaskSignal(
        id=item.id, label=item.label, type=SignalType(base=item.dtype, shape=shape, unit=item.units),
    )


def _port(document, composites, component, direction, port):
    return ModelAddress(
        document=document, composites=composites, element=PortElement(
            kind="port", end=ComponentPortAddress(owner="component", component=component, direction=direction, port=port),
        ),
    )


def selected_model(definition: SelectedRunDefinition, document: str) -> object:
    managed = definition.payload.get("trial_documents")
    path = "payload.trial_documents.model" if isinstance(managed, dict) else "payload.graph"
    if definition.document_ids.get(path) != document or definition.document_pointers.get(path):
        return None
    return managed.get("model") if isinstance(managed, dict) else definition.payload.get("graph")


def _selected(folders: ProjectFolderService, project_id: str, body: ModelProjectionRequest):
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
    definition = plan.selected_definitions.get(key)
    if definition is None or (model := selected_model(definition, body.document)) is None:
        raise ProjectFolderInvalid("The selected run does not use this Model")
    managed = definition.payload.get("trial_documents")
    objective = definition.payload.get("objective")
    if isinstance(managed, dict):
        if managed.get("python_task") is not None:
            selection = PythonTaskSelection.model_validate(managed["python_task"])
            template = resolve_python_task(selection, folders.python_tasks or builtin_python_tasks())
            signals = {} if isinstance(template, tuple) else template.signals
        else:
            task = TaskDocument.model_validate(managed["task"])
            signals = {signal.name: signal.type for signal in task.signals}
        binding = StudioTaskBindingSpec(exposed_data=[
            StudioTaskDataSpec(
                id=name, label=name, kind="signal", path=name,
                dtype={"integer": "int", "real": "float", "boolean": "bool"}[type_.base],
                expected_shape=["time", *type_.shape], bindable=True,
                units=next((name for name, unit in UNITS.items()
                            if name and name != "st" and unit.scale == 1 and unit.dimensions == type_.dimensions), None),
            ) for name, type_ in signals.items()
        ])
        objective = managed.get("objective")
    else:
        binding = StudioTaskBindingSpec.model_validate(definition.payload.get("task_binding_spec") or {})
    overrides = managed.get("input_overrides", {}) if isinstance(managed, dict) else {}
    return ModelDocument.model_validate(model), binding, objective, isinstance(managed, dict), overrides


def selected_task_signal(folders: ProjectFolderService, project_id: str, body: ModelProjectionRequest, signal: str) -> ModelTaskSignal:
    _, task, _, _, _ = _selected(folders, project_id, body)
    item = next((item for item in task.exposed_data if item.bindable and item.id == signal), None)
    if item is None:
        raise ProjectFolderInvalid("The selected task does not expose this signal")
    projected = _signal(item)
    return projected.model_copy(update={"type": projected.type.model_copy(update={"base": None})})


def model_projection(folders, project_id, body, components):
    model, task, objective, managed, overrides = _selected(folders, project_id, body)
    task_signals = tuple(_signal(item) for item in task.exposed_data if item.bindable)
    names = {signal.id for signal in task_signals}
    supplied = {port.name for port in model.ports.inputs if overrides.get(port.name, port.name) in names}
    try:
        lowered = lower_model_document(
            body.document, model, components, None if managed else task,
            supplied_inputs=frozenset(supplied) if managed else frozenset(),
        )
    except ValueError:
        # The selected Plan carries the addressed failure. Exposed task signals
        # are still known when the Model cannot be lowered.
        return ModelProjection(task_signals=task_signals, resolved_model=model)
    if managed:
        task_inputs = tuple(
            ModelAddress.model_validate({
                "document": body.document, "composites": [],
                "element": {"kind": "port", "end": {"owner": "inputs", "port": port.id}},
            }) for port in model.ports.inputs if port.name in supplied
        )
    else:
        task_inputs = tuple(
            address for path, address in lowered.origins.items() if path[:2] == ("task_binding_spec", "bindings")
        )
    try:
        resolution = resolve_graph_parameters(
            lowered.graph, prototypes_from_task_bindings(lowered.task) if lowered.task else {},
            components, allow_incomplete=True,
        )
    except ValueError:
        return ModelProjection(task_signals=task_signals, task_inputs=task_inputs, resolved_model=model)
    ports = []
    def collect(resolved, owner, composites):
        for port in _enumerate_graph_ports(resolved.graph, component_registry=components, resolution=resolved):
            value = port.value_schema
            if port.node_id is None:
                declarations = owner.ports.inputs if port.direction == "input" else owner.ports.outputs
                item = next((item for item in declarations if (item.id if composites else item.name) == port.port), None)
                if item is None:
                    continue
                address = ModelAddress.model_validate({
                    "document": body.document, "composites": composites,
                    "element": {"kind": "port", "end": {"owner": port.direction + "s", "port": item.id}},
                })
                unit = item.type.unit
            elif port.node_id in owner.interior.components:
                address = _port(body.document, composites, port.node_id, port.direction, port.port)
                unit = value.units
            else:
                continue
            ports.append(ModelPortProjection(address=address, type=SignalType(base=value.dtype, shape=value.shape, unit=unit)))
        for child, branch in resolved.subgraphs.items():
            component = owner.interior.components.get(child)
            if component is not None and component.kind == "composite":
                collect(branch, component, (*composites, child))
    collect(resolution, model, ())
    objective_ports = {}
    if managed and isinstance(objective, dict):
        document = ObjectiveDocument.model_validate(objective)
        for term in document.terms:
            for reference in references(term.measure):
                if not reference.startswith("{"):
                    continue
                try:
                    address = ModelAddress.from_text(reference)
                except ValueError:
                    continue
                if address.document == body.document and isinstance(address.element, PortElement):
                    objective_ports[reference] = address
    for selector in _objective_selectors(objective):
        selected = loss_service.resolve_probe_selector(selector, lowered.graph)
        if selected and selected["type"] == "port" and selected["node"] in model.interior.components:
            address = _port(body.document, (), selected["node"], "output", selected["port"])
            objective_ports[address.to_text()] = address
    return ModelProjection(
        ports=tuple(ports), task_signals=task_signals, task_inputs=task_inputs,
        objective_ports=tuple(objective_ports.values()), resolved_model=model,
    )
