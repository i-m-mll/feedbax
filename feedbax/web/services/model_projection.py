"""Project the selected run's compiler and task facts onto stored Model addresses."""

from collections.abc import Iterator

from feedbax.compiler.model_document import lower_model_document
from feedbax.compiler.prototypes import prototypes_from_task_bindings
from feedbax.compiler.resolution import resolve_graph_parameters
from feedbax.contracts.graph import StudioTaskBindingSpec, StudioTaskDataSpec
from feedbax.contracts.model_address import ComponentPortAddress, ModelAddress, PortElement
from feedbax.contracts.model_document import ModelDocument, SignalType
from feedbax.contracts.model_edit import ModelPortProjection, ModelProjection, ModelProjectionRequest, ModelTaskSignal
from feedbax.contracts.project_plan import RunKey, SelectedRunDefinition
from feedbax.contracts.task_document import TaskDocument
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
        task = TaskDocument.model_validate(managed["task"])
        binding = StudioTaskBindingSpec(exposed_data=[
            StudioTaskDataSpec(
                id=signal.name, label=signal.name, kind="signal", path=signal.id,
                dtype={"integer": "int", "real": "float", "boolean": "bool"}[signal.type.base],
                expected_shape=["time", *signal.type.shape], bindable=True,
            ) for signal in task.signals if signal.type.base in {"integer", "real", "boolean"}
        ])
        objective = managed.get("objective")
    else:
        binding = StudioTaskBindingSpec.model_validate(definition.payload.get("task_binding_spec") or {})
    return ModelDocument.model_validate(model), binding, objective, isinstance(managed, dict)


def selected_task_signal(folders: ProjectFolderService, project_id: str, body: ModelProjectionRequest, signal: str) -> ModelTaskSignal:
    _, task, _, _ = _selected(folders, project_id, body)
    item = next((item for item in task.exposed_data if item.bindable and item.id == signal), None)
    if item is None:
        raise ProjectFolderInvalid("The selected task does not expose this signal")
    projected = _signal(item)
    return projected.model_copy(update={"type": projected.type.model_copy(update={"base": None})})


def model_projection(folders, project_id, body, components):
    model, task, objective, managed = _selected(folders, project_id, body)
    task_signals = tuple(_signal(item) for item in task.exposed_data if item.bindable)
    try:
        lowered = lower_model_document(body.document, model, components, None if managed else task)
    except ValueError:
        # The selected Plan carries the addressed failure. Exposed task signals
        # are still known when the Model cannot be lowered.
        return ModelProjection(task_signals=task_signals, resolved_model=model)
    task_inputs = tuple(address for path, address in lowered.origins.items() if path[:2] == ("task_binding_spec", "bindings"))
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
    for selector in _objective_selectors(objective):
        selected = loss_service.resolve_probe_selector(selector, lowered.graph)
        if selected and selected["type"] == "port" and selected["node"] in model.interior.components:
            address = _port(body.document, (), selected["node"], "output", selected["port"])
            objective_ports[address.to_text()] = address
    return ModelProjection(
        ports=tuple(ports), task_signals=task_signals, task_inputs=task_inputs,
        objective_ports=tuple(objective_ports.values()), resolved_model=model,
    )
