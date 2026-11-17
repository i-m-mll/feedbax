"""Project the selected run's compiler and task facts onto stored Model addresses."""

from feedbax.compiler.model_document import lower_model_document
from feedbax.compiler.resolution import resolve_graph_parameters
from feedbax.contracts.model_address import ComponentPortAddress, ModelAddress, PortElement
from feedbax.contracts.model_document import SignalType
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.model_edit import ModelPortProjection, ModelProjection
from feedbax.expression_check import references
from feedbax.studio.schema import _enumerate_graph_ports
from feedbax.project_model_selection import selected_model as selected_model, selected_model_and_task, task_signal


def _port(document, composites, component, direction, port):
    return ModelAddress(
        document=document, composites=composites, element=PortElement(
            kind="port", end=ComponentPortAddress(owner="component", component=component, direction=direction, port=port),
        ),
    )


def model_projection(folders, project_id, body, components):
    model, task, objective, overrides = selected_model_and_task(folders, project_id, body)
    task_signals = tuple(task_signal(item) for item in task.exposed_data if item.bindable)
    names = {signal.id for signal in task_signals}
    supplied = {port.name for port in model.ports.inputs if overrides.get(port.name, port.name) in names}
    try:
        lowered = lower_model_document(
            body.document, model, components, supplied_inputs=frozenset(supplied),
        )
    except ValueError:
        # The selected Plan carries the addressed failure. Exposed task signals
        # are still known when the Model cannot be lowered.
        return ModelProjection(task_signals=task_signals, resolved_model=model)
    task_inputs = tuple(
        ModelAddress.model_validate({
            "document": body.document, "composites": [],
            "element": {"kind": "port", "end": {"owner": "inputs", "port": port.id}},
        }) for port in model.ports.inputs if port.name in supplied
    )
    try:
        resolution = resolve_graph_parameters(
            lowered.graph, {}, components, allow_incomplete=True,
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
    if isinstance(objective, dict):
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
    return ModelProjection(
        ports=tuple(ports), task_signals=task_signals, task_inputs=task_inputs,
        objective_ports=tuple(objective_ports.values()), resolved_model=model,
    )
