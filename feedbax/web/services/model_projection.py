"""Project the selected run's compiler and task facts onto stored Model addresses."""

from feedbax.compiler.model_document import lower_model_document
from feedbax.compiler.resolution import resolve_graph_parameters
from feedbax.contracts.model_address import ComponentPortAddress, ModelAddress, PortElement
from feedbax.contracts.model_document import ModelDocument, SignalType
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.model_edit import ModelPortProjection, ModelProjection
from feedbax.contracts.expression_check import references
from feedbax.compiler.graph_schema import _enumerate_graph_ports
from feedbax.contracts.project import BasedDocument
from feedbax.contracts.trial_bindings import selected_signal_id
from feedbax.planner.project_plan import RunKey
from feedbax.project_folder import model_edit_source
from feedbax.project_model_selection import selected_model as selected_model, selected_model_and_task, task_signal
from feedbax.compiler.model_differences import structural_differences
from feedbax.web.services.trial_scope import runs_scope, selected_value, value_model


def _port(composites, component, direction, port):
    return ModelAddress(
        composites=composites, element=PortElement(
            kind="port", end=ComponentPortAddress(owner="component", component=component, direction=direction, port=port),
        ),
    )


def _dumped(payload) -> dict:
    return ModelDocument.model_validate(payload).model_dump(mode="json")


def model_differences(folders, project_id, body):
    """The shown document's structural differences from its base, and the selected value's from the step's Model."""
    snapshot = folders.read(project_id, revision=body.revision)
    project = snapshot.project
    payload, layout = model_edit_source(snapshot, body.document)
    shown = _dumped(payload)
    from_base = from_value = ()
    entry = project.documents[body.document]
    if isinstance(entry, BasedDocument):
        base, base_layout = model_edit_source(snapshot, entry.base.document)
        from_base = structural_differences(_dumped(base), shown, base_layout.get("positions", {}))
    if body.rows:
        plan = folders.read_plan(project_id, revision=body.revision).plan
        key = RunKey(body.run.variant, body.run.step, body.rows[0])
        named = selected_value(project, key.step, runs_scope(project, plan, key, tuple(body.rows), body.axes or {}))
        if named is not None:
            from_value = structural_differences(
                shown, _dumped(value_model(project, key.step, named, body.document)), layout.get("positions", {}),
            )
    return {"base_differences": from_base, "value_differences": from_value}


def model_projection(folders, project_id, body, components):
    differences = model_differences(folders, project_id, body)
    model, task, objective, overrides = selected_model_and_task(folders, project_id, body)
    task_signals = tuple(task_signal(item) for item in task.exposed_data if item.bindable)
    names = {signal.label: signal.id for signal in task_signals}
    ids = set(names.values())
    supplied = {
        port.name for port in model.ports.inputs
        if selected_signal_id(port.id, port.name, names, overrides) in ids
    }
    try:
        lowered = lower_model_document(
            model, components, supplied_inputs=frozenset(supplied),
        )
    except ValueError:
        # The selected Plan carries the addressed failure. Exposed task signals
        # are still known when the Model cannot be lowered.
        return ModelProjection(task_signals=task_signals, resolved_model=model, **differences)
    task_inputs = tuple(
        ModelAddress.model_validate({
            "composites": [],
            "element": {"kind": "port", "end": {"owner": "inputs", "port": port.id}},
        }) for port in model.ports.inputs if port.name in supplied
    )
    try:
        resolution = resolve_graph_parameters(
            lowered.graph, {}, components, allow_incomplete=True,
        )
    except ValueError:
        return ModelProjection(task_signals=task_signals, task_inputs=task_inputs, resolved_model=model, **differences)
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
                    "composites": composites,
                    "element": {"kind": "port", "end": {"owner": port.direction + "s", "port": item.id}},
                })
                unit = item.type.unit
            elif port.node_id in owner.interior.components:
                address = _port(composites, port.node_id, port.direction, port.port)
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
                if isinstance(address.element, PortElement):
                    objective_ports[reference] = address
    return ModelProjection(
        ports=tuple(ports), task_signals=task_signals, task_inputs=task_inputs,
        objective_ports=tuple(objective_ports.values()), resolved_model=model, **differences,
    )
