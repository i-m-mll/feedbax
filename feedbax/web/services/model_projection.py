"""Project the selected run's compiler and task facts onto stored Model addresses."""

from typing import cast, get_args

from feedbax.compiler.model_document import lower_model_document
from feedbax.compiler.model_edit import conserving_port_type, model_ungroup_eligibility
from feedbax.compiler.model_ports import ModelRegistryPorts
from feedbax.compiler.resolution import resolve_graph_parameters
from feedbax.contracts.model_address import ComponentPortAddress, ModelAddress, PortElement
from feedbax.contracts.model_document import ModelDocument, PhysicalPortType, SignalType
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.model_edit import ModelPhysicalPortProjection, ModelPortProjection, ModelProjection
from feedbax.contracts.expression_check import references
from feedbax.compiler.graph_schema import _enumerate_graph_ports
from feedbax.contracts.project import BasedDocument
from feedbax.contracts.trial_bindings import selected_signal_id
from feedbax.planner.project_plan import RunKey
from feedbax.project_folder import ProjectFolderInvalid, model_edit_source
from feedbax.project_model_selection import selected_model as selected_model, selected_model_and_task, task_signal
from feedbax.compiler.model_differences import structural_differences
from feedbax.web.services.trial_scope import runs_scope, selected_value, value_model


def _port(composites, component, direction, port):
    return ModelAddress(
        composites=composites, element=PortElement(
            kind="port", end=ComponentPortAddress(owner="component", component=component, direction=direction, port=port),
        ),
    )


def _boundary_port(composites, direction, port):
    return ModelAddress.model_validate({
        "composites": composites,
        "element": {"kind": "port", "end": {"owner": "boundary" if direction == "physical" else direction + "s", "port": port}},
    })


def _declared_ports(authority: ModelRegistryPorts, owner, scope=()):
    for direction, ports in (("input", owner.ports.inputs), ("output", owner.ports.outputs)):
        for port in ports:
            yield ModelPortProjection(address=_boundary_port(scope, direction, port.id), type=port.type)
    for port in owner.ports.physical:
        yield ModelPhysicalPortProjection(address=_boundary_port(scope, "physical", port.id), type=port.type)
    for key, component in owner.interior.components.items():
        if component.kind == "composite":
            for direction, ports in (("input", component.ports.inputs), ("output", component.ports.outputs)):
                for port in ports:
                    yield ModelPortProjection(address=_port(scope, key, direction, port.id), type=port.type)
            for port in component.ports.physical:
                yield ModelPhysicalPortProjection(address=_port(scope, key, "physical", port.id), type=port.type)
            yield from _declared_ports(authority, component, (*scope, key))
            continue
        try:
            declarations = authority.layout(component.model_dump(mode="json"))
        except ValueError:
            continue  # A broken dynamic layout does not prevent reading the other occurrences.
        for (direction, port), declaration in declarations.items():
            if isinstance(declaration.type, SignalType):
                yield ModelPortProjection(address=_port(scope, key, direction, port), type=declaration.type)
            if direction == "physical":
                type_ = conserving_port_type(declaration, owner.interior.physical_domain)
                if type_ in get_args(PhysicalPortType):
                    yield ModelPhysicalPortProjection(
                        address=_port(scope, key, direction, port), type=cast(PhysicalPortType, type_),
                    )


def _dumped(payload) -> dict:
    return ModelDocument.model_validate(payload).model_dump(mode="json")


def model_differences(folders, project_id, body, snapshot, shown, layout):
    """The shown document's structural differences from its base, and the selected value's from the step's Model."""
    project = snapshot.project
    from_base = from_value = ()
    entry = project.documents[body.document]
    if isinstance(entry, BasedDocument):
        base, base_layout = model_edit_source(snapshot, entry.base.document)
        from_base = structural_differences(_dumped(base), shown, base_layout.get("positions", {}))
    if body.rows and body.run is not None:
        plan = folders.read_plan(project_id, revision=body.revision).plan
        key = RunKey(body.run.variant, body.run.step, body.rows[0])
        named = selected_value(project, key.step, runs_scope(project, plan, key, tuple(body.rows), body.axes or {}))
        if named is not None:
            from_value = structural_differences(
                shown, _dumped(value_model(project, key.step, named, body.document)), layout.get("positions", {}),
            )
    return {"base_differences": from_base, "value_differences": from_value}


def model_projection(folders, project_id, body, components):
    snapshot = folders.read(project_id, revision=body.revision)
    payload, layout = model_edit_source(snapshot, body.document)
    authored = ModelDocument.model_validate(payload)
    differences = model_differences(folders, project_id, body, snapshot, authored.model_dump(mode="json"), layout)
    selected = None
    if body.run is not None:
        try:
            selected = selected_model_and_task(folders, project_id, body)
        except ProjectFolderInvalid:
            # A broken pairing can leave the current Model independently readable.
            pass
    model, task, objective, overrides = selected or (authored, None, None, {})
    task_signals = tuple(task_signal(item) for item in task.exposed_data if item.bindable) if task is not None else ()
    names = {signal.label: signal.id for signal in task_signals}
    ids = set(names.values())
    bound = tuple(
        port for port in model.ports.inputs
        if selected_signal_id(port.id, port.name, names, overrides) in ids
    )
    supplied = {port.name for port in bound}
    authority = ModelRegistryPorts(components)
    ports: dict[str, ModelPortProjection] = {}
    physical_ports: list[ModelPhysicalPortProjection] = []
    for port in _declared_ports(authority, model):
        if isinstance(port, ModelPhysicalPortProjection):
            physical_ports.append(port)
        else:
            ports[port.address.to_text()] = port
    task_inputs = tuple(
        ModelAddress.model_validate({
            "composites": [],
            "element": {"kind": "port", "end": {"owner": "inputs", "port": port.id}},
        }) for port in bound
    )
    known = dict(
        task_signals=task_signals, task_inputs=task_inputs, resolved_model=model, physical_ports=tuple(physical_ports),
        ungroup=model_ungroup_eligibility(model), **differences,
    )
    try:
        lowered = lower_model_document(
            model, components, supplied_inputs=frozenset(supplied),
        )
    except ValueError:
        return ModelProjection(ports=tuple(ports.values()), **known)
    try:
        resolution = resolve_graph_parameters(
            lowered.graph, {}, components, allow_incomplete=True,
        )
    except ValueError:
        return ModelProjection(ports=tuple(ports.values()), **known)

    def collect(resolved, owner, composites):
        for port in _enumerate_graph_ports(resolved.graph, component_registry=components, resolution=resolved):
            value = port.value_schema
            if port.node_id is None:
                declarations = owner.ports.inputs if port.direction == "input" else owner.ports.outputs
                item = next((item for item in declarations if (item.id if composites else item.name) == port.port), None)
                if item is None:
                    continue
                address = _boundary_port(composites, port.direction, item.id)
            elif port.node_id in owner.interior.components:
                address = _port(composites, port.node_id, port.direction, port.port)
            else:
                continue
            key = address.to_text()
            declared = ports[key].type if key in ports else SignalType()
            ports[key] = ModelPortProjection(address=address, type=SignalType(
                base=declared.base or value.dtype, shape=declared.shape if declared.shape is not None else value.shape,
                unit=declared.unit or value.units,
            ))
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
        ports=tuple(ports.values()), objective_ports=tuple(objective_ports.values()), **known,
    )
