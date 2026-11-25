"""Project the selected run's compiler and task facts onto stored Model addresses."""

from collections import defaultdict
from typing import cast, get_args

from feedbax.compiler.model_document import inspect_model_signals
from feedbax.compiler.model_edit import conserving_port_type, model_ungroup_eligibility
from feedbax.compiler.model_ports import ModelRegistryPorts
from feedbax.compiler.model_signals import ModelSignalTable, ResolvedSignalLeaf
from feedbax.contracts.model_address import ComponentElement, ComponentPortAddress, FieldElement, ModelAddress, PortElement
from feedbax.contracts.model_document import ModelDocument, PhysicalPortType, SignalType
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.model_edit import (
    EvaluateModelRequest, ModelPhysicalPortProjection, ModelPortProjection, ModelProjection, ModelTaskInput, StepRuns,
)
from feedbax.contracts.expression_check import references
from feedbax.contracts.project import BasedDocument
from feedbax.planner.project_plan import RunKey
from feedbax.project_folder import ProjectFolderInvalid, model_edit_source
from feedbax.project_model_selection import (
    matched_task_inputs, model_task_signals, selected_model as selected_model, selected_model_and_task, task_signal,
)
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


def _publishes_origin(leaf: ResolvedSignalLeaf, origin: ModelAddress) -> bool:
    if leaf.address == origin:
        return True
    element = origin.element
    port = leaf.port.element
    return (
        leaf.origin == origin and leaf.port.composites == origin.composites
        and isinstance(element, FieldElement) and isinstance(element.parent, ComponentElement)
        and element.path[:1] == ("state",)
        and isinstance(port, PortElement) and isinstance(port.end, ComponentPortAddress)
        and port.end.component == element.parent.id
    )


def _objective_readers(objective: ObjectiveDocument | None, signals: ModelSignalTable) -> dict[str, dict[str, None]]:
    readers: dict[str, dict[str, None]] = defaultdict(dict)
    if objective is None:
        return readers
    readable = signals.readable_leaves
    for term in objective.terms:
        for reference in references(term.measure) | references(term.weight):
            if not reference.startswith("{"):
                continue
            try:
                address = ModelAddress.from_text(reference)
            except ValueError:
                continue
            leaf = signals.at(address)
            if leaf is None:
                continue
            places = [leaf]
            if leaf.origin is not None and leaf.origin != leaf.address:
                places.extend(item for item in readable if _publishes_origin(item, leaf.origin))
            for place in places:
                readers[place.address.to_text()][term.id] = None
                readers[place.port.to_text()][term.id] = None
    return readers


def evaluate_projection(folders, project_id, body, bundle):
    from feedbax.web.services.evaluate_sources import read_evaluate_model
    from feedbax.web.services.trial_authoring import _pairing

    key = RunKey(body.run.variant, body.run.step, body.run.row)
    plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
    reading = read_evaluate_model(
        folders, project_id, EvaluateModelRequest(
            revision=body.revision, run=body.run, scope=body.evaluate_scope or StepRuns(kind="step"),
        ), bundle.components, selected=plan, model_document=body.document,
    )
    model = reading.state.resolved_model
    definition = plan.selected_definitions.get(key)
    if definition is None:
        return model, (), None, {}
    try:
        documents, _, _ = _pairing(plan, key, bundle, definition=definition)
    except ValueError:
        return model, (), None, {}
    if definition.document_ids.get("inputs.subject") == body.document:
        model = documents.model
    return model, model_task_signals(folders, documents.task_source), documents.objective, documents.input_bindings


def model_projection(folders, project_id, body, bundle):
    snapshot = folders.read(project_id, revision=body.revision)
    payload, layout = model_edit_source(snapshot, body.document)
    authored = ModelDocument.model_validate(payload)
    differences = model_differences(folders, project_id, body, snapshot, authored.model_dump(mode="json"), layout)
    selected = None
    step = snapshot.project.steps.get(body.run.step) if body.run is not None else None
    if step is not None and step.kind == "evaluate":
        selected = evaluate_projection(folders, project_id, body, bundle)
    elif body.evaluate_scope is not None:
        raise ProjectFolderInvalid("Choose an Evaluate run for this Model selection.")
    elif body.run is not None:
        selected = selected_model_and_task(folders, project_id, body)
    model, task, objective, overrides = selected or (authored, (), None, {})
    model = model or authored
    task_signals = tuple(task_signal(item) for item in task)
    bound = matched_task_inputs(model, task, overrides)
    supplied = {port.name: signal.type for port, signal in bound}
    authority = ModelRegistryPorts(bundle.components)
    table = inspect_model_signals(model, bundle.components, supplied)
    readers = _objective_readers(objective, table)
    ports: list[ModelPortProjection] = []
    physical_ports: list[ModelPhysicalPortProjection] = []
    for port in _declared_ports(authority, model):
        if isinstance(port, ModelPhysicalPortProjection):
            physical_ports.append(port)
            continue
        leaves = table.at_port(port.address)
        if not leaves or leaves[0].path:
            base = table.structured_base(port.address)
            ports.append(ModelPortProjection(
                address=port.address,
                type=port.type.model_copy(update={"base": base}) if base is not None else port.type,
                readers=tuple(readers.get(port.address.to_text(), ())),
            ))
        for leaf in leaves:
            ports.append(ModelPortProjection(
                address=leaf.address, type=leaf.signal_type,
                readers=tuple(readers.get(leaf.address.to_text(), ())),
            ))
    return ModelProjection(
        ports=tuple(ports), physical_ports=tuple(physical_ports), problems=table.problems, task_signals=task_signals,
        task_inputs=tuple(
            ModelTaskInput(composites=(), element=_boundary_port((), "input", port.id).element, signal_id=signal.id)
            for port, signal in bound
        ), resolved_model=model,
        ungroup=model_ungroup_eligibility(model), **differences,
    )
