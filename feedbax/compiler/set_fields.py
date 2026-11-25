"""Lower a step's Set placements to one Model and the Task signals that switch it."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import product

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.contracts.expression_values import (
    CallValue, DrawValue, Expression, LiteralValue, NameValue, ValueBase, ValueType,
)
from feedbax.contracts.model_address import (
    AttachmentElement, ComponentElement, ComponentPortAddress, ComponentPortEnd, FieldElement, InputBoundaryEnd,
    ModelAddress, PortElement, SignalSource, WireElement,
)
from feedbax.contracts.model_attachment import SetField
from feedbax.contracts.model_document import (
    CompositeComponent, GraphInterior, InterventionAttachment, LeafComponent, ModelDocument, ModelWire,
    SignalInput, SignalType,
)
from feedbax.contracts.project import Document, apply_differences
from feedbax.contracts.project_sweep import CooccurringSetTrials, ListedSetTrials, SetEntryParts, TrialSetField
from feedbax.contracts.set_document import SetEntry, read_set
from feedbax.contracts.task_document import (
    Cell, Link, SetVariableChoices, Signal, TaskAdditions, TaskDocument, ValidationTrial, Variable, add_to_task,
)
from feedbax.contracts.window import Window


class SetFieldProblem(ValueError):
    def __init__(self, axis: str, set_name: str, entry: str | None, reason: str):
        self.axis = axis
        self.set_name = set_name
        self.entry = entry
        super().__init__(reason)


def _id(*parts: str) -> str:
    return "set-" + sha256("\0".join(parts).encode()).hexdigest()[:28]


def _call(function: str, *args: Expression) -> CallValue:
    return CallValue(function=function, args=args)


def _component_port(component: str, port: str) -> ComponentPortEnd:
    return ComponentPortEnd(owner="component", component=component, port=port)


_BOOL = ValueType(base=ValueBase.BOOLEAN)
_INT = ValueType(base=ValueBase.INTEGER)


def _entry_scope(axis: str, entry: str, source: SetEntry) -> SetEntry:
    refs = {variable.id: _id(axis, entry, variable.id) for variable in source.task_variables}

    def rename(value):
        if isinstance(value, dict):
            if value.get("kind") == "name" and isinstance(reference := value.get("reference"), str) and reference in refs:
                return value | {"reference": refs[reference]}
            return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        return value

    renamed = rename(source.model_dump(mode="json"))
    for variable in renamed.get("task_variables", ()):
        variable["id"] = refs[variable["id"]]
        variable["name"] = f"{entry} · {variable['name']}"
    for link in renamed.get("task_links", ()):
        link["variable"] = refs.get(link["variable"], link["variable"])
    return SetEntry.model_validate(renamed)


def _attachment_links(
    model: ModelDocument, links: tuple[Link, ...], components: ComponentRegistry,
) -> tuple[ModelDocument, tuple[Link, ...]]:
    writes, remaining = [], []
    for link in links:
        try:
            address = ModelAddress.from_text(link.model_field)
        except ValueError:
            remaining.append(link)
            continue
        at = address.element
        if isinstance(at, FieldElement) and isinstance(at.parent, AttachmentElement) and at.path[0] in {"fields", "window"}:
            value = NameValue(reference=link.variable).model_dump(mode="json")
            writes.append(SetField(op="set_field", address=address, value=value))
        else:
            remaining.append(link)
    return resolve_effective_model(model, components, tuple(writes)), tuple(remaining)


@dataclass(frozen=True)
class _Alternative:
    axis: str
    set_name: str
    entry: str
    predicate: Expression
    cooccurring: bool
    wire: ModelWire

    def problem(self, reason: str) -> SetFieldProblem:
        return SetFieldProblem(self.axis, self.set_name, self.entry, reason)


class _Union:
    def __init__(
        self, model: ModelDocument, task: TaskDocument, registry: ComponentRegistry, *, scope: tuple[str, ...] = (),
    ):
        self.model = model
        self.task = task
        self.registry = registry
        self.scope = scope
        self.variables = list(task.variables)
        self.signals = list(task.signals)
        self.links = list(task.links)
        self.inputs = list(model.ports.inputs)
        self.components = dict(model.interior.components)
        self.wires = dict(model.interior.wires)
        self.port_attachments = model.interior.attachments
        self.alternatives: dict[str, list[_Alternative]] = defaultdict(list)
        self.conditions: list[tuple[dict[str, int], ...]] = []
        self.curl_activation: dict[str, tuple[bool, list[Expression]]] = {}

    def variable(
        self, id: str, value_type: ValueType, value: Expression, *, choices: SetVariableChoices | None = None,
    ) -> NameValue:
        if any(var.name == id or var.id == id for var in self.variables):
            raise ValueError(f"Task variable {id!r} is already in use.")
        self.variables.append(Variable(id=id, name=id, type=value_type, value=value, set_choices=choices))
        return NameValue(reference=id)

    def gate(self, alt: _Alternative, gate_id: str, window: Window | None) -> InputBoundaryEnd:
        first, last = self.task.epochs[0].id, self.task.epochs[-1].id
        self.signals.append(Signal(
            id=gate_id, name=gate_id, type=_BOOL, absent=LiteralValue(value=False), window=window,
            cells=(Cell(id=gate_id, first_epoch=first, last_epoch=last, value=alt.predicate),),
        ))
        self.inputs.append(SignalInput(id=gate_id, name=gate_id, type=SignalType(base="bool", shape=())))
        return InputBoundaryEnd(owner="inputs", port=gate_id)

    def operands(self, attachment: InterventionAttachment, context: str) -> InterventionAttachment:
        fields = dict(attachment.fields)
        for key, value in fields.items():
            reference = _task_reference(value)
            if reference is None:
                continue
            variable = next((variable for variable in self.variables if variable.id == reference), None)
            if variable is None:
                raise ValueError(f"Intervention {attachment.id!r} links to missing Task variable {reference!r}.")
            id = _id(*self.scope, context, attachment.id, key)
            self.signals.append(Signal(
                id=id, name=id, type=variable.type,
                cells=(Cell(id=id, first_epoch=self.task.epochs[0].id, last_epoch=self.task.epochs[-1].id,
                            value=NameValue(reference=reference)),),
            ))
            self.inputs.append(SignalInput(
                id=id, name=id, type=SignalType(base=variable.type.base.value, shape=variable.type.shape),
            ))
            address = ModelAddress(
                composites=self.scope, element=PortElement(kind="port", end=InputBoundaryEnd(owner="inputs", port=id)),
            )
            fields[key] = NameValue(reference=address.to_text()).model_dump(mode="json")
        return attachment.model_copy(update={"fields": fields}) if fields != attachment.fields else attachment

    def added_components(self, alt: _Alternative, model: ModelDocument, added_wires: set[str]):
        nominal = self.model.interior
        added_nodes = model.interior.components.keys() - nominal.components.keys()
        for node in added_nodes:
            component = model.interior.components[node]
            if not isinstance(component, LeafComponent) or component.type != "feedbax.CurlField":
                raise alt.problem(f"Added component {node!r} has no qualified trial switch.")
            if node in self.components and self.components[node] != component:
                raise alt.problem(f"Added component {node!r} conflicts with another entry.")
            self.components[node] = component
        for id in added_wires:
            wire = model.interior.wires[id]
            if not isinstance(wire.target, ComponentPortEnd) or wire.target.component not in added_nodes:
                raise alt.problem(f"Added wire {id!r} does not feed a switchable added component.")
            if id in self.wires and self.wires[id] != wire:
                raise alt.problem(f"Added wire {id!r} conflicts with another entry.")
            self.wires[id] = wire

    def entry(self, axis: str, placement: TrialSetField, entry: str, predicate: Expression, authored: SetEntry) -> None:
        set_name = placement.axis.set
        cooccurring = isinstance(placement.sampling, CooccurringSetTrials)
        try:
            source = _entry_scope(axis, entry, authored)
            variant = ModelDocument.model_validate(
                apply_differences(self.model.model_dump(mode="json"), source.differences),
            )
            variant, links = _attachment_links(variant, source.task_links, self.registry)
            task = add_to_task(
                self.task.model_copy(update={"variables": tuple(self.variables), "links": tuple(self.links)}),
                (TaskAdditions(task_variables=source.task_variables, task_links=links),),
            )
            self.variables = list(task.variables)
            self.links = list(task.links)
        except ValueError as error:
            raise SetFieldProblem(axis, set_name, entry, str(error)) from error
        before, after = self.model.interior, variant.interior
        changes = {id: wire for id, wire in after.wires.items() if id in before.wires and before.wires[id] != wire}
        added = after.wires.keys() - before.wires.keys()
        stable = after.model_copy(update={"wires": before.wires, "components": before.components})
        if variant.model_copy(update={"interior": stable}) != self.model or not before.wires.keys() <= after.wires.keys():
            raise SetFieldProblem(axis, set_name, entry, "The entry has no qualified trial switch for its Model changes.")
        for id, node in before.components.items():
            other = after.components.get(id)
            if other != node and not self.switch_curl(id, node, other, predicate):
                raise SetFieldProblem(axis, set_name, entry, f"Component {id!r} has no qualified trial switch for the change.")
        if after.components.keys() != before.components.keys() or added:
            if not changes:
                raise SetFieldProblem(axis, set_name, entry, "The added branch has no existing wire it can switch.")
            exemplar = _Alternative(axis, set_name, entry, predicate, cooccurring, next(iter(changes.values())))
            self.added_components(exemplar, variant, added)
        for id, wire in changes.items():
            alt = _Alternative(axis, set_name, entry, predicate, cooccurring, wire)
            if before.wires[id].target != wire.target:
                raise alt.problem(f"Wire {id!r} changes its target, which has no qualified trial switch.")
            self.alternatives[id].append(alt)

    def switch_curl(self, id, node, other, predicate: Expression) -> bool:
        if not isinstance(node, LeafComponent) or not isinstance(other, LeafComponent) or node.type != "feedbax.CurlField":
            return False
        base = node.fields.get("active", False)
        active = other.fields.get("active", False)
        if type(base) is not bool or type(active) is not bool:
            return False
        fields = {key: value for key, value in other.fields.items() if key != "active"}
        if "active" in node.fields:
            fields["active"] = base
        if other.model_copy(update={"fields": fields}) != node:
            return False
        if base == active:
            return True
        param = _component_port(id, "params_override")
        address = _curl_address(id)
        if any(link.model_field == address for link in self.task.links) or any(
            wire.target == param for wire in self.model.interior.wires.values()
        ):
            return False
        _, selectors = self.curl_activation.setdefault(id, (base, []))
        selectors.append(predicate)
        return True

    def placement(self, axis: str, placement: TrialSetField, read: Callable[[str], Document | None]):
        try:
            entries = read_set(read, placement.axis.set).entries
            sample = placement.sampling
            by_id = tuple(entries)
            if isinstance(sample, ListedSetTrials):
                weights = (1.0,) * len(by_id)
                choices = tuple(range(1, len(by_id) + 1))
            elif isinstance(sample, CooccurringSetTrials):
                weights, choices = (), ()
            else:
                weights = (sample.none, *(sample.weights.get(entry, 1.0) for entry in by_id))
                choices = tuple(range(len(by_id) + 1))
            if not isinstance(sample, ListedSetTrials) and sample.weights.keys() - entries.keys():
                raise ValueError("Set sampling names an entry that the Set no longer has.")
            if isinstance(sample, CooccurringSetTrials):
                gates = []
                for entry in by_id:
                    parts = sample.weights.get(entry, SetEntryParts())
                    if not (parts.off or parts.on):
                        raise SetFieldProblem(axis, placement.axis.set, entry, "The switch has zero total parts.")
                    gates.append(self.variable(_id(axis, entry), _BOOL, DrawValue(
                        family="choice", args=(LiteralValue(value=False), LiteralValue(value=True)),
                        weights=(parts.off, parts.on),
                    ), choices=SetVariableChoices(set=placement.axis.set, entries=(None, entry))))
            else:
                if not any(weights):
                    raise ValueError("The choice has zero total parts.")
                selected = self.variable(axis, _INT, DrawValue(
                    family="choice", args=tuple(LiteralValue(value=value) for value in choices), weights=weights,
                ), choices=SetVariableChoices(set=placement.axis.set, entries=(None, *by_id)))
                if isinstance(sample, ListedSetTrials):
                    self.conditions.append(tuple({selected.reference: value} for value in choices))
                gates = [_call("eq", selected, LiteralValue(value=index)) for index in range(1, len(by_id) + 1)]
        except SetFieldProblem:
            raise
        except ValueError as error:
            raise SetFieldProblem(axis, placement.axis.set, None, str(error)) from error
        for entry, gate in zip(by_id, gates, strict=True):
            self.entry(axis, placement, entry, gate, entries[entry])

    def route(
        self, id: str, source: SignalSource, chain, active_source: SignalSource, active_chain,
        alt: _Alternative, window: Window | None = None,
    ) -> ComponentPortEnd:
        switch_id = _id(id, alt.axis, alt.entry)
        gate_id = _id(switch_id, "gate")
        self.components[switch_id] = LeafComponent(kind="leaf", name=switch_id, type="feedbax.Switch")
        self.wires[_id(switch_id, "base")] = ModelWire(
            source=source, target=_component_port(switch_id, "false_input"), attachments=chain,
        )
        self.wires[_id(switch_id, "entry")] = ModelWire(
            source=active_source, target=_component_port(switch_id, "true_input"), attachments=active_chain,
        )
        self.wires[_id(switch_id, "control")] = ModelWire(
            source=self.gate(alt, gate_id, window), target=_component_port(switch_id, "condition"),
        )
        return _component_port(switch_id, "output")

    def attach(self, id: str, source, chain, attachment: InterventionAttachment, alt: _Alternative):
        context = _id(*self.scope, id, alt.axis, alt.entry, attachment.id)
        if chain:
            stem = _id(context, "stem")
            self.components[stem] = LeafComponent(kind="leaf", name=stem, type="feedbax.Input")
            self.wires[stem] = ModelWire(source=source, target=_component_port(stem, "input"), attachments=chain)
            source = _component_port(stem, "output")
        segment = self.operands(attachment, context).model_copy(update={"window": None})
        return self.route(context, source, (), source, (segment,), alt, attachment.window)

    def source_chain(self, end: SignalSource, attachments):
        if not _needs_task(attachments):
            return attachments
        owner = _id(end.owner, getattr(end, "component", ""), end.port)
        affected = tuple(id for id, wire in self.wires.items() if wire.source == end)
        source, chain = end, ()
        for attachment in attachments:
            if isinstance(attachment, InterventionAttachment) and attachment.window is not None:
                representative = ModelWire(source=end, target=_component_port(owner, "input"))
                alt = _Alternative("", "", attachment.id, LiteralValue(value=True), False, representative)
                source = self.attach(owner, source, chain, attachment, alt)
                chain = ()
            else:
                if isinstance(attachment, InterventionAttachment):
                    attachment = self.operands(attachment, owner)
                chain = (*chain, attachment)
        if source == end:
            return chain
        if chain:
            out = _id(*self.scope, owner, "source")
            self.components[out] = LeafComponent(kind="leaf", name=out, type="feedbax.Input")
            self.wires[out] = ModelWire(source=source, target=_component_port(out, "input"), attachments=chain)
            source = _component_port(out, "output")
        for key in affected:
            self.wires[key] = self.wires[key].model_copy(update={"source": source})
        return ()

    def source_ports(self):
        for name, child in tuple(self.components.items()):
            if child.attachments.sources:
                chains = {
                    port: self.source_chain(_component_port(name, port), chain)
                    for port, chain in child.attachments.sources.items()
                }
                self.components[name] = child.model_copy(update={
                    "attachments": child.attachments.model_copy(update={"sources": chains}),
                })
        if self.port_attachments.sources:
            chains = {
                port: self.source_chain(InputBoundaryEnd(owner="inputs", port=port), chain)
                for port, chain in self.port_attachments.sources.items()
            }
            self.port_attachments = self.port_attachments.model_copy(update={"sources": chains})

    def child_graphs(self):
        for name, child in tuple(self.components.items()):
            if not isinstance(child, CompositeComponent) or not isinstance(child.interior, GraphInterior):
                continue
            if not _has_interventions(child):
                continue
            branch = self.model.model_copy(update={"interior": child.interior, "ports": child.ports})
            task = self.task.model_copy(update={"variables": tuple(self.variables), "signals": tuple(self.signals)})
            model, task = _Union(branch, task, self.registry, scope=(*self.scope, name)).finish()
            self.signals = list(task.signals)
            self.components[name] = child.model_copy(update={"interior": model.interior, "ports": model.ports})
            for port in model.ports.inputs[len(child.ports.inputs):]:
                self.inputs.append(port)
                self.wires[_id(*self.scope, name, port.id)] = ModelWire(
                    source=InputBoundaryEnd(owner="inputs", port=port.id), target=_component_port(name, port.id),
                )

    def finish(self) -> tuple[ModelDocument, TaskDocument]:
        for id in sorted(self.alternatives.keys() | {
            id for id, wire in self.model.interior.wires.items() if _needs_task(wire.attachments)
        }):
            alternatives = self.alternatives[id]
            base = self.model.interior.wires[id]
            insertions = [_insertions(base, alt.wire) for alt in alternatives]
            source, chain = base.source, base.attachments
            if all(item is not None for item in insertions):
                by_position = defaultdict(list)
                for alt, added in zip(alternatives, insertions, strict=True):
                    for position, attachment in added:
                        by_position[position].append((alt, attachment))
                chain = ()
                for position in range(len(base.attachments) + 1):
                    for alt, attachment in by_position[position]:
                        source = self.attach(id, source, chain, attachment, alt)
                        chain = ()
                    if position < len(base.attachments):
                        attachment = base.attachments[position]
                        if isinstance(attachment, InterventionAttachment) and attachment.window is not None:
                            alt = _Alternative("", "", attachment.id, LiteralValue(value=True), False, base)
                            source = self.attach(id, source, chain, attachment, alt)
                            chain = ()
                        else:
                            if isinstance(attachment, InterventionAttachment):
                                attachment = self.operands(attachment, id)
                            chain = (*chain, attachment)
            else:
                if len(alternatives) > 1 and (
                    any(alt.cooccurring for alt in alternatives) or len({alt.axis for alt in alternatives}) != 1
                ):
                    raise alternatives[-1].problem(f"Wire {id!r} can be rewired by two entries in the same trial.")
                for alt in alternatives:
                    if _has_window(base) or _has_window(alt.wire):
                        raise alt.problem("A windowed attachment must be inserted without rewiring its wire.")
                    source = self.route(id, source, chain, alt.wire.source, alt.wire.attachments, alt)
                    chain = ()
            self.wires[id] = base.model_copy(update={"source": source, "attachments": chain})
        self.source_ports()
        self.child_graphs()
        model = self.model.model_copy(update={
            "ports": self.model.ports.model_copy(update={"inputs": tuple(self.inputs)}),
            "interior": self.model.interior.model_copy(update={
                "components": self.components, "wires": self.wires, "attachments": self.port_attachments,
            }),
        })
        cohort = self.task.validation_trials
        if self.conditions:
            cohort = tuple(
                ValidationTrial(variables=row.variables | _combine_conditions(selected), epoch_steps=row.epoch_steps)
                for row in cohort for selected in product(*self.conditions)
            )
        links = self.links
        for node, (base, selectors) in self.curl_activation.items():
            enabled = selectors[0]
            for selector in selectors[1:]:
                enabled = _call("or", enabled, selector)
            active = _call("not", enabled) if base else enabled
            variable = self.variable(_id(node, "active"), _BOOL, active)
            links.append(Link(model_field=_curl_address(node), variable=variable.reference))
        task = self.task.model_copy(update={
            "variables": tuple(self.variables), "signals": tuple(self.signals),
            "validation_trials": cohort, "links": tuple(links),
        })
        return model, task


def _curl_address(component: str) -> str:
    return ModelAddress(composites=(), element=FieldElement(
        kind="field", parent=ComponentElement(kind="component", id=component), path=("fields", "active"),
    )).to_text()


def _combine_conditions(selected):
    return {id: value for condition in selected for id, value in condition.items()}


def _has_window(wire: ModelWire) -> bool:
    return any(isinstance(item, InterventionAttachment) and item.window is not None for item in wire.attachments)


def _task_reference(value) -> str | None:
    if not isinstance(value, dict) or value.get("kind") != "name":
        return None
    reference = NameValue.model_validate(value).reference
    try:
        ModelAddress.from_text(reference)
    except ValueError:
        return reference
    return None


def _needs_task(attachments) -> bool:
    return any(
        isinstance(attachment, InterventionAttachment)
        and (attachment.window is not None or any(_task_reference(value) is not None for value in attachment.fields.values()))
        for attachment in attachments
    )


def model_interventions(
    model: ModelDocument | CompositeComponent, scope: tuple[str, ...] = (),
):
    """The attachment addresses and operations authored on wires and source ports."""
    for name, wire in model.interior.wires.items():
        for item in wire.attachments:
            if isinstance(item, InterventionAttachment):
                yield ModelAddress(composites=scope, element=AttachmentElement(
                    kind="attachment", id=item.id, parent=WireElement(kind="wire", id=name),
                )), item
    for port, chain in model.interior.attachments.sources.items():
        for item in chain:
            if isinstance(item, InterventionAttachment):
                yield ModelAddress(composites=scope, element=AttachmentElement(
                    kind="attachment", id=item.id,
                    parent=PortElement(kind="port", end=InputBoundaryEnd(owner="inputs", port=port)),
                )), item
    for name, component in model.interior.components.items():
        for port, chain in component.attachments.sources.items():
            for item in chain:
                if isinstance(item, InterventionAttachment):
                    yield ModelAddress(composites=scope, element=AttachmentElement(
                        kind="attachment", id=item.id, parent=PortElement(
                            kind="port", end=ComponentPortAddress(
                                owner="component", component=name, direction="output", port=port,
                            ),
                        ),
                    )), item
        if isinstance(component, CompositeComponent):
            yield from model_interventions(component, (*scope, name))


def _has_interventions(model: ModelDocument | CompositeComponent) -> bool:
    return any(_needs_task((item,)) for _, item in model_interventions(model))


def _insertions(base: ModelWire, variant: ModelWire) -> list[tuple[int, InterventionAttachment]] | None:
    if base.source != variant.source:
        return None
    position = 0
    added = []
    for item in variant.attachments:
        if position < len(base.attachments) and item == base.attachments[position]:
            position += 1
        elif isinstance(item, InterventionAttachment):
            added.append((position, item))
        else:
            return None
    return added if position == len(base.attachments) else None


def lower_trial_sets(
    model: ModelDocument, task: TaskDocument, fields: Mapping[str, TrialSetField],
    read: Callable[[str], Document | None], components: ComponentRegistry,
) -> tuple[ModelDocument, TaskDocument]:
    model, links = _attachment_links(model, task.links, components)
    task = task.model_copy(update={"links": links}) if links != task.links else task
    union = _Union(model, task, components)
    for axis, placement in sorted(fields.items()):
        if placement.axis.where is not None:
            raise SetFieldProblem(axis, placement.axis.set, None, "Trial Set placements apply to all runs of the step.")
        union.placement(axis, placement, read)
    return union.finish()


def lower_trial_interventions(
    model: ModelDocument, task: TaskDocument, components: ComponentRegistry,
) -> tuple[ModelDocument, TaskDocument]:
    base, links = _attachment_links(model, task.links, components)
    if not _has_interventions(base):
        return model, task
    paired = task.model_copy(update={"links": links}) if links != task.links else task
    return _Union(base, paired, components).finish()
