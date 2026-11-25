"""Resolve the types and identities of an authored Model's signal leaves."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import jax.tree_util as jtu

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.graph_schema import _component_port_dimension
from feedbax.compiler.resolution import GraphParameterResolution
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    DIMENSIONLESS, Dimensions, Level, ScopedValue, SignalSpace, ValueBase, ValueType,
)
from feedbax.contracts.graph import ComponentSpec, StudioTaskDataSpec
from feedbax.contracts.model_address import (
    ComponentPortAddress, ComponentPortEnd, FieldElement, InputBoundaryEnd, ModelAddress, OutputBoundaryEnd,
    PortElement, SignalSource, SignalTarget, WireElement,
)
from feedbax.contracts.model_document import CompositeComponent, ModelDocument, SignalType
from feedbax.contracts.model_edit import ModelGraphProblem
from feedbax.contracts.model_values import SignalLeafPath, model_owner_class
from feedbax.contracts.portable import OutputSignals, PassedSignal, SignalProperties, StateSignal, signal_leaf_declaration


@dataclass(frozen=True)
class ResolvedSignalLeaf:
    port: ModelAddress
    path: SignalLeafPath
    name: str
    _facts: _Facts
    origin: ModelAddress | None

    @property
    def type(self) -> ValueType | None:
        facts = self._facts
        if facts.base is not None and facts.shape is not None:
            return ValueType(base=facts.base, shape=facts.shape, dimensions=facts.dimensions or DIMENSIONLESS,
                             space=facts.space)
        return None

    @property
    def dimensions(self) -> Dimensions | None:
        return self._facts.dimensions

    @property
    def unit(self) -> str | None:
        return self._facts.unit

    @property
    def signal_type(self) -> SignalType:
        facts = self._facts
        return SignalType(base=facts.base, shape=facts.shape, unit=facts.unit, space=facts.space)

    @property
    def address(self) -> ModelAddress:
        return _leaf_address(self.port, self.path)

    def accepts(self, actual: ValueType) -> bool:
        expected = self.type
        return expected is None or (
            expected.accepts_shape_and_base(actual) and expected.accepts_space(actual)
            and (self.dimensions is None or self.dimensions == actual.dimensions)
        )


@dataclass(frozen=True)
class ModelSignalTable:
    _ports: Mapping[str, tuple[ResolvedSignalLeaf, ...]] = field(default_factory=dict)
    problems: tuple[ModelGraphProblem, ...] = ()
    _claims: Mapping[str, Mapping[SignalLeafPath, _Facts]] = field(default_factory=dict)

    def at(self, address: ModelAddress) -> ResolvedSignalLeaf | None:
        element = address.element
        port, path = (address, ())
        if isinstance(element, FieldElement) and isinstance(element.parent, PortElement):
            port = ModelAddress(composites=address.composites, element=element.parent)
            path = element.path
        return next((leaf for leaf in self.at_port(port) if leaf.path == path), None)

    def at_port(self, address: ModelAddress) -> tuple[ResolvedSignalLeaf, ...]:
        return self._ports.get(address.to_text(), ())

    def connection_problem(
        self, scope: tuple[str, ...], source: SignalSource | None, target: SignalTarget,
        provided: SignalType, required: SignalType,
    ) -> str | None:
        """Compare a proposed source with the target's own claims before any wire it might replace."""
        offered = {} if source is None else {
            leaf.path: leaf._facts for leaf in self.at_port(_end_address(scope, source, "output"))
        }
        expected = self._claims.get(_end_address(scope, target, "input").to_text(), {(): _Facts.from_claim(required)})
        for path, facts in (offered or {(): _Facts.from_claim(provided)}).items():
            if (claim := expected.get(path)) is not None and (reason := claim.conflict(facts)):
                return f"Signal type does not fit: {reason}"
        return None

    @property
    def readable_leaves(self) -> tuple[ResolvedSignalLeaf, ...]:
        def output(leaf: ResolvedSignalLeaf) -> bool:
            end = _port_element(leaf.port).end
            return isinstance(end, OutputBoundaryEnd) or (
                isinstance(end, ComponentPortAddress) and end.direction == "output"
            )

        return tuple(leaf for leaves in self._ports.values() for leaf in leaves if output(leaf))


@dataclass(frozen=True)
class ModelMeasureScope:
    names: tuple[ScopedValue, ...]
    paths: dict[str, tuple[str, ...]]
    leaves: dict[str, ResolvedSignalLeaf]


def model_measure_scope(model: ModelDocument, signals: ModelSignalTable) -> ModelMeasureScope:
    """Expose one name per readable output leaf, retaining IDs and display paths separately."""
    owners: dict[tuple[str, ...], ModelDocument | CompositeComponent] = {(): model}
    labels: dict[tuple[str, ...], tuple[str, ...]] = {(): ()}
    names: list[ScopedValue] = []
    paths: dict[str, tuple[str, ...]] = {}
    leaves: dict[str, ResolvedSignalLeaf] = {}
    used: set[str] = set()

    def owner_at(scope: tuple[str, ...]) -> ModelDocument | CompositeComponent:
        if scope not in owners:
            parent = owner_at(scope[:-1])
            owner = parent.interior.components[scope[-1]]
            assert isinstance(owner, CompositeComponent)
            owners[scope] = owner
            labels[scope] = (*labels[scope[:-1]], owner.name)
        return owners[scope]

    ordered = sorted(signals.readable_leaves, key=lambda leaf: bool(
        leaf.port.composites or isinstance(_port_element(leaf.port).end, ComponentPortAddress)
    ))
    for leaf in ordered:
        if leaf.type is None:
            continue
        end = _port_element(leaf.port).end
        scope = leaf.port.composites
        owner = owner_at(scope)
        prefix = labels[scope]
        if isinstance(end, ComponentPortAddress):
            prefix = (*prefix, owner.interior.components[end.component].name)
        elif scope:
            prefix = (*prefix, "Outputs")
        path = (*prefix, leaf.name, *map(str, leaf.path))
        address = leaf.address.to_text()
        name = ".".join(path)
        if name in used:
            identity = ".".join((*scope, getattr(end, "component", end.port), end.port, *map(str, leaf.path)))
            name = f"{name} [{identity}]"
        used.add(name)
        names.append(ScopedValue(id=address, name=name, type=leaf.type, level=Level.STEP))
        paths[address] = path
        leaves[address] = leaf
    return ModelMeasureScope(tuple(names), paths, leaves)


def boundary_signal_port(scope: tuple[str, ...], direction: str, port: str) -> ModelAddress:
    end = InputBoundaryEnd(owner="inputs", port=port) if direction == "inputs" else OutputBoundaryEnd(
        owner="outputs", port=port,
    )
    return ModelAddress(composites=scope, element=PortElement(kind="port", end=end))


def _component_port(
    scope: tuple[str, ...], component: str, direction: Literal["input", "output"], port: str,
) -> ModelAddress:
    end = ComponentPortAddress(owner="component", component=component, direction=direction, port=port)
    return ModelAddress(composites=scope, element=PortElement(kind="port", end=end))


def _end_address(
    scope: tuple[str, ...], end: SignalSource | SignalTarget, direction: Literal["input", "output"],
) -> ModelAddress:
    if isinstance(end, ComponentPortEnd):
        return _component_port(scope, end.component, direction, end.port)
    return ModelAddress(composites=scope, element=PortElement(kind="port", end=end))


def _port_element(address: ModelAddress) -> PortElement:
    element = address.element
    assert isinstance(element, PortElement)
    return element


def _leaf_address(port: ModelAddress, path: SignalLeafPath) -> ModelAddress:
    if not path:
        return port
    return ModelAddress(composites=port.composites, element=FieldElement(kind="field", parent=_port_element(port), path=path))


_DTYPE_BASES = {
    "bool": ValueBase.BOOLEAN, "boolean": ValueBase.BOOLEAN,
    "int": ValueBase.INTEGER, "integer": ValueBase.INTEGER,
    "float": ValueBase.REAL, "real": ValueBase.REAL,
    **{f"{stem}{bits}": base for stem, base in (
        ("int", ValueBase.INTEGER), ("uint", ValueBase.INTEGER), ("float", ValueBase.REAL),
    ) for bits in (8, 16, 32, 64)},
    "bfloat16": ValueBase.REAL,
}


def _port_base(dtype: str | None) -> ValueBase | None:
    return _DTYPE_BASES.get(dtype) if dtype is not None else None


def _base_accepts(required: ValueBase | None, actual: ValueBase | None) -> bool:
    return required is None or actual is None or required == actual or (
        required == ValueBase.REAL and actual == ValueBase.INTEGER
    )


def _shape_accepts(required: tuple[int | str | None, ...] | None, actual: tuple[int, ...] | None) -> bool:
    return required is None or actual is None or (
        len(required) == len(actual) and all(type(a) is not int or a == b for a, b in zip(required, actual))
    )


def _concrete_shape(shape) -> tuple[int, ...] | None:
    return tuple(shape) if shape is not None and all(type(size) is int and size > 0 for size in shape) else None


def _prototype_types(prototype) -> dict[SignalLeafPath, tuple[ValueBase, tuple[int, ...]]]:
    result = {}
    for keys, value in jtu.tree_leaves_with_path(prototype):
        path = tuple(
            key.name if isinstance(key, jtu.GetAttrKey) else key.idx if isinstance(key, jtu.SequenceKey) else key.key
            for key in keys
        )
        if not all(type(part) is str or type(part) is int and part >= 0 for part in path):
            continue
        base = _port_base(getattr(getattr(value, "dtype", None), "name", None))
        shape = _concrete_shape(getattr(value, "shape", None))
        if base is not None and shape is not None:
            result[path] = base, shape
    return result


@dataclass(frozen=True)
class _Facts:
    base: ValueBase | None = None
    shape: tuple[int, ...] | None = None
    dimensions: Dimensions | None = None
    unit: str | None = None
    space: SignalSpace | None = None

    @classmethod
    def from_value(cls, value: ValueType) -> _Facts:
        unit = next((name for name, rule in UNITS.items() if name and name != "st" and rule.scale == 1
                     and rule.dimensions == value.dimensions), None)
        return cls(value.base, value.shape, value.dimensions, unit, value.space)

    @classmethod
    def from_claim(cls, claim: SignalType) -> _Facts:
        unit = UNITS.get(claim.unit) if claim.unit is not None else None
        return cls(_port_base(claim.base), _concrete_shape(claim.shape), unit.dimensions if unit else None,
                   claim.unit, claim.space)

    def fill(self, source: _Facts) -> _Facts:
        return _Facts(
            self.base if self.base is not None else source.base,
            self.shape if self.shape is not None else source.shape,
            self.dimensions if self.dimensions is not None else source.dimensions,
            self.unit if self.dimensions is not None else source.unit,
            self.space if self.space is not None else source.space,
        )

    def conflict(self, source: _Facts) -> str | None:
        if not _base_accepts(self.base, source.base):
            return f"expected {self.base}, received {source.base}"
        if not _shape_accepts(self.shape, source.shape):
            return f"expected shape {self.shape}, received {source.shape}"
        if self.dimensions is not None and source.dimensions is not None and self.dimensions != source.dimensions:
            return f"expected unit {self.unit or self.dimensions}, received {source.unit or source.dimensions}"
        if self.space is not None and source.space is not None and self.space != source.space:
            return f"expected {self.space} space, received {source.space} space"
        return None


def task_data_value_type(item: StudioTaskDataSpec) -> ValueType | None:
    shape = item.expected_shape
    if shape and shape[0] == "time":
        shape = shape[1:]
    base = _port_base(item.dtype)
    dims = _concrete_shape(shape)
    if base is None or dims is None:
        return None
    unit = UNITS.get(item.units or "")
    space = item.metadata.get("space")
    return ValueType(base=base, shape=dims, dimensions=unit.dimensions if unit else DIMENSIONLESS,
                     space=space if space in ("point", "free") else None)


@dataclass
class _Port:
    address: ModelAddress
    name: str
    prototypes: dict[SignalLeafPath, tuple[ValueBase, tuple[int, ...]]]
    claim: SignalType
    declarations: OutputSignals
    states: Mapping[str, SignalProperties]
    attachments: bool = False
    makes_value: bool = False
    paths: dict[SignalLeafPath, None] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.address.to_text()

    def __post_init__(self) -> None:
        self.paths.update(dict.fromkeys(self.prototypes))
        for path, declaration in self.declarations.items():
            if path or not isinstance(declaration, PassedSignal):
                self.paths.setdefault(path, None)


@dataclass(frozen=True)
class _Route:
    source: _Port
    target: _Port
    problem: ModelAddress
    makes_value: bool = False


@dataclass(frozen=True)
class _Incoming:
    source: tuple[str, SignalLeafPath]
    problem: ModelAddress
    origin: ModelAddress | None = None


@dataclass(frozen=True)
class _Local:
    facts: _Facts
    prototype: _Facts
    origin: ModelAddress
    makes_value: bool


@dataclass(frozen=True)
class _Resolved:
    facts: _Facts = _Facts()
    origin: ModelAddress | None = None
    conflict: str | None = None
    blocked: bool = False


class _Resolve:
    def __init__(self, components: ComponentRegistry, task_inputs: Mapping[str, ValueType]):
        self.components = components
        self.task_inputs = task_inputs
        self.ports: dict[str, _Port] = {}
        self.routes: list[_Route] = []
        self.problems: dict[str, ModelGraphProblem] = {}

    def problem(self, address: ModelAddress, reason: str) -> None:
        self.problems.setdefault(address.to_text(), ModelGraphProblem(address=address, reason=reason))

    def port(
        self, address, name, prototype, claim=None, *, declarations=None, states=None,
        attachments=False, makes_value=False,
    ) -> _Port:
        result = _Port(address, name, _prototype_types(prototype), claim or SignalType(), declarations or {},
                       states or {}, attachments, makes_value)
        self.ports[result.key] = result
        return result

    def collect(
        self, owner: ModelDocument | CompositeComponent, scope: tuple[str, ...],
        resolution: GraphParameterResolution | None, *, outer: dict[tuple[str, str], _Port] | None = None,
        physical: AcausalGraphSpec | None = None,
    ) -> None:
        outer = outer or {}
        graph = resolution.graph if resolution else None
        outputs = resolution.graph_outputs if resolution else {}
        for direction in ("inputs", "outputs"):
            for declaration in getattr(owner.ports, direction):
                port = declaration.id
                external = port if scope else declaration.name
                peer = outer.get((direction, port))
                prototype = None
                if resolution is not None:
                    assert graph is not None
                    if direction == "outputs":
                        prototype = outputs.get(external)
                    else:
                        if (target := graph.input_bindings.get(external)) is not None:
                            prototype = resolution.node_inputs.get(target)
                        if prototype is None and not scope and declaration.default is not None:
                            from feedbax.compiler.model_document import _generated

                            prototype = resolution.node_outputs.get((_generated("input", port), "output"))
                address = boundary_signal_port(scope, direction, port)
                claim = declaration.type if peer is None or direction == "outputs" else None
                boundary = self.port(
                    address, declaration.name, prototype, claim,
                    attachments=bool(owner.interior.attachments.sources.get(port)),
                    makes_value=direction == "inputs" and not scope,
                )
                if peer is not None:
                    boundary.prototypes.update(peer.prototypes)
                    boundary.paths.update(peer.paths)
                    source, target = (peer, boundary) if direction == "inputs" else (boundary, peer)
                    self.routes.append(_Route(source, target, address))

        for name, component in owner.interior.components.items():
            node = graph.nodes.get(name) if graph else physical.nodes.get(name) if physical else None
            if component.kind == "composite":
                if physical is not None:
                    if (branch := (physical.subgraphs or {}).get(name)) is not None:
                        self.collect(component, (*scope, name), None, physical=branch)
                    continue
                peers = {}
                for direction in ("inputs", "outputs"):
                    found = {} if resolution is None else getattr(resolution, "node_" + direction)
                    for declaration in getattr(component.ports, direction):
                        port = declaration.id
                        address = _component_port(scope, name, "input" if direction == "inputs" else "output", port)
                        peers[direction, port] = self.port(
                            address, declaration.name, found.get((name, port)),
                            declaration.type if direction == "inputs" else None, attachments=bool(component.attachments.sources.get(port)),
                        )
                child = (graph.subgraphs or {}).get(name) if graph is not None else None
                self.collect(component, (*scope, name), resolution.subgraphs.get(name) if resolution else None, outer=peers,
                             physical=child if isinstance(child, AcausalGraphSpec) else None)
                continue
            if node is None:
                node = self.authored_node(component)
                if node is None:
                    continue
            declared_owner = model_owner_class(node, self.components)
            if declared_owner is None or (ports := declared_owner.ports) is None:
                continue
            params = node.params
            if physical is not None:
                meta = self.components.get(node.type)
                assert meta is not None
                params = meta.params.require(params).model_dump(mode="python")
            elif resolution is not None and name not in resolution.node_configs:
                # Partial resolution may have failed before applying parameter defaults.
                params = None
            facts = declared_owner.model_facts
            states = {state.symbol: SignalProperties(state.units, state.space)
                      for state in (facts.states or ())} if facts is not None and facts.has_state(node) else {}
            output_declarations = (
                ports.declared_outputs(params) if params is not None else
                {} if callable(ports.output_leaves) else ports.output_leaves
            )
            for direction, declarations, names, prototypes in (
                ("input", ports.input_leaves, node.input_ports, resolution.node_inputs if resolution else {}),
                ("output", output_declarations, node.output_ports, resolution.node_outputs if resolution else {}),
            ):
                specs = ports.inputs if direction == "input" else ports.outputs
                for port in names or specs:
                    spec = specs.get(port)
                    if spec is not None and spec.kind == "conserving":
                        continue
                    address = _component_port(scope, name, "input" if direction == "input" else "output", port)
                    selected = {path[1:]: item for path, item in declarations.items() if path[0] == port}
                    registered = SignalType(base=spec.dtype, shape=tuple(spec.shape) if spec.shape is not None else None) if spec else SignalType()
                    prototype = prototypes.get((name, port))
                    if direction == "input" and prototype is None:
                        dimension = _component_port_dimension(
                            component_type=node.type, node_params=node.params, port=port, direction="input",
                        )
                        if dimension is not None:
                            dtype = node.params.get("dtype")
                            registered = SignalType(
                                base=dtype if _port_base(registered.base) is None and isinstance(dtype, str) else registered.base,
                                shape=tuple(dimension["shape"]),
                            )
                    leaf_port = self.port(
                        address, port, prototype, registered, declarations=selected, states=states,
                        attachments=bool(component.attachments.sources.get(port)), makes_value=direction == "output",
                    )
                    if physical is not None and params is not None and isinstance(key := params.get("port_name"), str):
                        boundary = outer.get((direction + "s", key))
                        if boundary is not None:
                            leaf_port.prototypes.update(boundary.prototypes)
                            leaf_port.paths.update(boundary.paths)

        for name, wire in owner.interior.wires.items():
            source = self.ports.get(_end_address(scope, wire.source, "output").to_text())
            target = self.ports.get(_end_address(scope, wire.target, "input").to_text())
            if source is None or target is None:
                continue
            address = ModelAddress(composites=scope, element=WireElement(kind="wire", id=name))
            self.routes.append(_Route(source, target, address, bool(wire.attachments)))

    def authored_node(self, component) -> ComponentSpec | None:
        declaration = self.components.get(component.type)
        if declaration is None or declaration.is_composite:
            return None
        params = dict(component.fields)
        policy = declaration.dynamic_port_policy
        if policy and policy.count_param in component.port_counts:
            params[policy.count_param] = component.port_counts[policy.count_param]
        try:
            params = declaration.params.require(params).model_dump(mode="json")
            layout = self.components.dynamic_port_layout(component.type, params)
        except ValueError:
            return None
        return ComponentSpec(
            type=component.type, revision=component.revision, params=params,
            input_ports=list(layout.input_ports if layout else declaration.input_ports),
            output_ports=list(layout.output_ports if layout else declaration.output_ports),
        )

    def connect(self) -> dict[tuple[str, SignalLeafPath], _Incoming]:
        for port in self.ports.values():
            if port.claim.shape is not None or _port_base(port.claim.base) is not None:
                if any(port.paths):
                    if isinstance(_port_element(port.address).end, InputBoundaryEnd | OutputBoundaryEnd):
                        self.problem(port.address, f"Port {port.name!r} states an array type but carries a structure")
                else:
                    port.paths.setdefault((), None)
        changed = True
        while changed:
            changed = False
            for route in self.routes:
                a, b = route.source, route.target
                for port, peer in ((a, b), (b, a)):
                    if not port.paths and peer.paths:
                        port.paths.update(peer.paths)
                        changed = True
        for port in self.ports.values():
            if not port.paths:
                port.paths[()] = None

        incoming = {}
        for route in self.routes:
            source, target = route.source, route.target
            if source.prototypes and target.prototypes and source.prototypes.keys() != target.prototypes.keys():
                self.problem(route.problem, f"Signal structures at {source.name!r} and {target.name!r} differ")
                continue
            for path in source.paths.keys() & target.paths.keys():
                origin = _leaf_address(target.address, path) if route.makes_value else (
                    _leaf_address(source.address, path) if source.attachments else None
                )
                incoming[target.key, path] = _Incoming((source.key, path), route.problem, origin)
        return incoming

    def local(self, port: _Port, path: SignalLeafPath, *, connected: bool) -> _Local:
        address = _leaf_address(port.address, path)
        end = _port_element(port.address).end
        claim = SignalType() if path else port.claim
        declaration = signal_leaf_declaration(port.declarations, path)
        origin = address
        properties = declaration if isinstance(declaration, SignalProperties) else SignalProperties()
        if isinstance(declaration, StateSignal) and declaration.symbol in port.states:
            assert isinstance(end, ComponentPortAddress)
            properties = port.states[declaration.symbol]
            origin = ModelAddress.at_state(port.address.composites, end.component, declaration.symbol)
        unit = claim.unit if claim.unit is not None else properties.unit
        dims = UNITS.get(unit) if unit is not None else None
        if unit is not None and dims is None:
            self.problem(address, f"Unknown expression unit {unit!r}")
        stated = _Facts.from_claim(claim.model_copy(update={"unit": unit, "space": claim.space or properties.space}))
        base, shape = port.prototypes.get(path, (None, None))
        proto = _Facts(base, shape)
        if not connected and shape is not None and not _shape_accepts(claim.shape, shape):
            self.problem(address, f"Resolved shape of {port.name!r} conflicts with its port")
        if not connected and not _base_accepts(stated.base, base) and not isinstance(end, ComponentPortAddress):
            self.problem(address, f"Resolved base of {port.name!r} conflicts with its port")
        if isinstance(end, InputBoundaryEnd) and not port.address.composites and not path:
            if (task_type := self.task_inputs.get(port.name)) is not None:
                offered = _Facts.from_value(task_type)
                if stated.conflict(offered) is None and _shape_accepts(claim.shape, offered.shape):
                    stated = stated.fill(offered)
        return _Local(stated, proto, origin, port.makes_value and not isinstance(declaration, PassedSignal))

    def finish(self) -> ModelSignalTable:
        incoming = self.connect()
        local = {}
        for key, port in self.ports.items():
            end = _port_element(port.address).end
            for path in port.paths:
                item = key, path
                local[item] = self.local(port, path, connected=item in incoming)
                if isinstance(end, ComponentPortAddress) and end.direction == "output":
                    declaration = signal_leaf_declaration(port.declarations, path)
                    if isinstance(declaration, PassedSignal):
                        upstream, *subpath = declaration.source
                        assert isinstance(upstream, str)
                        source = _component_port(port.address.composites, end.component, "input", upstream)
                        incoming[item] = _Incoming(
                            (source.to_text(), tuple(subpath)), port.address,
                            local[item].origin if declaration.keeps == "type" else None,
                        )

        resolved: dict[tuple[str, SignalLeafPath], _Resolved] = {}

        def advance(item, previous: _Resolved) -> _Resolved:
            here = local[item]
            edge = incoming.get(item)
            if previous.blocked:
                return _Resolved(here.prototype.fill(here.facts), blocked=True)
            conflict = here.facts.conflict(previous.facts) if edge else None
            if conflict is not None:
                return _Resolved(here.facts, conflict=conflict, blocked=True)
            if edge is None:
                return _Resolved(here.prototype.fill(here.facts), here.origin if here.makes_value else None)
            facts = here.facts.fill(previous.facts).fill(here.prototype)
            origin = edge.origin or previous.origin
            return _Resolved(facts, origin)

        for key in local:
            if key in resolved:
                continue
            chain, positions = [], {}
            item = key
            while item in local and item not in resolved and item not in positions:
                positions[item] = len(chain)
                chain.append(item)
                edge = incoming.get(item)
                if edge is None:
                    break
                item = edge.source
            previous = resolved.get(item, _Resolved())
            if item in positions and item in incoming:
                cycle = list(reversed(chain[positions[item]:]))
                # A lap seeds each declared property. A second propagates it past the cut.
                for index, member in enumerate(cycle * 2):
                    previous = advance(member, previous)
                    resolved[member] = previous
                    if previous.blocked:
                        remaining = cycle[index % len(cycle) + 1:] + cycle[:index % len(cycle)]
                        for follower in remaining:
                            previous = advance(follower, previous)
                            resolved[follower] = previous
                        break
                previous = resolved[item]
                chain = chain[:positions[item]]
            for member in reversed(chain):
                previous = advance(member, previous)
                resolved[member] = previous

        by_port = {}
        claims = {}
        for key, port in self.ports.items():
            leaves = []
            for path in port.paths:
                item = key, path
                row = resolved[item]
                if row.conflict is not None:
                    address = incoming[item].problem
                    self.problem(address, f"Signal type does not fit: {row.conflict}")
                facts = row.facts
                proto = local[item].prototype
                base = proto.base or facts.base
                shape = proto.shape if proto.shape is not None else facts.shape
                leaves.append(ResolvedSignalLeaf(
                    port.address, path, port.name, _Facts(base, shape, facts.dimensions, facts.unit, facts.space), row.origin,
                ))
            by_port[key] = tuple(leaves)
            claims[key] = {path: local[key, path].facts for path in port.paths}
        return ModelSignalTable(by_port, tuple(self.problems.values()), claims)


def resolve_model_signals(
    model: ModelDocument, resolution: GraphParameterResolution | None, components: ComponentRegistry,
    *, task_inputs: Mapping[str, ValueType] | None = None,
) -> ModelSignalTable:
    resolver = _Resolve(components, task_inputs or {})
    resolver.collect(model, (), resolution)
    return resolver.finish()
