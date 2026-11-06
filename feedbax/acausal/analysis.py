"""Pure-Python structural analysis for acausal element graphs."""

from __future__ import annotations

from dataclasses import dataclass
import math

from feedbax.acausal.base import (
    AcausalConnection,
    AcausalElement,
    AcausalEquation,
    AcausalVar,
    StateLayout,
)


@dataclass(frozen=True)
class StructuralNetwork:
    """One connected acausal physical network."""

    elements: tuple[str, ...]
    variables: tuple[str, ...]
    differential_variables: tuple[str, ...]
    massless_variables: tuple[str, ...]
    equations: tuple[str, ...]
    has_coordinate_reference: bool
    source_elements: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        """Return the balance counts used by diagnostics."""
        return {
            "equations": len(self.equations),
            "unknowns": len(self.differential_variables),
        }


@dataclass(frozen=True)
class StructuralAnalysis:
    """Structural facts computed before JAX vector-field construction."""

    layout: StateLayout
    params_dict: dict[str, float]
    physical_aliases: dict[str, str]
    coordinate_map: dict[str, tuple[str, tuple[tuple[str, int], ...]]]
    node_masses: dict[str, list[str]]
    free_inertial_pairs: tuple[tuple[str, str], ...]
    networks: tuple[StructuralNetwork, ...]
    through_equation_cycle: tuple[str, ...] | None = None


class GearConstraintError(ValueError):
    """An authored gear constraint cannot be reduced to a nonsingular tree."""


class CoordinateOwnershipError(ValueError):
    """A reduced coordinate has competing prescribed or grounded values."""

    def __init__(
        self, coordinate: str, grounds: tuple[str, ...], sources: tuple[str, ...]
    ) -> None:
        self.coordinate = coordinate
        self.grounds = grounds
        self.sources = sources
        if grounds:
            message = (
                f"Reduced coordinate {coordinate!r} is grounded by {list(grounds)!r} "
                f"and driven by prescribed motion sources {list(sources)!r}."
            )
        else:
            message = (
                f"Reduced coordinate {coordinate!r} has multiple prescribed motion "
                f"sources {list(sources)!r}."
            )
        super().__init__(message)


class UnionFind:
    """Disjoint-set (Union-Find) with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def groups(self) -> dict[str, list[str]]:
        """Return ``{root: [members]}``."""
        groups: dict[str, list[str]] = {}
        for name in self._parent:
            root = self.find(name)
            groups.setdefault(root, []).append(name)
        return groups


def analyze_system(
    elements: dict[str, AcausalElement],
    connections: list[AcausalConnection],
) -> StructuralAnalysis:
    """Analyze acausal structure without importing JAX or building closures."""

    all_vars: dict[str, AcausalVar] = {}
    variable_owners: dict[str, str] = {}
    params_dict: dict[str, float] = {}

    for elem in sorted(elements.values(), key=lambda item: item.name):
        for signal_input in elem.signal_inputs:
            fqn = f"{elem.name}.{signal_input}"
            variable_owners[fqn] = elem.name
            all_vars[fqn] = AcausalVar(
                name=fqn,
                is_differential=False,
            )
        for port in elem.ports.values():
            for slot in port.across_vars:
                fqn = f"{elem.name}.{port.name}.{slot}"
                variable_owners[fqn] = elem.name
                all_vars[fqn] = AcausalVar(name=fqn)
            through_fqn = f"{elem.name}.{port.name}.{port.through_var}"
            variable_owners[through_fqn] = elem.name
            all_vars[through_fqn] = AcausalVar(name=through_fqn, is_differential=False)
        params_dict.update(elem.params)

    uf = UnionFind()

    for conn in connections:
        port_a_obj = elements[conn.element_a].ports[conn.port_a]

        for slot in port_a_obj.across_vars:
            fqn_a = f"{conn.element_a}.{conn.port_a}.{slot}"
            fqn_b = f"{conn.element_b}.{conn.port_b}.{slot}"
            uf.union(fqn_a, fqn_b)

    eliminated: dict[str, str] = {}
    for root, members in uf.groups().items():
        for member in members:
            if member != root:
                all_vars[member].is_eliminated = True
                all_vars[member].alias_of = root
                eliminated[member] = root

    coordinate_owners: dict[str, list[str]] = {}
    authored_inputs: dict[str, tuple[str, int]] = {}
    output_map: dict[str, str] = {}
    gears: list[AcausalElement] = []

    for elem in sorted(elements.values(), key=lambda item: item.name):
        for slot_idx, signal_input in enumerate(elem.signal_inputs):
            fqn = f"{elem.name}.{signal_input}"
            authored_inputs[fqn] = (elem.name, slot_idx)

        if elem.element_type == "ground":
            for port in elem.ports.values():
                for slot in port.across_vars:
                    fqn = f"{elem.name}.{port.name}.{slot}"
                    canonical = _resolve(fqn, eliminated)
                    coordinate_owners.setdefault(canonical, []).append(fqn)

        elif elem.element_type == "force_source":
            for port in elem.ports.values():
                through_fqn = f"{elem.name}.{port.name}.{port.through_var}"
                authored_inputs[through_fqn] = (elem.name, 0)

        elif elem.element_type == "prescribed_motion":
            for port in elem.ports.values():
                for slot_idx, slot in enumerate(
                    elem.source_payload.slots if elem.source_payload else port.across_vars
                ):
                    fqn = f"{elem.name}.{port.name}.{slot}"
                    canonical = _resolve(fqn, eliminated)
                    coordinate_owners.setdefault(canonical, []).append(fqn)
                    authored_inputs[fqn] = (elem.name, slot_idx)

        elif elem.element_type == "sensor":
            if elem.sensor_output is not None:
                port_name, slot_name = elem.sensor_output
                fqn = f"{elem.name}.{port_name}.{slot_name}"
                output_map[elem.name] = fqn

        elif elem.element_type == "gear_ratio":
            gears.append(elem)

    physical_aliases = dict(eliminated)

    # Each physical across variable is q_i = s_i(params) * q_root.  Store
    # symbolic ratio products so trainable ratios remain part of the JAX graph.
    adjacency: dict[str, list[tuple[str, str, int]]] = {}
    incoming: set[str] = set()
    for elem in gears:
        for slot in elem.ports["flange_a"].across_vars:
            left = _resolve(f"{elem.name}.flange_a.{slot}", physical_aliases)
            right = _resolve(f"{elem.name}.flange_b.{slot}", physical_aliases)
            key = f"{elem.name}.ratio"
            if not math.isfinite(params_dict[key]) or params_dict[key] == 0:
                raise GearConstraintError(f"Gear ratio {key!r} must be finite and nonzero")
            adjacency.setdefault(left, []).append((right, key, 1))
            adjacency.setdefault(right, []).append((left, key, -1))
            incoming.add(right)

    node_coordinates: dict[str, tuple[str, tuple[tuple[str, int], ...]]] = {}
    remaining = set(adjacency)
    while remaining:
        component: set[str] = set()
        pending = [min(remaining)]
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(other for other, _, _ in adjacency[node])
        fixed = component & coordinate_owners.keys()
        candidates = fixed or (component - incoming) or component

        def mechanical_names(node):
            return tuple(
                sorted(
                    name
                    for name in all_vars
                    if _resolve(name, physical_aliases) == node
                    and elements[variable_owners[name]].element_type != "sensor"
                )
            )

        root = min(candidates, key=mechanical_names)
        node_coordinates[root] = (root, ())
        pending = [(root, None)]
        visited: set[str] = set()
        while pending:
            node, parent_key = pending.pop()
            if node in visited:
                raise GearConstraintError("Gear constraint cycle: use an independent gear tree")
            visited.add(node)
            _, factors = node_coordinates[node]
            for other, key, exponent in adjacency[node]:
                if key == parent_key:
                    continue
                if other in node_coordinates:
                    raise GearConstraintError("Gear constraint cycle: use an independent gear tree")
                node_coordinates[other] = (root, factors + ((key, exponent),))
                pending.append((other, key))
        remaining -= component
        for node in component - {root}:
            eliminated[node] = root
            all_vars[node].is_eliminated = True
            all_vars[node].alias_of = root
            all_vars[node].is_differential = False

    coordinate_map = {}
    for name in all_vars:
        node = _resolve(name, physical_aliases)
        coordinate_map[name] = node_coordinates.get(node, (node, ()))

    reduced_owners: dict[str, list[str]] = {}
    for node, owners in coordinate_owners.items():
        root, _ = coordinate_map[node]
        reduced_owners.setdefault(root, []).extend(owners)

    grounded: set[str] = set()
    for root, owners in reduced_owners.items():
        sources = [owner for owner in owners if owner in authored_inputs]
        grounds = [owner for owner in owners if owner not in authored_inputs]
        if len(sources) > 1 or (sources and grounds):
            raise CoordinateOwnershipError(
                root,
                tuple(sorted({variable_owners[owner] for owner in grounds})),
                tuple(sorted({variable_owners[owner] for owner in sources})),
            )
        if grounds:
            if len({_resolve(owner, physical_aliases) for owner in grounds}) > 1:
                raise GearConstraintError("Gear network has multiple grounded coordinates")
            grounded.add(root)
            all_vars[root].is_grounded = True
            all_vars[root].is_differential = False

    input_vars: dict[str, int] = {}
    input_specs: dict[str, tuple[str, int]] = {}
    for index, (name, spec) in enumerate(authored_inputs.items()):
        root, _ = coordinate_map[name]
        input_vars[root] = index
        input_specs[root] = spec
        all_vars[root].is_input = True
        all_vars[root].is_differential = False

    differential: list[str] = []
    for name, var in sorted(all_vars.items()):
        if (
            var.is_differential
            and not var.is_eliminated
            and not var.is_grounded
            and not var.is_input
        ):
            differential.append(name)

    layout = StateLayout(
        _vars=all_vars,
        _differential=differential,
        _algebraic=[],
        _eliminated=eliminated,
        _grounded=grounded,
        _inputs=input_vars,
        _input_specs=input_specs,
        _outputs=output_map,
        total_size=len(differential),
    )

    node_masses: dict[str, list[str]] = {}
    free_variables = set(differential)
    free_inertial_pairs: set[tuple[str, str]] = set()
    for elem in elements.values():
        if elem.element_type != "mass" or not elem.ports:
            continue
        mass_key = next(
            (key for key in elem.params if key.endswith((".mass", ".inertia"))), None
        )
        if mass_key is None:
            raise ValueError(f"Mass element {elem.name!r} has no mass/inertia parameter")
        for port in elem.ports.values():
            position = f"{elem.name}.{port.name}.{port.across_vars[0]}"
            velocity = f"{elem.name}.{port.name}.{port.across_vars[1]}"
            node = _resolve(position, physical_aliases)
            node_masses.setdefault(node, []).append(mass_key)
            root_pos, _ = coordinate_map[position]
            root_vel, _ = coordinate_map[velocity]
            if root_pos in free_variables and root_vel in free_variables:
                free_inertial_pairs.add((root_pos, root_vel))

    through_equations = [
        eq for elem in elements.values() for eq in elem.equations if eq.is_through_def
    ]
    through_equation_cycle = _through_equation_cycle(through_equations)

    return StructuralAnalysis(
        layout=layout,
        params_dict=params_dict,
        physical_aliases=physical_aliases,
        coordinate_map=coordinate_map,
        node_masses=node_masses,
        free_inertial_pairs=tuple(sorted(free_inertial_pairs)),
        networks=_build_networks(
            elements,
            connections,
            layout,
            variable_owners,
            reduced_owners,
            {name for pair in free_inertial_pairs for name in pair},
        ),
        through_equation_cycle=through_equation_cycle,
    )


def _resolve(name: str, eliminated: dict[str, str]) -> str:
    """Follow alias chain."""
    visited: set[str] = set()
    while name in eliminated:
        if name in visited:
            raise RuntimeError(f"Cyclic alias for '{name}'")
        visited.add(name)
        name = eliminated[name]
    return name


def _build_networks(
    elements: dict[str, AcausalElement],
    connections: list[AcausalConnection],
    layout: StateLayout,
    variable_owners: dict[str, str],
    coordinate_owners: dict[str, list[str]],
    inertial_roots: set[str],
) -> tuple[StructuralNetwork, ...]:
    element_groups: dict[str, set[str]] = {name: {name} for name in elements}

    def find(name: str) -> str:
        parent = next(root for root, members in element_groups.items() if name in members)
        return parent

    for conn in connections:
        left = find(conn.element_a)
        right = find(conn.element_b)
        if left == right:
            continue
        element_groups[left].update(element_groups.pop(right))

    networks: list[StructuralNetwork] = []
    for names in sorted(element_groups.values(), key=lambda item: sorted(item)):
        network_elements = tuple(sorted(names))
        variables = tuple(
            sorted(
                var_name
                for var_name in layout._vars
                if variable_owners[var_name] in names and not layout._vars[var_name].is_eliminated
            )
        )
        differential_variables = tuple(
            var_name for var_name in layout._differential if variable_owners[var_name] in names
        )
        equation_lhs = tuple(
            sorted(
                {
                    _resolve(eq.lhs_var, layout._eliminated)
                    for elem_name in names
                    for eq in elements[elem_name].equations
                    if not eq.is_through_def
                    and _resolve(eq.lhs_var, layout._eliminated) not in layout._grounded
                    and _resolve(eq.lhs_var, layout._eliminated) not in layout._inputs
                }
            )
        )
        networks.append(
            StructuralNetwork(
                elements=network_elements,
                variables=variables,
                differential_variables=differential_variables,
                massless_variables=tuple(
                    name for name in differential_variables if name not in inertial_roots
                ),
                equations=equation_lhs,
                has_coordinate_reference=not coordinate_owners.keys().isdisjoint(variables),
                source_elements=tuple(
                    sorted(name for name in names if elements[name].element_type == "force_source")
                ),
            )
        )
    return tuple(networks)


def _through_equation_cycle(equations: list[AcausalEquation]) -> tuple[str, ...] | None:
    try:
        _topo_sort_through_eqs(equations)
    except ValueError as exc:
        message = str(exc)
        if "Cyclic through-variable dependency" in message:
            return tuple(sorted(eq.lhs_var for eq in equations))
    return None


def _topo_sort_through_eqs(
    equations: list[AcausalEquation],
) -> list[AcausalEquation]:
    """Sort through-variable equations so dependencies are evaluated first."""
    by_lhs: dict[str, AcausalEquation] = {}
    for eq in equations:
        if eq.lhs_var in by_lhs:
            raise ValueError(f"Duplicate through-variable definition for {eq.lhs_var!r}")
        by_lhs[eq.lhs_var] = eq

    visited: set[str] = set()
    visiting: set[str] = set()
    order: list[AcausalEquation] = []

    def visit(lhs: str) -> None:
        if lhs in visited:
            return
        if lhs in visiting:
            raise ValueError(f"Cyclic through-variable dependency involving {lhs!r}")
        visiting.add(lhs)
        eq = by_lhs[lhs]
        for dep in eq.depends_on:
            if dep in by_lhs:
                visit(dep)
        visiting.remove(lhs)
        visited.add(lhs)
        order.append(eq)

    for lhs in sorted(by_lhs):
        visit(lhs)
    return order
