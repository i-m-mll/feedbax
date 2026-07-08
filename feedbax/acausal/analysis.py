"""Pure-Python structural analysis for acausal element graphs."""

from __future__ import annotations

from dataclasses import dataclass

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
    equations: tuple[str, ...]
    has_ground: bool
    source_elements: tuple[str, ...]
    across_source_elements: tuple[str, ...]

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
    node_through_vars: dict[str, list[str]]
    gear_infos: list[dict]
    eliminated_pre_gear: dict[str, str]
    networks: tuple[StructuralNetwork, ...]
    through_equation_cycle: tuple[str, ...] | None = None


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
    params_dict: dict[str, float] = {}

    for elem in elements.values():
        for signal_input in elem.signal_inputs:
            fqn = f"{elem.name}.{signal_input}"
            all_vars[fqn] = AcausalVar(
                name=fqn,
                is_differential=False,
            )
        for port in elem.ports.values():
            for slot in port.across_vars:
                fqn = f"{elem.name}.{port.name}.{slot}"
                all_vars[fqn] = AcausalVar(name=fqn)
            through_fqn = f"{elem.name}.{port.name}.{port.through_var}"
            all_vars[through_fqn] = AcausalVar(name=through_fqn, is_differential=False)
        params_dict.update(elem.params)

    uf = UnionFind()
    node_through_vars: dict[str, list[str]] = {}

    for conn in connections:
        port_a_obj = elements[conn.element_a].ports[conn.port_a]
        port_b_obj = elements[conn.element_b].ports[conn.port_b]

        for slot in port_a_obj.across_vars:
            fqn_a = f"{conn.element_a}.{conn.port_a}.{slot}"
            fqn_b = f"{conn.element_b}.{conn.port_b}.{slot}"
            uf.union(fqn_a, fqn_b)

        first_slot = port_a_obj.across_vars[0]
        fqn_a_first = f"{conn.element_a}.{conn.port_a}.{first_slot}"
        node_key = uf.find(fqn_a_first)

        through_a = f"{conn.element_a}.{conn.port_a}.{port_a_obj.through_var}"
        through_b = f"{conn.element_b}.{conn.port_b}.{port_b_obj.through_var}"
        node_through_vars.setdefault(node_key, [])
        if through_a not in node_through_vars[node_key]:
            node_through_vars[node_key].append(through_a)
        if through_b not in node_through_vars[node_key]:
            node_through_vars[node_key].append(through_b)

    eliminated: dict[str, str] = {}
    for root, members in uf.groups().items():
        for member in members:
            if member != root:
                all_vars[member].is_eliminated = True
                all_vars[member].alias_of = root
                eliminated[member] = root

    grounded: set[str] = set()
    input_vars: dict[str, int] = {}
    input_specs: dict[str, tuple[str, int]] = {}
    input_idx_counter = 0
    output_map: dict[str, str] = {}
    gear_infos: list[dict] = []

    for elem in elements.values():
        for slot_idx, signal_input in enumerate(elem.signal_inputs):
            fqn = f"{elem.name}.{signal_input}"
            input_vars[fqn] = input_idx_counter
            input_specs[fqn] = (elem.name, slot_idx)
            all_vars[fqn].is_input = True
            input_idx_counter += 1

        if elem.element_type == "ground":
            for port in elem.ports.values():
                for slot in port.across_vars:
                    fqn = f"{elem.name}.{port.name}.{slot}"
                    canonical = _resolve(fqn, eliminated)
                    grounded.add(canonical)
                    all_vars[canonical].is_grounded = True
                    all_vars[canonical].is_differential = False

        elif elem.element_type == "force_source":
            for port in elem.ports.values():
                through_fqn = f"{elem.name}.{port.name}.{port.through_var}"
                input_vars[through_fqn] = input_idx_counter
                input_specs[through_fqn] = (elem.name, 0)
                all_vars[through_fqn].is_input = True
                input_idx_counter += 1

        elif elem.element_type == "prescribed_motion":
            for port in elem.ports.values():
                for slot_idx, slot in enumerate(port.across_vars):
                    fqn = f"{elem.name}.{port.name}.{slot}"
                    canonical = _resolve(fqn, eliminated)
                    input_vars[canonical] = input_idx_counter
                    input_specs[canonical] = (elem.name, slot_idx)
                    all_vars[canonical].is_input = True
                    all_vars[canonical].is_differential = False
                    input_idx_counter += 1

        elif elem.element_type == "sensor":
            if elem.sensor_output is not None:
                port_name, slot_name = elem.sensor_output
                fqn = f"{elem.name}.{port_name}.{slot_name}"
                canonical = _resolve(fqn, eliminated)
                output_map[elem.name] = canonical

        elif elem.element_type == "gear_ratio":
            gear_infos.append({"element": elem, "ratio_key": f"{elem.name}.ratio"})

    eliminated_pre_gear = dict(eliminated)

    for gear_info in gear_infos:
        elem = gear_info["element"]
        for slot in elem.ports["flange_a"].across_vars:
            fqn_a = f"{elem.name}.flange_a.{slot}"
            fqn_b = f"{elem.name}.flange_b.{slot}"
            canon_a = _resolve(fqn_a, eliminated)
            canon_b = _resolve(fqn_b, eliminated)
            if canon_b != canon_a:
                eliminated[canon_b] = canon_a
                all_vars[canon_b].is_eliminated = True
                all_vars[canon_b].alias_of = canon_a
                all_vars[canon_b].is_differential = False

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

    through_equations = [
        eq for elem in elements.values() for eq in elem.equations if eq.is_through_def
    ]
    through_equation_cycle = _through_equation_cycle(through_equations)

    return StructuralAnalysis(
        layout=layout,
        params_dict=params_dict,
        node_through_vars=node_through_vars,
        gear_infos=gear_infos,
        eliminated_pre_gear=eliminated_pre_gear,
        networks=_build_networks(elements, connections, layout),
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
                if var_name.split(".", 1)[0] in names and not layout._vars[var_name].is_eliminated
            )
        )
        differential_variables = tuple(
            var_name for var_name in layout._differential if var_name.split(".", 1)[0] in names
        )
        equation_lhs = tuple(
            sorted(
                eq.lhs_var
                for elem_name in names
                for eq in elements[elem_name].equations
                if not eq.is_through_def
            )
        )
        networks.append(
            StructuralNetwork(
                elements=network_elements,
                variables=variables,
                differential_variables=differential_variables,
                equations=equation_lhs,
                has_ground=any(elements[name].element_type == "ground" for name in names),
                source_elements=tuple(
                    sorted(name for name in names if elements[name].element_type == "force_source")
                ),
                across_source_elements=tuple(
                    sorted(
                        name for name in names if elements[name].element_type == "prescribed_motion"
                    )
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
