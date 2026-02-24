"""Equation assembly algorithm for acausal systems.

The assembler takes a set of elements and connections and produces:

1. A ``StateLayout`` that maps variable names to indices in a flat state
   vector.
2. A compiled **vector field** function ``(t, y, args) -> dy`` suitable for
   ``diffrax`` solvers and fully JAX-traceable.
3. A dictionary of initial parameter values.

Algorithm outline
-----------------
1. Collect variables from every element/port.
2. Process connections with a Union-Find to identify shared (across) nodes.
3. Handle special elements: Ground, ForceSource, PrescribedMotion, Sensors,
   GearRatio.
4. Assign indices to surviving differential variables.
5. Build the compiled vector field via closures over pre-computed indices.

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0.  See LICENSE for details.
"""

from __future__ import annotations

import logging
from typing import Callable

import jax.numpy as jnp

from feedbax.acausal.base import (
    AcausalConnection,
    AcausalElement,
    AcausalEquation,
    AcausalVar,
    StateLayout,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

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
        g: dict[str, list[str]] = {}
        for x in self._parent:
            root = self.find(x)
            g.setdefault(root, []).append(x)
        return g


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assemble_system(
    elements: dict[str, AcausalElement],
    connections: list[AcausalConnection],
) -> tuple[StateLayout, Callable, dict[str, float], Callable]:
    """Assemble acausal elements and connections into a compiled vector field.

    Args:
        elements: Named elements.
        connections: Port-level connections between elements.

    Returns:
        layout: ``StateLayout`` mapping variable names to state-vector slots.
        vector_field_fn: ``(t, y, args) -> dy`` ready for diffrax.
        params_dict: Initial parameter values.
        sensor_fn: ``(y, input_vals, params_dict) -> outputs`` for sensors.
    """

    # ---- Step 1: Collect all variables and parameters --------------------
    all_vars: dict[str, AcausalVar] = {}
    params_dict: dict[str, float] = {}

    for elem in elements.values():
        for port in elem.ports.values():
            for slot in port.across_vars:
                fqn = f"{elem.name}.{port.name}.{slot}"
                all_vars[fqn] = AcausalVar(name=fqn)
            through_fqn = f"{elem.name}.{port.name}.{port.through_var}"
            all_vars[through_fqn] = AcausalVar(
                name=through_fqn, is_differential=False,
            )
        params_dict.update(elem.params)

    # ---- Step 2: Union-Find over across variables -----------------------
    uf = UnionFind()
    # Track which through-vars share a node so we can build force balances
    # node_key -> list of (element_name, port_name, through_var_fqn)
    node_through_vars: dict[str, list[str]] = {}

    for conn in connections:
        port_a_obj = elements[conn.element_a].ports[conn.port_a]
        port_b_obj = elements[conn.element_b].ports[conn.port_b]

        # Union across vars pair-wise
        for slot in port_a_obj.across_vars:
            fqn_a = f"{conn.element_a}.{conn.port_a}.{slot}"
            fqn_b = f"{conn.element_b}.{conn.port_b}.{slot}"
            uf.union(fqn_a, fqn_b)

        # Register through vars at the shared node.  The node key is the
        # across-variable group root of the first across var (they are all
        # in the same group now).
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

    # ---- Step 3: Identify canonical across variables --------------------
    eliminated: dict[str, str] = {}
    across_groups = uf.groups()
    for root, members in across_groups.items():
        for m in members:
            if m != root:
                all_vars[m].is_eliminated = True
                all_vars[m].alias_of = root
                eliminated[m] = root

    # ---- Step 4: Handle special elements --------------------------------
    grounded: set[str] = set()
    input_vars: dict[str, int] = {}  # var_fqn -> input_index
    input_specs: dict[str, tuple[str, int]] = {}  # var_fqn -> (element, slot_index)
    input_idx_counter = 0
    output_map: dict[str, str] = {}  # sensor_label -> var_fqn
    gear_infos: list[dict] = []

    for elem in elements.values():
        if elem.element_type == "ground":
            # Ground all across vars connected to this element's port
            for port in elem.ports.values():
                for slot in port.across_vars:
                    fqn = f"{elem.name}.{port.name}.{slot}"
                    canonical = _resolve(fqn, eliminated)
                    grounded.add(canonical)
                    all_vars[canonical].is_grounded = True
                    all_vars[canonical].is_differential = False

        elif elem.element_type == "force_source":
            # The through var of the source's port becomes a causal input
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
            gear_infos.append({
                "element": elem,
                "ratio_key": f"{elem.name}.ratio",
            })

    # Snapshot eliminated before gear processing -- the pre-gear version
    # is needed to resolve physical nodes correctly (gear elimination must
    # NOT merge force-balance nodes).
    eliminated_pre_gear = dict(eliminated)

    # Handle gear ratios: eliminate port_b across vars in favour of
    # port_a across vars * ratio.
    for gi in gear_infos:
        elem = gi["element"]
        for slot in elem.ports["flange_a"].across_vars:
            fqn_a = f"{elem.name}.flange_a.{slot}"
            fqn_b = f"{elem.name}.flange_b.{slot}"
            canon_a = _resolve(fqn_a, eliminated)
            canon_b = _resolve(fqn_b, eliminated)
            # Mark b as eliminated, pointing to a (scaled -- handled at VF
            # build time via special gear lookup).
            if canon_b != canon_a:
                eliminated[canon_b] = canon_a
                all_vars[canon_b].is_eliminated = True
                all_vars[canon_b].alias_of = canon_a
                all_vars[canon_b].is_differential = False

    # ---- Step 5: Build differential variable list -----------------------
    differential: list[str] = []
    for name, var in sorted(all_vars.items()):
        if (
            var.is_differential
            and not var.is_eliminated
            and not var.is_grounded
            and not var.is_input
        ):
            differential.append(name)

    algebraic: list[str] = []  # Not yet needed for index-1 DAE support

    layout = StateLayout(
        _vars=all_vars,
        _differential=differential,
        _algebraic=algebraic,
        _eliminated=eliminated,
        _grounded=grounded,
        _inputs=input_vars,
        _input_specs=input_specs,
        _outputs=output_map,
        total_size=len(differential),
    )

    # ---- Step 6: Build the compiled vector field ------------------------
    vf_fn, sensor_fn = _make_vector_field(
        layout=layout,
        elements=elements,
        connections=connections,
        node_through_vars=node_through_vars,
        gear_infos=gear_infos,
        eliminated_pre_gear=eliminated_pre_gear,
    )

    return layout, vf_fn, params_dict, sensor_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(name: str, eliminated: dict[str, str]) -> str:
    """Follow alias chain."""
    visited: set[str] = set()
    while name in eliminated:
        if name in visited:
            raise RuntimeError(f"Cyclic alias for '{name}'")
        visited.add(name)
        name = eliminated[name]
    return name


# ---------------------------------------------------------------------------
# Vector field builder
# ---------------------------------------------------------------------------

def _make_vector_field(
    layout: StateLayout,
    elements: dict[str, AcausalElement],
    connections: list[AcausalConnection],
    node_through_vars: dict[str, list[str]],
    gear_infos: list[dict],
    eliminated_pre_gear: dict[str, str] | None = None,
) -> tuple[Callable, Callable]:
    """Build a JAX-traceable vector field function.

    The returned functions are:
        - ``vector_field(t, y, args) -> dy``
        - ``sensor_fn(y, input_vals, params_dict) -> dict``
    where ``args = (input_array, params_dict)``.

    All Python-level indexing is pre-computed so that the function body
    contains only ``jnp`` operations and integer-literal array indexing,
    making it fully traceable.
    """

    diff_vars = layout._differential
    eliminated = layout._eliminated
    grounded = layout._grounded
    input_vars = layout._inputs

    # ---- Pre-compute index maps (all plain Python ints) -----------------

    # var_name -> how to look it up at runtime
    # Each entry is one of:
    #   ("state", int)        -- read from y[i]
    #   ("ground", None)      -- always 0
    #   ("input", int)        -- read from input_vals[i]
    #   ("param", int)        -- read from param_vals[i]
    #   ("alias", str)        -- resolve to another var first
    var_lookup: dict[str, tuple[str, object]] = {}

    for vname in diff_vars:
        var_lookup[vname] = ("state", diff_vars.index(vname))
    for vname in grounded:
        var_lookup[vname] = ("ground", None)
    for vname, idx in input_vars.items():
        var_lookup[vname] = ("input", idx)

    # Through vars that are not in diff/grounded/input are computed on the fly
    # -- they get resolved later during VF evaluation.

    # Collect sorted param keys
    params_keys = sorted(
        set().union(*(elem.params.keys() for elem in elements.values()))
    )

    # ---- Collect through-variable equations per element ------------------
    through_eqs: list[AcausalEquation] = []
    for elem in elements.values():
        for eq in elem.equations:
            if eq.is_through_def:
                through_eqs.append(eq)

    # Gear torque transmission via node-B force balance + power conservation.
    # At node B (massless): sum of non-gear through vars = -gear.flange_b.torque
    # Power conservation: gear.flange_a.torque = ratio * gear.flange_b.torque
    _epg = eliminated_pre_gear or eliminated
    for gi in gear_infos:
        elem = gi["element"]
        ratio_key = gi["ratio_key"]
        tau_a = f"{elem.name}.flange_a.torque"
        tau_b = f"{elem.name}.flange_b.torque"

        # Find through vars at node B (using pre-gear elimination so we don't
        # accidentally merge nodes A and B).
        fqn_b_first = (
            f"{elem.name}.flange_b.{elem.ports['flange_b'].across_vars[0]}"
        )
        canon_b_pre = _resolve(fqn_b_first, _epg)
        node_b_through: list[str] = []
        for _nk, tvars in node_through_vars.items():
            resolved_nk = _resolve(_nk, _epg)
            if resolved_nk == canon_b_pre:
                node_b_through.extend(tvars)
        # Exclude the gear's own flange_b through var
        node_b_sources = tuple(
            sorted(set(tv for tv in node_b_through if tv != tau_b))
        )

        # tau_b = -(sum of other through vars at node B)
        through_eqs.append(AcausalEquation(
            lhs_var=tau_b,
            rhs_fn=lambda vals, _tvars=node_b_sources: (
                -sum((vals.get(tv, 0.0) for tv in _tvars), 0.0)
            ),
            depends_on=node_b_sources,
            is_through_def=True,
        ))
        # tau_a = ratio * tau_b
        through_eqs.append(AcausalEquation(
            lhs_var=tau_a,
            rhs_fn=lambda vals, _tb=tau_b, _r=ratio_key: (
                vals[_r] * vals[_tb]
            ),
            depends_on=(tau_b, ratio_key),
            is_through_def=True,
        ))

    # ---- Build node force-balance information ---------------------------
    # For each mass/inertia element, find which through vars sum at its node
    mass_infos: list[dict] = []
    for elem in elements.values():
        if elem.element_type != "mass":
            continue
        for port in elem.ports.values():
            # Find the velocity equation (the one with rhs_fn=None)
            vel_slot = port.across_vars[1]  # vel or angular_vel
            vel_fqn = f"{elem.name}.{port.name}.{vel_slot}"
            canon_vel = _resolve(vel_fqn, eliminated)

            if canon_vel not in diff_vars:
                continue  # grounded or eliminated

            vel_idx = diff_vars.index(canon_vel)

            # Find the position var too
            pos_slot = port.across_vars[0]
            pos_fqn = f"{elem.name}.{port.name}.{pos_slot}"
            canon_pos = _resolve(pos_fqn, eliminated)

            # Find the mass/inertia parameter
            mass_param_candidates = [
                k for k in elem.params
                if k.endswith(".mass") or k.endswith(".inertia")
            ]
            if not mass_param_candidates:
                raise ValueError(
                    f"Mass element '{elem.name}' has no mass/inertia parameter"
                )
            mass_param_key = mass_param_candidates[0]

            # Collect through vars at this node by looking up which node
            # this port's across var belongs to.  Use eliminated_pre_gear
            # so gear elimination doesn't merge distinct physical nodes.
            first_across = f"{elem.name}.{port.name}.{port.across_vars[0]}"
            canon_first = _resolve(first_across, _epg)

            # Find all through vars that share this node
            connected_through: list[str] = []
            for _node_key, tvars in node_through_vars.items():
                # Check if any across var in this node resolves to canon_first
                # We need to check the node_key resolution
                resolved_node = _resolve(_node_key, _epg)
                if resolved_node == canon_first:
                    connected_through.extend(tvars)

            # Also include the element's own through var
            own_through = f"{elem.name}.{port.name}.{port.through_var}"
            if own_through not in connected_through:
                connected_through.append(own_through)

            # Remove duplicates and the mass element's own through var
            # (it does not contribute force to itself)
            through_sources = [
                tv for tv in set(connected_through) if tv != own_through
            ]

            mass_infos.append({
                "vel_idx": vel_idx,
                "mass_key": mass_param_key,
                "through_sources": through_sources,
                "elem_name": elem.name,
            })

    # ---- Build position-derivative pairs --------------------------------
    pos_vel_pairs: list[tuple[int, int]] = []
    for elem in elements.values():
        if elem.element_type != "mass":
            continue
        for port in elem.ports.values():
            pos_slot = port.across_vars[0]
            vel_slot = port.across_vars[1]
            pos_fqn = f"{elem.name}.{port.name}.{pos_slot}"
            vel_fqn = f"{elem.name}.{port.name}.{vel_slot}"
            canon_pos = _resolve(pos_fqn, eliminated)
            canon_vel = _resolve(vel_fqn, eliminated)
            if canon_pos in diff_vars and canon_vel in diff_vars:
                pos_vel_pairs.append(
                    (diff_vars.index(canon_pos), diff_vars.index(canon_vel))
                )
            elif canon_pos in grounded:
                pass  # grounded pos, no derivative needed
            elif canon_pos in input_vars:
                pass  # prescribed, no derivative needed

    # ---- Pre-compute through-equation evaluation info -------------------
    # We need to topologically sort through equations (some depend on others)
    through_eq_order = _topo_sort_through_eqs(through_eqs)

    # For each through equation, pre-compute how to resolve each dependency
    through_eval_infos: list[dict] = []
    for eq in through_eq_order:
        info: dict = {
            "lhs_var": eq.lhs_var,
            "rhs_fn": eq.rhs_fn,
            "depends_on": eq.depends_on,
        }
        through_eval_infos.append(info)

    # ---- Build gear ratio lookup ----------------------------------------
    # Use pre-gear elimination so the scaled alias is preserved even after
    # the gear constraint merges nodes in the final eliminated map.
    gear_ratio_scale: dict[str, tuple[str, str]] = {}
    for gi in gear_infos:
        elem = gi["element"]
        ratio_key = gi["ratio_key"]
        for slot in elem.ports["flange_a"].across_vars:
            fqn_b = f"{elem.name}.flange_b.{slot}"
            fqn_a = f"{elem.name}.flange_a.{slot}"
            canon_b = _resolve(fqn_b, _epg)
            canon_a = _resolve(fqn_a, _epg)
            if canon_b != canon_a:
                gear_ratio_scale[canon_b] = (canon_a, ratio_key)

    # ---- Sensor evaluation info ----------------------------------------
    mass_through_vars: set[str] = set()
    for elem in elements.values():
        if elem.element_type != "mass":
            continue
        for port in elem.ports.values():
            mass_through_vars.add(
                f"{elem.name}.{port.name}.{port.through_var}"
            )

    sensor_eval_infos: list[dict] = []
    for label, var_fqn in layout._outputs.items():
        canon = _resolve(var_fqn, eliminated)
        if canon in diff_vars:
            continue
        sum_vars: tuple[str, ...] | None = None
        sensor_elem = elements.get(label)
        if (
            sensor_elem is not None
            and sensor_elem.element_type == "sensor"
            and sensor_elem.sensor_output is not None
        ):
            port_name, slot_name = sensor_elem.sensor_output
            port = sensor_elem.ports[port_name]
            if slot_name == port.through_var:
                first_across = (
                    f"{sensor_elem.name}.{port_name}.{port.across_vars[0]}"
                )
                canon_first = _resolve(first_across, _epg)
                connected_through: list[str] = []
                for _node_key, tvars in node_through_vars.items():
                    resolved_node = _resolve(_node_key, _epg)
                    if resolved_node == canon_first:
                        connected_through.extend(tvars)
                sensor_through = (
                    f"{sensor_elem.name}.{port_name}.{port.through_var}"
                )
                through_sources = [
                    tv for tv in set(connected_through)
                    if tv != sensor_through and tv not in mass_through_vars
                ]
                sum_vars = tuple(sorted(through_sources))
        sensor_eval_infos.append({
            "label": label,
            "var_name": canon,
            "sum_vars": sum_vars,
        })

    # ---- Freeze all pre-computed data -----------------------------------
    # Convert mass_infos through_sources to index-based lookups
    for mi in mass_infos:
        mi["through_sources_frozen"] = tuple(mi["through_sources"])

    pos_vel_pairs_frozen = tuple(pos_vel_pairs)
    mass_infos_frozen = tuple(
        {k: v for k, v in mi.items()} for mi in mass_infos
    )
    through_eval_frozen = tuple(through_eval_infos)
    eliminated_items = tuple(eliminated.items())
    sensor_eval_frozen = tuple(sensor_eval_infos)

    # ---- The actual vector field ----------------------------------------

    def _build_vals(y, input_vals, param_vals):
        # Build a value dict for evaluating through-variable equations.
        # We use a plain dict so through-eq lambdas can look up by name.
        vals: dict[str, object] = {}

        # State variables
        for i, vname in enumerate(diff_vars):
            vals[vname] = y[i]

        # Grounded variables
        for vname in grounded:
            vals[vname] = 0.0

        # Input variables
        for vname, idx in input_vars.items():
            vals[vname] = input_vals[idx]

        # Eliminated (alias) variables -- resolve to canonical (handle chains)
        updated = True
        while updated:
            updated = False
            for alias, canon in eliminated_items:
                if alias in vals or canon not in vals:
                    continue
                if alias in gear_ratio_scale:
                    _, ratio_key = gear_ratio_scale[alias]
                    vals[alias] = vals[canon] * param_vals[ratio_key]
                else:
                    vals[alias] = vals[canon]
                updated = True

        # Parameters
        for pkey in params_keys:
            vals[pkey] = param_vals[pkey]

        # Evaluate through-variable equations in topological order
        for eq_info in through_eval_frozen:
            lhs = eq_info["lhs_var"]
            fn = eq_info["rhs_fn"]
            vals[lhs] = fn(vals)

        return vals

    def vector_field(t, y, args):
        input_vals, param_vals = args
        vals = _build_vals(y, input_vals, param_vals)

        # Build dy
        dy = jnp.zeros_like(y)

        # Position derivatives: d(pos)/dt = vel
        for pos_idx, vel_idx in pos_vel_pairs_frozen:
            dy = dy.at[pos_idx].set(y[vel_idx])

        # Velocity derivatives: d(vel)/dt = net_force / mass
        for mi in mass_infos_frozen:
            vel_idx = mi["vel_idx"]
            mass_val = param_vals[mi["mass_key"]]
            # Sum through-variable contributions at this node
            net_force = 0.0
            for tv_name in mi["through_sources_frozen"]:
                # Resolve aliases
                resolved = tv_name
                while resolved in eliminated:
                    resolved = eliminated[resolved]
                if resolved in vals:
                    net_force = net_force + vals[resolved]
                elif tv_name in vals:
                    net_force = net_force + vals[tv_name]
            dy = dy.at[vel_idx].set(net_force / mass_val)

        return dy

    def sensor_fn(y, input_vals, param_vals):
        vals = _build_vals(y, input_vals, param_vals)
        outputs: dict[str, object] = {}
        for info in sensor_eval_frozen:
            label = info["label"]
            var_name = info["var_name"]
            if var_name in vals:
                outputs[label] = vals[var_name]
                continue
            sum_vars = info["sum_vars"]
            if sum_vars is None:
                continue
            total = 0.0
            for tv_name in sum_vars:
                resolved = tv_name
                while resolved in eliminated:
                    resolved = eliminated[resolved]
                if resolved in vals:
                    total = total + vals[resolved]
                elif tv_name in vals:
                    total = total + vals[tv_name]
            outputs[label] = total
        return outputs

    return vector_field, sensor_fn


# ---------------------------------------------------------------------------
# Through-equation topological sort
# ---------------------------------------------------------------------------

def _topo_sort_through_eqs(
    equations: list[AcausalEquation],
) -> list[AcausalEquation]:
    """Sort through-variable equations so dependencies are evaluated first.

    Through-variable equations may depend on other through variables (e.g.
    ``force_b = -force_a``).  We topologically sort them so each equation's
    dependencies are available by the time it runs.
    """
    by_lhs: dict[str, AcausalEquation] = {}
    for eq in equations:
        by_lhs[eq.lhs_var] = eq

    through_vars = set(by_lhs.keys())
    visited: set[str] = set()
    order: list[AcausalEquation] = []

    def visit(var: str) -> None:
        if var in visited:
            return
        visited.add(var)
        eq = by_lhs.get(var)
        if eq is None:
            return
        for dep in eq.depends_on:
            if dep in through_vars:
                visit(dep)
        order.append(eq)

    for var in through_vars:
        visit(var)

    return order
