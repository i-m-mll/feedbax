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
from typing import Callable, Iterable

import jax.numpy as jnp

from feedbax.acausal.analysis import (
    StructuralAnalysis,
    analyze_system,
    _resolve,
    _topo_sort_through_eqs,
)
from feedbax.acausal.base import (
    AcausalConnection,
    AcausalElement,
    AcausalEquation,
    StateLayout,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assemble_system(
    elements: dict[str, AcausalElement],
    connections: list[AcausalConnection],
    analysis: StructuralAnalysis | None = None,
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

    analysis = analysis or analyze_system(elements, connections)

    # ---- Step 6: Build the compiled vector field ------------------------
    vf_fn, sensor_fn = _make_vector_field(
        layout=analysis.layout,
        elements=elements,
        connections=connections,
        node_through_vars=analysis.node_through_vars,
        gear_infos=analysis.gear_infos,
        eliminated_pre_gear=analysis.eliminated_pre_gear,
    )

    return analysis.layout, vf_fn, analysis.params_dict, sensor_fn


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

    diff_var_indices = {vname: idx for idx, vname in enumerate(diff_vars)}

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

    mass_through_vars: set[str] = set()
    sensor_through_vars: set[str] = set()
    for elem in elements.values():
        if elem.element_type == "mass":
            for port in elem.ports.values():
                mass_through_vars.add(
                    f"{elem.name}.{port.name}.{port.through_var}"
                )
        elif elem.element_type == "sensor":
            for port in elem.ports.values():
                sensor_through_vars.add(
                    f"{elem.name}.{port.name}.{port.through_var}"
                )

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
            sorted(
                set(
                    tv for tv in node_b_through
                    if tv != tau_b
                    and tv not in mass_through_vars
                    and tv not in sensor_through_vars
                )
            )
        )

        # tau_b = -(sum of other through vars at node B)
        through_eqs.append(AcausalEquation(
            lhs_var=tau_b,
            rhs_fn=lambda vals, _tvars=node_b_sources: (
                -sum((vals[tv] for tv in _tvars), 0.0)
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

            vel_idx = diff_var_indices[canon_vel]

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
                tv for tv in set(connected_through)
                if tv != own_through and tv not in sensor_through_vars
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
                    (diff_var_indices[canon_pos], diff_var_indices[canon_vel])
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

    through_lhs = {eq.lhs_var for eq in through_eqs}
    available_vars = set(diff_vars) | set(grounded) | set(input_vars) | set(params_keys) | through_lhs

    def _is_available(var_name: str) -> bool:
        if var_name in available_vars:
            return True
        try:
            resolved = _resolve(var_name, eliminated)
        except RuntimeError:
            return False
        return resolved in available_vars or var_name in gear_ratio_scale

    def _require_available(var_names: Iterable[str], *, context: str) -> None:
        missing = sorted({var_name for var_name in var_names if not _is_available(var_name)})
        if missing:
            raise ValueError(
                f"Unresolved acausal through-variable dependency in {context}: {missing}"
            )

    for eq in through_eq_order:
        _require_available(eq.depends_on, context=f"through equation {eq.lhs_var!r}")

    # ---- Sensor evaluation info ----------------------------------------
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
        _require_available(
            mi["through_sources_frozen"],
            context=f"force balance for {mi['elem_name']!r}",
        )

    pos_vel_pairs_frozen = tuple(pos_vel_pairs)
    mass_infos_frozen = tuple(
        {k: v for k, v in mi.items()} for mi in mass_infos
    )
    through_eval_frozen = tuple(through_eval_infos)
    eliminated_items = tuple(eliminated.items())
    sensor_eval_frozen = tuple(sensor_eval_infos)
    for info in sensor_eval_frozen:
        sum_vars = info["sum_vars"]
        if sum_vars is not None:
            _require_available(sum_vars, context=f"sensor output {info['label']!r}")

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
                else:
                    raise KeyError(f"Unresolved through-variable contribution {tv_name!r}")
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
                else:
                    raise KeyError(f"Unresolved sensor through-variable contribution {tv_name!r}")
            outputs[label] = total
        return outputs

    return vector_field, sensor_fn
