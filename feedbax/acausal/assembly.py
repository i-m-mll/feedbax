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

from typing import Callable

import equinox as eqx
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
    StateLayout,
)


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
    vf_fn, sensor_fn = _make_vector_field(analysis, elements)

    return analysis.layout, vf_fn, analysis.params_dict, sensor_fn


# ---------------------------------------------------------------------------
# Vector field builder
# ---------------------------------------------------------------------------


def _make_vector_field(
    analysis: StructuralAnalysis,
    elements: dict[str, AcausalElement],
) -> tuple[Callable, Callable]:
    """Pull physical effort and inertia back through q_i = s_i(params) q."""
    layout = analysis.layout
    coordinates = analysis.coordinate_map
    physical_aliases = analysis.physical_aliases
    indices = {name: idx for idx, name in enumerate(layout._differential)}
    through_eqs = _topo_sort_through_eqs(
        [eq for elem in elements.values() for eq in elem.equations if eq.is_through_def]
    )
    node_efforts: dict[str, set[str]] = {}
    node_masses: dict[str, list[str]] = {}
    velocity_for_node: dict[str, str] = {}
    position_pairs: set[tuple[int, int]] = set()
    for elem in elements.values():
        for port in elem.ports.values():
            position = f"{elem.name}.{port.name}.{port.across_vars[0]}"
            velocity = f"{elem.name}.{port.name}.{port.across_vars[1]}"
            node = _resolve(position, physical_aliases)
            velocity_for_node[node] = velocity
            node_efforts.setdefault(node, set())
            if elem.element_type not in {
                "mass",
                "sensor",
                "gear_ratio",
                "ground",
                "prescribed_motion",
            }:
                node_efforts[node].add(f"{elem.name}.{port.name}.{port.through_var}")
            if elem.element_type == "mass":
                mass_keys = [
                    key for key in elem.params if key.endswith(".mass") or key.endswith(".inertia")
                ]
                if not mass_keys:
                    raise ValueError(f"Mass element '{elem.name}' has no mass/inertia parameter")
                node_masses.setdefault(node, []).append(mass_keys[0])
                root_pos, _ = coordinates[position]
                root_vel, _ = coordinates[velocity]
                if root_pos in indices and root_vel in indices:
                    position_pairs.add((indices[root_pos], indices[root_vel]))

    # Physical nodes contribute once, even when several inertias share a flange.
    groups: dict[int, list[str]] = {}
    for node, velocity in velocity_for_node.items():
        root, _ = coordinates[velocity]
        if root in indices:
            groups.setdefault(indices[root], []).append(node)
    groups = {idx: sorted(nodes) for idx, nodes in groups.items()}
    available = (
        set(layout._differential)
        | layout._grounded
        | set(layout._inputs)
        | set(analysis.params_dict)
        | {eq.lhs_var for eq in through_eqs}
    )

    def require(names, context):
        missing = sorted(
            {
                name
                for name in names
                if name not in available and coordinates.get(name, (name, ()))[0] not in available
            }
        )
        if missing:
            raise ValueError(
                f"Unresolved acausal through-variable dependency in {context}: {missing}"
            )

    for eq in through_eqs:
        require(eq.depends_on, f"through equation {eq.lhs_var!r}")
    for node, efforts in node_efforts.items():
        require(efforts, f"force balance for {node!r}")

    effort_sensors = {}
    for label in layout._outputs:
        elem = elements[label]
        port_name, slot = elem.sensor_output
        port = elem.ports[port_name]
        if slot != port.through_var:
            continue
        position = f"{elem.name}.{port_name}.{port.across_vars[0]}"
        node = _resolve(position, physical_aliases)
        root, factors = coordinates[velocity_for_node[node]]
        if root not in layout._grounded:
            if root in layout._inputs:
                raise ValueError(
                    f"Net effort sensor {label!r} requires acceleration of prescribed "
                    f"coordinate {root!r}; prescribed velocity inputs do not supply acceleration"
                )
            if root not in indices or not any(
                node_masses.get(member) for member in groups[indices[root]]
            ):
                raise ValueError(
                    f"Net effort sensor {label!r} has no solved acceleration for free "
                    f"coordinate {root!r}: its reduced group requires inertia"
                )
        effort_sensors[label] = (node, root, factors)

    def scale(factors, params):
        value = 1.0
        for key, exponent in factors:
            ratio = jnp.asarray(params[key], dtype=jnp.result_type(params[key], 1.0))
            ratio = eqx.error_if(
                ratio,
                (ratio == 0) | ~jnp.isfinite(ratio),
                f"Gear ratio {key!r} must be finite and nonzero",
            )
            value = value * ratio**exponent
        return value

    def build_vals(y, inputs, params):
        vals = dict(params)
        vals.update((name, y[idx]) for name, idx in indices.items())
        vals.update((name, 0.0) for name in layout._grounded)
        vals.update((name, inputs[idx]) for name, idx in layout._inputs.items())
        for name, (root, factors) in coordinates.items():
            if root in vals:
                vals[name] = scale(factors, params) * vals[root]
        for eq in through_eqs:
            vals[eq.lhs_var] = eq.rhs_fn(vals)
        return vals

    def accelerations(vals, params):
        result = {}
        for idx, nodes in groups.items():
            effort = 0.0
            inertia = 0.0
            for node in nodes:
                _, factors = coordinates[velocity_for_node[node]]
                weight = scale(factors, params)
                effort = effort + weight * sum(vals[tv] for tv in sorted(node_efforts[node]))
                inertia = inertia + weight**2 * sum(
                    params[key] for key in node_masses.get(node, ())
                )
            if not any(node_masses.get(node) for node in nodes):
                continue
            result[idx] = effort / inertia
        return result

    def vector_field(t, y, args):
        inputs, params = args
        vals = build_vals(y, inputs, params)
        dy = jnp.zeros_like(y)
        for pos_idx, vel_idx in sorted(position_pairs):
            dy = dy.at[pos_idx].set(y[vel_idx])
        for idx, acceleration in accelerations(vals, params).items():
            dy = dy.at[idx].set(acceleration)
        return dy

    def sensor_fn(y, inputs, params):
        vals = build_vals(y, inputs, params)
        acceleration = accelerations(vals, params)
        outputs = {}
        for label, name in layout._outputs.items():
            if label not in effort_sensors:
                outputs[label] = vals[name]
                continue
            node, root, factors = effort_sensors[label]
            if root in layout._grounded:
                local_acceleration = 0.0
            else:
                local_acceleration = acceleration[indices[root]] * scale(factors, params)
            outputs[label] = (
                sum(params[key] for key in node_masses.get(node, ())) * local_acceleration
            )
        return outputs

    return vector_field, sensor_fn
