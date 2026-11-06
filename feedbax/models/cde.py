"""Build a neural controlled differential equation controller as a native graph."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from feedbax.runtime.graph import Graph


def cde_controller(
    obs_dim: int,
    hidden_dim: int,
    out_size: int,
    vf_width: int = 64,
    vf_depth: int = 2,
    decay: float = 0.1,
    use_anti_nf: bool = True,
    alpha: float = 1.0,
    dtype: Any = jnp.float32,
    *,
    key: PRNGKeyArray,
) -> Graph:
    """Compile and initialize a native neural CDE controller."""
    from feedbax.compiler import GraphDocument, compile_graph
    from feedbax.compiler.cde_templates import cde_controller_spec
    from feedbax.component_registry import ComponentRegistry

    spec = cde_controller_spec(
        obs_dim, hidden_dim, out_size, vf_width, vf_depth, decay, use_anti_nf, alpha, dtype
    )
    field_key, readout_key, h0_key, gate_key = jr.split(key, 4)
    keys = {("vf",): field_key, ("linear",): readout_key}
    if use_anti_nf:
        keys[("gru_gate",)] = gate_key
    graph = compile_graph(
        GraphDocument(graph=spec),
        ComponentRegistry(load_user_components=False),
        initialization_keys=keys,
    ).graph
    return eqx.tree_at(
        lambda model: model.nodes["h0"].value,
        graph,
        0.01 * jr.normal(h0_key, (hidden_dim,), dtype=dtype),
    )
