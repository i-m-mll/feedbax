"""Build a neural controlled differential equation controller as a native graph."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from feedbax.component_registry.cde_templates import (
    cde_controller_spec as cde_controller_spec,
    cde_initialization_keys as cde_initialization_keys,
)
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
    from feedbax.component_registry import ComponentRegistry

    spec = cde_controller_spec(
        obs_dim, hidden_dim, out_size, vf_width, vf_depth, decay, use_anti_nf, alpha, dtype, key=key
    )
    return compile_graph(
        GraphDocument(graph=spec),
        ComponentRegistry(load_user_components=False),
        initialization_keys=cde_initialization_keys(key, use_anti_nf=use_anti_nf),
    ).graph
