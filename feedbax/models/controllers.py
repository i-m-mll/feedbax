"""Graph construction for recurrent controller compositions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox import Module
from jaxtyping import PRNGKeyArray

from feedbax._vendor.jax_cookbook import identity as identity_func
from feedbax.runtime.components import GRU, LSTM, named_activation
from feedbax.runtime.graph import Graph

if TYPE_CHECKING:
    from feedbax.models.networks import PopulationStructure


def recurrent_controller(
    input_size: int,
    hidden_size: int,
    *,
    out_size: int | None = None,
    encoding_size: int | None = None,
    hidden_type: str | type[Module] = eqx.nn.GRUCell,
    cell_params: Mapping[str, object] | None = None,
    use_bias: bool = True,
    hidden_nonlinearity: str | Callable = identity_func,
    out_nonlinearity: str | Callable = identity_func,
    hidden_noise_std: float | None = None,
    population_structure: PopulationStructure | Mapping[str, object] | None = None,
    dtype: object = jnp.float32,
    key: PRNGKeyArray,
) -> Graph:
    """Compile a staged recurrent controller from the built-in graph template.

    Choose a built-in cell by name or pass the GRU, LSTM, or LeakyRNNCell
    constructor. Supply cell options through ``cell_params``. Omitting
    ``out_size`` binds the controller output directly to the hidden signal,
    with the requested output activation. For manual stepping, pass the
    recurrent carry returned by :meth:`Graph.step` to the next step.
    The carry includes the hidden signal after any activation and noise.
    """
    from feedbax.compiler.graph import GraphDocument, compile_graph
    from feedbax.compiler.templates import recurrent_controller_template_graph
    from feedbax.component_registry import ComponentRegistry
    from feedbax.models.networks import (
        LeakyRNNCell,
        VanillaRNN,
        normalize_recurrent_cell_type,
    )

    if isinstance(hidden_type, str):
        cell_type = normalize_recurrent_cell_type(hidden_type)
    elif hidden_type in (eqx.nn.GRUCell, GRU):
        cell_type = "GRU"
    elif hidden_type in (eqx.nn.LSTMCell, LSTM):
        cell_type = "LSTM"
    elif hidden_type in (LeakyRNNCell, VanillaRNN):
        cell_type = "VanillaRNN"
    else:
        raise TypeError("hidden_type must be a built-in recurrent cell name or constructor")

    def activation_name(value: str | Callable) -> str:
        if value is identity_func:
            return "identity"
        if value is jnp.tanh:
            return "tanh"
        return named_activation(value)

    spec = recurrent_controller_template_graph(
        input_size=input_size,
        hidden_size=hidden_size,
        out_size=out_size,
        encoding_size=encoding_size,
        cell_type=cell_type,
        cell_params={"use_bias": use_bias, **(cell_params or {})},
        hidden_nonlinearity=activation_name(hidden_nonlinearity),
        out_nonlinearity=activation_name(out_nonlinearity),
        hidden_noise_std=hidden_noise_std,
        encoder_use_bias=use_bias if population_structure is not None else True,
        readout_use_bias=use_bias if population_structure is not None else True,
        readout_initialization="zero_bias",
        population_structure=population_structure,
        dtype=jnp.dtype(dtype if dtype is not None else jnp.float32).name,
    )
    keys = dict(zip(("cell", "encoder", "readout"), jr.split(key, 3), strict=True))
    registry = ComponentRegistry()
    return compile_graph(
        GraphDocument(graph=spec),
        registry,
        initialization_keys={name: keys[name] for name in keys if name in spec.nodes},
    ).graph
