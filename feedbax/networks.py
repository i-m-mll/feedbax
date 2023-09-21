"""Neural network architectures."""

from itertools import zip_longest
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from feedbax.utils import interleave_unequal


class SimpleMultiLayerNet(eqx.Module):
    """A series of layers of the same type with nonlinearities of the same type.
    
    NOTE: Could just use `eqx.nn.MLP` in case of linear layers with fixed nonlinearity.
    """
    layers: list 
    
    def __init__(
        self, 
        sizes: Tuple[int, ...], 
        layer_type: eqx.Module = eqx.nn.Linear,
        use_bias=(), 
        nonlinearity=jnp.tanh, 
        output_nonlinearity=None, 
        linear_final_layer=False,  # replace the final layer with a linear layer
        key=jrandom.PRNGKey(0)
    ):
        keys = jrandom.split(key, len(sizes) - 1)
        
        if bool(use_bias) is use_bias:
            use_bias = (use_bias,) * (len(sizes) - 1)
            
        layers = [layer_type(m, n, key=k, use_bias=b) 
                  for m, n, k, b in zip(sizes[:-1], sizes[1:], keys, use_bias)]
        
        nonlinearities = [nonlinearity] * (len(sizes) - 2) 
        if output_nonlinearity is not None:
            nonlinearities += [output_nonlinearity]
        
        # TODO: makes a diff to use eqx.nn.Sequential?
        self.layers = list(interleave_unequal(layers, nonlinearities))
        
        if linear_final_layer:
            self.layers[-1] = eqx.nn.Linear(sizes[-2], sizes[-1], key=keys[-1])

    def __call__(self, x):        
        for layer in self.layers:
            x = layer(x)
        return x


class RNN(eqx.Module):
    """From https://docs.kidger.site/equinox/examples/train_rnn/"""
    hidden_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array
    # activation: Callable

    def __init__(self, in_size, out_size, hidden_size, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)
        # if activation is None:
        #     self.activation = lambda x: x
        # else:
        #     self.activation = activation

    def __call__(self, input, state):
        state = self.init_state()
        # TODO: flatten leaves before concatenating 
        input = jnp.concatenate(jax.tree_leaves(input))
        state = self.cell(input, state)
        #state = jax.nn.tanh(state)
        return self.linear(state) + self.bias, state
    
    def init_state(self, state=None):
        if state is None:
            return jnp.zeros(self.hidden_size)
        else:
            return jnp.array(state)