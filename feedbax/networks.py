"""Neural network architectures."""

from functools import cached_property
from itertools import zip_longest
from typing import Callable, Optional, Tuple

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
        key,
        layer_type: eqx.Module = eqx.nn.Linear,
        use_bias=(), 
        nonlinearity=jnp.tanh, 
        output_nonlinearity=None, 
        linear_final_layer=False,  # replace the final layer with a linear layer
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
    out_nonlinearity: Callable 
    noise_std: Optional[float]

    def __init__(
            self, 
            in_size, 
            out_size, 
            hidden_size, 
            key, 
            out_nonlinearity=lambda x: x,
            noise_std=None,
        ):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)
        self.out_nonlinearity = out_nonlinearity       
        self.noise_std = noise_std
        
        # initialize cached properties
        self._add_noise  

    def __call__(self, input, state, key=None):
        state = self.init_state()
        # TODO: flatten leaves before concatenating `tree_map(ravel, leaves)`
        input = jnp.concatenate(jax.tree_leaves(input))
        state = self.cell(input, state)
        state = self._add_noise(state, key)
        output = self.out_nonlinearity(self.linear(state) + self.bias)
        
        return output, state
    
    @cached_property
    def _add_noise(self):
        if self.noise_std is not None:
            return self.__add_noise
        else:
            return lambda state, _: state
    
    def __add_noise(self, state, key):
        noise = self.noise_std * jrandom.normal(key, state.shape) 
        return state + noise
    
    def init_state(self, state=None):
        if state is None:
            return jnp.zeros(self.hidden_size)
        else:
            return jnp.array(state)