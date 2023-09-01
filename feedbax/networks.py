"""Neural network architectures."""

from itertools import zip_longest
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from feedbax.utils import interleave_unequal


class SimpleMultiLayerNet(eqx.Module):
    """A series of layers of the same type with nonlinearities of the same type."""
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
            
        layers = [eqx.nn.Linear(m, n, key=k, use_bias=b) 
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
    