"""Neural network architectures."""

from itertools import zip_longest

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from feedbax.utils import interleave_unequal


class SimpleNet(eqx.Module):
    """A simple feedforward neural network."""
    layers: list 
    
    def __init__(self, sizes, use_bias=(), nonlinearity=jnp.tanh, output_nonlinearity=None, key=jrandom.PRNGKey(0)):
        keys = jrandom.split(key, len(sizes) - 1)
        
        if bool(use_bias) is use_bias:
            use_bias = (use_bias,) * (len(sizes) - 1)
            
        layers = [eqx.nn.Linear(m, n, key=k, use_bias=b) 
                  for m, n, k, b in zip(sizes[:-1], sizes[1:], keys, use_bias)]
        
        nonlinearities = [nonlinearity] * (len(sizes) - 2) 
        if output_nonlinearity is not None:
            nonlinearities += [output_nonlinearity]
            
        self.layers = list(interleave_unequal(layers, nonlinearities))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x