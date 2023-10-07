"""Neural network architectures.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
from itertools import zip_longest
import logging
import math
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from feedbax.utils import interleave_unequal


logger = logging.getLogger(__name__)


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


class RNNCell(eqx.Module):
    """
    
    Based on `eqx.nn.GRUCell` and the leaky RNN from
    
        [1] G. R. Yang, M. R. Joglekar, H. F. Song, W. T. Newsome, 
            and X.-J. Wang, “Task representations in neural networks trained 
            to perform many cognitive tasks,” Nat Neurosci, vol. 22, no. 2, 
            pp. 297–306, Feb. 2019, doi: 10.1038/s41593-018-0310-2.

    TODO: 
    - If `diffrax` varies `dt`, then `dt` could be passed to update `alpha`.
    """
    weight_hh: jax.Array
    weight_ih: jax.Array
    bias: Optional[jax.Array]
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    use_noise: bool = eqx.field(static=True)
    noise_strength: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        use_bias: bool = True,
        use_noise: bool = False,
        noise_strength: float = 0.01,
        dt: float = 1,
        tau: float = 1,
        *,  # this forces the user to pass the following as keyword arguments
        key: jrandom.PRNGKeyArray,
        **kwargs
    ):
        ihkey, hhkey, bkey = jrandom.split(key, 3)
        lim = math.sqrt(1 / hidden_size)
        
        self.weight_ih = jrandom.uniform(
            ihkey, (hidden_size, input_size), minval=-lim, maxval=lim,
        )
        self.weight_hh = jrandom.uniform(
            hhkey, (hidden_size, hidden_size), minval=-lim, maxval=lim,
        )
        
        if use_bias:
            self.bias = jrandom.uniform(
                bkey, (hidden_size,), minval=-lim, maxval=lim,
            )
        else:
            self.bias = None  
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.use_noise = use_noise
        self.noise_strength = noise_strength
        self.dt = dt
        self.tau = tau
        
    def __call__(self, input: jax.Array, state: jax.Array, *, key=None):
        """Vanilla RNN cell."""
        if self.use_bias:
            bias = self.bias
        else:
            bias = 0
            
        if self.use_noise:
            noise = self.noise_std * jrandom.normal(key, state.shape) 
        else:
            noise = 0
                
        state = (1 - self.alpha) * state + self.alpha * jnp.tanh(
            jnp.dot(self.weight_ih, input) 
            + jnp.dot(self.weight_hh, state)
            + bias 
            + noise 
        )
        
        return state  #! 0D PyTree
    
    @cached_property
    def alpha(self):
        return self.dt / self.tau
    
    @cached_property
    def noise_std(self, noise_strength):
        if self.use_noise:
            return math.sqrt(2 / self.alpha) * noise_strength
        else:
            return None
    

class RNN(eqx.Module):
    """From https://docs.kidger.site/equinox/examples/train_rnn/"""
    out_size: int = eqx.field(static=True)
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array
    out_nonlinearity: Callable 
    noise_std: Optional[float]

    def __init__(
        self, 
        cell: eqx.Module,
        out_size: int, 
        out_nonlinearity=lambda x: x,
        noise_std=None,
        *,
        key: jrandom.PRNGKeyArray, 
    ):
        self.out_size = out_size
        self.cell = cell
        self.linear = eqx.nn.Linear(cell.hidden_size, out_size, use_bias=False, key=key)
        self.bias = jnp.zeros(out_size)
        self.out_nonlinearity = out_nonlinearity       
        self.noise_std = noise_std
        
        # initialize cached properties
        self._add_noise  

    def __call__(self, input, state, key=None):
        # state = self.init()
        # TODO: flatten leaves before concatenating `tree_map(ravel, leaves)`
        input = jnp.concatenate(jax.tree_leaves(input))
        state = self.cell(input, state)
        state = self._add_noise(state, key)
        output = self.out_nonlinearity(self.linear(state) + self.bias)
        
        return output, state
    
    @cached_property
    def _add_noise(self):
        #? this might be overkill; equinox uses simple conditionals in `__call__` for similar
        # TODO: timeit the difference
        if self.noise_std is not None:
            return self.__add_noise
        else:
            return lambda state, _: state
    
    def __add_noise(self, state, key):
        noise = self.noise_std * jrandom.normal(key, state.shape) 
        return state + noise
    
    def init(self, state=None):
        if state is None:
            return jnp.zeros(self.cell.hidden_size)
        else:
            return jnp.array(state)