"""Neural network architectures.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
import logging
import math
from typing import Callable, Optional, Protocol, Tuple, Type, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PyTree

from feedbax.utils import interleave_unequal


logger = logging.getLogger(__name__)


class NetworkState(eqx.Module):
    """State of a neural network."""
    activity: PyTree[Float[Array, "unit"]]
    output: PyTree
    

@runtime_checkable
class RNNCell(Protocol):
    """Specifies the interface expected from RNN cell instances.
    
    Based on `eqx.nn.GRUCell` and `eqx.nn.LSTMCell`.
    
    Neither mypy nor typeguard currently complain if the `Type[RNNCell]` 
    argument to `RNNCellWithReadout` doesn't satisfy this protocol. I'm 
    not sure if this is because protocols aren't compatible with `Type`, 
    though no errors are raised to suggest that is so.
    
    I'm leaving this in here because it is currently harmless, and it at 
    least functions as documentation for the interface expected from an
    RNN cell. 
    
    Alternatively, we could maybe use a protocol `RNNCellType` that 
    defines `__call__` for `__init__`, whose return satisfies another protocol 
    `RNNCell` that specifies the fields expected in the returned class.
    """
    hidden_size: int
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        use_bias: bool, 
        *, 
        key: jax.Array,
        **kwargs, 
    ):
        ...
    
    def __call__(
        self, 
        input: jax.Array, 
        state: jax.Array, 
    ) -> jax.Array:
        ...
        

class RNNCellWithReadout(eqx.Module):
    """A single step of an RNN with a linear readout layer and noise.
    
    Derived from https://docs.kidger.site/equinox/examples/train_rnn/"""
    out_size: int 
    cell: eqx.Module
    readout: eqx.Module
    out_nonlinearity: Callable[[Float], Float]
    noise_std: Optional[float]
    persistence: bool

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        out_size: int, 
        cell_type: Type[RNNCell] = eqx.nn.GRUCell,
        readout_type: Type[eqx.Module] = eqx.nn.Linear,
        use_bias: bool = True,
        out_nonlinearity: Callable[[Float], Float] = lambda x: x,
        noise_std: Optional[float] = None,
        persistence: bool = True,
        *,
        key: jax.Array, 
    ):
        key1, key2 = jr.split(key, 2)
        self.out_size = out_size
        self.cell = cell_type(input_size, hidden_size, use_bias=use_bias, key=key1)
        self.readout = readout_type(hidden_size, out_size, use_bias=True, key=key2)
        self.out_nonlinearity = out_nonlinearity       
        self.noise_std = noise_std
        self.persistence = persistence
    
    @jax.named_scope("fbx.RNNCellWithReadout")
    def __call__(
        self, 
        input, 
        state: NetworkState, 
        key: jax.Array,
    ) -> NetworkState:
        if not self.persistence:
            state = self.init()
        if not isinstance(input, jnp.ndarray):
            # TODO: flatten each leaf before concatenating 
            input = jnp.concatenate(jax.tree_leaves(input))
        activity = self.cell(input, state.activity)
        if self.noise_std is not None:
            #! this will only affect recurrent computations if `persistence` is `True`
            noise = self.noise_std * jr.normal(key, activity.shape) 
            activity = activity + noise
        
        return NetworkState(activity, self._output(activity))
    
    def _output(self, activity):
        return self.out_nonlinearity(self.readout(activity))
    
    def init(self, activity=None, output=None):
        if activity is None:
            activity = jnp.zeros(self.cell.hidden_size)
        else:
            activity = jnp.array(activity)
            
        if output is None:
            output = self._output(activity)
        else:
            output = jnp.array(activity)
        
        return NetworkState(activity, output)


class RNNCellWithReadoutAndInput(eqx.Module):
    """Adds a linear input layer to `RNNCellWithReadout`.
    
    Since the linear layer is "stateless", we use the same `init` method 
    and the same `__call__` signature as `RNNCellWithReadout`, so instances 
    of the two classes are interchangeable as model components.
    """
    out_size: int 
    input_layer: eqx.Module
    rnn_cell_with_readout: RNNCellWithReadout
    cell: eqx.Module
    readout: eqx.Module
    
    def __init__(
        self, 
        input_size: int, 
        rnn_input_size: int,
        hidden_size: int,
        out_size: int, 
        cell_type: Type[RNNCell] = eqx.nn.GRUCell,
        *,
        key: jax.Array, 
        **kwargs
    ):
        key1, key2 = jr.split(key, 2)
        self.input_layer = eqx.nn.Linear(input_size, rnn_input_size, key=key1)
        self.rnn_cell_with_readout = RNNCellWithReadout(
            rnn_input_size,
            hidden_size,
            out_size,
            cell_type=cell_type, 
            key=key2,
            **kwargs,
        )
        self.out_size = out_size
        
        # expose some attributes of `RNNCellWithReadout`
        # to maintain references in the model PyTree
        self.cell = self.rnn_cell_with_readout.cell
        self.readout = self.rnn_cell_with_readout.readout
        
        
    @jax.named_scope("fbx.RNNCellWithReadout")
    def __call__(
        self, 
        input, 
        state: NetworkState, 
        key: jax.Array,
    ) -> NetworkState:
        # TODO: flatten each leaf before concatenating `tree_map(ravel, leaves)`
        if not isinstance(input, jnp.ndarray):
            input = jnp.concatenate(jax.tree_leaves(input))
        input = jnp.concatenate(jax.tree_leaves(input))
        rnn_input = self.input_layer(input)
        output = self.rnn_cell_with_readout(rnn_input, state, key)
        return output

    def init(self, activity=None, output=None):
        return self.rnn_cell_with_readout.init(activity, output)


class LeakyRNNCell(eqx.Module):
    """Custom `RNNCell` with persistent, leaky state.
    
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
    input_size: int 
    hidden_size: int 
    use_bias: bool 
    use_noise: bool 
    noise_strength: float 
    dt: float 
    tau: float 
    
    @jax.named_scope("fbx.RNNCell")
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        use_bias: bool = True,
        use_noise: bool = False,
        noise_strength: float = 0.01,
        dt: float = 1.,
        tau: float = 1.,
        *,  # this forces the user to pass the following as keyword arguments
        key: jax.Array,
        **kwargs
    ):
        ihkey, hhkey, bkey = jr.split(key, 3)
        lim = math.sqrt(1 / hidden_size)
        
        self.weight_ih = jr.uniform(
            ihkey, (hidden_size, input_size), minval=-lim, maxval=lim,
        )
        self.weight_hh = jr.uniform(
            hhkey, (hidden_size, hidden_size), minval=-lim, maxval=lim,
        )
        
        if use_bias:
            self.bias = jr.uniform(
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
        
    def __call__(
        self, 
        input: jax.Array, 
        state: jax.Array,
        key: jax.Array
    ):
        """Vanilla RNN cell."""
        if self.use_bias:
            bias = self.bias
        else:
            bias = 0
            
        if self.use_noise:
            noise = self.noise_std * jr.normal(key, state.shape) 
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
        nonlinearity: Callable[[Float], Float] = jnp.tanh, 
        output_nonlinearity: Optional[Callable[[Float], Float]] = None, 
        linear_final_layer: bool = False,  # replace the final layer with a linear layer
        *,
        key: jax.Array,
    ):
        keys = jr.split(key, len(sizes) - 1)
        
        if bool(use_bias) is use_bias:
            use_bias = (use_bias,) * (len(sizes) - 1)
            
        layers = [layer_type(m, n, key=key, use_bias=b) 
                  for m, n, key, b in zip(sizes[:-1], sizes[1:], keys, use_bias)]
        
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