"""Neural network architectures.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections import OrderedDict
from functools import cached_property, wraps
import logging
import math
from typing import (
    TYPE_CHECKING,
    Callable, 
    Dict, 
    Optional, 
    Protocol, 
    Sequence, 
    Tuple, 
    Type, 
    TypeVar,
    Union,
    runtime_checkable,
)

import equinox as eqx
from equinox import field
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PyTree

from feedbax.intervene import AbstractIntervenor
from feedbax.model import AbstractModel, AbstractModelState
from feedbax.utils import interleave_unequal  

StateT = TypeVar("StateT", bound=AbstractModelState)
    

logger = logging.getLogger(__name__)


@runtime_checkable
class RNNCellProto(Protocol):
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


class NetworkState(AbstractModelState):
    """State of a neural network."""
    activity: PyTree[Float[Array, "unit"]]
    output: Optional[PyTree]


class RNNCell(AbstractModel[NetworkState]):
    """A single step of an RNN with a linear readout layer and noise.
    
    Derived from https://docs.kidger.site/equinox/examples/train_rnn/"""
    cell: eqx.Module
    noise_std: Optional[float]
    hidden_size: int
    
    intervenors: Dict[str, AbstractIntervenor] 

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        cell_type: Type[RNNCellProto] = eqx.nn.GRUCell,
        use_bias: bool = True,
        noise_std: Optional[float] = None,
        *,
        key: jax.Array, 
    ):
        key1, key2 = jr.split(key, 2)
        self.cell = cell_type(input_size, hidden_size, use_bias=use_bias, key=key1)  
        self.noise_std = noise_std
        self.hidden_size = self.cell.hidden_size
    
    # @jax.named_scope("fbx.RNNCell")
    # def __call__(
    #     self, 
    #     input, 
    #     state: NetworkState, 
    #     key: jax.Array,
    # ) -> NetworkState:
    #     if not isinstance(input, jnp.ndarray):
    #         input, _ = ravel_pytree(input)
    #     activity = self.cell(input, state.activity)
    #     if self.noise_std is not None:
    #         noise = self.noise_std * jr.normal(key, activity.shape) 
    #         activity = activity + noise
        
    #     return NetworkState(activity, None)
    
    @property
    def model_spec(self):
        return OrderedDict({
            'cell': (
                lambda self: self.cell,
                lambda state: state.activity,
                lambda input, _: ravel_pytree(input)[0],
            ),
            'noise': (
                lambda self: self._add_state_noise,
                lambda state: state.activity,
                lambda _, state: state.activity,
            ),
            # 'readout': (
            #     lambda self: self._output,
            #     lambda state: state.output,
            #     lambda _, state: state.activity,
            # )
        })
        
    @property
    def memory_spec(self):
        return NetworkState(
            activity=True,
            output=True,
        )        
    
    def init(self, activity=None, output=None):
        if activity is None:
            activity = jnp.zeros(self.cell.hidden_size)
        else:
            activity = jnp.array(activity)
        
        return NetworkState(activity, output=output)    


class RNNCellWithReadout(AbstractModel[NetworkState]):
    """A single step of an RNN with a linear readout layer and noise.
    
    Derived from https://docs.kidger.site/equinox/examples/train_rnn/"""
    out_size: int 
    cell: eqx.Module
    readout: eqx.Module
    out_nonlinearity: Callable[[Float], Float]
    noise_std: Optional[float]
    hidden_size: int
    intervenors: Dict[str, AbstractIntervenor]

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        out_size: int, 
        cell_type: Type[RNNCellProto] = eqx.nn.GRUCell,
        readout_type: Type[eqx.Module] = eqx.nn.Linear,
        use_bias: bool = True,
        out_nonlinearity: Callable[[Float], Float] = lambda x: x,
        noise_std: Optional[float] = None,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: jax.Array, 
    ):
        key1, key2 = jr.split(key, 2)
        self.out_size = out_size
        self.cell = cell_type(input_size, hidden_size, use_bias=use_bias, key=key1)
        self.readout = readout_type(hidden_size, out_size, use_bias=True, key=key2)
        self.out_nonlinearity = out_nonlinearity       
        self.noise_std = noise_std
        self.hidden_size = self.cell.hidden_size
        self.intervenors = self._get_intervenors_dict(intervenors)
    
    def _output(self, activity, state, *, key):
        return self.out_nonlinearity(self.readout(activity))
    
    def _add_state_noise(self, input, state, *, key):
        if self.noise_std is None:
            return state
        return state + self.noise_std * jr.normal(key, state.shape)      
    
    @property
    def model_spec(self):
        return OrderedDict({
            'cell': (
                lambda self: self.cell,
                lambda input, _: ravel_pytree(input)[0],
                lambda state: state.activity,
            ),
            'noise': (
                lambda self: self._add_state_noise,
                lambda _, state: state.activity,
                lambda state: state.activity,
            ),
            'readout': (
                lambda self: self._output,
                lambda _, state: state.activity,
                lambda state: state.output,
            )
        })
        
    @property
    def memory_spec(self):
        return NetworkState(
            activity=True,
            output=True,
        )        
    
    def init(self, activity=None, output=None):
        if activity is None:
            activity = jnp.zeros(self.cell.hidden_size)
        else:
            activity = jnp.array(activity)
            
        if output is None:
            output = jnp.zeros(self.out_size)
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
    hidden_size: int
    
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
        self.hidden_size = self.rnn_cell_with_readout.hidden_size
        
        # expose some attributes of `RNNCellWithReadout`
        # to maintain references in the model PyTree
        self.cell = self.rnn_cell_with_readout.cell
        self.readout = self.rnn_cell_with_readout.readout
        
        
    @jax.named_scope("fbx.RNNCellWithReadoutAndInput")
    def __call__(
        self, 
        input, 
        state: NetworkState, 
        key: jax.Array,
    ) -> NetworkState:
        if not isinstance(input, jnp.ndarray):
            input, _ = ravel_pytree(input)
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
         

def wrap_stateless_network(net: eqx.Module):
    """Make a stateless network trivially compatible with state-passing.
    
    TODO: should be `wrap_stateless_module` probably
    """
    @wraps(net)
    def wrapped(input, state, *args, **kwargs):
        return net(input, *args, **kwargs)
    
    return wrapped


def n_layer_linear(
    hidden_sizes: Sequence[int], 
    input_size: int, 
    out_size: int, 
    use_bias: bool = True, 
    nonlinearity: Callable[[Float], Float] = jnp.tanh,
    *, 
    key
):
    """A simple n-layer linear network with nonlinearity."""
    keys = jr.split(key, len(hidden_sizes) + 1)
    sizes = (input_size,) + tuple(hidden_sizes) + (out_size,)
    layers = [
        eqx.nn.Linear(size0, size1, use_bias=use_bias, key=keys[i])
        for i, (size0, size1) in enumerate(
            zip(sizes[:-1], sizes[1:])
        )
    ]
    return eqx.nn.Sequential(
        interleave_unequal(layers, [nonlinearity] * len(hidden_sizes))
    )


def two_layer_linear(
    hidden_size, 
    input_size, 
    out_size, 
    use_bias=True, 
    nonlinearity=jnp.tanh,
    *, 
    key
):
    """A two-layer linear network with nonlinearity.
    
    Just a convenience over `n_layer` since two-layer readouts may be more 
    common than other n-layer readouts for RNNs.
    """
    return n_layer_linear(
        (hidden_size,), 
        input_size,
        out_size,
        use_bias=use_bias,
        nonlinearity=nonlinearity,
        key=key,
    )