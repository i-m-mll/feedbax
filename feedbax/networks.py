"""Neural network architectures.

TODO:

- Rename vague `activity` to `hidden`, `cell`, or `rnn`.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections import OrderedDict
from functools import cached_property
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
from feedbax.model import AbstractModel, AbstractModelState, wrap_stateless_module
from feedbax.utils import interleave_unequal, n_positional_args  

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
    """State of a neural network.
    
    TODO:
    - Rename to `RNNCellState`.
    """
    hidden: PyTree[Float[Array, "unit"]]
    output: Optional[PyTree]
    encoding: Optional[PyTree]


class SimpleNetwork(AbstractModel[NetworkState]):
    """A single step of a noisy network with optional encoding and readout layers.
    
    Ultimately derived from https://docs.kidger.site/equinox/examples/train_rnn/
    
    TODO:
    - The user should use `partial` if they want to change `use_bias` for the 
      encoder or readout.
    """
    out_size: int 
    hidden: eqx.Module
    hidden_size: int
    noise_std: Optional[float]
    hidden_nonlinearity: Optional[Callable[[Float], Float]] = None
    encoder: Optional[eqx.Module] = None
    encoding_size: Optional[int] = None
    readout: Optional[eqx.Module] = None
    out_nonlinearity: Optional[Callable[[Float], Float]] = None 
    
    intervenors: Dict[str, AbstractIntervenor]

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        out_size: Optional[int] = None, 
        encoding_size: Optional[int] = None,
        hidden_type: Type[RNNCellProto] = eqx.nn.GRUCell,
        encoder_type: Type[eqx.Module] = eqx.nn.Linear,
        readout_type: Type[eqx.Module] = eqx.nn.Linear,
        use_bias: bool = True,
        hidden_nonlinearity: Callable[[Float], Float] = None,
        out_nonlinearity: Callable[[Float], Float] = lambda x: x,
        noise_std: Optional[float] = None,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: jax.Array, 
    ):
        """
        If an integer is passed for `encoding_size`, input encoding is enabled.
        Otherwise network inputs are passed directly to the hidden layer.
            
        If an integer is passed for `out_size`, readout is enabled. Otherwise
        the network's outputs are the outputs of the "hidden" units.
        
        In principle `hidden_type` can be a multi-layer network class, but must
        be instantiated as `hidden_type(input_size, hidden_size, use_bias, *,
        key)`, and the called instance should only return its output values, as
        usual.
        """
        key1, key2, key3 = jr.split(key, 3)

        if encoding_size is not None:
            self.encoder = encoder_type(input_size, encoding_size, use_bias=True, key=key2)
            self.encoding_size = encoding_size
            self.hidden = hidden_type(encoding_size, hidden_size, use_bias=use_bias, key=key1)
        else:
            self.hidden = hidden_type(input_size, hidden_size, use_bias=use_bias, key=key1)
        
        self.hidden_size = hidden_size
        self.hidden_nonlinearity = hidden_nonlinearity
        self.noise_std = noise_std 
        
        if out_size is not None:
            readout = readout_type(hidden_size, out_size, use_bias=True, key=key3)
            self.readout = eqx.tree_at(
                lambda layer: layer.bias,
                readout,
                jnp.zeros_like(readout.bias),
            )
            self.out_nonlinearity = out_nonlinearity  
            self.out_size = out_size
        else:
            self.out_size = hidden_size

        self.intervenors = self._get_intervenors_dict(intervenors)
    
    def _output(self, hidden, state, *, key):
        return self.out_nonlinearity(self.readout(hidden))
        
    def _encode(self, input, state, *, key):
        return self.encoder(input)
    
    def _add_state_noise(self, input, state, *, key):
        if self.noise_std is None:
            return state
        return state + self.noise_std * jr.normal(key, state.shape)      
    
    @cached_property
    def model_spec(self):
        """
        Inspects the instantiated hidden layer to determine if it is a stateful
        network (e.g. an RNN). If not (e.g. Linear), it wraps the layer so that
        it plays well with the state-passing of `AbstractModel`. Assumes that 
        stateful layers will take 2 positional arguments, and stateless layers
        only 1.
        """
        
        if n_positional_args(self.hidden) == 1:
            hidden_module = lambda self: wrap_stateless_module(self.hidden)
            if isinstance(self.hidden, eqx.nn.Linear):
                logger.warning("Network hidden layer is linear but no hidden "
                                "nonlinearity is defined")
        else:
            hidden_module = lambda self: self.hidden
        
        if self.encoder is None:
            spec = OrderedDict({
                'hidden': (
                    hidden_module,
                    lambda input, _: ravel_pytree(input)[0],
                    lambda state: state.hidden,
                ),
            })
        else:
            spec = OrderedDict({
                'encoder': (
                    lambda self: self._encode,
                    lambda input, _: ravel_pytree(input)[0],
                    lambda state: state.encoding,
                ),
                'hidden': (
                    hidden_module,
                    lambda input, state: state.encoding,
                    lambda state: state.hidden,
                ),
            }) 
        
        if self.hidden_nonlinearity is not None:
            spec |= {
                'hidden_nonlinearity': (
                    lambda self: lambda input, state, key: \
                        self.hidden_nonlinearity(state),
                    lambda input, state: None,
                    lambda state: state.hidden,
                ),
            }
        
        if self.noise_std is not None:
            spec |= {
                'hidden_noise': (
                    lambda self: self._add_state_noise,
                    lambda _, state: state.hidden,
                    lambda state: state.hidden,
                ),
            }
        
        if self.readout is not None:
            spec |= {
                'readout': (
                    lambda self: self._output,
                    lambda _, state: state.hidden,
                    lambda state: state.output,
                )
            }
        
        return spec
        
    @property
    def memory_spec(self):
        return NetworkState(
            hidden=True,
            output=True,
            encoding=True,
        )        
    
    def init(self, *, key: Optional[jax.Array] = None):
        return NetworkState(
            hidden=jnp.zeros(self.hidden_size), 
            output=jnp.zeros(self.out_size), 
            encoding=jnp.zeros(self.encoding_size),
        )




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