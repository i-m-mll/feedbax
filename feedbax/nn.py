"""Neural network architectures.

TODO:

- Rename vague `activity` to `hidden`, `cell`, or `rnn`.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
import logging
import math
from typing import (
    Optional, 
    Protocol, 
    Type, 
    Union,
    runtime_checkable,
)

import equinox as eqx
from equinox import field
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.intervene import AbstractIntervenor
from feedbax.model import wrap_stateless_callable
from feedbax.misc import (
    identity_func, 
    interleave_unequal, 
    n_positional_args,  
)
from feedbax._staged import AbstractStagedModel, ModelStage
from feedbax.state import AbstractState, StateT


logger = logging.getLogger(__name__)


@runtime_checkable
class RNNCellProto(Protocol):
    """Specifies the interface expected from RNN cell instances.
    
    Based on `eqx.nn.GRUCell` and `eqx.nn.LSTMCell`.
    
    !!! dev-note "Development note"
        Neither mypy nor typeguard currently complain if the `Type[RNNCell]` 
        argument to `SimpleStagedNetwork` doesn't satisfy this protocol. I'm 
        not sure if this is because protocols aren't compatible with `Type`, 
        though no errors are raised to suggest that's so.
        
        I'm leaving this in here because it seems harmless, and for now it 
        functions as documentation for the interface expected from an
        RNN cell. 
    """
    hidden_size: int
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        use_bias: bool, 
        *, 
        key: PRNGKeyArray,
        **kwargs, 
    ):
        ...
    
    def __call__(
        self, 
        input: Float[Array, "channel"], 
        state: Float[Array, "unit"],
    ) -> Float[Array, "unit"]:
        ...


def orthogonal_gru_cell(input_size, hidden_size, use_bias=True, scale=1.0, *, key):
    """Returns an `eqx.nn.GRUCell` with orthogonal weight matrix initialization.
    """
    net = eqx.nn.GRUCell(input_size, hidden_size, use_bias=use_bias, key=key)
    initializer = jax.nn.initializers.orthogonal(scale=scale, column_axis=-1) 
    ortho_weight_hh = jnp.concatenate(
        [initializer(k, (hidden_size, hidden_size)) for k in jr.split(key, 3)], 
        axis=0,
    )
    net = eqx.tree_at(
        lambda net: net.weight_hh,
        net,
        ortho_weight_hh,
    )
    return net


class NetworkState(AbstractState):
    """Type of state PyTree operated on by [`SimpleStagedNetwork`][feedbax.nn.SimpleStagedNetwork] instances.
    
    Attributes:
        hidden: The (output) activity of the hidden layer units.
        output: The activity of the readout layer, if the network has one.
        encoding: The activity of the encoding layer, if the network has one.
    """
    hidden: PyTree[Float[Array, "unit"]]
    output: Optional[PyTree[Array]] = None
    encoding: Optional[PyTree[Array]] = None 


class SimpleStagedNetwork(AbstractStagedModel[NetworkState]):
    """A single step of a neural network layer, with optional encoder and readout layers.
    
    Attributes:
        hidden_size: The number of units in the hidden layer.
        out_size: The number of readout units, if the network has a readout layer. Otherwise
            this is equal to `hidden_size`.
        encoding_size: The number of encoder units, if the network has an encoder layer.
        hidden: The module implementing the hidden layer.
        hidden_nonlinearity: The nonlinearity applied to the hidden layer output.
        encoder: The module implementing the encoder layer, if present.
        readout: The module implementing the readout layer, if present.
    """
    out_size: int 
    hidden: eqx.Module
    hidden_size: int
    hidden_noise_std: Optional[float]
    hidden_nonlinearity: Optional[Callable[[Float], Float]] = None
    encoder: Optional[eqx.Module] = None
    encoding_size: Optional[int] = None
    readout: Optional[eqx.Module] = None
    out_nonlinearity: Optional[Callable[[Float], Float]] = None 
    
    intervenors: Mapping[str, AbstractIntervenor]

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
        out_nonlinearity: Callable[[Float], Float] = identity_func,
        hidden_noise_std: Optional[float] = None,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Mapping[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: PRNGKeyArray, 
    ):
        """
        !!! Note
            If an integer is passed for `encoding_size`, input encoding is enabled.
            Otherwise network inputs are passed directly to the hidden layer.
                
            If an integer is passed for `out_size`, readout is enabled. Otherwise
            the network's outputs are the outputs of the hidden units.
            
            In principle `hidden_type` can be class defining a multi-layer network, 
            as long as it is instantiated as `hidden_type(input_size, hidden_size, use_bias, *,
            key)`.
            
            Use `partial` to set `use_bias` for the encoder or readout types, before 
            passing them to this constructor.
        
        Arguments:
            input_size: The number of input channels in the network.
                If `encoder_type` is not `None`, this is the number of inputs 
                to the encoder layer—otherwise, the hidden layer.
            hidden_size: The number of units in the hidden layer.
            out_size: The number of readout units. If `None`, do not add a readout layer.
            encoding_size: The number of encoder units. If `None`, do not add an encoder layer.
            hidden_type: The type of hidden layer to use. 
            encoder_type: The type of encoder layer to use.
            use_bias: Whether the hidden layer should have a bias term.
            hidden_nonlinearity: A function to apply unitwise to the hidden layer output. This is 
                typically not used if `hidden_type` is `GRUCell` or `LSTMCell`.
            out_nonlinearity: A function to apply unitwise to the readout layer output.
            hidden_noise_std: Standard deviation of Gaussian noise to add to the hidden layer output.
            intervenors: [Intervenors][feedbax.intervene.AbstractIntervenor] to add 
                to the model at construction time.
            key: Random key for initialising the network.
        """
        key1, key2, key3 = jr.split(key, 3)

        if encoding_size is not None:
            self.encoder = encoder_type(input_size, encoding_size, key=key2)
            self.encoding_size = encoding_size
            self.hidden = hidden_type(encoding_size, hidden_size, use_bias=use_bias, key=key1)
        else:
            self.hidden = hidden_type(input_size, hidden_size, use_bias=use_bias, key=key1)
        
        self.hidden_size = hidden_size
        self.hidden_nonlinearity = hidden_nonlinearity
        self.hidden_noise_std = hidden_noise_std 
        
        if out_size is not None:
            readout = readout_type(hidden_size, out_size, key=key3)
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
    
    # def _output(self, hidden, state, *, key):
    #     return self.out_nonlinearity(self.readout(hidden))
        
    # def _encode(self, input, state, *, key):
    #     return self.encoder(input)
    
    def _add_hidden_noise(self, input, state, *, key):
        if self.hidden_noise_std is None:
            return state
        return state + self.hidden_noise_std * jr.normal(key, state.shape)      
    
    @property
    def model_spec(self) -> OrderedDict[str, ModelStage]:
        """Specifies the network model stages: layers, nonlinearities, and noise.
        
        Only includes stages for the encoding layer, readout layer, hidden noise, and
        hidden nonlinearity, if the user respectively requests them at the time of 
        construction.
        
        !!! NOTE
            Inspects the instantiated hidden layer to determine if it is a stateful
            network (e.g. an RNN). If not (e.g. Linear), it wraps the layer so that
            it plays well with the state-passing of `AbstractStagedModel`. This assumes 
            that stateful layers will take 2 positional arguments, and stateless layers
            only 1.
        """
        
        if n_positional_args(self.hidden) == 1:
            hidden_module = lambda self: wrap_stateless_callable(self.hidden)
            if isinstance(self.hidden, eqx.nn.Linear):
                logger.warning("Network hidden layer is linear but no hidden "
                                "nonlinearity is defined")
        else:
            # #TODO: revert this!
            # def tmp(self):
            #     def wrapper(input, state, *, key):
            #         return self.hidden(input, jnp.zeros_like(state))
            #     return wrapper
            # hidden_module = lambda self: tmp(self)
            hidden_module = lambda self: self.hidden 
            
        if self.encoder is None:
            spec = OrderedDict({
                'hidden': ModelStage(
                    callable=hidden_module,
                    where_input=lambda input, _: ravel_pytree(input)[0],
                    where_state=lambda state: state.hidden,
                ),
            })
        else:
            spec = OrderedDict({
                'encoder': ModelStage(
                    callable=lambda self: \
                        lambda input, state, *, key: self.encoder(input),
                    where_input=lambda input, _: ravel_pytree(input)[0],
                    where_state=lambda state: state.encoding,
                ),
                'hidden': ModelStage(
                    callable=hidden_module,
                    where_input=lambda input, state: state.encoding,
                    where_state=lambda state: state.hidden,
                ),
            }) 
        
        if self.hidden_nonlinearity is not None:
            spec |= {
                'hidden_nonlinearity': ModelStage(
                    callable=lambda self: wrap_stateless_callable(
                        self.hidden_nonlinearity, pass_key=False
                    ),
                    where_input=lambda input, state: state.hidden,
                    where_state=lambda state: state.hidden,
                ),
            }
        
        if self.hidden_noise_std is not None:
            spec |= {
                'hidden_noise': ModelStage(
                    callable=lambda self: self._add_hidden_noise,
                    where_input=lambda input, state: state.hidden,
                    where_state=lambda state: state.hidden,
                ),
            }
        
        if self.readout is not None:
            spec |= {
                'readout': ModelStage(
                    callable=lambda self: wrap_stateless_callable(self.readout),
                    where_input=lambda input, state: state.hidden,
                    where_state=lambda state: state.output,
                ),
                'out_nonlinearity': ModelStage(
                    callable=lambda self: wrap_stateless_callable(
                        self.out_nonlinearity, pass_key=False
                    ),
                    where_input=lambda input, state: state.output,
                    where_state=lambda state: state.output,
                ),
            }
        
        return spec
        
    @property
    def memory_spec(self):
        return NetworkState(
            hidden=True,
            output=True,
            encoding=True,
        )        
    
    def init(self, *, key: Optional[PRNGKeyArray] = None):
        if self.out_size is None:
            output = None
        else:
            output = jnp.zeros(self.out_size)
            
        if self.encoding_size is None:
            encoding = None
        else:
            encoding = jnp.zeros(self.encoding_size)
        
        # TODO: Try eval_shape
        return NetworkState(
            hidden=jnp.zeros(self.hidden_size), 
            output=output, 
            encoding=encoding,
        )


# TODO: Convert this to AbstractStagedModule
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
    weight_hh: Array
    weight_ih: Array
    bias: Optional[Array]
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
        key: PRNGKeyArray,
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
        input: Array, 
        state: Array,
        key: PRNGKeyArray
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