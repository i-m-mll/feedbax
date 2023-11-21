"""Modules that compose other modules into control contexts.

For example, classes defined here could be used to to model a sensorimotor 
loop, a body, or (perhaps) multiple interacting bodies. 

TODO:
- Maybe this should be renamed... mainly because it might be confused with 
`AbstractTask` in terms of its purpose.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
import logging
from typing import Callable, Generic, Sequence, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree
import numpy as np

from feedbax.channel import Channel, ChannelState
from feedbax.intervene import AbstractIntervenor
from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.networks import NetworkState 
from feedbax.task import AbstractTask
from feedbax.types import CartesianState2D
from feedbax.utils import tree_sum_n_features


logger = logging.getLogger(__name__)


N_DIM = 2


class AbstractModelState(eqx.Module):
    ...
    

StateT = TypeVar("StateT", bound=AbstractModelState)


class AbstractModel(eqx.Module, Generic[StateT]):
    """
    TODO:
    - It might make sense to implement `__call__` here, and then get subclasses
      to implement another method (`step`?). 
      - For example, this would allow all the `intervenors` to be called at the 
        start of each `__call__`, regardless of what type of model we're 
        dealing with; we'd need an `intervenors: AbstractVar` field in that 
        case. However, I'm not sure yet that this makes sense.
    """
        
    @abstractmethod
    def __call__(
        self, 
        input, 
        state: StateT, 
        key: jax.Array,
    ) -> StateT:
        ...
        
    # @abstractmethod
    # def init(
    #     self,
    #     state, 
    # ) -> State:
    #     ...
        
    @abstractproperty
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered by callers."""
        ...


class SimpleFeedbackState(AbstractModelState):
    mechanics: MechanicsState
    network: NetworkState
    feedback: ChannelState


class SimpleFeedback(AbstractModel[SimpleFeedbackState]):
    """Simple feedback loop with a single RNN and single mechanical system.
    
    TODO:
    - PyTree of force inputs (in addition to control inputs) to mechanics
    """
    net: eqx.Module  
    mechanics: Mechanics 
    feedback_channel: Channel
    delay: int 
    feedback_leaves_func: Callable[[PyTree], PyTree]  
    # TODO: could do AbstractIntervenor[SimpleFeedbackState] but maybe that's too restrictive
    intervenors: Sequence[AbstractIntervenor] 
    
    def __init__(
        self, 
        net: eqx.Module, 
        mechanics: Mechanics, 
        delay: int = 0, 
        feedback_leaves_func: Callable[[MechanicsState], PyTree] = \
            lambda mechanics_state: mechanics_state.system,
        intervenors: Sequence[AbstractIntervenor] = (),
    ):
        self.net = net
        self.mechanics = mechanics
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
        self.feedback_channel = Channel(delay)
        self.feedback_leaves_func = feedback_leaves_func
        self.intervenors = intervenors
    
    @jax.named_scope("fbx.SimpleFeedback")
    def __call__(
        self, 
        input,  # AbstractTaskInput 
        state: SimpleFeedbackState, 
        key: jax.Array,
    ) -> SimpleFeedbackState:
        
        # eqx.tree_pprint(
        #     jax.tree_map(
        #         lambda x: jnp.any(jnp.isnan(x)),
        #         state,
        #         is_leaf=lambda x: isinstance(x, jnp.ndarray),
        #     ),
        #     short_arrays=False,
        # )
        
        for intervenor in self.intervenors:
            # whether this modifies the state depends on `input`
            #? it should be OK to use a single key across unique intervenors
            state = intervenor(state, input, key=key)
        
        key1, key2 = jr.split(key)
        
        feedback_state = self.feedback_channel(
            self.feedback_leaves_func(state.mechanics),
            state.feedback,
            key1,
        )
        
        network_state = self.net(
            (input, feedback_state.output), 
            state.network, 
            key2
        )
        
        mechanics_state = self.mechanics(
            network_state.output, 
            state.mechanics
        )        
        
        return SimpleFeedbackState(
            mechanics_state, 
            network_state,
            feedback_state,
        )
    
    def init(
        self, 
        effector_state: CartesianState2D,
        # mechanics_state: MechanicsState = None, 
        # network_state: NetworkState = None, 
        # feedback_state: ChannelState = None,
    ): 
        mechanics_state = self.mechanics.init(effector_state=effector_state)
        return SimpleFeedbackState(
            mechanics=mechanics_state,
            network=self.net.init(),
            feedback=self.feedback_channel.init(
                self.feedback_leaves_func(mechanics_state)
            ),
        )
    
    @property
    def memory_spec(self) -> SimpleFeedbackState:
        """Specifies which states should typically be remembered by callers.
        
        For example, `fbx.Iterator` stores trajectories of states, however it
        doesn't usually make sense to store `states.feedback.queue` for every
        timestep, because it contains info that is already available to
        `Iterator` if `states.mechanics` is stored at every timestep. If the
        feedback delay is 5 steps, `Iterator` would otherwise end up with 5 
        extra copies of all the parts of `states.mechanics` that are part of 
        the feedback.
        
        NOTE: It makes sense for this to be here since it has to do with the
        logic of the feedback loop, i.e. that queue is just transient internal 
        memory of another variable in the loop. 
        """
        return SimpleFeedbackState(
            mechanics=True, 
            network=True,
            feedback=ChannelState(output=True, queue=False)
        )

    @staticmethod
    def get_nn_input_size(
        task: AbstractTask, 
        mechanics: Mechanics, 
        feedback_leaves_func=lambda mechanics_state: mechanics_state.system,
    ) -> int:
        """Determine how many scalar input features the neural network needs.
        
        This is a static method because its logic (number of network inputs =
        number of task inputs + number of feedback inputs from `mechanics`) 
        is related to the structure of `SimpleFeedback`. However, it is 
        not an instance method because we want to construct the network
        before we construct `SimpleFeedback`.
        """
        example_feedback = feedback_leaves_func(mechanics.init())
        n_feedback = tree_sum_n_features(example_feedback)
        example_trial_spec = task.get_train_trial(key=jr.PRNGKey(0))[0]
        n_task_inputs = tree_sum_n_features(example_trial_spec.input)
        return n_feedback + n_task_inputs
    
    
def add_intervenors(
    model: SimpleFeedback, 
    intervenors: Sequence[AbstractIntervenor], 
    keep_existing: bool = True,
) -> SimpleFeedback:
    """Add intervenors to a model, returning the updated model.
    
    TODO:
    - Could this be generalized to `AbstractModel[StateT]`?
    """
    if keep_existing:
        intervenors = model.intervenors + tuple(intervenors)
    
    return SimpleFeedback(
        net=model.net,
        mechanics=model.mechanics,
        delay=model.delay,
        feedback_leaves_func=model.feedback_leaves_func,
        intervenors=intervenors,
    )