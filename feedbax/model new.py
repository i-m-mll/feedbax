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
from functools import partial
import logging
from typing import Any, Callable, Generic, Sequence, TypeVar

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
from feedbax.state import CartesianState2D
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
        
    @abstractmethod
    def init(
        self,
        state, 
    ) -> StateT:
        ...
        
    @abstractproperty
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered by callers."""
        ...


class SimpleFeedbackState(AbstractModelState):
    mechanics: MechanicsState
    network: NetworkState
    feedback: ChannelState


class ModuleWrapper(eqx.Module):
    where: Callable[[SimpleFeedbackState], PyTree]
    module: eqx.Module
    module_input: Callable[[PyTree, SimpleFeedbackState], PyTree]  

    def __call__(self, state, input, key):
        return eqx.tree_at(
            self.where, 
            state, 
            self.module(
                self.module_input(input, state), 
                self.where(state), 
                key=key
            ),
        )
        

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
    
    wheres: Any
    modules: Any
    modules_: list[ModuleWrapper]
    module_inputs: Any
    module_tuples: Any
    
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
             
        
        # module_labels = ['get_feedback', 'network_step', 'mechanics_step']
        self.wheres = [
            lambda state: state.feedback,
            lambda state: state.network,
            lambda state: state.mechanics,
        ]
        
        self.modules = [
            self.feedback_channel,
            self.net,
            self.mechanics,
        ]
        
        self.module_inputs = [
            lambda _, state: feedback_leaves_func(state.mechanics),
            lambda input, state: (input, state.feedback.output),
            lambda _, state: state.network.output,
        ]
        
        self.modules_ = jax.tree_map(
            lambda *args: ModuleWrapper(*args),
            self.wheres, self.modules, self.module_inputs
        )
        
        self.module_tuples = tuple(zip(
            self.wheres, 
            self.modules, 
            self.module_inputs,
        ))

    def _eval_module(self, i, val):
        state, keys = val
        where, module, module_input = self.module_tuples[i]
        state = eqx.tree_at(
            where,
            state, 
            module(module_input(input, state), where(state), key=keys[i]),
        )
        return state, keys
    
    @partial(jax.jit, static_argnames=('module_tuples',))
    def _eval_modules(self, state, keys, module_tuples):
        state, _ = jax.lax.fori_loop(
            0, 
            len(module_tuples),
            self._eval_module,
            (state, keys)
        )
        return state
    
    @jax.named_scope("fbx.SimpleFeedback")
    def __call__(
        self, 
        input,  # AbstractTaskInput 
        state: SimpleFeedbackState, 
        key: jax.Array,
    ) -> SimpleFeedbackState:
        
        for intervenor in self.intervenors:
            # whether thisz modifies the state depends on `input`
            #? it should be OK to use a single key across unique intervenors
            state = intervenor(state, input, key=key)
        
        keys = jr.split(key, len(self.module_tuples))  

        # 0. THIS DOES NOT WORK        
        for i, module_ in enumerate(self.modules_):
            state = module_(state, input, keys[i])
        
        # 1. THIS DOES NOT WORK
        # state = self._eval_modules(state, keys, self.module_tuples)
        
        # 2. THIS DOES NOT WORK
        # states = [state]
        #        
        # for i in range(len(self.module_tuples)):
        #     where, module, module_input = self.module_tuples[i]
        #     states.append(eqx.tree_at(
        #         where, 
        #         states[-1], 
        #         module(
        #             module_input(input, states[-1]), 
        #             where(states[-1]), 
        #             key=keys[i]
        #         ),
        #     ))

        return state
    
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

def remove_intervenors(
    model: SimpleFeedback
) -> SimpleFeedback:
    """Return a model with no intervenors."""
    return add_intervenors(model, intervenors=(), keep_existing=False)