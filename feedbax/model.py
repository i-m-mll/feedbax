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
from collections import OrderedDict
import logging
from typing import Callable, Dict, Generic, Optional, Sequence, Tuple, TypeVar, Union

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
from feedbax.task import AbstractTask, AbstractTaskInput
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
    - Could insert intervenors as stages in the model, but I think it makes 
    more sense for modules to be designed-in, which then provides the user 
    with labels for stages at which interventions can be applied, resulting 
    in a dict of interventions that mirrors the dict of stages.
    """
    net: eqx.Module  
    mechanics: Mechanics 
    feedback_channel: Channel
    delay: int 
    feedback_leaves_func: Callable[[PyTree], PyTree]  
    
    # stage_spec: Dict[str, Tuple[
    #     Callable[[SimpleFeedbackState], PyTree],
    #     eqx.Module,
    #     Sequence[Callable[[AbstractTaskInput, SimpleFeedbackState], PyTree]],
    # ]]
    intervenors: Dict[str, Sequence[AbstractIntervenor]]
    stages: Tuple[Tuple[
        Sequence[AbstractIntervenor],
        Callable[[SimpleFeedbackState], PyTree],
        eqx.Module,
        Sequence[Callable[[AbstractTaskInput, SimpleFeedbackState], PyTree]],
    ]]
    
    def __init__(
        self, 
        net: eqx.Module, 
        mechanics: Mechanics, 
        delay: int = 0, 
        feedback_leaves_func: Callable[[MechanicsState], PyTree] = \
            lambda mechanics_state: mechanics_state.system,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                              Dict[str, Sequence[AbstractIntervenor]]]] = None,
    ):
        self.net = net
        self.mechanics = mechanics
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
        self.feedback_channel = Channel(delay)
        self.feedback_leaves_func = feedback_leaves_func
        
        intervention_dict = jax.tree_map(
            lambda _: ((),), 
            self.stages_spec,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        
        if intervenors is not None:
            if isinstance(intervenors, Sequence):
                # By default, place interventions in the first stage.
                intervention_dict.update({'get_feedback': tuple(intervenors)})
            elif isinstance(intervenors, dict):
                intervention_dict.update(jax.tree_map(tuple, intervenors))
            else:
                raise ValueError("intervenors not a sequence or dict of sequences")
        
        self.intervenors = intervention_dict
        
        # To construct actual stages, join intervenors with stages_spec.
        self.stages = tuple(jax.tree_map(
            lambda x, y: y + x, 
            self.stages_spec, 
            self.intervenors,
            is_leaf=lambda x: isinstance(x, tuple),
        ).values())
        
    @property
    def stages_spec(self):
        """Specifies the stages of the model in terms of substate operations.
        
        Note that the module has to be given as a function that obtains it from `self`, 
        or otherwise it won't use modules that have been updated during training.
        I'm not sure why the references are kept across model updates...
        """
        return OrderedDict({
            'get_feedback': (
                lambda state: state.feedback,  # substate it operates on
                lambda self: self.feedback_channel,  # module that operates on it
                lambda _, state: self.feedback_leaves_func(state.mechanics),  # inputs
            ),
            'network_step': (
                lambda state: state.network,
                lambda self: self.net,
                lambda input, state: (input, state.feedback.output),
            ),
            'mechanics_step': (
                lambda state: state.mechanics,
                lambda self: self.mechanics,
                lambda _, state: state.network.output,
            ),
        })
    
    @jax.named_scope("fbx.SimpleFeedback")
    def __call__(
        self, 
        input,  # AbstractTaskInput 
        state: SimpleFeedbackState, 
        key: jax.Array,
    ) -> SimpleFeedbackState:
        
        keys = jr.split(key, len(self.stages))
        
        for (intervenors, where, module, module_input), key \
            in zip(self.stages, keys):
            
            key1, key2 = jr.split(key)
            
            for intervenor in intervenors:
                # whether this modifies the state depends on `input`
                state = intervenor(state, input, key=key1)
                
            state = eqx.tree_at(
                where, 
                state, 
                module(self)(
                    module_input(input, state), 
                    where(state), 
                    key=key2,
                ),
            )
        
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
    intervenors: Union[Sequence[AbstractIntervenor],
                       Dict[str, Sequence[AbstractIntervenor]]], 
    keep_existing: bool = True,
) -> SimpleFeedback:
    """Add intervenors to a model, returning the updated model.
    
    TODO:
    - Could this be generalized to `AbstractModel[StateT]`?
    """
    if keep_existing:
        if isinstance(intervenors, Sequence):
            # If a sequence is given, append to the first stage.
            first_stage_label = model.intervenors.popitem(last=False)[0]
            intervenors_dict = eqx.tree_at(
                lambda intervenors: intervenors[first_stage_label],
                model.intervenors,
                model.intervenors[first_stage_label] + tuple(intervenors),
            )
        elif isinstance(intervenors, dict):
            intervenors_dict = eqx.tree_map(
                lambda x, y: x + tuple(y),
                model.intervenors,
                intervenors,
            )
        else:
            raise ValueError("intervenors not a sequence or dict of sequences")
    
    return SimpleFeedback(
        net=model.net,
        mechanics=model.mechanics,
        delay=model.delay,
        feedback_leaves_func=model.feedback_leaves_func,
        intervenors=intervenors_dict,
    )

def remove_intervenors(
    model: SimpleFeedback
) -> SimpleFeedback:
    """Return a model with no intervenors."""
    return add_intervenors(model, intervenors=(), keep_existing=False)