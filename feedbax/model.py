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
import copy
from functools import cached_property
import logging
import os
from typing import (
    TYPE_CHECKING,
    Callable, 
    Dict, 
    Generic, 
    Optional, 
    Sequence, 
    Tuple, 
    TypeVar, 
    Union,
)

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree
import numpy as np

from feedbax.channel import Channel, ChannelState
from feedbax.intervene import AbstractIntervenor
if TYPE_CHECKING:
    from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.task import AbstractTask, AbstractTaskInput
from feedbax.utils import tree_sum_n_features, tree_pformat_indent

if TYPE_CHECKING:
    from feedbax.networks import NetworkState


logger = logging.getLogger(__name__)


N_DIM = 2


class AbstractModelState(eqx.Module):
    ...
    

StateT = TypeVar("StateT", bound=AbstractModelState)


class AbstractModel(eqx.nn.StatefulLayer, Generic[StateT]):
    """Base class for compositional state-dependent models.
    
    To define a new model, the following should be implemented:
    
        1. A concrete subclass of `AbstractModelState` that defines the PyTree
           structure of the full model state. The fields may be types of 
           `AbstractModelState` as well, in the case of nested models.
        2. A concrete subclass of this class (`AbstractModel`) with:
            i. the appropriate components for doing state transformations;
            either `eqx.Module`/`AbstractModel`-typed fields, or instance 
            methods, each with the signature `input, state, *, key`;
            ii. a `model_spec` property that specifies a series of named 
            operations on parts of the full model state, using those component 
            modules; and
            iii. an `init` method that returns an initial full model state.
            
    The motivation behind the level of abstraction of this class is that it
    gives a name to each functional stage of a composite model, so that after 
    model instantiation or training, the user can choose to insert additional 
    state interventions to be executed prior to any of the named stages. 
    """
    intervenors: AbstractVar[Dict[str, Sequence[AbstractIntervenor]]]
    
    def __call__(
        self, 
        input, # AbstractTaskInput 
        state: StateT, 
        key: jax.Array,
    ) -> StateT:
        
        with jax.named_scope(type(self).__name__):
            
            keys = jr.split(key, len(self._stages))
            
            for label, key in zip(self._stages, keys):
                intervenors, module, module_input, substate_where = \
                    self._stages[label]
                
                key1, key2 = jr.split(key)
                
                if os.environ.get('FEEDBAX_DEBUG', False) == "True": 
                    debug_strs = [tree_pformat_indent(x, indent=4) for x in 
                        (
                            module(self),
                            module_input(input, state),
                            substate_where(state),
                        )
                    ]
                    
                    logger.debug(
                        f"Module: {type(self).__name__} \n"
                        f"Stage: {label} \n"
                        f"Stage module:\n{debug_strs[0]}\n"
                        f"Input:\n{debug_strs[1]}\n"
                        f"Substate:\n{debug_strs[2]}\n"
                    )                  
                
                for intervenor in intervenors:
                    # Whether this modifies the state depends on `input`.
                    state = intervenor(input, state, key=key1)
                
                state = eqx.tree_at(
                    substate_where, 
                    state, 
                    module(self)(
                        module_input(input, state), 
                        substate_where(state), 
                        key=key2,
                    ),
                )
        
        return state

    @abstractproperty
    def model_spec(self) -> OrderedDict[str, Tuple[
        eqx.Module,
        Callable[[StateT], PyTree],
        Sequence[Callable[[AbstractTaskInput, StateT], PyTree]],
    ]]:
        """Specifies the model in terms of substate operations.
        
        Each tuple in the dict has the following elements:
        
            1. A callable (e.g. an `eqx.Module` or method) that takes `input`
               and a substate of the model state and returns an updated 
               substate;
            2. A function that selects the parts of the model input and state 
               PyTrees that are passed to the module, i.e. its `input`; and
            3. A function that selects the part of the model state PyTree to 
               be updated, i.e. the part of the model state passed as the
               second (`state`) argument to the callable, and updated by the
               PyTree returned by the callable.
               
        It's necessary to use `OrderedDict` because `jax.tree_util` still
        sorts `dict` keys, and we can't have that.
        """
        ...
  
    @cached_property 
    def _stages(self) -> Tuple[Tuple[
        Sequence[AbstractIntervenor],
        eqx.Module,
        Callable[[StateT], PyTree],
        Sequence[Callable[[AbstractTaskInput, StateT], PyTree]],
    ]]: 
        """Zips up the user-defined intervenors with `model_spec`."""
        #! should not be referred to in `__init__`, at least before defining `intervenors`
        return jax.tree_map(
            lambda x, y: (y,) + x, 
            self.model_spec, 
            jax.tree_map(tuple, self.intervenors, 
                        is_leaf=lambda x: isinstance(x, list)),
            is_leaf=lambda x: isinstance(x, tuple),
        )
    
    def _get_intervenors_dict(
        self, intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                          Dict[str, Sequence[AbstractIntervenor]]]]
    ):
        intervenors_dict = jax.tree_map(
            lambda _: [], 
            self.model_spec,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        
        if intervenors is not None:
            if isinstance(intervenors, Sequence):
                # By default, place interventions in the first stage.
                intervenors_dict.update({'get_feedback': list(intervenors)})
            elif isinstance(intervenors, dict):
                intervenors_dict.update(
                    jax.tree_map(list, intervenors, 
                                 is_leaf=lambda x: isinstance(x, Sequence)))
            else:
                raise ValueError("intervenors not a sequence or dict of sequences")
        
        return intervenors_dict
    
    @abstractmethod
    def init(
        self,
        **kwargs,
    ) -> StateT:
        """Return an initial state for the model."""
        ...
    
    @abstractproperty
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered by callers."""
        ...
        

class SimpleFeedbackState(AbstractModelState):
    mechanics: "MechanicsState"
    network: "NetworkState"
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
    mechanics: "Mechanics"
    feedback_channel: Channel
    delay: int 
    feedback_leaves_func: Callable[[PyTree], PyTree]  
    
    intervenors: Dict[str, Sequence[AbstractIntervenor]]
    
    def __init__(
        self, 
        net: eqx.Module, 
        mechanics: "Mechanics", 
        delay: int = 0, 
        feedback_leaves_func: Callable[["MechanicsState"], PyTree] \
            = lambda mechanics_state: mechanics_state.plant.skeleton,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        key: Optional[jax.Array] = None,
    ):
        self.net = net
        self.mechanics = mechanics
        self.feedback_leaves_func = feedback_leaves_func
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1
        self.feedback_channel = Channel(delay)
        self.intervenors = self._get_intervenors_dict(intervenors)
        
    @property
    def model_spec(self):
        """Specifies the stages of the model in terms of state operations.
        
        Note that the module has to be given as a function that obtains it from `self`, 
        otherwise it won't use modules that have been updated during training.
        I'm not sure why the references are kept across model updates...
        """
        return OrderedDict({
            'get_feedback': (
                lambda self: self.feedback_channel,  # module that operates on it
                lambda _, state: self.feedback_leaves_func(state.mechanics),  # inputs
                lambda state: state.feedback,  # substate it operates on
            ),
            'nn_step': (
                lambda self: self.net,
                lambda input, state: (input, state.feedback.output),
                lambda state: state.network,                
            ),
            'mechanics_step': (
                lambda self: self.mechanics,
                lambda _, state: state.network.output,
                lambda state: state.mechanics,
            ),
        })        

    def init(
        self, 
        mechanics = None,
        network = None, 
        feedback = None,
    ): 
        """Return an initial state for the model.
        
        TODO:
        - We can probably use `eval_shape` more generally when an init is not 
          available, which may be the case for simpler modules. The 
          `model_spec` could be used to determine the relevant info flow.
        """
        
        if mechanics is None:
            mechanics = dict()
        if network is None:
            network = dict()
        
        mechanics_state = self.mechanics.init(**mechanics)

        return SimpleFeedbackState(
            mechanics=mechanics_state,
            network=self.net.init(**network),
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
        
        In particular, this information will be used by `Iterator`, but will
        be ignored by `SimpleIterator`, which remembers the full state 
        indiscriminately to favour speed over memory usage.
        
        NOTE: It makes sense for this to be here since it has to do with the
        logic of the feedback loop, i.e. that queue is just transient internal 
        memory of another variable in the loop. 
        """
        return SimpleFeedbackState(
            mechanics=self.mechanics.memory_spec, 
            network=self.net.memory_spec,
            feedback=ChannelState(output=True, queue=False)
        )

    @staticmethod
    def get_nn_input_size(
        task: AbstractTask, 
        mechanics: "Mechanics", 
        feedback_leaves_func=lambda mechanics_state: mechanics_state.plant.skeleton,
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
    model: AbstractModel, 
    intervenors: Union[Sequence[AbstractIntervenor],
                       Dict[str, Sequence[AbstractIntervenor]]], 
    keep_existing: bool = True,
    *,
    key: jax.Array
) -> AbstractModel:
    """Return an updated model with added intervenors.
    
    TODO:
    - Could this be generalized to `AbstractModel[StateT]`?
    """
    if keep_existing:
        if isinstance(intervenors, Sequence):
            # If a sequence is given, append to the first stage.
            first_stage_label = next(iter(model.intervenors))
            intervenors_dict = eqx.tree_at(
                lambda intervenors: intervenors[first_stage_label],
                model.intervenors,
                model.intervenors[first_stage_label] + list(intervenors),
            )
        elif isinstance(intervenors, dict):
            intervenors_dict = copy.deepcopy(model.intervenors)
            for label, new_intervenors in intervenors.items():
                intervenors_dict[label] += list(new_intervenors)
        else:
            raise ValueError("intervenors not a sequence or dict of sequences")

    return eqx.tree_at(
        lambda model: model.intervenors,
        model, 
        intervenors_dict,
    )

def remove_intervenors(
    model: SimpleFeedback
) -> SimpleFeedback:
    """Return a model with no intervenors."""
    return add_intervenors(model, intervenors=(), keep_existing=False)