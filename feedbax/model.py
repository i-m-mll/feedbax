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
from collections.abc import Mapping
import copy
from functools import cached_property, wraps
import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable, 
    Dict, 
    Generic, 
    Optional, 
    Sequence, 
    Tuple,
    Type, 
    TypeVar, 
    Union,
)

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, PyTree
import numpy as np

from feedbax.channel import Channel, ChannelState
from feedbax.intervene import AbstractIntervenor, AbstractIntervenorInput
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
    """
    
    TODO:
    - Should this be a generic of `StateT | Array`, or something?
      - i.e. what about models that return a single array or some other
        kind of PyTree? As in linear regression, PCA...
    """
    
    @abstractmethod
    def __call__(
        self,
        input,
        state: StateT, 
        key: jax.Array,
    ) -> StateT:
        """Update the model state given inputs and a prior state."""
        ...

    @abstractproperty
    def _step(self) -> "AbstractModel[StateT]":
        """Interface to a single model step.
        
        For non-iterated models, this should trivially return `step`.
        """
        ...
    
    def state_consistency_update(
        self, 
        state: StateT,
    ) -> StateT:
        """Make sure the model state is self-consistent.
        
        The default behaviour is to just return the same state. However, in
        models where it is customary to initialize (say) the effector state 
        but not the plant configuration state, this method should be used 
        to ensure that the configuration state is initialized properly. 
        
        This avoids having to define how both states should be initialized
        in `AbstractTask`, which would require knowledge not only of the 
        structure of the state, but also of the model interface that 
        provides particular operations for modifying the state. Though, I'm not 
        sure that would be a bad thing, as the model state objects tend to 
        mirror the operations done on them anyway.
        
        However, this approach also has the advantage that these consistency
        checks can be arbitrarily complex and important to the model 
        functioning properly; it is useful for the model to be able to 
        render a state consistent, with respect to its other normal operations.
        """
        return state
    
    @abstractmethod
    def init(
        self,
        *,
        key: Optional[jax.Array] = None,
    ) -> StateT:
        """Return an initial state for the model."""
        ...
    

class ModelInput(eqx.Module):
    input: AbstractTaskInput
    intervene: Mapping[AbstractIntervenorInput]


class AbstractStagedModel(AbstractModel[StateT]):
    """Base class for state-dependent models whose stages can be intervened upon.
    
    To define a new model, the following should be implemented:
    
        1. A concrete subclass of `AbstractModelState` that defines the PyTree
           structure of the full model state. The fields may be types of 
           `AbstractModelState`, in the case of nested models.
        2. A concrete subclass of this class with:
            i. the appropriate components for doing state transformations;
            either `eqx.Module`/`AbstractModel`-typed fields, or instance 
            methods, each with the signature `input, state, *, key`;
            ii. a `model_spec` property that specifies a series of named 
            operations on parts of the full model state, using those component 
            modules; and
            iii. an `init` method that returns an initial model state.
            
    The motivation behind the level of abstraction of this class is that it
    gives a name to each functional stage of a composite model, so that after 
    model instantiation or training, the user can choose to insert additional 
    state interventions to be executed prior to any of the named stages. 
    """
    intervenors: AbstractVar[Dict[str, Sequence[AbstractIntervenor]]]
    
    def __call__(
        self, 
        input: ModelInput,
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
                    if intervenor.label in input.intervene:
                        params = input.intervene[intervenor.label]
                    else:
                        params = None
                    state = intervenor(params, state, key=key1)
                
                state = eqx.tree_at(
                    substate_where, 
                    state, 
                    module(self)(
                        #! Constructing this each time might be a performance issue
                        ModelInput(module_input(input, state), input.intervene), 
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
    
    @abstractproperty
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered by callers."""
        ...
        
    @property 
    def _step(self):
        """This assumes all staged models will specify single update steps,
        not iterated models.
        
        This allows classes like `AbstractTask` and `TaskTrainer` to refer
        to methods of a single step of the model, without having to know whether
        the model step is wrapped in an iterator; `AbstractIterator` concretes will 
        return `self._step` instead.
        """
        return self 
    
    @property 
    def _all_intervenor_labels(self):
        model_leaves = jax.tree_util.tree_leaves(
            self, 
            is_leaf=lambda x: isinstance(x, AbstractIntervenor)
        )
        labels = [leaf.label for leaf in model_leaves 
                  if isinstance(leaf, AbstractIntervenor)]
        return labels
        

class SimpleFeedbackState(AbstractModelState):
    mechanics: "MechanicsState"
    network: "NetworkState"
    feedback: PyTree[ChannelState]
    
    
class FeedbackSpec(eqx.Module):
    where: Callable[[AbstractModelState], PyTree[Array]]
    delay: int = 0
    noise_std: Optional[float] = None
    

DEFAULT_FEEDBACK_SPEC = FeedbackSpec(
    where=lambda mechanics_state: mechanics_state.plant.skeleton,
)


def _convert_feedback_spec(
    feedback_spec: Union[PyTree[FeedbackSpec], PyTree[Mapping[str, Any]]] 
) -> PyTree[FeedbackSpec]:
    
    if not isinstance(feedback_spec, PyTree[FeedbackSpec]):
        feedback_spec_flat, feedback_spec_def = \
            eqx.tree_flatten_one_level(feedback_spec)
        
        if isinstance(feedback_spec_flat, PyTree[Mapping]):
            feedback_specs_flat = jax.tree_map(
                lambda spec: FeedbackSpec(**spec), 
                feedback_spec_flat,
                is_leaf=lambda x: isinstance(x, Mapping),
            )
            return jtu.tree_unflatten(
                feedback_spec_def,
                feedback_specs_flat,
            )
        else:
            return [FeedbackSpec(**feedback_spec)] 
    
    return feedback_spec
        
        
class SimpleFeedback(AbstractStagedModel[SimpleFeedbackState]):
    """Simple feedback loop with a single RNN and single mechanical model.
    
    TODO:
    - PyTree of force inputs (in addition to control inputs) to mechanics
    - It might make more sense to handle multiple feedback channels inside 
      a single `Channel` object. That could minimize `tree_map` calls 
      from this class.
    - Could insert intervenors as stages in the model, but I think it makes 
    more sense for modules to be designed-in, which then provides the user 
    with labels for stages at which interventions can be applied, resulting 
    in a dict of interventions that mirrors the dict of stages.
    """
    net: eqx.Module  
    mechanics: "Mechanics"
    feedback_channels: PyTree[Channel, 'T']
    feedback_specs: PyTree[FeedbackSpec, 'T']  
    intervenors: Dict[str, Sequence[AbstractIntervenor]]
    
    def __init__(
        self, 
        net: eqx.Module, 
        mechanics: "Mechanics", 
        feedback_spec: Union[PyTree[FeedbackSpec], PyTree[Mapping[str, Any]]] = \
            DEFAULT_FEEDBACK_SPEC,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: Optional[jax.Array] = None,
    ):
        self.net = net
        self.mechanics = mechanics
        self.intervenors = self._get_intervenors_dict(intervenors)

        # If `feedback_spec` is given as a `PyTree[Mapping]`, convert to 
        # `PyTree[FeedbackSpec]`. 
        #        
        # Allow nesting of mappings to one level, to allow the user to provide
        # (say) a dict of dicts.
        feedback_specs = _convert_feedback_spec(feedback_spec)
        
        init_mechanics_state = mechanics.init()
        
        def _build_feedback_channel(spec: FeedbackSpec):
            return Channel(spec.delay, spec.noise_std, jnp.nan).change_input(
                spec.where(init_mechanics_state)
            )
        
        self.feedback_channels = jax.tree_map(
            lambda spec: _build_feedback_channel(spec),
            feedback_specs,
            is_leaf=lambda x: isinstance(x, FeedbackSpec),
        )
        self.feedback_specs = feedback_specs

    def update_feedback(
        self, 
        input: "MechanicsState", 
        state: PyTree[ChannelState, 'T'], 
        *, 
        key: Optional[Array] = None
    ) -> PyTree[ChannelState, 'T']:
        """Send current feedback states through channels, and return delayed feedback."""
        # TODO: separate keys for the different channels
        return jax.tree_map(
            lambda channel, spec, state: channel(spec.where(input), state, key=key),
            self.feedback_channels,
            self.feedback_specs,
            state,
            is_leaf=lambda x: isinstance(x, Channel),
        )
        
    @property
    def model_spec(self):
        """Specifies the stages of the model in terms of state operations.
        
        Note that the module has to be given as a function that obtains it from `self`, 
        otherwise it won't use modules that have been updated during training.
        I'm not sure why the references are kept across model updates...
        """
        return OrderedDict({
            'update_feedback': (
                lambda self: self.update_feedback,  
                lambda input, state: state.mechanics,  
                lambda state: state.feedback,  
            ),
            'nn_step': (
                lambda self: self.net,
                lambda input, state: (
                    input, 
                    jax.tree_map(
                        lambda state: state.output,
                        state.feedback,
                        is_leaf=lambda x: isinstance(x, ChannelState),
                    ),
                ),
                lambda state: state.network,                
            ),
            'mechanics_step': (
                lambda self: self.mechanics,
                lambda input, state: state.network.output,
                lambda state: state.mechanics,
            ),
        })       

    def init(
        self, 
        *,
        key: Optional[jax.Array] = None,
    ): 
        """Return an initial state for the model.
        
        TODO:
        - We can probably use `eval_shape` more generally when an init is not 
          available, which may be the case for simpler modules. The 
          `model_spec` could be used to determine the relevant info flow.
        """
        mechanics_state = self.mechanics.init()

        return SimpleFeedbackState(
            mechanics=mechanics_state,
            network=self.net.init(),
            feedback=jax.tree_map(
                lambda channel: channel.init(),
                self.feedback_channels,
                is_leaf=lambda x: isinstance(x, Channel),
            ),
        )
    
    @property
    def memory_spec(self) -> SimpleFeedbackState:
        """Specifies which states should typically be remembered by callers.
        
        For example, `fbx.Iterator` stores trajectories of states, however it
        doesn't usually make sense to store `states.feedback.queue` for every
        timestep, because it contains info that is already available at the level of
        `Iterator` if `states.mechanics` is stored at every timestep. If the
        feedback delay is 5 steps, `Iterator` could end up with 5 
        extra copies of all the parts of `states.mechanics` that are part of 
        the feedback. So it may be better not to store `states.feedback.queue`.
        
        In particular, this information will be used by `Iterator`, but will
        be ignored by `SimpleIterator`, which remembers the full state 
        indiscriminately---this is faster, but may use more memory. 
        
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
        feedback_spec: Union[PyTree[FeedbackSpec], PyTree[Mapping[str, Any]]] = \
            DEFAULT_FEEDBACK_SPEC,
    ) -> int:
        """Determine how many scalar input features the neural network needs.
        
        This is a static method because its logic (number of network inputs =
        number of task inputs + number of feedback inputs from `mechanics`) 
        is related to the structure of `SimpleFeedback`. However, it is 
        not an instance method because we want to construct the network
        before we construct `SimpleFeedback`.
        """
        init_mechanics_state = mechanics.init()
        example_feedback = jax.tree_map(
            lambda spec: spec.where(init_mechanics_state),
            _convert_feedback_spec(feedback_spec),
            is_leaf=lambda x: isinstance(x, FeedbackSpec),
        )
        n_feedback = tree_sum_n_features(example_feedback)
        example_trial_spec = task.get_train_trial(key=jr.PRNGKey(0))[0]
        n_task_inputs = tree_sum_n_features(example_trial_spec.input)
        return n_feedback + n_task_inputs
    
    def state_consistency_update(
        self, 
        state: SimpleFeedbackState
    ) -> SimpleFeedbackState:
        """Adjust the state 
        
        Update the plant configuration state, given that the user has 
        initialized the effector state.
        
        Also fill the feedback queues with the initial feedback states. This 
        is less problematic than passing all zeros.
        
        TODO: 
        - Check which of the two is initialized, and update the other one.
          Might require initializing them to NaN or something in `init`.
        """
        state = eqx.tree_at(
            lambda state: state.mechanics.plant.skeleton,
            state, 
            self.mechanics.plant.skeleton.inverse_kinematics(
                state.mechanics.effector
            ),
        )
        
        # If the PyTree of feedback channel states is full of NaNs, fill the channel queues
        # with the current values of the states to be fed back. By initializing `Channel` with
        # NaN and then performing this check/fill, we avoid passing zeros as feedback, which 
        # is more incorrect.
        def _fill_feedback_queues(state):
            return eqx.tree_at(
                lambda state: state.feedback,
                state,
                jax.tree_map(
                    lambda channel_state, spec, channel: eqx.tree_at(
                        lambda channel_state: channel_state.queue,
                        channel_state,
                        channel.delay * (spec.where(state.mechanics),),
                    ),
                    state.feedback,
                    self.feedback_specs,
                    self.feedback_channels,
                    is_leaf=lambda x: isinstance(x, ChannelState),
                ),
            )
        
        feedback_state_isnan = jax.flatten_util.ravel_pytree(
            jax.tree_map(lambda x: jnp.isnan(x), state.feedback)
        )[0]        
        
        state = jax.lax.cond(
            jnp.all(feedback_state_isnan),
            _fill_feedback_queues,
            lambda state: state,
            state,
        )
     
        return state 
    

def add_intervenor(
    model: AbstractStagedModel[StateT], 
    intervenor: AbstractIntervenor,
    stage_name: Optional[str] = None,
    **kwargs
) -> AbstractStagedModel[StateT]:
    """Return an updated model with an added intervenor.
    
    This is just a convenience for passing a single intervenor to `add_intervenor`.
    """
    if stage_name is not None:
        return add_intervenors(model, {stage_name: [intervenor]}, **kwargs)
    else:
        return add_intervenors(model, [intervenor], **kwargs)


def add_intervenors(
    model: AbstractStagedModel[StateT], 
    intervenors: Union[Sequence[AbstractIntervenor],
                       Dict[str, Sequence[AbstractIntervenor]]], 
    where: Callable[[AbstractStagedModel[StateT]], Any] = lambda model: model,
    keep_existing: bool = True,
    *,
    key: Optional[jax.Array] = None
) -> AbstractStagedModel[StateT]:
    """Return an updated model with added intervenors.
    
    Intervenors are added to `where(model)`, which by default is 
    just `model` itself.
    
    If intervenors are passed as a sequence, they are added to the first stage
    specified in `where(model).model_spec`, otherwise they should be passed in
    a dict where the keys refer to particular stages in the model spec.
    
    TODO:
    - Could this be generalized to `AbstractModel[StateT]`?
    """
    if keep_existing:
        existing_intervenors = where(model).intervenors
        
        if isinstance(intervenors, Sequence):
            # If a sequence is given, append to the first stage.
            first_stage_label = next(iter(existing_intervenors))
            intervenors_dict = eqx.tree_at(
                lambda intervenors: intervenors[first_stage_label],
                existing_intervenors,
                existing_intervenors[first_stage_label] + list(intervenors),
            )
        elif isinstance(intervenors, dict):
            intervenors_dict = copy.deepcopy(existing_intervenors)
            for label, new_intervenors in intervenors.items():
                intervenors_dict[label] += list(new_intervenors)
        else:
            raise ValueError("intervenors not a sequence or dict of sequences")

    for k in intervenors_dict:
        if k not in where(model).model_spec:
            raise ValueError(f"{k} is not a valid model stage for intervention")

    return eqx.tree_at(
        lambda model: where(model).intervenors,
        model, 
        intervenors_dict,
    )

def remove_intervenors(
    model: SimpleFeedback
) -> SimpleFeedback:
    """Return a model with no intervenors."""
    return add_intervenors(model, intervenors=(), keep_existing=False)


def wrap_stateless_callable(callable: Callable, pass_key=True):
    """Makes a 'stateless' callable trivially compatible with state-passing.
    
    `AbstractModel` defines everything in terms of transformations of parts of
    a state PyTree. In each case, the substate that is operated on is passed
    to the module that returns the updated substate. However, in some cases
    the new substate does not depend on the previous substate. For example,
    a linear network layer takes some inputs and returns some outputs, and
    on the next iteration, the linear layer's outputs do not conventionally 
    depend on its previous outputs, like an RNN cell's would.
    """
    if pass_key:
        @wraps(callable)
        def wrapped(input, state, *args, **kwargs):
            return callable(input, *args, **kwargs)
        
    else:
        @wraps(callable)
        def wrapped(input, state, *args, key: Optional[Array] = None, **kwargs):
            return callable(input, *args, **kwargs)
    
    return wrapped


def model_spec_format(
    model: AbstractStagedModel, 
    indent: int = 2, 
    newlines: bool = False,
) -> str:
    """Return a string representation of the model specification tree.
    
    Show what is called by `model`, and by any `AbstractStagedModel`s it calls.
    
    This assumes that the model spec is a tree/DAG. If there are cycles in 
    the model spec, this will recurse until an exception is raised.
    
    Args:
        model: The staged model to format.
        indent: Number of spaces to indent each nested level of the tree.
        newlines: Whether to add an extra blank line between each line.
    """
       
    def get_spec_strs(model: AbstractStagedModel):
        spec_strs = []

        for label, (intervenors, module_func, _, _) in model._stages.items():
            intervenor_str = ''.join([
                f"intervenor: {type(intervenor).__name__}\n" for intervenor in intervenors
            ])

            callable = module_func(model)
            
            # Get a meaningful label for `BoundMethod`s
            if (func := getattr(callable, '__func__', None)) is not None:
                owner = type(getattr(callable, '__self__')).__name__
                spec_str = f"{label}: {owner}.{func.__name__}"

            else: 
                spec_str = f"{label}: {type(callable).__name__}"
                
            spec_strs += [intervenor_str + spec_str]

            if isinstance(callable, AbstractStagedModel):
                spec_strs += [
                    ' ' * indent + spec_str
                    for spec_str in get_spec_strs(callable)
                ]              
            
        return spec_strs 
    
    nl = '\n\n' if newlines else '\n'
    
    return nl.join(get_spec_strs(model))

