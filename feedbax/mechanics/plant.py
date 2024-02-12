"""

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
import logging
from typing import Optional, Tuple, TypeVar, Union

import equinox as eqx
from equinox import AbstractVar
import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import Float, PyTree
from feedbax.intervene import AbstractIntervenor
from feedbax.mechanics.muscle import AbstractMuscle, AbstractMuscleState
from feedbax.mechanics.skeleton.arm import TwoLink
from feedbax.mechanics.skeleton.skeleton import AbstractSkeleton, AbstractSkeletonState

from feedbax.staged import AbstractStagedModel, ModelStageSpec
from feedbax.state import AbstractState, StateBounds, clip_state


logger = logging.getLogger(__name__)
    
    
class PlantState(AbstractState):
    skeleton: AbstractSkeletonState
    muscles: Optional[AbstractMuscleState] = None


class AbstractPlant(AbstractStagedModel[PlantState]):
    """
    Static updates are specified in `model_spec` (i.e. `__call__`), and 
    dynamic updates (i.e. parts of the ODE/vector field) in `dynamics_spec`.
    
    TODO:
    - Could inherit from `AbstractDynamicalSystem` as well?
    """
    
    clip_states: AbstractVar[bool]
    skeleton: AbstractVar[AbstractSkeleton]
    muscle_model: AbstractVar[Optional[AbstractMuscle]]
    
    def vector_field(self, t, state, input):       
        d_state = jax.tree_map(jnp.zeros_like, state)
        
        for vf, input_func, state_where in self.dynamics_spec.values():
            d_state = eqx.tree_at(
                state_where,
                d_state,
                vf(t, state_where(state), input_func(input, state))
            )
            
        return d_state 

    @abstractproperty
    def model_spec(self):
        ...
    
    @abstractproperty 
    def dynamics_spec(self):
        ... 
    
    @abstractproperty 
    def memory_spec(self):
        ...
    
    @abstractmethod
    def init(self, *, key: Optional[jax.Array] = None) -> PlantState:
        ...
        
    @abstractproperty 
    def input_size(self) -> int:
        ...
        
    @property 
    def bounds(self) -> PyTree[StateBounds]:
        return PlantState(
            skeleton=self.skeleton.bounds,
            muscles=StateBounds(
                low=None,
                high=None,
            ),
        )
    
    def _clip_state(self, input, state, *, key: Optional[Array] = None):
        if self.clip_states:
            return clip_state(input, state)
        else:
            return state
        

class DirectForceInput(AbstractPlant):
    """A skeleton that is controlled by direct forces or torques.
    
    This is essentially a wrapper for `AbstractSkeleton`, to make sure a
    skeleton conforms with the interface expected by `Mechanics`. 
    
    TODO:
    - Make state clipping optional
    """
    
    skeleton: AbstractSkeleton 
    clip_states: bool 
    intervenors: Mapping[str, AbstractIntervenor]
    muscle_model: None = None
    
    def __init__(
        self, 
        skeleton, 
        clip_states=True, 
        intervenors=None, 
        *, 
        key=None
    ):
        self.skeleton = skeleton
        self.clip_states = clip_states
        self.intervenors = self._get_intervenors_dict(intervenors)
        
    @cached_property
    def dynamics_spec(self):
        return dict({
            "skeleton": (
                self.skeleton,
                lambda input, state: input,
                lambda state: state.skeleton,
            ),
        })
    
    @cached_property
    def model_spec(self):
        """Simple plants have no state updates apart from the skeletal ODE."""
        return OrderedDict({
            "clip_skeleton_state": ModelStageSpec(
                callable=lambda self: self._clip_state,
                where_input=lambda input, state: self.bounds.skeleton,
                where_state=lambda state: state.skeleton,
            ),
        })
        
    @property
    def memory_spec(self):
        """A simple plant has no muscles, and no muscle state to remember."""
        return PlantState(
            skeleton=True,
            muscles=False,
        )
    
    def init(self, *, key: Optional[jax.Array] = None) -> PlantState:
        return PlantState(
            skeleton=self.skeleton.init(),
            muscles=None,
        )
    
    @property
    def input_size(self):
        return self.skeleton.control_size
    


class MuscledArm(AbstractPlant):
    """"""
    skeleton: AbstractSkeleton 
    muscle_model: AbstractMuscle
    activator: eqx.Module
    clip_states: bool
    n_muscles: int
    moment_arms: Float[Array, "links=2 muscles"]
    theta0: Float[Array, "links=2 muscles"] 
    l0: Float[Array, "muscles"] 
    f0: Float[Array, "muscles"] 
    
    intervenors: Mapping[str, AbstractIntervenor] 
    
    def __init__(
        self, 
        muscle_model, 
        activator,
        skeleton=TwoLink(), 
        clip_states: bool = True,
        moment_arms=jnp.array(((2.0, -2.0, 0.0, 0.0, 1.50, -2.0),  # [cm]
                               (0.0, 0.0, 2.0, -2.0, 2.0, -1.50))),
        theta0=2 * jnp.pi * jnp.array(((15.0, 4.88, 0.0, 0.0, 4.5, 2.12), # [rad]
                                       (0.0, 0.0, 80.86, 109.32, 92.96, 91.52))) / 360.,  
        l0=jnp.array((7.32, 3.26, 6.4, 4.26, 5.95, 4.04)),  # [cm]
        f0=1., #31.8 * jnp.array((22., 12., 18., 14., 5., 10.)),  # [N] = [N/cm^2] * [cm^2]
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Mapping[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: Optional[jax.Array] = None,
    ):
        self.skeleton = skeleton
        self.activator = activator
        self.clip_states = clip_states
        
        if not theta0.shape[1] == l0.shape[0] == moment_arms.shape[1]:
            raise ValueError(
                "moment_arms, theta0, and l0 must have the same number of "
                "columns (i.e. number of muscles)"
            )
        
        self.moment_arms = moment_arms  
        self.theta0 = theta0
        self.l0 = l0
        self.f0 = f0
        self.n_muscles = moment_arms.shape[1]
        # Make sure the muscle model has the right number of muscles.
        self.muscle_model = muscle_model.change_n_muscles(self.n_muscles)
        
        self.intervenors = self._get_intervenors_dict(intervenors)        
    
    @cached_property
    def model_spec(self):
        return OrderedDict({
            "clip_skeleton_state": ModelStageSpec(
                callable=lambda self: self._clip_state,
                where_input=lambda input, state: self.bounds.skeleton,
                where_state=lambda state: state.skeleton,
            ),
            "muscle_geometry": ModelStageSpec(
                callable=lambda self: self._muscle_geometry,
                where_input=lambda input, state: state.skeleton,
                where_state=lambda state: (
                    state.muscles.length,
                    state.muscles.velocity,
                ),
            ),
            "clip_muscle_state": ModelStageSpec(
                # Activation shouldn't be below 0, and length has an UB.
                callable=lambda self: self._clip_state,
                where_input=lambda input, state: self.bounds.muscles,
                where_state=lambda state: state.muscles,
            ),
            "muscle_tension": ModelStageSpec(
                callable=lambda self: self.muscle_model,
                where_input=lambda input, state: state.muscles.activation,
                where_state=lambda state: state.muscles,
            ),
            "muscle_torques": ModelStageSpec(
                callable=lambda self: self._muscle_torques,
                where_input=lambda input, state: state.muscles,
                where_state=lambda state: state.skeleton.torque,
            ),
        })
    
    @cached_property
    def dynamics_spec(
        self
    ) -> Mapping[str, Tuple[eqx.Module, Callable, Callable]]:
        
        return dict({
            "muscle_activation": (
                self.activator,  # ODE/vector field
                lambda input, state: input,  # input to the ODE
                lambda state: state.muscles.activation,  # state we're returning derivatives for
            ),
            
            #! is this applying the torques twice? since arm will do `input_torque + state.torque`
            "skeleton": (
                self.skeleton,
                lambda input, state: state.skeleton.torque,
                lambda state: state.skeleton,
            ),
        })
    
    def _muscle_geometry(
        self, 
        input: AbstractSkeletonState, 
        state: Tuple[Array, Array], 
        *, 
        key=None
    ):
        skeleton_state = input
        length = self._muscle_length(skeleton_state.angle)
        velocity = self._muscle_velocity(skeleton_state.d_angle)
        
        return (length, velocity)

    def _muscle_length(self, angle: Array) -> Array:
        # TODO: should this be a function? how general is it?
        moment_arms, l0, theta0 = self.moment_arms, self.l0, self.theta0
        l = 1 + (moment_arms[0] * (theta0[0] - angle[0]) + moment_arms[1] * (theta0[1] - angle[1])) / l0
        return l
    
    def _muscle_velocity(self, d_angle: Array) -> Array:
        moment_arms, l0 = self.moment_arms, self.l0
        v = (moment_arms[0] * d_angle[0] + moment_arms[1] * d_angle[1]) / l0
        return v

    def _muscle_torques(self, input, state, *, key=None) -> Array:
        torque = self.moment_arms @ (self.f0 * input.tension)
        return torque
        
    @property
    def memory_spec(self) -> PlantState:
        return PlantState(
            skeleton=True,
            muscles=True,
        )
    
    def init(self, *, key: Optional[jax.Array] = None) -> PlantState:
        return PlantState(
            skeleton=self.skeleton.init(),
            muscles=self.muscle_model.init(),
        )
    
    @property
    def input_size(self) -> int:
        return self.n_muscles
    
    @property 
    def bounds(self) -> PyTree[StateBounds]:
        return PlantState(
            skeleton=self.skeleton.bounds,
            muscles=self.muscle_model.bounds,
        )