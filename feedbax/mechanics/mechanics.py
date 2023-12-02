"""Discretize and solve mechanical models.

TODO:
- Maybe use generics for `MechanicsState.system`, e.g. so we can type 
  `system_state`

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
from functools import cached_property
import logging
from typing import Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union

import diffrax as dfx
import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from feedbax.intervene import AbstractIntervenor
from feedbax.mechanics.muscle import AbstractMuscleState, VirtualMuscle
from feedbax.mechanics.plant import AbstractPlant, PlantState
from feedbax.mechanics.skeleton import AbstractSkeleton, AbstractSkeletonState

from feedbax.dynamics import AbstractDynamicalSystem
from feedbax.model import AbstractModel, AbstractModelState
from feedbax.state import AbstractState, CartesianState2D, StateBounds


logger = logging.getLogger(__name__)


N_DIM = 2


class MechanicsState(AbstractModelState):
    plant: PlantState
    effector: CartesianState2D
    solver: PyTree 


class Mechanics(AbstractModel[MechanicsState]):
    """Discretizes and iterates the solution of a system with an effector.
        
    TODO: 
    - Make sure total system forces are in `system.force`/`system.torque`...
      - It doesn't make sense to update them in the solver step, and they are 
        a combination of the converted effector forces and the effect of the 
        inputs (either directly, or through muscles).
    """
    plant: AbstractPlant 
    dt: float 
    solver: dfx.AbstractSolver
    clip_states: bool
    
    intervenors: Dict[str, AbstractIntervenor] 
    
    def __init__(
        self, 
        plant: AbstractPlant,
        dt: float, 
        solver=dfx.Euler, 
        clip_states=True,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: Optional[jax.Array] = None,
    ):
        self.plant = plant
        self.solver = solver()
        self.dt = dt        
        self.clip_states = clip_states
        self.intervenors = self._get_intervenors_dict(intervenors)         
    
    @property
    def model_spec(self):
        return OrderedDict({
            "convert_effector_force": (
                lambda self: self.plant.skeleton.update_state_given_effector_force,
                lambda _, state: state.effector.force,
                lambda state: state.plant.skeleton,
            ),
            "plant_update": (  # dependent (non-ODE) updates specified by the plant
                lambda self: self.plant,
                lambda input, state: (input, state.plant),
                lambda state: state.plant,
            ),
            "solver_step": (
                lambda self: self._solver_step,
                lambda input, _: input,
                lambda state: (state.plant, state.solver),
            ),
            "clip_states": (
                lambda self: self._get_clipped_states,
                lambda *_: None,
                lambda state: state.plant,
            ),
            "get_effector": (
                lambda self: \
                    lambda input, _, key=None: self.plant.skeleton.effector(input),
                lambda _, state: state.plant.skeleton,
                lambda state: state.effector,
            )
        })

    @cached_property 
    def term(self) -> dfx.AbstractTerm:
        
        def _vector_field(self, t, state, input):       
            d_state = jax.tree_map(jnp.zeros, state)
            
            for vf, input_func, state_where in self.plant.dynamics_spec.values():
                d_state = eqx.tree_at(
                    state_where(d_state),
                    d_state,
                    vf(t, state_where(state), input_func(input, state))
                )
            
            return state 
        
        return dfx.ODETerm(_vector_field) 
    
    
    def _solver_step(
        self, 
        input, 
        state: Tuple[PlantState, PyTree],
        *,
        key: Optional[jax.Array] = None,
    ):
        plant_state, solver_state = state
        
        plant_state, _, _, solver_state, _ = self.solver.step(
            self.term, 
            0, 
            self.dt, 
            plant_state, 
            input, 
            solver_state, 
            made_jump=False,
        )
        
        return (plant_state, solver_state)
    
    def _get_clipped_states(self, input, state, *, key: Optional[jax.Array] = None):
        # TODO: This gets passed `mechanics_state.plant`. Maybe move it into `AbstractPlant`.
        if self.clip_states:
            return clip_state(state, self.plant.bounds)
        else: 
            return state
    
    @property 
    def memory_spec(self):
        return MechanicsState(
            plant=True,
            effector=True,
            solver=False,
        )
    
    def init(
        self, 
        plant=None,
        effector: CartesianState2D = None,
        # solver=None,
        key=None,
    ):
        """Returns an initial state for use with the `Mechanics` module.
        
        If system state happens to be passed, it takes precedence over 
        passed effector state. If neither is passed, the default system state
        is used.
        """
        if effector is None and plant is None:
            plant_state = self.plant.init()
            effector_state = self.plant.skeleton.effector(plant_state.skeleton)
            
        elif plant is not None:
            if effector is not None:
                logger.warning("Both `plant` and `effector` inits provided "
                            "to `Mechanics`; initializing from `plant` "
                            "values")
            plant_state = self.plant.init(**plant)
            effector_state = self.plant.skeleton.effector(plant_state.skeleton)
        
        elif effector is not None:
            skeleton_state = self.plant.skeleton.inverse_kinematics(effector)
            plant_state = self.plant.init(skeleton=skeleton_state)
            effector_state = self.plant.skeleton.effector(plant_state.skeleton)

        init_input = jnp.zeros((self.plant.input_size,))
        solver_state = self.solver.init(
            self.term, 0, self.dt, plant_state, init_input
        )


        return MechanicsState(
            plant=plant_state,
            effector=effector_state, 
            solver=solver_state,
        )
        
    
    def n_vars(self, leaves_func):
        """
        TODO: Given a function that returns a PyTree of leaves of `mechanics_state`,
        return the sum of the sizes of the last dimensions of the leaves.
        
        Alternatively, just return an empty `mechanics_state`.
        
        This is useful to automatically determine the number of feedback inputs 
        during model construction, when a `mechanics_state` instance isn't yet available.
        
        See `get_model` in notebook 8.
        """
        # utils.tree_sum_n_features
        ...
      

StateT = TypeVar("StateT", bound=AbstractState)
  
        
def clip_state(
    state: StateT, 
    bounds: StateBounds[StateT],
) -> StateT:
    """Constrain a state to the given bounds.
    
    TODO: 
    - Maybe we can `tree_map` this, but I'm not sure it matters,
      especially since it might require we make a bizarre
      `StateBounds[Callable]` for the operations...
    """
    if bounds.low is not None:
        state = _clip_state_to_bound(
            state, bounds.low, bounds.filter_spec.low, jnp.greater
        )
    if bounds.high is not None:
        state = _clip_state_to_bound(
            state, bounds.high, bounds.filter_spec.high, jnp.less
        )
    return state


def _clip_state_to_bound(
    state: StateT, 
    bound: StateT,
    filter_spec: PyTree[bool],
    op: Callable,
) -> StateT:
    """A single (one-sided) clipping operation."""
    states_to_clip, states_other = eqx.partition(
        state,
        filter_spec,
    )    
    
    states_clipped = jax.tree_map(
        lambda x, y: jnp.where(op(x, y), x, y),
        states_to_clip,
        bound,
    )
    
    return eqx.combine(states_other, states_clipped)