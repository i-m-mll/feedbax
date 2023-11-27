"""Discretize and solve mechanical models.

TODO:
- Maybe use generics for `MechanicsState.system`, e.g. so we can type 
  `system_state`

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
import logging
from typing import Callable, Dict, Optional, Sequence, TypeVar, Union

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from feedbax.intervene import AbstractIntervenor

from feedbax.mechanics.system import System
from feedbax.model import AbstractModel, AbstractModelState
from feedbax.state import AbstractState, CartesianState2D, StateBounds


logger = logging.getLogger(__name__)


N_DIM = 2


class MechanicsState(AbstractModelState):
    system: PyTree[Array]
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
    system: System 
    dt: float 
    term: dfx.AbstractTerm 
    solver: dfx.AbstractSolver
    clip_states: bool
    
    intervenors: Dict[str, AbstractIntervenor] 
    
    def __init__(
        self, 
        system, 
        dt, 
        solver=dfx.Euler, 
        clip_states=True,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: Optional[jax.Array] = None,
    ):
        self.system = system
        self.term = dfx.ODETerm(self.system.vector_field)
        self.solver = solver()
        self.dt = dt        
        self.clip_states = clip_states
        self.intervenors = self._get_intervenors_dict(intervenors)
    
    @property
    def model_spec(self):
        return OrderedDict({
            "convert_effector_force": (
                lambda self: self.system.update_state_given_effector_force,
                lambda state: state.system, 
                lambda _, state: state.effector.force,
            ),
            "solver_step": (
                lambda self: self._solver_step,
                lambda state: (state.system, state.solver),
                lambda input, _: input,
            ),
            "clip_states": (
                lambda self: self._get_clipped_states,
                lambda state: state.system,
                lambda *_: None,
            ),
            "get_effector": (
                lambda self: \
                    lambda input, _, key=None: self.system.effector(input),
                lambda state: state.effector,
                lambda _, state: state.system,
            )
        })
    
    def _solver_step(
        self, 
        input, 
        state,
        *,
        key: Optional[jax.Array] = None,
    ):
        system_state, solver_state = state
        system_state, _, _, solver_state, _ = self.solver.step(
            self.term, 
            0, 
            self.dt, 
            system_state, 
            input, 
            solver_state, 
            made_jump=False,
        )
        
        return (system_state, solver_state)
    
    def _get_clipped_states(self, input, state, *, key: Optional[jax.Array] = None):
        if self.clip_states:
            return clip_state(state, self.system.bounds)
        else: 
            return state
    
    @property 
    def memory_spec(self):
        return MechanicsState(
            system=True,
            effector=True,
            solver=False,
        )
    
    def init(
        self, 
        system=None,
        effector: CartesianState2D = None,
        solver=None,
        key=None,
    ):
        """Returns an initial state for use with the `Mechanics` module.
        
        If system state happens to be passed, it takes precedence over 
        passed effector state. If neither is passed, the default system state
        is used.
        """
        if effector is None and system is None:
            system = self.system.init()
        
        if system is not None:
            if effector is not None:
                logger.warning("Both `system` and `effector` inits provided "
                            "to `Mechanics`; initializing from `system` "
                            "values")
            system_state = system
            effector_state = self.system.effector(system_state)

        if effector is not None:
            system_state = self.system.inverse_kinematics(effector)
            effector_state = self.system.effector(system_state)
        
        if solver is None:
            init_input = jnp.zeros((self.system.control_size,))
            solver_state = self.solver.init(
                self.term, 0, self.dt, system_state, init_input
            )
        else:
            # I don't know that this would ever be useful.
            solver_state = solver
        
        return MechanicsState(
            system=system_state,
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