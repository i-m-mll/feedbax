"""Discretize and solve mechanical models.

TODO:
- Maybe use generics for `MechanicsState.system`, e.g. so we can type 
  `system_state`

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import Callable, Optional, TypeVar

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from feedbax.mechanics.system import System
from feedbax.state import AbstractState, CartesianState2D, StateBounds


logger = logging.getLogger(__name__)


N_DIM = 2


class MechanicsState(eqx.Module):
    system: PyTree[Array]
    effector: CartesianState2D
    solver: PyTree 


class Mechanics(eqx.Module):
    """Discretizes and iterates the solution of a system with an effector.
    
    TODO:
    - Could subclass `AbstractModel`; rename `MechanicsModel`.
    """
    system: System 
    dt: float 
    term: dfx.AbstractTerm 
    solver: dfx.AbstractSolver
    clip_states: bool
    
    def __init__(self, system, dt, solver=dfx.Euler, clip_states=True):
        self.system = system
        self.term = dfx.ODETerm(self.system.vector_field)
        self.solver = solver()
        self.dt = dt        
        self.clip_states = clip_states
    
    @jax.named_scope("fbx.Mechanics")
    def __call__(self, input, state: MechanicsState, *, key: Optional[jax.Array] = None):
        # using (0, dt) for (tprev, tnext) seems fine if there's no t dependency in the system
        
        # given effector force, update system state
        state = eqx.tree_at(
            lambda state: state.system,
            state,
            self.system.update_state_given_effector_force(
                state.system, 
                state.effector.force,
            )
        )
        
        # evolve system state
        system_state, _, _, solver_state, _ = self.solver.step(
            self.term, 
            0, 
            self.dt, 
            state.system, 
            input, 
            state.solver, 
            made_jump=False,
        )
        
        # TODO: make sure total system forces are in `system.force`/`system.torque`...
        # It doesn't make sense to update them in the solver step,
        # and they are a combination of the converted effector forces and the 
        # effect of the inputs (either directly, or through muscles).
        
        if self.clip_states:
            system_state = clip_state(system_state, self.system.bounds)
        
        effector_state = self.system.effector(system_state)
        
        return MechanicsState(system_state, effector_state, solver_state)
    
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