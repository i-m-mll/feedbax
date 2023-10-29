"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from feedbax.mechanics.system import System
from feedbax.types import CartesianState2D


logger = logging.getLogger(__name__)


N_DIM = 2


class MechanicsState(eqx.Module):
    system: PyTree[Array]
    effector: CartesianState2D
    solver: PyTree 


class Mechanics(eqx.Module):
    system: System 
    dt: float 
    term: dfx.AbstractTerm 
    solver: dfx.AbstractSolver
    
    def __init__(self, system, dt, solver=dfx.Euler):
        self.system = system
        self.term = dfx.ODETerm(self.system.vector_field)
        self.solver = solver()
        self.dt = dt        
    
    @jax.named_scope("fbx.Mechanics")
    def __call__(self, input, state: MechanicsState):
        # using (0, dt) for (tprev, tnext) seems fine if there's no t dependency in the system
        system_state, _, _, solver_state, _ = self.solver.step(
            self.term, 
            0, 
            self.dt, 
            state.system, 
            input, 
            state.solver, 
            made_jump=False,
        )
        effector_state = self.system.effector(system_state)
        return MechanicsState(system_state, effector_state, solver_state)
    
    def init(
        self, 
        system_state=None,
        effector_state=None,
        solver_state=None,
        #input=None, 
        key=None
    ):
        """Returns an initial state for use with the `Mechanics` module.
        
        TODO: There are a couple of options how to switch between initializing 
        from effector state versus configuration state. We could 
        
            1) Initialize based on which of the two is provided. Then we'd have 
               to give precedence to one of them, or raise an error, if both 
               are provided. The init of other modules would need to 
            2) Let the user control a switch, in this module or elsewhere,
               that determines which to use, and raise an error if the 
               provided arguments don't match the switch.
            3) Assume that only effector init makes sense, since otherwise 
               `Task` might need to know about the inner workings of systems,
               when perhaps it should only know about behaviour.
        
        TODO:
        - Should we allow the user to pass input for constructing `solver_state`?
        """
        if effector_state is None:
            effector_state = CartesianState2D(
                jnp.zeros(N_DIM), 
                jnp.zeros(N_DIM)
            )
        #! assumes zero initial velocity; TODO convert initial velocity also
        system_state = self.system.init(effector_state)
        init_input = jnp.zeros((self.system.control_size,))
        solver_state = self.solver.init(
            self.term, 0, self.dt, system_state, init_input
        )
        
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