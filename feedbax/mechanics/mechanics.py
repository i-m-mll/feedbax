"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from feedbax.mechanics.system import System
from feedbax.state import AbstractState
from feedbax.utils import tree_get_idx


logger = logging.getLogger(__name__)


class MechanicsState(AbstractState):
    system: PyTree[Array]
    effector: PyTree[Array]
    solver: PyTree 


class Mechanics(eqx.Module):
    system: System 
    dt: float = eqx.field(static=True)
    term: dfx.AbstractTerm = eqx.field(static=True)
    solver: Optional[dfx.AbstractSolver] 
    
    def __init__(self, system, dt, solver=None):
        self.system = system
        self.term = dfx.ODETerm(self.system.vector_field)
        if solver is None:
            self.solver = dfx.Tsit5()
        else:
            self.solver = solver
        self.dt = dt        
    
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
        # TODO: this should be `self.system.effector(system_state)` or something,
        # otherwise, `Mechanics` knows that effector = last link, which might not be true
        effector_state = tree_get_idx(
            self.system.forward_kinematics(system_state),
            -1  # last link
        )
        return MechanicsState(system_state, effector_state, solver_state)
    
    def init(self, effector_state, input=None, key=None):
        # TODO the tuple structure of pos-vel should be introduced in data generation, and kept throughout
        #! assumes zero initial velocity; TODO convert initial velocity also
        system_state = self.system.init(effector_state)
        args = inputs_empty = jnp.zeros((self.system.control_size,))
 
        return MechanicsState(
            system_state,  # self.system.init()
            effector_state, 
            self.solver.init(self.term, 0, self.dt, system_state, args),
        )