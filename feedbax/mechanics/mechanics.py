"""Discretize and solve mechanical models.

TODO:
- Maybe use generics for `MechanicsState.system`, e.g. so we can type 
  `system_state`

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import cached_property
import logging
from typing import Optional, Type, Union

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
from feedbax.intervene import AbstractIntervenor
from feedbax.mechanics.plant import AbstractPlant, PlantState

from feedbax.model import wrap_stateless_callable
from feedbax.staged import AbstractStagedModel, ModelStage
from feedbax.state import AbstractState, CartesianState2D


logger = logging.getLogger(__name__)


N_DIM = 2


class MechanicsState(AbstractState):
    plant: PlantState
    effector: CartesianState2D
    solver: PyTree 


class Mechanics(AbstractStagedModel[MechanicsState]):
    """Discretizes and iterates the solution of a system with an effector.
        
    TODO: 
    - What does this class do that is distinct from `Plant`?
        - In principle we could amalgamate `term`, `solver`, and `_dynamics_step` 
          into `Plant`... or into `AbstractDynamicalSystem` 
        - This class manages the effector. Is it just an effector wrapper?
    - Make sure total system forces are in `system.force`/`system.torque`...
      - It doesn't make sense to update them in the solver step, and they are 
        a combination of the converted effector forces and the effect of the 
        inputs (either directly, or through muscles).
    """
    plant: AbstractPlant 
    dt: float 
    solver: dfx.AbstractSolver
    
    intervenors: Mapping[str, AbstractIntervenor] 
    
    def __init__(
        self, 
        plant: AbstractPlant,
        dt: float, 
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler, 
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Mapping[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        self.plant = plant
        self.solver = solver_type()
        self.dt = dt        
        self.intervenors = self._get_intervenors_dict(intervenors)         
    
    @property
    def model_spec(self):
        return OrderedDict({
            "convert_effector_force": ModelStage(
                callable=lambda self: self.plant.skeleton.update_state_given_effector_force,
                where_input=lambda input, state: state.effector.force,
                where_state=lambda state: state.plant.skeleton,
            ),
            "statics_step": ModelStage(  
                # the `plant` module directly implements non-ODE operations 
                callable=lambda self: self.plant,
                where_input=lambda input, state: input,
                where_state=lambda state: state.plant,
            ),
            "dynamics_step": ModelStage(
                callable=lambda self: self._dynamics_step,
                where_input=lambda input, state: input,
                where_state=lambda state: state,
            ),
            "get_effector": ModelStage(
                callable=lambda self: \
                    wrap_stateless_callable(self.plant.skeleton.effector, pass_key=False),
                where_input=lambda input, state: state.plant.skeleton,
                where_state=lambda state: state.effector,
            )
        })

    @cached_property 
    def term(self) -> dfx.AbstractTerm:
        """The total vector field for the plant"""
        return dfx.ODETerm(self.plant.vector_field) 
    
    def _dynamics_step(
        self, 
        input, 
        state: MechanicsState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        
        plant_state, _, _, solver_state, _ = self.solver.step(
            self.term, 
            0, 
            self.dt, 
            state.plant, 
            input, 
            state.solver, 
            made_jump=False,
        )
        
        return eqx.tree_at(
            lambda state: (state.plant, state.solver),
            state,
            (plant_state, solver_state),
        )
    
    @property 
    def memory_spec(self):
        return MechanicsState(
            plant=True,
            effector=True,
            solver=False,
        )
    
    def init(
        self, 
        *,
        key=None,
    ):
        """Returns an initial state for use with the `Mechanics` module.
        """            
           
        plant_state = self.plant.init()
        init_input = jnp.zeros((self.plant.input_size,))

        return MechanicsState(
            plant=plant_state,
            effector=self.plant.skeleton.effector(plant_state.skeleton), 
            solver=self.solver.init(
                self.term, 
                0, 
                self.dt, 
                plant_state, 
                init_input
            ),
        )
        
    
    def n_vars(self, where):
        """
        TODO: Given a function that returns a PyTree of leaves of `mechanics_state`,
        return the sum of the sizes of the last dimensions of the leaves.
        
        Alternatively, just return an empty `mechanics_state`.
        
        This is useful to automatically determine the number of feedback inputs 
        during model construction, when a `mechanics_state` instance isn't yet available.
        
        See `get_model` in notebook 8.
        """
        # tree.tree_sum_n_features
        ...
      

