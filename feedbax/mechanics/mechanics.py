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
from typing import Callable, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union

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
from feedbax.state import CartesianState2D, StateBounds


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
    
    intervenors: Dict[str, AbstractIntervenor] 
    
    def __init__(
        self, 
        plant: AbstractPlant,
        dt: float, 
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler, 
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: Optional[jax.Array] = None,
    ):
        self.plant = plant
        self.solver = solver_type()
        self.dt = dt        
        self.intervenors = self._get_intervenors_dict(intervenors)         
    
    @property
    def model_spec(self):
        return OrderedDict({
            "convert_effector_force": (
                lambda self: self.plant.skeleton.update_state_given_effector_force,
                lambda input, state: state.effector.force,
                lambda state: state.plant.skeleton,
            ),
            "statics_step": (  
                # the `plant` module directly implements non-ODE operations 
                lambda self: self.plant,
                lambda input, state: input,
                lambda state: state.plant,
            ),
            "dynamics_step": (
                lambda self: self._dynamics_step,
                lambda input, state: input,
                lambda state: (state.plant, state.solver),
            ),
            "get_effector": (
                lambda self: \
                    lambda input, state, key=None: \
                        self.plant.skeleton.effector(input),
                lambda input, state: state.plant.skeleton,
                lambda state: state.effector,
            )
        })

    @cached_property 
    def term(self) -> dfx.AbstractTerm:
        """The total vector field for the plant"""
        return dfx.ODETerm(self.plant.vector_field) 
    
    def _dynamics_step(
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
        # if plant is not None:
        #     plant_state = self.plant.init(**plant)
        #     effector_state = self.plant.skeleton.effector(plant_state.skeleton)
        
        # elif effector is not None:
        #     skeleton_state = self.plant.skeleton.inverse_kinematics(effector)
        #     plant_state = self.plant.init(skeleton=skeleton_state)
        #     effector_state = self.plant.skeleton.effector(plant_state.skeleton)
            
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
      

