"""Discretize and step plant models.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import cached_property
import logging
from typing import Optional, Self, Type, Union

import diffrax as dfx  # type: ignore
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
from feedbax.intervene import AbstractIntervenor
from feedbax.mechanics.plant import AbstractPlant, PlantState

from feedbax._model import wrap_stateless_keyless_callable
from feedbax._staged import AbstractStagedModel, ModelStage
from feedbax.state import AbstractState, CartesianState


logger = logging.getLogger(__name__)


class MechanicsState(AbstractState):
    """Type of state PyTree operated on by `Mechanics` instances.

    Attributes:
        plant: The state of the plant.
        effector: The state of the end effector.
        solver: The state of the Diffrax solver.
    """

    plant: PlantState
    effector: CartesianState
    solver: PyTree


class Mechanics(AbstractStagedModel[MechanicsState]):
    """Discretizes the dynamics of a plant, and iterates along with the plant statics.

    Attributes:
        plant: The plant model.
        dt: The time step duration.
        solver: The Diffrax solver.
        intervenors: The intervenors associated with each stage of the model.
    """

    plant: AbstractPlant
    dt: float
    solver: dfx.AbstractSolver

    intervenors: Mapping[Optional[str], Sequence[AbstractIntervenor]]

    def __init__(
        self,
        plant: AbstractPlant,
        dt: float,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,
        intervenors: Optional[
            Union[
                Sequence[AbstractIntervenor], 
                Mapping[Optional[str], Sequence[AbstractIntervenor]]
            ]
        ] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """
        Arguments:
            plant: The plant model.
            dt: The time step duration.
            solver_type: The type of Diffrax solver to use.
            intervenors: The intervenors associated with each stage of the model.
        """
        self.plant = plant
        self.solver = solver_type()
        self.dt = dt
        self.intervenors = self._get_intervenors_dict(intervenors)

    @property
    def model_spec(self) -> OrderedDict[str, ModelStage[Self, MechanicsState]]:
        """Specifies the stages of the model."""
        Stage = ModelStage[Self, MechanicsState]

        return OrderedDict(
            {
                "convert_effector_force": Stage(
                    callable=lambda self: self.plant.skeleton.update_state_given_effector_force,  # type: ignore
                    where_input=lambda input, state: state.effector.force,
                    where_state=lambda state: state.plant.skeleton,
                ),
                "kinematics_update": Stage(
                    # the `plant` module directly implements non-ODE operations
                    callable=lambda self: self.plant,
                    where_input=lambda input, state: input,
                    where_state=lambda state: state.plant,
                ),
                "dynamics_step": Stage(
                    callable=lambda self: self.dynamics_step,
                    where_input=lambda input, state: input,
                    where_state=lambda state: state,
                ),
                "get_effector": Stage(
                    callable=lambda self: wrap_stateless_keyless_callable(
                        self.plant.skeleton.effector
                    ),
                    where_input=lambda input, state: state.plant.skeleton,
                    where_state=lambda state: state.effector,
                ),
            }
        )

    @cached_property
    def _term(self) -> dfx.AbstractTerm:
        """The Diffrax term for the aggregate vector field of the plant."""
        return dfx.ODETerm(self.plant.vector_field)

    def dynamics_step(
        self,
        input: PyTree[Array],
        state: MechanicsState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> MechanicsState:
        """Return an updated state after a single step of plant dynamics."""
        plant_state, _, _, solver_state, _ = self.solver.step(
            self._term,
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
    def memory_spec(self) -> PyTree[bool]:
        return MechanicsState(plant=True, effector=True, solver=False)  # type: ignore

    def init(
        self,
        *,
        key: PRNGKeyArray,
    ):
        """Returns an initial state for use with the `Mechanics` module."""

        plant_state = self.plant.init(key=key)
        init_input = jnp.zeros((self.plant.input_size,))

        return MechanicsState(
            plant=plant_state,
            effector=self.plant.skeleton.effector(plant_state.skeleton),
            solver=self.solver.init(self._term, 0, self.dt, plant_state, init_input),
        )

    # def n_vars(self, where):
    #     """
    #     TODO: Given a function that returns a PyTree of leaves of `mechanics_state`,
    #     return the sum of the sizes of the last dimensions of the leaves.

    #     Alternatively, just return an empty `mechanics_state`.

    #     This is useful to automatically determine the number of feedback inputs
    #     during model construction, when a `mechanics_state` instance isn't yet available.

    #     See `get_model` in notebook 8.
    #     """
    #     # tree.tree_sum_n_features
    #     ...
