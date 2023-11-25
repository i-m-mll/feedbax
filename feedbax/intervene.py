"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from abc import abstractmethod
import logging
from typing import Callable, TypeVar

import equinox as eqx
from equinox import AbstractClassVar, AbstractVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from feedbax.state import AbstractState, CartesianState2D


logger = logging.getLogger(__name__)


StateT = TypeVar("StateT", bound=AbstractState)


class AbstractIntervenor(eqx.Module):
    """A module that intervenes on a model's state.
    
    Instances of the concrete subclasses of this class are intended to be
    registered with the `register_intervention` method of an instance of an
    `AbstractModel`, which causes the model to invoke the intervention 
    during model execution.
    
    NOTE: 
    - Interestingly, this has the same call signature as `AbstractModel`.
      This might be relevant to generalizing the `AbstractModel` class and 
      the notion of model-model interactions.
    
    TODO:
    - If `eqx.pytree_at` has much overhead, it might be good to have a way to
      "compile" multiple intervenors into a single `eqx.pytree_at` call.    
    """
    label: AbstractVar[str]
    _where: AbstractClassVar[Callable[[AbstractIntervenor, PyTree], PyTree]]
    
    @abstractmethod
    def __call__(
        self, 
        state: StateT, 
        input: PyTree,  # task inputs
        *,
        key: jax.Array,
    ) -> StateT:
        """Return a modified state PyTree."""
        ...
    
    def substate_update(
        self,
        state: StateT, 
        substate: PyTree,      
    ) -> StateT:
        """Return a modified state PyTree."""
        return eqx.tree_at(
            self._where, 
            state, 
            substate,
        )
    
    def substate(self, state: StateT) -> PyTree:
        return self._where(state)


class EffectorVelDepForceField(eqx.Module):
    """Apply a force field to an effector."""
    label: str = "effector_force_field"
    
    amplitude: float = 1.0   
    _where = lambda tree: tree.mechanics.effector.force
    
    
    def __call__(
        self, 
        state: StateT, 
        input: PyTree,  # task inputs
        key: jax.Array,
    ) -> StateT:
        ...
    
    def _force(self, vel: jax.Array):
        """Return velocity-dependent forces."""
        rot_forces = self._rot_scale * vel[..., ::-1]
        expand_force = self._expand_scale * vel 
        return rot_forces + expand_force
    

class EffectorCurlForceField(AbstractIntervenor):
    """Apply a curl force field to an effector.
    
    TODO:
    - Special case of `EffectorVelDepForceField`?
    """
    label: str = "effector_curl_field"
    _where = lambda self, tree: tree.mechanics.effector
    
    amplitude: float   # TODO: allow asymmetry 
    direction: str
    _scale: jax.Array
    
    def __init__(self, amplitude: float = 1.0, direction: str = "cw"):
        self.amplitude = amplitude
        self.direction = direction.lower()
        
        if self.direction == "cw":
            _signs = jnp.array([1, -1])
        elif self.direction == "ccw":
            _signs = jnp.array([-1, 1])
        else:
            raise ValueError(f"Invalid curl field direction: {direction}")
        
        self._scale = amplitude * _signs
    
    def __call__(
        self, 
        state: StateT, 
        input: PyTree,  # task inputs
        key: jax.Array,
    ) -> StateT:
        """Return a modified state PyTree."""
        effector_state = state.mechanics.effector
        effector_force = (
            effector_state.force
            + self._curl_force(state.mechanics.effector.vel)
        )

        return self.substate_update(
            state,
            CartesianState2D(
                pos=effector_state.pos,
                vel=effector_state.vel,
                force=effector_force,
            ),
        )

    def _curl_force(self, vel: jax.Array):
        """Return a curl force field."""
        # flip x and y
        return self._scale * vel[..., ::-1]
    
    
class NetworkConstantPerturbation(AbstractIntervenor):
    label: str = "network_perturbation"
    _where = lambda self, tree: tree.network
    
    unit_spec: PyTree
    
    def __init__(
        self, 
        unit_spec, # network state PyTree with constant perturbations
    ):
        self.unit_spec = unit_spec 
        
    def __call__(
        self, 
        state: StateT,
        input: PyTree,  # task inputs
        key: jax.Array,
    ) -> StateT:
        """Return a modified state PyTree."""
        network_state = state.network 
        
        network_state = jax.tree_map(
            lambda x, y: x + y,
            network_state,
            self.unit_spec,
        )
        
        return self.substate_update(
            state,
            network_state,
        )
        