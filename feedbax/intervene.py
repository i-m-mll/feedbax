"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
import logging
from typing import TypeVar

import equinox as eqx
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
    

class EffectorCurlForceField(eqx.Module):
    """Apply a curl force field to an effector.
    
    TODO:
    - amalgamate amplitude and _signs, and make this a special case of 
      `EffectorForceField`
    """
    amplitude: float   # TODO: allow asymmetry 
    direction: str
    _signs: jax.Array
    _where = lambda self, tree: tree.mechanics.effector
    label = "effector_curl_field"
    
    def __init__(self, amplitude: float = 1.0, direction: str = "cw"):
        self.direction = direction.lower()
        if self.direction == "cw":
            self._signs = jnp.array([1, -1])
        elif self.direction == "ccw":
            self._signs = jnp.array([-1, 1])
        else:
            raise ValueError(f"Invalid curl field direction: {direction}")
        self.amplitude = amplitude
    
    def __call__(
        self, 
        state: StateT, 
        input: PyTree,  # task inputs
        key: jax.Array,
    ) -> StateT:
        """Return a modified state PyTree."""
        effector_state = self._where(state)
        effector_force = (
            effector_state.force
            + self._curl_force(state.mechanics.effector.vel)
        )
    
        state = eqx.tree_at(
            self._where, 
            state, 
            CartesianState2D(
                pos=effector_state.pos,
                vel=effector_state.vel,
                force=effector_force,
            ),
        )
        return state

    def _curl_force(self, vel: jax.Array):
        """Return a curl force field."""
        # flip x and y
        return (self.amplitude * vel[..., ::-1]) * self._signs
    
    
class EffectorForceField(eqx.Module):
    """Apply a force field to an effector."""
    amplitude: float = 1.0   
    _where = lambda tree: tree.mechanics.effector.force
    label = "effector_force_field"
    
    def __call__(
        self, 
        state: StateT, 
        input: PyTree,  # task inputs
        key: jax.Array,
    ) -> StateT:
        ...