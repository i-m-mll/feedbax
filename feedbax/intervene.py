"""Modules that modify parts of PyTrees.

TODO:
- The type signature is backwards, compared to `AbstractModel`.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

#! Equinox 0.11.2 doesn't support stringified annotations of abstract vars
#! This is why we can't properly annotate `_in` and `_out` fields below.
# from __future__ import annotations

from abc import abstractmethod
import logging
from typing import TYPE_CHECKING, Callable, Generic, LiteralString, Optional, TypeVar

import equinox as eqx
from equinox import AbstractClassVar, AbstractVar, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

if TYPE_CHECKING:
    from feedbax.networks import NetworkState

from feedbax.state import AbstractState, CartesianState2D


logger = logging.getLogger(__name__)


StateT = TypeVar("StateT", bound=AbstractState)


class AbstractIntervenor(eqx.Module, Generic[StateT]):
    """Base class for modules that intervene on a model's state.
    
    Concrete subclasses must define `_out` and `_in` class variables, which 
    are callables that return the substate to be updated and the substate to 
    use as input to the update method, respectively.
    
    Instances of the concrete subclasses of this class are intended to be
    registered with the `register_intervention` method of an instance of an
    `AbstractModel`, which causes the model to invoke the intervention 
    during model execution.
    
    NOTE: 
    - Interestingly, this has the same call signature as `AbstractModel`.
      This might be relevant to generalizing the `AbstractModel` class and 
      the notion of model-model interactions.
        - Each "stage" of an `AbstractModel` is a modification of a substate.
    - `_get_updated_substate` in `AbstractClampIntervenor` and 
      `_get_substate_to_add` in `AbstractAdditiveIntervenor` serve a similar 
      role and have the same signature, but are defined separately for the 
      sake of clearer naming.
      
    TODO:
    - Should `Intervenor` only operate on a specific array in a state PyTree?    
      That is, should `_out` and `_in` return `Array` instead of `PyTree`?
    """
    
    label: AbstractVar[str]
    
    @abstractmethod
    def _out(self, state: StateT) -> PyTree:
        """Return the substate to be updated."""
        ...
    
    @abstractmethod
    def _in(self, state: StateT) -> PyTree:
        """Return the substate to be used as input."""
        ...
    
    def update_substate(
        self,
        state: StateT, 
        substate: PyTree,      
    ) -> StateT:
        """Return a modified state PyTree."""
        return eqx.tree_at(
            self._out, 
            state, 
            substate,
        )


HasSubstateT = TypeVar("HasSubstateT", bound=PyTree)


class AbstractClampIntervenor(AbstractIntervenor[StateT]):
    """Base class for intervenors that replace part of the substate."""
    
    def __call__(
        self, 
        input: PyTree,  # task inputs
        state: HasSubstateT, 
        *,
        key: jax.Array,
    ) -> HasSubstateT:
        """Return a modified state PyTree."""
        new_substate = self._get_updated_substate(self._in(state), key=key)
        return self.update_substate(state, new_substate)
    
    @abstractmethod
    def _get_updated_substate(
        self, 
        substate_in: PyTree, 
        *, 
        key: Optional[jax.Array] = None
    ) -> PyTree:
        """Return a modified substate PyTree."""
        ...
    

class AbstractAdditiveIntervenor(AbstractIntervenor[StateT]):
    """Base class for intervenors that add updates to a substate."""
    
    def __call__(
        self, 
        input: PyTree,  # task inputs
        state: HasSubstateT, 
        *,
        key: jax.Array,
    ) -> HasSubstateT:
        """Return a modified state PyTree."""
        new_substate = jax.tree_map(
            lambda x, y: x + y,
            self._out(state),
            self._get_substate_to_add(self._in(state), key=key),
        )
        return self.update_substate(state, new_substate)

    @abstractmethod
    def _get_substate_to_add(
        self, 
        substate_in: PyTree, 
        *, 
        key: Optional[jax.Array] = None
    ) -> PyTree:
        """Return a modified substate PyTree."""
        ...


class AddNoise(AbstractAdditiveIntervenor[StateT]):
    """Add noise to a substate. Default is standard normal noise."""
    amplitude: float = 1.0
    noise_func: Callable = jax.random.normal
    
    label: str = "noise"
    
    def _out(self, state: StateT):
        return state
    
    def _in(self, state: StateT):
        return state
    
    def _get_substate_to_add(
        self, 
        substate_in: PyTree, 
        *, 
        key: jax.Array
    ):
        return jax.tree_map(
            lambda x: self.noise_func(
                key=key, 
                shape=x.shape, 
                dtype=x.dtype,
            ) * self.amplitude,
            substate_in,
        ) 


class EffectorVelDepForceField(AbstractAdditiveIntervenor):
    """Apply a force field to an effector."""
    _rot_scale: jax.Array
    _expand_scale: jax.Array
    
    label: str = "effector_force_field"
    
    amplitude: float = 1.0   
    _out = lambda tree: tree.mechanics.effector.force
    _in = lambda tree: tree.mechanics.effector.vel

    def _get_substate_to_add(self, vel: jax.Array):
        """Return velocity-dependent forces."""
        rot_forces = self._rot_scale * vel[..., ::-1]
        expand_force = self._expand_scale * vel 
        return rot_forces + expand_force
    

class EffectorCurlForceField(AbstractAdditiveIntervenor):
    """Apply a curl force field to an effector.
    
    TODO:
    - Special case of `EffectorVelDepForceField`?
    """
    amplitude: float   # TODO: allow asymmetry 
    direction: str
    _scale: jax.Array# = field(static=True)
    
    label: str = "effector_curl_field"
    
    _in = lambda self, tree: tree.mechanics.effector.vel 
    _out = lambda self, tree: tree.mechanics.effector.force
        
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

    def _get_substate_to_add(self, vel: Array, *, key: jax.Array):
        """Returns curl forces."""
        return self._scale * vel[..., ::-1]  
    
    
class NetworkConstantInputPerturbation(AbstractAdditiveIntervenor["NetworkState"]):
    """Adds a constant input to a network unit's states.
    
    NOTE:
    Originally this was intended to modify an arbitrary PyTree of network 
    states. See `NetworkClamp` for more details.
    
    Args:
        unit_spec (PyTree[Array]): A PyTree with the same structure as the
        network state, to be added to the network state.
    """
    unit_spec: PyTree
    
    label: str = "network_input_perturbation"
    
    def _in(self, tree: "NetworkState"):
        return tree.hidden
    
    def _out(self, tree: "NetworkState"):
        return tree.hidden
    
    def __init__(self, unit_spec: PyTree):
        self.unit_spec = jax.tree_map(jnp.nan_to_num, unit_spec)
    
    def _get_substate_to_add(self, network_state: PyTree, *, key: jax.Array):
        """Return a modified network state PyTree."""
        return self.unit_spec


class NetworkClamp(AbstractClampIntervenor["NetworkState"]):
    """Replaces part of a network's state with constant values.

    NOTE:
    Originally this was intended to modify an arbitrary PyTree of network 
    states, which is why we use `tree_map` to replace parts of `network_state`
    with parts of `unit_spec`. However, currently it is only used to modify
    `NetworkState.hidden`, which is a single JAX array. Thus `unit_spec`
    should also be a single JAX array.

    Args:
        unit_spec (PyTree[Array]): A PyTree with the same structure as the
        network state, where all array values are NaN except for the constant
        values to be clamped.
    """
    unit_spec: PyTree
    
    label: str = "network_clamp"
    
    def _in(self, state: "NetworkState"):
        return state.hidden
    
    def _out(self, state: "NetworkState"):
        return state.hidden
    
    def _get_updated_substate(self, network_state: PyTree, *, key: jax.Array):
        """Return a modified network state PyTree."""
        return jax.tree_map(
            lambda x, y: jnp.where(jnp.isnan(y), x, y),
            network_state,
            self.unit_spec,
        )

