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
from jaxtyping import Array, Float, PyTree

if TYPE_CHECKING:
    from feedbax.model import AbstractStagedModel
    from feedbax.mechanics.mechanics import MechanicsState
    from feedbax.networks import NetworkState

from feedbax.state import AbstractState, CartesianState2D


logger = logging.getLogger(__name__)


StateT = TypeVar("StateT", bound=AbstractState)

class AbstractIntervenorInput(eqx.Module):
    active: AbstractVar[bool]
    
    
InputT = TypeVar("InputT", bound=AbstractIntervenorInput)    


class AbstractIntervenor(eqx.Module, Generic[StateT, InputT]):
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
    
    params: AbstractVar[InputT]
    in_where: AbstractVar[Callable[[StateT], PyTree]]
    out_where: AbstractVar[Callable[[StateT], PyTree]]
    label: AbstractVar[str]
    
    def __call__(self, input, state, *, key):
        """Return a modified state PyTree."""
        params = eqx.combine(input, self.params)
        return eqx.tree_at(
            self.out_where,
            state, 
            self.intervene(params, state)
        )
    
    @abstractmethod
    def intervene(params, state):
        ...        
    

class CurlForceFieldParams(AbstractIntervenorInput):
    amplitude: Optional[float] = 0.
    active: Optional[bool] = False
    

class CurlForceField(AbstractIntervenor["MechanicsState", CurlForceFieldParams]):
    """Apply a curl force field to an effector.
    
    Positive amplitude corresponds to a counterclockwise curl, in keeping
    with the conventional sign of rotation in the XY plane.
    
    TODO:
    - Special case of `EffectorVelDepForceField`?
    """
    
    params: CurlForceFieldParams = CurlForceFieldParams()
    in_where: Callable[["MechanicsState"], Float[Array, '... ndim=2']] = lambda state: state.effector.vel 
    out_where: Callable[["MechanicsState"], Float[Array, '... ndim=2']] = lambda state: state.effector.force
    label: str = "CurlForceField"
    
    def intervene(self, params: CurlForceFieldParams, state: "MechanicsState"):
        return params.amplitude * jnp.array([-1, 1]) * self.in_where(state)[..., ::-1]
    
    # amplitude: float   # TODO: allow asymmetry 
    # _scale: jax.Array
    
    # label: str = "effector_curl_field"
    
    # in_where: Callable[[MechanicsState], PyTree]
    # lambda self, tree: tree.effector.vel 
    # _out = lambda self, tree: tree.effector.force
        
    # def __init__(self, amplitude: float = 1.0):
    #     self.amplitude = amplitude
    #     self._scale = amplitude * jnp.array([-1, 1])

    # def _get_substate_to_add(self, vel: Array, *, key: jax.Array):
    #     """Returns curl forces."""
    #     return self._scale * vel[..., ::-1]  
    
