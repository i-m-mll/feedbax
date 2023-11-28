"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging

from equinox import AbstractVar
import jax
from jax import Array
from feedbax.mechanics.muscle import AbstractMuscleState

from feedbax.model import AbstractModel, AbstractModelState


logger = logging.getLogger(__name__)


class AbstractSkeletonState(AbstractModelState):
    ...
    
# class AbstractPlantState(AbstractModelState):
#     ...


# class AbstractPlant(AbstractModel):
    
#     @property
#     def model_spec(self):
#         ...
        
#     @property 
#     def memory_spec(self):
#         ...
        
#     def init(self):
#         ...


class MuscledPlantState(AbstractModelState):
    skeleton: AbstractSkeletonState
    muscles: AbstractMuscleState
    

class MuscledPlant(AbstractModel):
    
    @property
    def model_spec(self):
        ...
        
    @property 
    def memory_spec(self):
        ...
        
    def init(self):
        ...