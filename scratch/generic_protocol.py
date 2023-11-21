

from typing import Protocol, TypeVar, runtime_checkable

import equinox as eqx
from typeguard import check_type


class AbstractSystemState(eqx.Module):
    ...
    
    
StateT = TypeVar("StateT", bound=AbstractSystemState)


@runtime_checkable
class System(Protocol[StateT]):
    def vector_field(
        self,
        state: StateT,
    ) -> StateT:
        """Vector field of the system."""
        ...


class SystemState(AbstractSystemState):
    a: int 
    

class SomeSystem(eqx.Module):
    def vector_field(
        self, 
        state: SystemState
    ) -> SystemState:
        return SystemState(0)


system = SomeSystem()

check_type(system, System[SystemState])