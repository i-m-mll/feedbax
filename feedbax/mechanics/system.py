""""""

from typing import Any, Protocol, TypeVar

from jaxtyping import PyTree


T = TypeVar("T")


# TODO maybe this should be `AbstractSystem(eqx.Module)` instead
class System(Protocol):
    def vector_field(
        self, 
        t: float, 
        y: PyTree[T], 
        args: PyTree,  # controls
) -> PyTree[T]:
        """Vector field of the system."""
        pass

    @property
    def control_size(self) -> int:
        """Number of control inputs."""
        pass