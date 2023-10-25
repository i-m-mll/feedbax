"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
from typing import Any, Protocol, TypeVar

import equinox as eqx
from jaxtyping import Array, Float, PyTree


logger = logging.getLogger(__name__)


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
        ...

    @property
    def control_size(self) -> int:
        """Number of control inputs."""
        ...
    
    def init(self) -> PyTree[T]:
        """Initial state of the system."""
        ...