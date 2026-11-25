"""Base classes for continuous dynamical systems.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from typing import Generic

from equinox import Module
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from feedbax.runtime.state import StateT


class AbstractDynamicalSystem(Module, Generic[StateT]):
    """Base class for continuous dynamical systems.

    ??? dev-note "Development note"
        The signature of `vector_field` matches that expected by
        `diffrax.ODETerm`. However, the argument that is called
        `args` in the Diffrax documentation, we call `input`, since
        we use it for input signals (e.g. forces from the controller)

        Vector fields for biomechanical models are generally not
        time-dependent. That is, the argument `t` to `vector_field` is
        typically unused. This is apparent in the way we alias `vector_field`
        to `__call__`, which is the call surface expected of components.
    """

    def __call__(
        self,
        input_: PyTree[Array],
        state: StateT,
        key: PRNGKeyArray,
    ) -> StateT:
        """Alias for `vector_field`, with a modified signature."""
        return self.vector_field(jnp.array(0.), state, input_)

    @abstractmethod
    def vector_field(
        self,
        t: Scalar,
        state: StateT,
        input: PyTree[Array],  # controls
    ) -> StateT:
        """Returns scalar (e.g. time) derivatives of the system's states."""
        ...

    @abstractproperty
    def input_size(self) -> int:
        """Number of input variables."""
        ...

    @abstractmethod
    def init(self, *, key: PRNGKeyArray) -> StateT:
        """Returns the initial state of the system."""
        ...
