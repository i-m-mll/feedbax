"""Queue-based modules for modeling distant, possibly noisy connections.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable
import dataclasses
import logging
from typing import Generic, Optional, Tuple, TypeVar

import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.graph import Component
from feedbax.noise import Normal
from feedbax._tree import random_split_like_tree


logger = logging.getLogger(__name__)


class ChannelState(Module):
    """State for a delay/noise channel.

    Attributes:
        output: The current output of the channel.
        queue: A tuple of previous inputs to the channel, with most recent last.
        noise: The noise added to the current output, if any.
    """

    output: PyTree[Array, "T"]
    queue: Tuple[Optional[PyTree[Array, "T"]], ...]
    noise: Optional[PyTree[Array, "T"]]


StateT = TypeVar("StateT")


class ChannelSpec(Module, Generic[StateT]):
    """Specifies how to build a feedback channel from a state.

    Attributes:
        where: A function that selects the subtree of feedback states.
        delay: The number of previous inputs to store in the queue.
        noise_func: Optional noise function.
    """

    where: Callable[[StateT], PyTree[Array]]
    delay: int = 0
    noise_func: Optional[Callable[[PRNGKeyArray, Array], Array]] = None


class Channel(Component):
    """A noisy delay line."""

    input_ports = ("input",)
    output_ports = ("output",)

    delay: int
    noise_func: Optional[Callable[[PRNGKeyArray, Array], Array]]
    add_noise: bool
    input_proto: PyTree[Array]
    init_value: float
    state_index: StateIndex
    _initial_state: ChannelState = field(static=True)

    def __init__(
        self,
        delay: int,
        noise_func: Optional[Callable[[PRNGKeyArray, Array], Array]] = Normal(),
        add_noise: bool = True,
        input_proto: Optional[PyTree[Array]] = None,
        init_value: float = 0.0,
    ):
        if not isinstance(delay, int):
            raise ValueError("Delay must be an integer")
        self.delay = delay
        self.noise_func = noise_func
        self.add_noise = add_noise
        if input_proto is None:
            input_proto = jnp.zeros(1)
        self.input_proto = input_proto
        self.init_value = init_value

        self._initial_state = self._initial_state_value(input_proto)
        self.state_index = StateIndex(self._initial_state)

    def _initial_state_value(self, input_proto: PyTree[Array]) -> ChannelState:
        input_init = jt.map(lambda x: jnp.full_like(x, self.init_value), input_proto)
        if not self.add_noise or self.noise_func is None:
            noise_init = None
        else:
            noise_init = input_init
        queue = self.delay * (input_init,)
        return ChannelState(output=input_init, queue=queue, noise=noise_init)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        channel_state: ChannelState = state.get(self.state_index)
        input_value = inputs["input"]

        if self.delay > 0:
            output = channel_state.queue[0]
            new_queue = channel_state.queue[1:] + (input_value,)
        else:
            output = input_value
            new_queue = channel_state.queue

        noise = None
        if self.add_noise and self.noise_func is not None:
            noise = jt.map(
                self.noise_func,
                random_split_like_tree(key, output),
                output,
            )
            output = jt.map(lambda x, y: x + y, output, noise)

        new_state = ChannelState(output=output, queue=new_queue, noise=noise)
        state = state.set(self.state_index, new_state)
        return {"output": output}, state

    def change_input(self, input_proto: PyTree[Array]) -> "Channel":
        """Return a similar Channel with a changed input structure."""
        new_initial_state = self._initial_state_value(input_proto)
        new_state_index = StateIndex(new_initial_state)
        return dataclasses.replace(
            self,
            input_proto=input_proto,
            state_index=new_state_index,
            _initial_state=new_initial_state,
        )


def toggle_channel_noise(tree, enabled: Optional[bool] = None):
    """Disable/enable noise in all Channel leaves of a PyTree."""
    if enabled is None:
        replace_fn = lambda x: not x
    else:
        replace_fn = lambda _: enabled

    return eqx.tree_at(
        lambda channel: channel.add_noise,
        tree,
        replace_fn,
        is_leaf=lambda x: isinstance(x, Channel),
    )
