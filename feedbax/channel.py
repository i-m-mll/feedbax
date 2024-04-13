"""Queue-based modules for modeling distant, possibly noisy connections.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import logging
from typing import Generic, Optional, Self, Tuple, Union

import equinox as eqx
from equinox import Module, field
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.intervene import AbstractIntervenor
from feedbax.noise import Normal
from feedbax._staged import AbstractStagedModel, ModelStage
from feedbax.state import AbstractState, StateT
from feedbax._tree import random_split_like_tree


logger = logging.getLogger(__name__)


class ChannelState(AbstractState):
    """Type of state PyTree operated on by [`Channel`][feedbax.channel.Channel] instances.

    Attributes:
        output: The current output of the channel.
        queue: A tuple of previous inputs to the channel, with the most recent appearing last.
        noise: The noise added to the current output, if any.
    """

    output: Optional[PyTree[Array, "T"]]
    queue: Tuple[Optional[PyTree[Array, "T"]], ...]
    noise: Optional[PyTree[Array, "T"]] = None


class ChannelSpec(Module, Generic[StateT]):
    """Specifies how to build a [`Channel`][feedbax.channel.Channel], with respect to the state PyTree of its owner.

    Attributes:
        where: A function that selects the subtree of feedback states.
        delay: The number of previous inputs to store in the queue.
        noise_std: The standard deviation of the noise to add to the output.
    """

    where: Callable[[StateT], PyTree[Array]]
    delay: int = 0
    noise_func: Optional[Callable[[PRNGKeyArray, Array], Array]] = None


class Channel(AbstractStagedModel[ChannelState]):
    """A noisy queue.

    !!! NOTE
        This can be used for modeling an axon, tract, wire, or other delayed
        and semi-reliable connection between model components.

    Attributes:
        delay: The number of previous inputs stored in the queue. May be zero.
        noise_std: The standard deviation of the noise added to the output.
            If `None`, no noise is added.
        input_proto: A PyTree of arrays with the same structure/shapes as the
            inputs to the channel will have.
        intervenors: [Intervenors][feedbax.intervene.AbstractIntervenor] to add
            to the model at construction time.
    """

    delay: int
    add_noise: bool = True
    noise_func: Callable[[PRNGKeyArray, Array], Array] = Normal()
    input_proto: PyTree[Array] = field(default_factory=lambda: jnp.zeros(1))
    init_value: float = 0
    intervenors: Mapping[str, Sequence[AbstractIntervenor]] = field(init=False)

    # def __init__(
    #     self,
    #     delay: int,
    #     noise_std: Optional[float] = None,
    #     init_value: float = 0.,  # jnp.nan
    #     input_proto: PyTree[Array] = jnp.zeros(1),
    #     intervenors: Optional[
    #         Union[
    #             Sequence[AbstractIntervenor], Mapping[str, Sequence[AbstractIntervenor]]
    #         ]
    #     ] = None,
    # ):
    #     self.delay = delay  # otherwise when delay=0, nothing is stored
    #     self.noise_std = noise_std
    #     self.init_value = init_value
    #     self.input_proto = input_proto

    def __post_init__(self):
        self.intervenors = self._get_intervenors_dict({})

    def __check_init__(self):
        if not isinstance(self.delay, int):
            raise ValueError("Delay must be an integer")

    def _update_queue(
        self, input: PyTree[Array], state: ChannelState, *, key: PRNGKeyArray
    ):
        return ChannelState(
            output=state.queue[0],
            queue=state.queue[1:] + (input,),
        )

    def _update_queue_zerodelay(
        self, input: PyTree[Array], state: ChannelState, *, key: PRNGKeyArray
    ):
        return ChannelState(
            output=input,
            queue=state.queue,
        )

    def _add_noise(self, input, state, *, key):
        noise = jax.tree_map(
            self.noise_func,
            random_split_like_tree(key, input),
            input,
        )
        output = jax.tree_map(lambda x, y: x + y, input, noise)
        return noise, output

    @property
    def model_spec(self) -> OrderedDict[str, ModelStage[Self, ChannelState]]:
        """Returns an `OrderedDict` that specifies the stages of the channel model.

        Always includes a queue input-output stage. Optionally includes
        a stage that adds noise to the output, if `noise_std` was not
        `None` at construction time.
        """
        Stage = ModelStage[Self, ChannelState]

        update_queue = (
            self._update_queue
            if self.delay > 0
            else self._update_queue_zerodelay
        )

        spec = OrderedDict(
            {
                "update_queue": Stage(
                    callable=lambda self: update_queue,
                    where_input=lambda input, state: input,
                    where_state=lambda state: state,
                ),
            }
        )

        if self.add_noise:
            spec |= {
                "add_noise": Stage(
                    callable=lambda self: self._add_noise,
                    where_input=lambda input, state: state.output,
                    where_state=lambda state: (state.noise, state.output),
                ),
            }

        return spec

    @property
    def memory_spec(self) -> PyTree[bool]:
        return ChannelState(
            output=True,
            queue=False,  # type: ignore
            noise=False,
        )

    def init(self, *, key: PRNGKeyArray) -> ChannelState:
        """Returns an empty `ChannelState` for the channel."""
        input_init = jax.tree_map(
            lambda x: jnp.full_like(x, self.init_value), self.input_proto
        )
        if self.add_noise:
            noise_init = input_init
        else:
            noise_init = None

        return ChannelState(
            output=input_init,
            queue=self.delay * (input_init,),
            noise=noise_init,
        )

    def change_input(self, input_proto: PyTree[Array]) -> "Channel":
        """Returns a similar `Channel` with a changed input structure."""
        return eqx.tree_at(lambda channel: channel.input_proto, self, input_proto)
