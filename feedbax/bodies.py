"""Compositions of mechanics, controllers, and channels into sensorimotor loops.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

import equinox as eqx
from equinox import Module
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.channel import Channel, ChannelSpec, ChannelState
from feedbax.filters import FilterState, FirstOrderFilter
from feedbax.graph import Component, Graph, Wire
from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.nn import NetworkState, SimpleStagedNetwork
from feedbax.noise import Normal
from feedbax._tree import tree_sum_n_features

if TYPE_CHECKING:
    from feedbax.task import AbstractTask

logger = logging.getLogger(__name__)


T = TypeVar("T")


class SimpleFeedbackState(Module):
    """State for the SimpleFeedback graph."""

    mechanics: MechanicsState
    net: NetworkState
    feedback: PyTree[ChannelState]
    efferent: ChannelState
    force_filter: FilterState


def _convert_feedback_spec(
    feedback_spec: Union[
        PyTree[ChannelSpec, "T"], PyTree[Mapping[str, Any], "T"]
    ]
) -> PyTree[ChannelSpec, "T"]:
    if isinstance(feedback_spec, ChannelSpec):
        return feedback_spec

    leaves = jt.leaves(
        feedback_spec, is_leaf=lambda x: isinstance(x, (ChannelSpec, Mapping))
    )
    if leaves and all(isinstance(x, ChannelSpec) for x in leaves):
        return feedback_spec
    if leaves and all(isinstance(x, Mapping) for x in leaves):
        return jt.map(
            lambda spec: ChannelSpec(**spec),
            feedback_spec,
            is_leaf=lambda x: isinstance(x, Mapping),
        )

    raise ValueError(f"{type(feedback_spec)} is not a valid feedback spec")


class FeedbackChannels(Component):
    """Bundle of feedback channels with a shared mechanics input."""

    input_ports = ("mechanics",)
    output_ports = ("feedback",)

    channels: PyTree[Channel]
    specs: PyTree[ChannelSpec]

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        mechanics_state = inputs["mechanics"]

        channels_flat, treedef = jt.flatten(
            self.channels, is_leaf=lambda x: isinstance(x, Channel)
        )
        specs_flat = jt.leaves(self.specs, is_leaf=lambda x: isinstance(x, ChannelSpec))
        keys = jr.split(key, len(channels_flat)) if channels_flat else ()

        outputs_flat = []
        for channel, spec, key_i in zip(channels_flat, specs_flat, keys):
            channel_input = spec.where(mechanics_state)
            out, state = channel({"input": channel_input}, state, key=key_i)
            outputs_flat.append(out["output"])

        outputs = jt.unflatten(treedef, outputs_flat)
        return {"feedback": outputs}, state

    def state_view(self, state: eqx.nn.State) -> PyTree[ChannelState]:
        channels_flat, treedef = jt.flatten(
            self.channels, is_leaf=lambda x: isinstance(x, Channel)
        )
        states_flat = [state.get(ch.state_index) for ch in channels_flat]
        return jt.unflatten(treedef, states_flat)

    def fill_queues(
        self,
        state: eqx.nn.State,
        mechanics_state: MechanicsState,
    ) -> eqx.nn.State:
        channels_flat, treedef = jt.flatten(
            self.channels, is_leaf=lambda x: isinstance(x, Channel)
        )
        specs_flat = jt.leaves(self.specs, is_leaf=lambda x: isinstance(x, ChannelSpec))

        for channel, spec in zip(channels_flat, specs_flat):
            channel_state: ChannelState = state.get(channel.state_index)
            value = spec.where(mechanics_state)
            queue = channel.delay * (value,)
            channel_state = eqx.tree_at(
                lambda s: (s.queue, s.output),
                channel_state,
                (queue, value),
            )
            state = state.set(channel.state_index, channel_state)
        return state


class SimpleFeedback(Graph):
    """Graph of feedback channels, a neural network, and mechanics."""

    _feedback_specs: PyTree[ChannelSpec]
    feedback_channels: FeedbackChannels
    mechanics: Mechanics
    net: SimpleStagedNetwork
    efferent_channel: Channel
    force_lp: Optional[FirstOrderFilter]

    def __init__(
        self,
        net: SimpleStagedNetwork,
        mechanics: Mechanics,
        feedback_spec: Union[
            PyTree[ChannelSpec], PyTree[Mapping[str, Any]]
        ] = ChannelSpec(
            where=lambda mechanics_state: mechanics_state.plant.skeleton,  # type: ignore
        ),
        motor_delay: int = 0,
        motor_noise_func: Callable[[PRNGKeyArray, Array], Array] = Normal(),
        tau_rise: float = 0.0,
        tau_decay: float = 0.0,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        feedback_specs = _convert_feedback_spec(feedback_spec)

        plant_state = mechanics.plant.init(key=jr.PRNGKey(0))
        effector = mechanics.plant.skeleton.effector(plant_state.skeleton)
        example_mechanics_state = MechanicsState(
            plant=plant_state,
            effector=effector,
            solver=None,
        )

        def _build_feedback_channel(spec: ChannelSpec):
            return Channel(
                delay=spec.delay,
                noise_func=spec.noise_func,
                init_value=0.0,
            ).change_input(spec.where(example_mechanics_state))

        feedback_channels = jt.map(
            lambda spec: _build_feedback_channel(spec),
            feedback_specs,
            is_leaf=lambda x: isinstance(x, ChannelSpec),
        )
        feedback = FeedbackChannels(feedback_channels, feedback_specs)

        efferent = Channel(
            delay=motor_delay,
            noise_func=motor_noise_func,
            init_value=0.0,
            input_proto=jnp.zeros(net.out_size),
        )

        if tau_rise == 0.0 and tau_decay == 0.0:
            force_filter = None
        else:
            force_filter = FirstOrderFilter(
                tau_rise=tau_rise,
                tau_decay=tau_decay,
                dt=mechanics.dt,
                init_value=0.0,
                input_proto=jnp.zeros(net.out_size),
            )

        nodes = {
            "feedback": feedback,
            "net": net,
            "efferent": efferent,
            "mechanics": mechanics,
        }
        wires = [
            Wire("feedback", "feedback", "net", "feedback"),
            Wire("net", "output", "efferent", "input"),
        ]

        if force_filter is None:
            wires.append(Wire("efferent", "output", "mechanics", "force"))
        else:
            nodes["force_filter"] = force_filter
            wires.append(Wire("efferent", "output", "force_filter", "input"))
            wires.append(Wire("force_filter", "output", "mechanics", "force"))

        # Cycle: mechanics output feeds feedback input
        wires.append(Wire("mechanics", "effector", "feedback", "mechanics"))

        def _state_view(node_states):
            force_filter_state = node_states.get(
                "force_filter", FilterState(output=None, solver=None)
            )
            return SimpleFeedbackState(
                mechanics=node_states["mechanics"],
                net=node_states["net"],
                feedback=node_states["feedback"],
                efferent=node_states["efferent"],
                force_filter=force_filter_state,
            )

        def _consistency_update(state):
            mechanics_state: MechanicsState = state.get(mechanics.state_index)
            new_skeleton = mechanics.plant.skeleton.inverse_kinematics(
                mechanics_state.effector
            )
            mechanics_state = eqx.tree_at(
                lambda s: s.plant.skeleton,
                mechanics_state,
                new_skeleton,
            )
            state = state.set(mechanics.state_index, mechanics_state)
            return feedback.fill_queues(state, mechanics_state)

        super().__init__(
            nodes=nodes,
            wires=tuple(wires),
            input_ports=("input",),
            output_ports=("effector",),
            input_bindings={"input": ("net", "input")},
            output_bindings={"effector": ("mechanics", "effector")},
            state_view_fn=_state_view,
            state_consistency_fn=_consistency_update,
        )

        self._feedback_specs = feedback_specs
        self.feedback_channels = feedback
        self.mechanics = mechanics
        self.net = net
        self.efferent_channel = efferent
        self.force_lp = force_filter

    @staticmethod
    def get_nn_input_size(
        task: "AbstractTask",
        mechanics: Mechanics,
        feedback_spec: Union[
            PyTree[ChannelSpec[MechanicsState]], PyTree[Mapping[str, Any]]
        ] = ChannelSpec(
            where=lambda mechanics_state: mechanics_state.plant.skeleton
        ),
    ) -> int:
        plant_state = mechanics.plant.init(key=jr.PRNGKey(0))
        effector = mechanics.plant.skeleton.effector(plant_state.skeleton)
        example_mechanics_state = MechanicsState(
            plant=plant_state,
            effector=effector,
            solver=None,
        )
        example_feedback = jt.map(
            lambda spec: spec.where(example_mechanics_state),
            _convert_feedback_spec(feedback_spec),
            is_leaf=lambda x: isinstance(x, ChannelSpec),
        )
        n_feedback = tree_sum_n_features(example_feedback)
        example_trial_spec = task.get_train_trial_with_intervenor_params(
            key=jr.PRNGKey(0)
        )
        n_task_inputs = tree_sum_n_features(example_trial_spec.inputs)
        return n_feedback + n_task_inputs
