"""Compositions of mechanics, controllers, and channels into sensorimotor loops.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast

import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from equinox import Module
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.runtime.channel import Channel, ChannelSpec, ChannelState, SignalChannel, _noise_model
from feedbax.runtime.feedback import (
    FEEDBACK_INPUT,
    FEEDBACK_OUTPUT,
    AssembleFeedback,
    SelectState,
    StateSelection,
    StateSelector,
    feedback_node_names,
    feedback_wires,
)
from feedbax.runtime.filters import FilterState, FirstOrderFilter, LowPassFilter
from feedbax.runtime.graph import Component, ComponentBinding, Graph, Wire
from feedbax.runtime.noise import Normal
from feedbax.runtime.trial_history import trial_history_plan

if TYPE_CHECKING:
    from feedbax.tasks import AbstractTask

T = TypeVar("T")


def _tree_sum_n_features(tree: PyTree[Array]) -> int:
    return jt.reduce(lambda x, y: x + y, jt.map(lambda x: x.shape[-1], tree))


class SimpleFeedbackState(Module):
    """State for the mechanics, native controller, and both signal paths."""

    mechanics: MechanicsState
    net: PyTree
    feedback: PyTree[ChannelState]
    efferent: ChannelState
    force_filter: FilterState


def _convert_feedback_spec(
    feedback_spec: Union[PyTree[ChannelSpec, "T"], PyTree[Mapping[str, Any], "T"]],
) -> PyTree[ChannelSpec, "T"]:
    if isinstance(feedback_spec, ChannelSpec):
        return feedback_spec

    def is_spec(value):
        return isinstance(value, ChannelSpec) or (
            isinstance(value, Mapping) and callable(value.get("where"))
        )

    leaves = jt.leaves(feedback_spec, is_leaf=is_spec)
    if not leaves or not all(is_spec(value) for value in leaves):
        raise ValueError(f"{type(feedback_spec)} is not a valid feedback spec")
    return jt.map(
        lambda value: value if isinstance(value, ChannelSpec) else ChannelSpec(**value),
        feedback_spec,
        is_leaf=is_spec,
    )


class SimpleFeedback(Graph):
    """Graph of feedback channels, a neural network, and mechanics."""

    def __init__(
        self,
        net: Component,
        mechanics: Mechanics,
        feedback_spec: Union[PyTree[ChannelSpec], PyTree[Mapping[str, Any]]] = ChannelSpec(
            where=StateSelector(("plant.skeleton",)),
        ),
        motor_delay: int = 0,
        motor_noise_func: Callable[[PRNGKeyArray, Array], Array] | None = Normal(),
        tau_rise: float = 0.0,
        tau_decay: float = 0.0,
        *,
        key: Optional[PRNGKeyArray] = None,
        motor_input_proto: Array | None = None,
    ):
        specs, layout = jt.flatten(
            _convert_feedback_spec(feedback_spec),
            is_leaf=lambda value: isinstance(value, ChannelSpec),
        )
        example_state = mechanics.state_index.init
        nodes: dict[str, Component] = {"feedback_values": AssembleFeedback(layout)}
        for (selector_name, channel_name), spec in zip(feedback_node_names(len(specs)), specs):
            selector_type = SelectState if isinstance(spec.where, StateSelector) else StateSelection
            nodes[selector_name] = selector_type(spec.where)
            channel_type = Channel if _noise_model(spec.noise_func) != "custom" else SignalChannel
            nodes[channel_name] = channel_type(
                delay=spec.delay,
                noise_func=spec.noise_func,
                input_proto=spec.where(example_state),
                noise_role="sensory_feedback",
                noise_timing="pre_controller",
            )

        efferent_type = Channel if _noise_model(motor_noise_func) != "custom" else SignalChannel
        efferent = efferent_type(
            delay=motor_delay,
            noise_func=motor_noise_func,
            init_value=0.0,
            input_proto=(
                jnp.zeros(mechanics.plant.input_size)
                if motor_input_proto is None
                else motor_input_proto
            ),
            noise_role="motor_command",
            noise_timing="post_controller",
        )

        if tau_rise == 0.0 and tau_decay == 0.0:
            force_filter = None
        else:
            filter_type = (
                FirstOrderFilter if hasattr(efferent.input_proto, "shape") else LowPassFilter
            )
            force_filter = filter_type(
                tau_rise=tau_rise,
                tau_decay=tau_decay,
                dt=mechanics.dt,
                init_value=0.0,
                input_proto=efferent.input_proto,
            )

        nodes.update(net=net, efferent=efferent, mechanics=mechanics)
        if force_filter is not None:
            nodes["force_filter"] = force_filter
        wires = feedback_wires(len(specs), force_filter=force_filter is not None)
        super().__init__(
            nodes=nodes,
            wires=wires,
            input_ports=("input",),
            output_ports=("effector",),
            input_bindings={"input": FEEDBACK_INPUT},
            output_bindings={"effector": FEEDBACK_OUTPUT},
            component_bindings=_component_bindings(net),
            state_consistency_fn=trial_history_plan(nodes, wires),
            state_view_fn=feedback_state_view(nodes, wires),
        )

    @property
    def mechanics(self) -> Mechanics:
        return cast(Mechanics, self.nodes["mechanics"])

    @property
    def net(self) -> Component:
        return self.nodes["net"]

    @property
    def efferent_channel(self) -> SignalChannel:
        return cast(SignalChannel, self.nodes["efferent"])

    @property
    def force_lp(self) -> Optional[LowPassFilter]:
        return cast(LowPassFilter | None, self.nodes.get("force_filter"))

    @staticmethod
    def get_nn_input_size(
        task: "AbstractTask",
        mechanics: Mechanics,
        feedback_spec: Union[
            PyTree[ChannelSpec[MechanicsState]], PyTree[Mapping[str, Any]]
        ] = ChannelSpec(where=StateSelector(("plant.skeleton",))),
    ) -> int:
        example_mechanics_state = mechanics.state_index.init
        example_feedback = jt.map(
            lambda spec: spec.where(example_mechanics_state),
            _convert_feedback_spec(feedback_spec),
            is_leaf=lambda x: isinstance(x, ChannelSpec),
        )
        n_feedback = _tree_sum_n_features(example_feedback)
        example_trial_spec = task.sample_trial(key=jr.PRNGKey(0))
        n_task_inputs = _tree_sum_n_features(example_trial_spec.inputs)
        return n_feedback + n_task_inputs


def feedback_state_view(
    nodes: Mapping[str, Component],
    wires: Sequence[Wire],
    *,
    input_bindings: Mapping[str, tuple[str, str]] | None = None,
    output_bindings: Mapping[str, tuple[str, str]] | None = None,
) -> Callable[[dict[str, PyTree]], PyTree] | None:
    """Expose the state projection of the canonical feedback graph."""
    assembly = nodes.get("feedback_values")
    if not isinstance(assembly, AssembleFeedback):
        if (
            set(nodes) == {"select", "channel"}
            and isinstance(nodes["select"], StateSelection)
            and isinstance(nodes["channel"], SignalChannel)
            and tuple(wires) == (Wire("select", "output", "channel", "input"),)
            and input_bindings == {"mechanics": ("select", "state")}
            and output_bindings == {"feedback": ("channel", "output")}
        ):
            return _selected_channel_state
        return None
    channel_names = tuple(channel for _, channel in feedback_node_names(len(assembly.input_ports)))
    expected = feedback_wires(len(channel_names), force_filter="force_filter" in nodes)
    if not set(expected).issubset(wires):
        return None
    return _FeedbackStateView(assembly.structure, channel_names)


def _selected_channel_state(node_states: dict[str, PyTree]) -> PyTree:
    return node_states["channel"]


@dataclass(frozen=True)
class _FeedbackStateView:
    structure: Any
    channel_names: tuple[str, ...]

    def __call__(self, node_states: dict[str, PyTree]) -> SimpleFeedbackState:
        return SimpleFeedbackState(
            mechanics=node_states["mechanics"],
            net=node_states.get("net"),
            feedback=self.structure.unflatten([node_states[name] for name in self.channel_names]),
            efferent=node_states["efferent"],
            force_filter=node_states.get("force_filter", FilterState(output=None, solver=None)),
        )


def _component_bindings(net: Component) -> dict[str, ComponentBinding]:
    if not isinstance(net, Graph):
        return {}
    from feedbax.component_registry import ComponentRegistry
    from feedbax.runtime.graph import Subgraph

    owner = ComponentRegistry().get(f"feedbax.{Subgraph.__name__}")
    if owner is None:
        raise RuntimeError("The built-in Subgraph owner is unavailable")
    return {"net": ComponentBinding(owner.name, owner.revision)}
