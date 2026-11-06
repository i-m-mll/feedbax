"""Seed declared delayed channels from the initial recurrent source value."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from equinox.nn import State

from feedbax.runtime.channel import Channel
from feedbax.runtime.components import Ravel
from feedbax.runtime.feedback import SelectState
from feedbax.runtime.graph import Component, Graph, Wire


def trial_history_plan(
    nodes: Mapping[str, Component], wires: Sequence[Wire]
) -> _TrialHistoryPlan | None:
    children = tuple(
        name
        for name, node in nodes.items()
        if isinstance(node, Graph) and node.state_consistency_fn is not None
    )
    routes = []
    for wire in wires:
        initializer = wire.recurrent_initializer or {}
        if "seed_channel_history" not in initializer:
            continue
        seed_history = initializer["seed_channel_history"]
        if not isinstance(seed_history, bool):
            raise ValueError(f"{wire} seed_channel_history must be a boolean")
        if not seed_history:
            continue
        if wire.temporality != "recurrent" or initializer.get("kind") != "state_output":
            raise ValueError(f"{wire} seed_channel_history requires a recurrent state_output")
        channel = wire.target_node
        adapter = channel if isinstance(nodes[channel], (Ravel, SelectState)) else None
        if adapter is not None:
            targets = [
                edge.target_node
                for edge in wires
                if edge.source_node == adapter
                and edge.source_port == "output"
                and edge.target_port == "input"
                and edge.temporality == "instant"
                and isinstance(nodes[edge.target_node], Channel)
            ]
            if len(targets) != 1:
                raise ValueError(
                    f"{wire} seed_channel_history requires one Channel after the selection or flattening node"
                )
            channel = targets[0]
        if not isinstance(nodes[channel], Channel):
            raise ValueError(
                f"{wire} seed_channel_history must feed a Channel, optionally through Ravel or SelectState"
            )
        routes.append((channel, wire.source_node, wire.source_port, adapter))

    if not routes and not children:
        return None
    return _TrialHistoryPlan(children, tuple(routes))


@dataclass(frozen=True)
class _TrialHistoryPlan:
    children: tuple[str, ...]
    routes: tuple[tuple[str, str, str, str | None], ...]

    def __call__(self, graph: Graph, state: State) -> State:
        for name in self.children:
            state = graph.nodes[name].state_consistency_update(state)

        sources = dict.fromkeys(source for _, source, _, _ in self.routes)
        for source in sources:
            if source not in self.children:
                state = graph.nodes[source].state_consistency_update(state)

        samples = {}
        for channel_name, source_name, source_port, adapter_name in self.routes:
            source = graph.nodes[source_name]
            outputs = source.initial_outputs(source.state_view(state))
            if source_port not in outputs:
                raise ValueError(
                    f"Channel {channel_name!r} seed_channel_history requires an initial "
                    f"output from {source_name}.{source_port}"
                )
            value = outputs[source_port]
            adapter = graph.nodes[adapter_name] if adapter_name is not None else None
            if isinstance(adapter, SelectState):
                value = adapter.where(value)
            elif isinstance(adapter, Ravel):
                value = adapter.flatten(value)
            samples[channel_name] = value

        for channel_name, sample in samples.items():
            state = cast(Channel, graph.nodes[channel_name]).fill_history(state, sample)
        return state
