"""Selection and wiring for explicit feedback paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, ClassVar

import jax.tree as jt
from jaxtyping import PyTree
from pydantic import BaseModel, ConfigDict
from pydantic import Field as ConfigField

from feedbax.contracts.component import PortType
from feedbax.contracts.model_values import ModelFacts
from feedbax.contracts.portable import ComponentOptions, Ports
from feedbax.runtime.graph import Component, GraphNode, Wire
from feedbax.runtime.prototypes import _required_input


@dataclass(frozen=True)
class StateSelector:
    """Select one state subtree or a tuple of subtrees by dotted attribute paths."""

    paths: tuple[str, ...]

    def __post_init__(self):
        if not self.paths or any(
            not isinstance(path, str) or not all(part.isidentifier() for part in path.split("."))
            for path in self.paths
        ):
            raise ValueError("StateSelector requires nonempty dotted attribute paths")

    def __call__(self, state: PyTree) -> PyTree:
        values = []
        for path in self.paths:
            try:
                value = attrgetter(path)(state)
            except AttributeError as exc:
                raise ValueError(
                    f"StateSelector path {path!r} has no attribute on {type(state).__name__}"
                ) from exc
            leaves = jt.leaves(value)
            if not leaves or not all(
                hasattr(leaf, "shape") and hasattr(leaf, "dtype") for leaf in leaves
            ):
                raise ValueError(
                    f"StateSelector path {path!r} must select a state or signal containing array leaves"
                )
            values.append(value)
        return values[0] if len(values) == 1 else tuple(values)


def _extract_state_selector_params(component: SelectState, model: Any) -> Any:
    if not isinstance(component.where, StateSelector):
        raise ValueError(
            "Portable SelectState requires a StateSelector with declared attribute paths"
        )
    return {"paths": list(component.where.paths)}


def state_selector_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    state = _required_input(inputs, "state", component="SelectState")
    return {"output": StateSelector(tuple(params["paths"]))(state)}


class _SelectStateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    paths: Any = ConfigField(default=["effector"], json_schema_extra={"feedbax": {"type": "array"}})


class StateSelection(GraphNode):
    """Expose a state selector as a stateless graph node."""

    ports: ClassVar[Ports] = Ports(
        inputs={"state": PortType(dtype="state")}, outputs={"output": PortType(dtype="any")}
    )

    input_ports: ClassVar = tuple(ports.inputs)
    output_ports: ClassVar = tuple(ports.outputs)

    where: Callable

    def __call__(self, inputs, state, *, key):
        return {"output": self.where(inputs["state"])}, state


class SelectState(StateSelection, Component):
    """Output the parts of a state found at the configured paths."""

    model_facts = ModelFacts()

    ports: ClassVar = StateSelection.ports

    config: ClassVar[type[BaseModel]] = _SelectStateConfig

    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=state_selector_output_prototype
    )

    @classmethod
    def from_config(cls, config, **context):
        return cls(StateSelector(tuple(config.paths)))

    def configuration_values(self):
        return _extract_state_selector_params(self, self.config.model_construct())


def feedback_node_names(count: int) -> tuple[tuple[str, str], ...]:
    if count == 1:
        return (("feedback_select", "feedback"),)
    return tuple((f"feedback_select_{index}", f"feedback_{index}") for index in range(count))


def feedback_wires(count: int, *, force_filter: bool) -> tuple[Wire, ...]:
    """Connect the explicit sensory and efferent paths of a feedback graph."""
    wires = []
    for index, (selector, channel) in enumerate(feedback_node_names(count)):
        wires.extend(
            (
                Wire(
                    "mechanics",
                    "state",
                    selector,
                    "state",
                    "recurrent",
                    {
                        "kind": "state_output",
                        "scope": "trial",
                        "source": "state_initializer",
                        "state_slot": "state",
                        "seed_channel_history": True,
                    },
                ),
                Wire(selector, "output", channel, "input"),
                Wire(channel, "output", "feedback_values", f"in_{index}"),
            )
        )
    wires.extend(
        (
            Wire("feedback_values", "output", "net", "feedback"),
            Wire("net", "output", "efferent", "input"),
        )
    )
    if force_filter:
        wires.extend(
            (
                Wire("efferent", "output", "force_filter", "input"),
                Wire("force_filter", "output", "mechanics", "force"),
            )
        )
    else:
        wires.append(Wire("efferent", "output", "mechanics", "force"))
    return tuple(wires)


FEEDBACK_INPUT = ("net", "input")
FEEDBACK_OUTPUT = ("mechanics", "effector")
