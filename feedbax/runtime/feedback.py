"""Selection, assembly, and wiring for explicit feedback paths."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import attrgetter
from typing import Any

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

from feedbax.runtime.graph import Component, Wire


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


class SelectState(Component):
    """Expose a state selector as a stateless graph node."""

    input_ports = ("state",)
    output_ports = ("output",)
    where: Callable

    def __call__(self, inputs, state, *, key):
        return {"output": self.where(inputs["state"])}, state


class AssembleFeedback(Component):
    """Reassemble independent channel values in the controller's feedback tree."""

    input_ports: tuple[str, ...]
    output_ports = ("output",)
    structure: Any = eqx.field(static=True)

    def __init__(self, structure):
        self.structure = structure
        self.input_ports = tuple(f"in_{index}" for index in range(structure.num_leaves))

    def assemble(self, inputs: Mapping[str, PyTree]) -> PyTree:
        return self.structure.unflatten([inputs[port] for port in self.input_ports])

    def __call__(self, inputs, state, *, key):
        return {"output": self.assemble(inputs)}, state

    @classmethod
    def from_layout(cls, layout):
        def decode(value):
            if value is None:
                return 0
            if not isinstance(value, dict) or len(value) != 1:
                raise ValueError(
                    "Feedback layout expects null leaves or tagged tuple, list, and dict nodes"
                )
            ((kind, children),) = value.items()
            if kind == "tuple":
                return tuple(decode(child) for child in children)
            if kind == "list":
                return [decode(child) for child in children]
            if kind == "dict":
                return {name: decode(child) for name, child in children.items()}
            raise ValueError(f"Unsupported feedback layout node {kind!r}")

        return cls(jt.structure(decode(layout)))

    def to_layout(self):
        def encode(value):
            if isinstance(value, tuple) and type(value) is tuple:
                return {"tuple": [encode(child) for child in value]}
            if isinstance(value, list) and type(value) is list:
                return {"list": [encode(child) for child in value]}
            if (
                isinstance(value, dict)
                and type(value) is dict
                and all(isinstance(key, str) for key in value)
            ):
                return {"dict": {name: encode(child) for name, child in value.items()}}
            if type(value) is int and value == 0:
                return None
            raise ValueError("Portable feedback layouts support tuple, list, and string-keyed dict")

        return encode(self.structure.unflatten([0] * self.structure.num_leaves))


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
