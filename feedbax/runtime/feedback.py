"""Selection, assembly, and wiring for explicit feedback paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, ClassVar

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree
from pydantic import BaseModel, ConfigDict
from pydantic import Field as ConfigField

from feedbax.contracts.component import DynamicPortPolicy, PortType
from feedbax.contracts.portable import ComponentOptions, ConfigBinding, Ports
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


def _extract_feedback_assembly_params(component: AssembleFeedback, model: Any) -> Any:
    return {"layout": component.to_layout(), "n_inputs": component.structure.num_leaves}


def _build_feedback_assembly(params: Mapping[str, Any]) -> AssembleFeedback:
    assembly = AssembleFeedback.from_layout(params.get("layout"))
    if len(assembly.input_ports) != params["n_inputs"]:
        raise ValueError("AssembleFeedback n_inputs must match its layout leaf count")
    return assembly


def state_selector_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    state = _required_input(inputs, "state", component="SelectState")
    return {"output": StateSelector(tuple(params["paths"]))(state)}


def feedback_assembly_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    assembly = _build_feedback_assembly(params)
    for port in assembly.input_ports:
        _required_input(inputs, port, component="AssembleFeedback")
    return {"output": assembly.assemble(inputs)}


def _build_select_state(params: Mapping[str, Any]) -> SelectState:
    return SelectState(StateSelector(tuple(params["paths"])))


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
    """Portable SelectState with a reconstructible typed configuration."""

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


class _AssembleFeedbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    n_inputs: int = ConfigField(default=1, ge=1)
    layout: Any = ConfigField(default=None, json_schema_extra={"feedbax": {"type": "object"}})


class AssembleFeedback(Component):
    """Reassemble independent channel values in the controller's feedback tree."""

    config: ClassVar[type[BaseModel]] = _AssembleFeedbackConfig

    ports: ClassVar[Ports] = Ports(
        inputs={},
        outputs={"output": PortType(dtype="any")},
        dynamic=DynamicPortPolicy(
            count_param="n_inputs",
            count_mode="integer",
            direction="input",
            fixed_output_ports=("output",),
            generated_name_template="in_{index}",
            dynamic_port_type=PortType(dtype="any"),
        ),
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=feedback_assembly_output_prototype
    )

    config_bindings: ClassVar = (ConfigBinding("structure.num_leaves", name="n_inputs"),)

    structure: Any = eqx.field(static=True)

    def __init__(self, structure):
        self.structure = structure

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

    @classmethod
    def from_config(cls, config, **context):
        return _build_feedback_assembly(config.model_dump(mode="python"))

    def configuration_values(self):
        return _extract_feedback_assembly_params(self, self.config.model_construct())


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
