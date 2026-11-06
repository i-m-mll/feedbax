"""Queue-based modules for modeling distant, possibly noisy connections.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Generic, Optional, Tuple, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from equinox import Module, field
from equinox.nn import State, StateIndex
from jaxtyping import Array, PRNGKeyArray, PyTree

import feedbax._vendor.jax_cookbook.tree as jtree
from feedbax.component_registry import component
from feedbax.component_registry.declarations import ParameterField
from feedbax.component_registry.prototypes import (
    array_prototype,
    migrate_stateful_v1,
    prototype_params,
    stateful_passthrough_output_prototype,
)
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.migrations import ComponentMigration
from feedbax.runtime.graph import Component
from feedbax.runtime.noise import CompositeNoise, Multiplicative, Normal

logger = logging.getLogger(__name__)


class ChannelState(Module):
    """State for a delay/noise channel.

    Attributes:
        output: The current output of the channel.
        queue: A tuple of previous inputs to the channel, with most recent last.
        noise: The noise added to the current output, zero when disabled.
            None if the state has no noise slot.
    """

    output: PyTree[Array, "T"]
    queue: Tuple[Optional[PyTree[Array, "T"]], ...]
    noise: Optional[PyTree[Array, "T"]]


StateT = TypeVar("StateT")
T = TypeVar("T")
ModelT = TypeVar("ModelT")


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


def _noise_model(noise_func: Callable | None) -> str:
    if noise_func is None:
        return "none"
    if isinstance(noise_func, Normal):
        return "additive_gaussian"
    if isinstance(noise_func, Multiplicative):
        return "signal_dependent_gaussian"
    if isinstance(noise_func, CompositeNoise):
        if (
            len(noise_func.terms) == 2
            and isinstance(noise_func.terms[0], Multiplicative)
            and isinstance(noise_func.terms[1], Normal)
        ):
            return "signal_dependent_plus_additive"
    return "custom"


def _channel_noise_values(channel: Any) -> dict[str, Any]:
    from feedbax._vendor.jax_cookbook import identity
    from feedbax.runtime.channel import _noise_model
    from feedbax.runtime.noise import CompositeNoise, Multiplicative, Normal

    additive_std = 0.0
    signal_dependent_std = 0.0
    noise_func = channel.noise_func
    actual_model = _noise_model(noise_func)
    declared_model = channel.noise_model
    allowed = {
        "none": {"none"},
        "additive_gaussian": {"none", "additive_gaussian"},
        "signal_dependent_gaussian": {"none", "signal_dependent_gaussian"},
        "signal_dependent_plus_additive": {
            "none",
            "signal_dependent_gaussian",
            "additive_gaussian",
            "signal_dependent_plus_additive",
        },
    }
    if actual_model not in allowed.get(declared_model, ()):
        raise ValueError("Channel noise function does not match a portable noise model")
    terms = noise_func.terms if isinstance(noise_func, CompositeNoise) else (noise_func,)
    for term in terms:
        if term is None:
            continue
        if type(term) is Normal and term.mean == 0.0:
            additive_std = float(term.std)
        elif (
            type(term) is Multiplicative
            and type(term.noise_func) is Normal
            and term.noise_func.mean == 0.0
            and term.scale_func is identity
        ):
            signal_dependent_std = float(term.noise_func.std)
        else:
            raise ValueError("Channel noise parameters do not represent a portable Gaussian term")
    include_zero = noise_func is not None and (
        (actual_model == "additive_gaussian" and not additive_std)
        or (actual_model == "signal_dependent_gaussian" and not signal_dependent_std)
        or (
            actual_model == "signal_dependent_plus_additive"
            and not (additive_std and signal_dependent_std)
        )
    )
    if include_zero and declared_model != actual_model:
        raise ValueError("Channel noise model cannot recover the declared zero-strength terms")
    return {
        "noise_model": declared_model,
        "include_zero_noise_terms": include_zero,
        "noise_std": additive_std,
        "additive_noise_std": additive_std,
        "signal_dependent_noise_std": signal_dependent_std,
        "add_noise": bool(channel.add_noise),
        "noise_role": getattr(channel, "noise_role", None),
        "noise_timing": getattr(channel, "noise_timing", None),
    }


def _extract_channel_params(component: Any, model: Any) -> Any:
    values = {
        "delay": int(component.delay),
        **_channel_noise_values(component),
        "input_shape": None,
        "input_dtype": None,
        "init_value": component.init_value,
        **(
            prototype_params(component.input_proto)
            if hasattr(component.input_proto, "shape")
            else {}
        ),
    }
    return type(model).model_validate({name: values[name] for name in type(model).model_fields})


def _noise_params(params: Mapping[str, Any]) -> dict[str, Any]:
    noncanonical = sorted(
        set(params)
        & {
            "noise",
            "std",
            "additive_std",
            "signal_dependent_std",
        }
    )
    if noncanonical:
        raise ValueError(
            "Use canonical Channel noise parameters noise_model, noise_std, "
            "additive_noise_std, and signal_dependent_noise_std; unsupported: "
            + ", ".join(noncanonical)
        )
    model = str(params.get("noise_model", "additive_gaussian"))
    noise_std = params.get("noise_std", 0.0)
    additive_std = params.get("additive_noise_std", noise_std)
    if model == "additive_gaussian" and noise_std not in (None, 0, 0.0):
        additive_std = noise_std
    signal_dependent_std = params.get("signal_dependent_noise_std", 0.0)
    add_noise_default = model != "none"
    add_noise = bool(params.get("add_noise", add_noise_default))
    return {
        "model": model,
        "add_noise": add_noise,
        "additive_std": float(additive_std or 0.0),
        "include_zero_terms": bool(params.get("include_zero_noise_terms", False)),
        "signal_dependent_std": float(signal_dependent_std or 0.0),
        "role": params.get("noise_role"),
        "timing": params.get("noise_timing"),
    }


def _build_noise_func(noise: Mapping[str, Any]):
    model = str(noise["model"])
    additive_std = float(noise["additive_std"])
    signal_dependent_std = float(noise["signal_dependent_std"])
    supported = {
        "none",
        "additive_gaussian",
        "signal_dependent_gaussian",
        "signal_dependent_plus_additive",
    }
    if model not in supported:
        raise ValueError(
            f"Unsupported Channel noise_model {model!r}; expected one of "
            "'none', 'additive_gaussian', 'signal_dependent_gaussian', or "
            "'signal_dependent_plus_additive'"
        )
    if model == "none":
        return None
    include_zero = noise["include_zero_terms"]
    if model == "additive_gaussian":
        return Normal(std=additive_std) if additive_std or include_zero else None
    if model == "signal_dependent_gaussian":
        return (
            Multiplicative(Normal(std=signal_dependent_std))
            if signal_dependent_std or include_zero
            else None
        )
    if model == "signal_dependent_plus_additive":
        signal = (
            Multiplicative(Normal(std=signal_dependent_std))
            if signal_dependent_std or include_zero
            else None
        )
        additive = Normal(std=additive_std) if additive_std or include_zero else None
        if signal is not None and additive is not None:
            return signal + additive
        return signal if signal is not None else additive
    raise AssertionError(f"Unhandled Channel noise_model {model!r}")


def _build_channel(params: Mapping[str, Any]) -> Channel:
    delay = int(params.get("delay", 0))
    noise = _noise_params(params)
    noise_func = _build_noise_func(noise)
    input_proto = params.get("resolved_input_proto")
    if input_proto is None:
        input_proto = array_prototype(params)
    return Channel(
        delay=delay,
        noise_func=noise_func,
        add_noise=bool(noise["add_noise"]),
        input_proto=input_proto,
        init_value=float(params.get("init_value", 0.0)),
        noise_model=str(noise["model"]),
        noise_role=None if noise["role"] is None else str(noise["role"]),
        noise_timing=None if noise["timing"] is None else str(noise["timing"]),
    )


@component(
    "Channel",
    builder=_build_channel,
    extractor=_extract_channel_params,
    override_reason="noise composition and prototype shape require authored recovery",
    param_schema_version="2",
    category="Channels",
    description="Delay and noise for a signal.",
    parameter_fields=[
        ParameterField(name="init_value", type="float", default=0.0, required=False),
        ParameterField(name="delay", type="int", default=5, min=0, required=True),
        ParameterField(
            name="noise_model",
            type="enum",
            options=[
                "none",
                "additive_gaussian",
                "signal_dependent_gaussian",
                "signal_dependent_plus_additive",
            ],
            default="additive_gaussian",
            required=False,
        ),
        ParameterField(name="noise_std", type="float", default=0.01, min=0, required=False),
        ParameterField(
            name="additive_noise_std",
            type="float",
            default=0.0,
            min=0,
            required=False,
        ),
        ParameterField(
            name="signal_dependent_noise_std",
            type="float",
            default=0.0,
            min=0,
            required=False,
        ),
        ParameterField(
            name="include_zero_noise_terms",
            type="bool",
            default=False,
            required=False,
        ),
        ParameterField(name="add_noise", type="bool", default=True, required=False),
        ParameterField(name="noise_role", type="str", default=None, required=False),
        ParameterField(name="noise_timing", type="str", default=None, required=False),
        ParameterField(name="input_shape", type="array", default=None, required=False),
        ParameterField(name="input_dtype", type="str", default=None, required=False),
    ],
    icon="Signal",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=stateful_passthrough_output_prototype,
    build_output_prototypes={"resolved_input_proto": "output"},
    migrations=(
        ComponentMigration(
            source_type="Channel",
            target_type="Channel",
            owner="feedbax",
            source_param_schema_version="1",
            target_param_schema_version="2",
            migration_id="feedbax.component.Channel.params.v1-to-v2",
            migrate_params=migrate_stateful_v1,
            description="Preserve stateful declarations with optional inferred input dtype.",
        ),
    ),
)
class Channel(Component):
    """A noisy delay line."""

    input_ports = ("input",)
    output_ports = ("output",)

    delay: int
    noise_func: Optional[Callable[[PRNGKeyArray, Array], Array]]
    add_noise: bool
    input_proto: PyTree[Array]
    init_value: float
    noise_model: str = field(default="additive_gaussian", static=True)
    noise_role: Optional[str] = field(default=None, static=True)
    noise_timing: Optional[str] = field(default=None, static=True)
    state_index: StateIndex

    def __init__(
        self,
        delay: int,
        noise_func: Optional[Callable[[PRNGKeyArray, Array], Array]] = Normal(),
        add_noise: bool = True,
        input_proto: Optional[PyTree[Array]] = None,
        init_value: float = 0.0,
        noise_model: str | None = None,
        noise_role: Optional[str] = None,
        noise_timing: Optional[str] = None,
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
        self.noise_model = noise_model if noise_model is not None else _noise_model(noise_func)
        self.noise_role = noise_role
        self.noise_timing = noise_timing

        initial_state = self._initial_state_value(input_proto)
        self.state_index = StateIndex(initial_state)

    def _initial_state_value(self, input_proto: PyTree[Array]) -> ChannelState:
        input_init = jt.map(lambda x: jnp.full_like(x, self.init_value), input_proto)
        noise_init = jt.map(jnp.zeros_like, input_init) if self.noise_func is not None else None
        return self._state_with_history(input_init, noise_init)

    def _state_with_history(self, value: PyTree, noise: PyTree | None) -> ChannelState:
        return ChannelState(output=value, queue=self.delay * (value,), noise=noise)

    def fill_history(self, state: State, value: PyTree) -> State:
        """Use one sample for the channel output and every slot of its history."""
        return state.set(
            self.state_index, self._state_with_history(value, state.get(self.state_index).noise)
        )

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

        if self.add_noise and self.noise_func is not None:
            noise = jt.map(
                self.noise_func,
                jtree.random_split_like_tree(key, output),
                output,
            )
            output = jt.map(lambda x, y: x + y, output, noise)
        else:
            noise = jt.map(jnp.zeros_like, channel_state.noise)

        new_state = ChannelState(
            output=output,
            queue=new_queue,
            # Older disabled carries have no slot to record a sample.
            noise=noise if channel_state.noise is not None else None,
        )
        state = state.set(self.state_index, new_state)
        return {"output": output}, state

    def change_input(self, input_proto: PyTree[Array]) -> "Channel":
        """Return a similar Channel with a changed input structure."""
        return Channel(
            delay=self.delay,
            noise_func=self.noise_func,
            add_noise=self.add_noise,
            input_proto=input_proto,
            init_value=self.init_value,
            noise_model=self.noise_model,
            noise_role=self.noise_role,
            noise_timing=self.noise_timing,
        )


def toggle_channel_noise(tree: ModelT, enabled: Optional[bool] = None) -> ModelT:
    """Set the noise flag on every Channel in a PyTree, or invert each flag.

    The returned tree can use the same carry as the original.
    """

    def replace_channel(leaf):
        if not isinstance(leaf, Channel):
            return leaf
        add_noise = not leaf.add_noise if enabled is None else enabled
        return eqx.tree_at(lambda channel: channel.add_noise, leaf, replace=add_noise)

    return jt.map(replace_channel, tree, is_leaf=lambda leaf: isinstance(leaf, Channel))
