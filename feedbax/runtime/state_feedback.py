"""Declarative state feedback selector components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import jax.numpy as jnp
from equinox import field
from equinox.nn import State
from jaxtyping import PRNGKeyArray, PyTree

from feedbax.contracts.component import PortType
from feedbax.contracts.portable import ComponentOptions, ConfigBinding, Ports
from feedbax.runtime.graph import Component


@dataclass(frozen=True)
class _IndexRange:
    start: int
    stop: int

    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))


@dataclass(frozen=True)
class _StateSlice:
    selection: _IndexRange | tuple[int, ...]
    offset: int | None = None
    delay: int | None = None
    block_size: int | None = None
    compact: bool = False

    def indices(self) -> tuple[int, ...]:
        offset = (self.offset or 0) + (self.delay or 0) * (self.block_size or 0)
        selection = self.selection
        indices = selection.indices() if isinstance(selection, _IndexRange) else selection
        return tuple(offset + index for index in indices)

    def to_param(self) -> dict[str, Any] | list[int]:
        selection = self.selection
        if isinstance(selection, _IndexRange):
            if self.compact:
                return [selection.start, selection.stop]
            param: dict[str, Any] = {"start": selection.start, "stop": selection.stop}
        else:
            param = {"indices": list(selection)}
        if self.offset is not None:
            param["offset"] = self.offset
        if self.delay is not None:
            param["delay"] = self.delay
        if self.block_size is not None:
            param["block_size"] = self.block_size
        return param


_TRANSFORM_ALIASES = {
    "target_minus_state": "target_minus",
    "target_relative": "target_minus",
    "negative": "negate",
}


@dataclass(frozen=True)
class StateFeedbackChannel:
    """One output channel in a state feedback selector."""

    state_slice: str
    transform: str | None = "identity"
    target_slice: _IndexRange | tuple[int, ...] | None = None
    slice_key: str = "slice"

    @property
    def operation(self) -> str:
        transform = self.transform if self.transform is not None else "identity"
        return _TRANSFORM_ALIASES.get(transform, transform)

    def to_param(self) -> dict[str, Any]:
        param: dict[str, Any] = {self.slice_key: self.state_slice}
        if self.transform is not None:
            param["transform"] = self.transform
        if isinstance(self.target_slice, _IndexRange):
            param["target_slice"] = [self.target_slice.start, self.target_slice.stop]
        elif self.target_slice is not None:
            param["target_slice"] = list(self.target_slice)
        return param


_DEFAULT_STATE_SLICES = (
    ("position", _StateSlice(_IndexRange(0, 2))),
    ("velocity", _StateSlice(_IndexRange(2, 4))),
)
_DEFAULT_CHANNELS = (
    StateFeedbackChannel("position"),
    StateFeedbackChannel("velocity"),
)


class StateFeedbackSelector(Component):
    """Select and transform named slices from a vector state.

    The selector is intentionally generic: state slices may be absolute indices
    or delayed-block-relative slices, and channel transforms describe how those
    selected arrays are combined with optional target input.
    """

    ports: ClassVar[Ports] = Ports(
        inputs={
            "state": PortType(dtype="vector"),
            "target": PortType(dtype="vector"),
        },
        outputs={"feedback": PortType(dtype="vector")},
    )
    component_options: ClassVar = ComponentOptions(
        output_prototype_fn=lambda params, inputs: state_feedback_output_prototype(params, inputs)
    )

    config_bindings: ClassVar[tuple[ConfigBinding, ...]] = (
        ConfigBinding("state_slices_param", name="state_slices"),
        ConfigBinding("channels_param", name="channels"),
    )

    state_slices: tuple[tuple[str, _StateSlice], ...] = field(static=True)
    channels: tuple[StateFeedbackChannel, ...] = field(static=True)
    expected_state_dim: int | None = field(static=True)
    output_size: int = field(static=True)

    def __init__(
        self,
        *,
        state_slices: Mapping[str, Any] | None = None,
        channels: list[Any] | tuple[Any, ...] | None = None,
        expected_state_dim: int | None = None,
        output_size: int | None = None,
    ):
        if state_slices is not None and not isinstance(state_slices, Mapping):
            raise ValueError("state_slices must be an object mapping names to slice specs.")
        if channels is not None and not isinstance(channels, (list, tuple)):
            raise ValueError("channels must be a list of selector channel specs.")
        parsed_slices = (
            tuple((name, _parse_state_slice(name, spec)) for name, spec in state_slices.items())
            if state_slices
            else _DEFAULT_STATE_SLICES
        )
        parsed_channels = (
            tuple(_parse_channel(item) for item in channels) if channels else _DEFAULT_CHANNELS
        )
        slice_map = dict(parsed_slices)
        missing = [
            channel.state_slice
            for channel in parsed_channels
            if channel.state_slice not in slice_map
        ]
        if missing:
            raise ValueError(
                "StateFeedbackSelector channels reference unknown state slice(s): "
                + ", ".join(repr(name) for name in missing)
            )
        inferred_output_size = sum(
            len(slice_map[channel.state_slice].indices()) for channel in parsed_channels
        )
        if output_size is not None and int(output_size) != inferred_output_size:
            raise ValueError(
                f"StateFeedbackSelector output_size {output_size} does not match "
                f"channel width {inferred_output_size}."
            )
        self.state_slices = parsed_slices
        self.channels = parsed_channels
        self.expected_state_dim = None if expected_state_dim is None else int(expected_state_dim)
        self.output_size = inferred_output_size

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        del key
        vector = jnp.asarray(inputs["state"])
        if self.expected_state_dim is not None and vector.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"Expected a {self.expected_state_dim}D state vector; got shape {vector.shape}."
            )
        target = inputs.get("target")
        slice_map = {name: selection.indices() for name, selection in self.state_slices}
        parts = [
            self._channel_value(channel, vector, target, slice_map) for channel in self.channels
        ]
        if not parts:
            feedback = jnp.zeros(vector.shape[:-1] + (0,), dtype=vector.dtype)
        else:
            feedback = jnp.concatenate(parts, axis=-1)
        return {"feedback": feedback}, state

    def _channel_value(
        self,
        channel: StateFeedbackChannel,
        vector: PyTree,
        target: PyTree | None,
        slice_map: dict[str, tuple[int, ...]],
    ) -> PyTree:
        values = jnp.take(
            vector,
            jnp.asarray(slice_map[channel.state_slice], dtype=jnp.int32),
            axis=-1,
        )
        operation = channel.operation
        if operation == "identity":
            return values
        if operation == "negate":
            return -values
        if operation == "target_minus":
            if target is None:
                raise ValueError(
                    "StateFeedbackSelector channel transform 'target_minus' needs target input."
                )
            target_values = _select_target(target, channel.target_slice, values.shape[-1])
            target_values = jnp.broadcast_to(target_values, values.shape)
            return target_values - values
        if operation == "state_minus_target":
            if target is None:
                raise ValueError(
                    "StateFeedbackSelector channel transform 'state_minus_target' needs target input."
                )
            target_values = _select_target(target, channel.target_slice, values.shape[-1])
            target_values = jnp.broadcast_to(target_values, values.shape)
            return values - target_values
        raise ValueError(f"Unknown state feedback transform {operation!r}.")

    @property
    def state_slices_param(self) -> dict[str, Any]:
        return {name: selection.to_param() for name, selection in self.state_slices}

    @property
    def channels_param(self) -> list[dict[str, Any]]:
        return [channel.to_param() for channel in self.channels]

    def to_params(self) -> dict[str, Any]:
        """Return GraphSpec parameters that rebuild this selector."""

        params: dict[str, Any] = {
            "state_slices": self.state_slices_param,
            "channels": self.channels_param,
            "output_size": self.output_size,
        }
        if self.expected_state_dim is not None:
            params["expected_state_dim"] = self.expected_state_dim
        return params


def state_feedback_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return output prototypes for a state feedback selector spec."""

    selector = build_state_feedback_selector(params)
    state_proto = inputs.get("state")
    dtype = getattr(state_proto, "dtype", jnp.float32)
    return {"feedback": jnp.zeros((selector.output_size,), dtype=dtype)}


def build_state_feedback_selector(params: Mapping[str, Any]) -> StateFeedbackSelector:
    """Build a selector from GraphSpec node params."""

    return StateFeedbackSelector(
        state_slices=params.get("state_slices"),
        channels=params.get("channels"),
        expected_state_dim=params.get("expected_state_dim"),
        output_size=params.get("output_size"),
    )


def _parse_channel(item: Any) -> StateFeedbackChannel:
    if isinstance(item, str):
        item = {"slice": item, "transform": "identity"}
    if not isinstance(item, Mapping):
        raise ValueError(f"Invalid state feedback channel spec {item!r}.")
    slice_key = "slice" if "slice" in item else "state_slice"
    state_slice = item.get(slice_key)
    if not isinstance(state_slice, str) or not state_slice:
        raise ValueError(f"State feedback channel needs a slice name; got {item!r}.")
    transform = str(item["transform"]) if "transform" in item else None
    target_slice = item.get("target_slice")
    channel = StateFeedbackChannel(
        state_slice=state_slice,
        transform=transform,
        target_slice=_parse_target_slice(target_slice) if target_slice is not None else None,
        slice_key=slice_key,
    )
    if channel.operation not in {"identity", "negate", "target_minus", "state_minus_target"}:
        raise ValueError(f"Unknown state feedback transform {channel.operation!r}.")
    return channel


def _parse_state_slice(name: str, spec: Any) -> _StateSlice:
    if isinstance(spec, Mapping):
        offset = int(spec.get("offset") or 0) if "offset" in spec else None
        delay = None
        block_size = None
        if "delay" in spec:
            if "block_size" not in spec:
                raise ValueError(f"State slice {name!r} uses delay without block_size.")
            delay = int(spec["delay"])
            if delay < 0:
                raise ValueError(f"State slice {name!r} delay must be non-negative.")
            block_size = int(spec["block_size"])
        if "indices" in spec:
            raw_indices = spec["indices"]
            if not isinstance(raw_indices, (list, tuple)):
                raise ValueError(f"State slice {name!r} indices must be a list.")
            return _StateSlice(
                tuple(int(index) for index in raw_indices), offset, delay, block_size
            )
        if "start" in spec and "stop" in spec:
            selection = _parse_state_range(name, spec["start"], spec["stop"])
            return _StateSlice(selection, offset, delay, block_size)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return _StateSlice(_parse_state_range(name, *spec), compact=True)
    raise ValueError(
        f"State slice {name!r} must be a [start, stop] pair or object with "
        "'indices' or 'start'/'stop'."
    )


def _parse_state_range(name: str, start: Any, stop: Any) -> _IndexRange:
    selection = _IndexRange(int(start), int(stop))
    if selection.stop < selection.start:
        raise ValueError(f"State slice {name!r} stop must be >= start.")
    return selection


def _parse_target_slice(value: Any) -> _IndexRange | tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("target_slice must be a [start, stop] pair or explicit index list.")
    if len(value) == 2:
        return _IndexRange(int(value[0]), int(value[1]))
    return tuple(int(item) for item in value)


def _select_target(
    target: PyTree, selection: _IndexRange | tuple[int, ...] | None, width: int
) -> PyTree:
    target_array = jnp.asarray(target)
    if selection is None:
        if target_array.shape[-1] != width:
            raise ValueError(
                f"StateFeedbackSelector target width {target_array.shape[-1]} "
                f"does not match selected state width {width}."
            )
        return target_array
    indices = selection.indices() if isinstance(selection, _IndexRange) else selection
    if len(indices) != width:
        raise ValueError(
            f"StateFeedbackSelector target_slice width {len(indices)} "
            f"does not match selected state width {width}."
        )
    return jnp.take(target_array, jnp.asarray(indices, dtype=jnp.int32), axis=-1)
