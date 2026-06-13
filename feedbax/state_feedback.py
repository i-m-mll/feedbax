"""Declarative state feedback selector components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from equinox import field
from equinox.nn import State
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

from feedbax.graph import Component


_DEFAULT_STATE_SLICES = {
    "position": {"start": 0, "stop": 2},
    "velocity": {"start": 2, "stop": 4},
}
_DEFAULT_CHANNELS = [
    {"slice": "position", "transform": "identity"},
    {"slice": "velocity", "transform": "identity"},
]


@dataclass(frozen=True)
class StateFeedbackChannel:
    """One output channel in a state feedback selector."""

    state_slice: str
    transform: str = "identity"
    target_slice: tuple[int, ...] | None = None


class StateFeedbackSelector(Component):
    """Select and transform named slices from a vector state.

    The selector is intentionally generic: state slices may be absolute indices
    or delayed-block-relative slices, and channel transforms describe how those
    selected arrays are combined with optional target input.
    """

    input_ports = ("state", "target")
    output_ports = ("feedback",)

    state_slices: tuple[tuple[str, tuple[int, ...]], ...] = field(static=True)
    channels: tuple[StateFeedbackChannel, ...] = field(static=True)
    expected_state_dim: int | None = field(static=True)
    output_size: int = field(static=True)
    state_slices_param: dict[str, Any] = field(static=True)
    channels_param: list[dict[str, Any]] = field(static=True)

    def __init__(
        self,
        *,
        state_slices: Mapping[str, Any] | None = None,
        channels: list[Any] | tuple[Any, ...] | None = None,
        expected_state_dim: int | None = None,
        output_size: int | None = None,
    ):
        state_slices_param = dict(state_slices or _DEFAULT_STATE_SLICES)
        channels_param = [
            _normalize_channel_param(item) for item in (channels or _DEFAULT_CHANNELS)
        ]
        parsed_slices = tuple(
            (name, _indices_from_slice_spec(name, spec))
            for name, spec in state_slices_param.items()
        )
        parsed_channels = tuple(_parse_channel(item) for item in channels_param)
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
            len(slice_map[channel.state_slice]) for channel in parsed_channels
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
        self.state_slices_param = state_slices_param
        self.channels_param = channels_param

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
        slice_map = dict(self.state_slices)
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
        if channel.transform == "identity":
            return values
        if channel.transform == "negate":
            return -values
        if channel.transform == "target_minus":
            if target is None:
                raise ValueError(
                    "StateFeedbackSelector channel transform 'target_minus' needs target input."
                )
            target_values = _select_target(target, channel.target_slice, values.shape[-1])
            target_values = jnp.broadcast_to(target_values, values.shape)
            return target_values - values
        if channel.transform == "state_minus_target":
            if target is None:
                raise ValueError(
                    "StateFeedbackSelector channel transform 'state_minus_target' needs target input."
                )
            target_values = _select_target(target, channel.target_slice, values.shape[-1])
            target_values = jnp.broadcast_to(target_values, values.shape)
            return values - target_values
        raise ValueError(f"Unknown state feedback transform {channel.transform!r}.")

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

    selector = StateFeedbackSelector(
        state_slices=_mapping_param(params.get("state_slices"), _DEFAULT_STATE_SLICES),
        channels=_list_param(params.get("channels"), _DEFAULT_CHANNELS),
        expected_state_dim=params.get("expected_state_dim"),
        output_size=params.get("output_size"),
    )
    state_proto = inputs.get("state")
    dtype = getattr(state_proto, "dtype", jnp.float32)
    return {"feedback": jnp.zeros((selector.output_size,), dtype=dtype)}


def build_state_feedback_selector(params: Mapping[str, Any]) -> StateFeedbackSelector:
    """Build a selector from GraphSpec node params."""

    return StateFeedbackSelector(
        state_slices=_mapping_param(params.get("state_slices"), _DEFAULT_STATE_SLICES),
        channels=_list_param(params.get("channels"), _DEFAULT_CHANNELS),
        expected_state_dim=params.get("expected_state_dim"),
        output_size=params.get("output_size"),
    )


def _mapping_param(value: Any, default: Mapping[str, Any]) -> Mapping[str, Any]:
    if value is None:
        return default
    if not isinstance(value, Mapping):
        raise ValueError("state_slices must be an object mapping names to slice specs.")
    return value


def _list_param(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError("channels must be a list of selector channel specs.")
    return list(value)


def _normalize_channel_param(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"slice": item, "transform": "identity"}
    if isinstance(item, Mapping):
        return dict(item)
    raise ValueError(f"Invalid state feedback channel spec {item!r}.")


def _parse_channel(item: Mapping[str, Any]) -> StateFeedbackChannel:
    state_slice = item.get("slice", item.get("state_slice"))
    if not isinstance(state_slice, str) or not state_slice:
        raise ValueError(f"State feedback channel needs a slice name; got {item!r}.")
    transform = str(item.get("transform", "identity"))
    aliases = {
        "target_minus_state": "target_minus",
        "target_relative": "target_minus",
        "negative": "negate",
    }
    transform = aliases.get(transform, transform)
    if transform not in {"identity", "negate", "target_minus", "state_minus_target"}:
        raise ValueError(f"Unknown state feedback transform {transform!r}.")
    target_slice = item.get("target_slice")
    return StateFeedbackChannel(
        state_slice=state_slice,
        transform=transform,
        target_slice=_indices_from_target_slice(target_slice) if target_slice is not None else None,
    )


def _indices_from_slice_spec(name: str, spec: Any) -> tuple[int, ...]:
    if isinstance(spec, Mapping):
        offset = int(spec.get("offset", 0) or 0)
        if "delay" in spec:
            if "block_size" not in spec:
                raise ValueError(f"State slice {name!r} uses delay without block_size.")
            delay = int(spec["delay"])
            if delay < 0:
                raise ValueError(f"State slice {name!r} delay must be non-negative.")
            offset += delay * int(spec["block_size"])
        if "indices" in spec:
            raw_indices = spec["indices"]
            if not isinstance(raw_indices, (list, tuple)):
                raise ValueError(f"State slice {name!r} indices must be a list.")
            return tuple(offset + int(index) for index in raw_indices)
        if "start" in spec and "stop" in spec:
            start = offset + int(spec["start"])
            stop = offset + int(spec["stop"])
            if stop < start:
                raise ValueError(f"State slice {name!r} stop must be >= start.")
            return tuple(range(start, stop))
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        start, stop = (int(value) for value in spec)
        if stop < start:
            raise ValueError(f"State slice {name!r} stop must be >= start.")
        return tuple(range(start, stop))
    raise ValueError(
        f"State slice {name!r} must be a [start, stop] pair or object with "
        "'indices' or 'start'/'stop'."
    )


def _indices_from_target_slice(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, stop = (int(item) for item in value)
        return tuple(range(start, stop))
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    raise ValueError("target_slice must be a [start, stop] pair or explicit index list.")


def _select_target(target: PyTree, indices: tuple[int, ...] | None, width: int) -> PyTree:
    target_array = jnp.asarray(target)
    if indices is None:
        if target_array.shape[-1] != width:
            raise ValueError(
                f"StateFeedbackSelector target width {target_array.shape[-1]} "
                f"does not match selected state width {width}."
            )
        return target_array
    if len(indices) != width:
        raise ValueError(
            f"StateFeedbackSelector target_slice width {len(indices)} "
            f"does not match selected state width {width}."
        )
    return jnp.take(target_array, jnp.asarray(indices, dtype=jnp.int32), axis=-1)
