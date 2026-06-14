"""Affine feedback controller components."""

from __future__ import annotations

from typing import Any, Mapping

from equinox import field
from equinox.nn import State, StateIndex
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.runtime.graph import Component


class AffineFeedbackController(Component):
    """Time-varying affine feedback controller.

    Computes ``command = K[t] @ feedback + b[t]``. If a ``reference`` input is
    supplied, ``feedback`` is interpreted as observed state and the controller
    instead uses ``reference - feedback``. A feedforward command may be supplied
    either as a parameter schedule or through the ``feedforward`` input port.
    """

    input_ports = ("feedback", "reference", "feedforward")
    output_ports = ("command",)

    gain: Array
    bias: Array | None
    feedforward: Array | None
    schedule_policy: str = field(static=True)
    input_size: int = field(static=True)
    output_size: int = field(static=True)
    is_time_varying: bool = field(static=True)
    state_index: StateIndex
    _initial_state: int = field(static=True)

    def __init__(
        self,
        *,
        gain: PyTree,
        bias: PyTree | None = None,
        feedforward: PyTree | None = None,
        schedule_policy: str = "hold",
    ):
        gain_array = jnp.asarray(gain)
        if gain_array.ndim not in (2, 3):
            raise ValueError(
                "AffineFeedbackController gain must have shape "
                "(output_size, input_size) or (n_steps, output_size, input_size)."
            )
        if schedule_policy not in {"hold", "error"}:
            raise ValueError("schedule_policy must be 'hold' or 'error'.")

        self.gain = gain_array
        self.bias = _optional_vector_schedule("bias", bias, gain_array.shape[-2])
        self.feedforward = _optional_vector_schedule(
            "feedforward",
            feedforward,
            gain_array.shape[-2],
        )
        self.schedule_policy = schedule_policy
        self.input_size = int(gain_array.shape[-1])
        self.output_size = int(gain_array.shape[-2])
        self.is_time_varying = gain_array.ndim == 3
        self._initial_state = 0
        self.state_index = StateIndex(self._initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        del key
        step = state.get(self.state_index)
        state = state.set(self.state_index, step + 1)

        feedback = jnp.asarray(inputs["feedback"])
        if feedback.shape[-1] != self.input_size:
            raise ValueError(
                f"AffineFeedbackController expected feedback width {self.input_size}; "
                f"got shape {feedback.shape}."
            )
        if "reference" in inputs:
            reference = jnp.broadcast_to(jnp.asarray(inputs["reference"]), feedback.shape)
            feedback = reference - feedback

        gain = _schedule_value("gain", self.gain, step, rank=2, policy=self.schedule_policy)
        command = jnp.einsum("oi,...i->...o", gain, feedback)

        bias = _schedule_value("bias", self.bias, step, rank=1, policy=self.schedule_policy)
        if bias is not None:
            command = command + bias

        if self.feedforward is not None and "feedforward" in inputs:
            raise ValueError(
                "AffineFeedbackController feedforward is ambiguous: provide either "
                "a feedforward parameter schedule or the feedforward input, not both."
            )
        feedforward = (
            jnp.asarray(inputs["feedforward"])
            if "feedforward" in inputs
            else _schedule_value(
                "feedforward",
                self.feedforward,
                step,
                rank=1,
                policy=self.schedule_policy,
            )
        )
        if feedforward is not None:
            command = command + feedforward

        return {"command": command}, state

    def to_params(self) -> dict[str, Any]:
        """Return GraphSpec parameters that rebuild this controller."""

        params: dict[str, Any] = {
            "gain": self.gain.tolist(),
            "schedule_policy": self.schedule_policy,
        }
        if self.bias is not None:
            params["bias"] = self.bias.tolist()
        if self.feedforward is not None:
            params["feedforward"] = self.feedforward.tolist()
        return params


def build_affine_feedback_controller(params: Mapping[str, Any]) -> AffineFeedbackController:
    """Build an affine feedback controller from GraphSpec parameters."""

    if "gain" not in params:
        raise ValueError("AffineFeedbackController requires 'gain'.")
    return AffineFeedbackController(
        gain=params["gain"],
        bias=params.get("bias"),
        feedforward=params.get("feedforward"),
        schedule_policy=str(params.get("schedule_policy", "hold")),
    )


def affine_feedback_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return output prototypes for an affine feedback controller spec."""

    del inputs
    gain = jnp.asarray(params["gain"])
    if gain.ndim not in (2, 3):
        raise ValueError("AffineFeedbackController gain must be a matrix or matrix schedule.")
    return {"command": jnp.zeros((int(gain.shape[-2]),), dtype=gain.dtype)}


def _optional_vector_schedule(name: str, value: PyTree | None, output_size: int) -> Array | None:
    if value is None:
        return None
    array = jnp.asarray(value)
    if array.ndim not in (1, 2):
        raise ValueError(
            f"AffineFeedbackController {name} must have shape "
            "(output_size,) or (n_steps, output_size)."
        )
    if array.shape[-1] != output_size:
        raise ValueError(
            f"AffineFeedbackController {name} output width {array.shape[-1]} "
            f"does not match gain output width {output_size}."
        )
    return array


def _schedule_value(
    name: str,
    value: Array | None,
    step: PyTree,
    *,
    rank: int,
    policy: str,
) -> Array | None:
    if value is None:
        return None
    if value.ndim == rank:
        return value
    if value.ndim != rank + 1:
        raise ValueError(f"AffineFeedbackController {name} has invalid rank {value.ndim}.")

    if policy == "error":
        try:
            step_int = int(step)
        except TypeError:
            step_int = None
        if step_int is not None and step_int >= value.shape[0]:
            raise ValueError(
                f"AffineFeedbackController {name} schedule length {value.shape[0]} "
                f"was exhausted at step {step_int}."
            )
    index = jnp.minimum(jnp.asarray(step, dtype=jnp.int32), value.shape[0] - 1)
    return jnp.take(value, index, axis=0)
