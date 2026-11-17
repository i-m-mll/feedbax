"""Affine feedback controller components."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field
from equinox.nn import State, StateIndex
from jaxtyping import Array, PRNGKeyArray, PyTree
from pydantic import BaseModel, ConfigDict
from pydantic import Field as ConfigField

from feedbax.contracts.component import PortType
from feedbax.contracts.portable import ComponentOptions, Ports
from feedbax.runtime.graph import Component


class _AffineFeedbackControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    gain: Any = ConfigField(default=[[[1.0]]], json_schema_extra={"feedbax": {"type": "array"}})
    bias: Any = ConfigField(default=None, json_schema_extra={"feedbax": {"type": "array"}})
    feedforward: Any = ConfigField(default=None, json_schema_extra={"feedbax": {"type": "array"}})
    schedule_policy: Literal["hold", "error"] = ConfigField(
        default="hold"
    )


class AffineFeedbackController(Component):
    """Time-varying affine feedback controller.

    Computes ``command = K[t] @ feedback + b[t]``. If a ``reference`` input is
    supplied, ``feedback`` is interpreted as observed state and the controller
    instead uses ``reference - feedback``. A feedforward command may be supplied
    either as a parameter schedule or through the ``feedforward`` input port.

    Schedules use the controller state's zero-based step. After a schedule's
    last sample, ``hold`` reuses it and ``error`` raises on the next sample.
    """

    config: ClassVar[type[BaseModel]] = _AffineFeedbackControllerConfig

    ports: ClassVar[Ports] = Ports(
        inputs={
            "feedback": PortType(dtype="vector"),
            "reference": PortType(dtype="vector"),
            "feedforward": PortType(dtype="vector"),
        },
        outputs={"command": PortType(dtype="vector")},
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        prototype_depends_on_connected_inputs=True,
        output_prototype_fn=lambda params, inputs: affine_feedback_output_prototype(params, inputs),
    )

    gain: Array
    bias: Array | None
    feedforward: Array | None
    schedule_policy: str = field(static=True)
    input_size: int = field(static=True)
    output_size: int = field(static=True)
    is_time_varying: bool = field(static=True)
    state_index: StateIndex

    def __init__(
        self,
        *,
        gain: PyTree = _AffineFeedbackControllerConfig.model_fields["gain"].default,
        bias: PyTree | None = _AffineFeedbackControllerConfig.model_fields["bias"].default,
        feedforward: PyTree | None = _AffineFeedbackControllerConfig.model_fields[
            "feedforward"
        ].default,
        schedule_policy: str = _AffineFeedbackControllerConfig.model_fields[
            "schedule_policy"
        ].default,
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
        initial_state = 0
        self.state_index = StateIndex(initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        del key
        step = state.get(self.state_index)
        command = self._command(inputs, step)
        return {"command": command}, state.set(self.state_index, step + 1)

    def _command(self, inputs: Mapping[str, PyTree], step: Array | int) -> Array:
        feedback = jnp.asarray(inputs["feedback"])
        if feedback.ndim == 0 or feedback.shape[-1] != self.input_size:
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

        return command

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

    if "feedback" not in inputs:
        from feedbax.contracts.portable import MissingPrototypeInput

        raise MissingPrototypeInput("Affine feedback output requires input prototype 'feedback'.")
    controller = build_affine_feedback_controller(params)
    command = jax.eval_shape(lambda sample: controller._command(sample, step=0), dict(inputs))
    return {"command": jnp.zeros(command.shape, dtype=command.dtype)}


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
    step: Array | int,
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

    length = value.shape[0]
    if length == 0:
        raise ValueError(f"AffineFeedbackController {name} schedule must contain a sample.")
    if policy == "hold":
        step = jnp.minimum(step, length - 1)
    else:
        message = f"AffineFeedbackController {name} schedule length {length} was exhausted"
        if isinstance(step, jax.core.Tracer):
            step = eqx.error_if(step, step >= length, f"{message}.", on_error="raise")
        elif (step_int := int(step)) >= length:
            raise ValueError(f"{message} at step {step_int}.")
    return value[step]
