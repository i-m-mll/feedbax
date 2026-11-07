"""Affine value composition components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import jax.numpy as jnp
from equinox import field
from equinox.nn import State
from jaxtyping import Array, PRNGKeyArray, PyTree
from pydantic import BaseModel, ConfigDict
from pydantic import Field as ConfigField

from feedbax.contracts.component import PortType
from feedbax.contracts.portable import ComponentOptions, ConfigBinding, Ports
from feedbax.runtime.graph import Component
from feedbax.runtime.prototypes import _required_input

AFFINE_VALUE_COMPOSER_FEATURE_RULES = frozenset({"identity", "target_relative_difference"})


def _as_array_param(value: PyTree, shape: tuple[int, ...], name: str) -> Array:
    array = jnp.asarray(value, dtype=jnp.float32)
    if array.shape != shape:
        raise ValueError(f"AffineValueComposer {name} must have shape {shape}; got {array.shape}")
    return array


def _as_slice(raw: Any, *, name: str) -> tuple[int, int]:
    if isinstance(raw, Mapping):
        start = raw.get("start")
        stop = raw.get("stop")
    elif isinstance(raw, Sequence) and not isinstance(raw, str) and len(raw) == 2:
        start, stop = raw
    else:
        raise ValueError(
            f"AffineValueComposer {name} must be a [start, stop] pair or object "
            "with 'start' and 'stop'"
        )
    try:
        result = (int(start), int(stop))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AffineValueComposer {name} entries must be integers") from exc
    if result[0] < 0 or result[1] <= result[0]:
        raise ValueError(f"AffineValueComposer {name} must satisfy 0 <= start < stop; got {result}")
    return result


def _normalize_feature_rules(rules: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(rules, Sequence) or isinstance(rules, str) or not rules:
        raise ValueError("AffineValueComposer feature_rules must be a non-empty list")
    normalized = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise ValueError(f"AffineValueComposer feature_rules[{index}] must be an object")
        kind = str(rule.get("kind", "identity"))
        if kind not in AFFINE_VALUE_COMPOSER_FEATURE_RULES:
            options = sorted(AFFINE_VALUE_COMPOSER_FEATURE_RULES)
            raise ValueError(
                f"AffineValueComposer feature_rules[{index}].kind {kind!r} is unsupported; "
                f"expected one of {options!r}"
            )
        state_slice = _as_slice(rule.get("state_slice"), name=f"feature_rules[{index}].state_slice")
        normalized_rule: dict[str, Any] = {
            "kind": kind,
            "state_slice": state_slice,
        }
        if kind == "target_relative_difference":
            target_slice = _as_slice(
                rule.get("target_slice"),
                name=f"feature_rules[{index}].target_slice",
            )
            if _slice_size(target_slice) != _slice_size(state_slice):
                raise ValueError(
                    "AffineValueComposer feature_rules"
                    f"[{index}] target_relative_difference requires equal state and target "
                    f"slice sizes; got state_slice={state_slice}, target_slice={target_slice}"
                )
            normalized_rule["target_slice"] = target_slice
        elif rule.get("target_slice") is not None:
            raise ValueError(
                f"AffineValueComposer feature_rules[{index}] may only set target_slice "
                "for target_relative_difference"
            )
        normalized.append(normalized_rule)
    return tuple(normalized)


def _slice_size(slice_spec: tuple[int, int]) -> int:
    return int(slice_spec[1] - slice_spec[0])


def feature_dim_for_rules(feature_rules: Sequence[Mapping[str, Any]]) -> int:
    """Return the concatenated feature width for normalized or raw feature rules."""

    return sum(_slice_size(rule["state_slice"]) for rule in _normalize_feature_rules(feature_rules))


def _validate_slice_bounds(
    rules: Sequence[Mapping[str, Any]],
    *,
    state_size: int,
    target_size: int | None,
) -> None:
    for index, rule in enumerate(rules):
        _, state_stop = rule["state_slice"]
        if state_stop > state_size:
            raise ValueError(
                "AffineValueComposer feature_rules"
                f"[{index}].state_slice {rule['state_slice']} exceeds state size {state_size}"
            )
        if rule["kind"] == "target_relative_difference":
            if target_size is None:
                raise ValueError(
                    "AffineValueComposer target_relative_difference requires input "
                    "prototype 'target'"
                )
            _, target_stop = rule["target_slice"]
            if target_stop > target_size:
                raise ValueError(
                    "AffineValueComposer feature_rules"
                    f"[{index}].target_slice {rule['target_slice']} exceeds target size "
                    f"{target_size}"
                )


def _require_vector(value: Any, *, name: str) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError(f"AffineValueComposer {name} must be an array with a trailing dimension")
    return value


def _extract_features(
    state_value: Array,
    target_value: Array | None,
    feature_rules: Sequence[Mapping[str, Any]],
) -> Array:
    _require_vector(state_value, name="state")
    target_size = None
    if target_value is not None:
        _require_vector(target_value, name="target")
        target_size = int(target_value.shape[-1])
    _validate_slice_bounds(
        feature_rules,
        state_size=int(state_value.shape[-1]),
        target_size=target_size,
    )

    features = []
    for rule in feature_rules:
        state_start, state_stop = rule["state_slice"]
        feature = state_value[..., state_start:state_stop]
        if rule["kind"] == "target_relative_difference":
            if target_value is None:
                raise ValueError(
                    "AffineValueComposer target_relative_difference requires input 'target'"
                )
            target_start, target_stop = rule["target_slice"]
            target_feature = target_value[..., target_start:target_stop]
            try:
                jnp.broadcast_shapes(feature.shape, target_feature.shape)
            except ValueError as exc:
                raise ValueError(
                    "AffineValueComposer target_relative_difference cannot broadcast "
                    f"state slice shape {feature.shape} with target slice shape "
                    f"{target_feature.shape}"
                ) from exc
            feature = feature - target_feature
        features.append(feature)
    return jnp.concatenate(features, axis=-1)


def _affine_delta(features: Array, gain: Array, *, output_block_size: int) -> Array:
    gain = jnp.asarray(gain, dtype=features.dtype)
    expected = (int(output_block_size), int(features.shape[-1]))
    if gain.shape[-2:] != expected:
        raise ValueError(
            "AffineValueComposer gain must end with "
            f"(output_block_size, feature_dim)={expected}; got {gain.shape}"
        )
    if gain.ndim == 2:
        return jnp.einsum("...f,of->...o", features, gain)
    return jnp.einsum("...f,...of->...o", features, gain)


def affine_value_composer_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Infer the composed value prototype from input prototypes."""

    for port in ("base", "state"):
        if port not in inputs:
            raise ValueError(
                f"AffineValueComposer output prototype requires input prototype {port!r}"
            )
    base = _require_vector(inputs["base"], name="base prototype")
    state = _require_vector(inputs["state"], name="state prototype")
    target = inputs.get("target")
    target_size = (
        None if target is None else int(_require_vector(target, name="target prototype").shape[-1])
    )
    output_block_size = int(params.get("output_block_size", 1))
    if int(base.shape[-1]) != output_block_size:
        raise ValueError(
            "AffineValueComposer base prototype trailing dimension "
            f"{base.shape[-1]} does not match output_block_size {output_block_size}"
        )
    feature_rules = _normalize_feature_rules(params.get("feature_rules"))
    _validate_slice_bounds(
        feature_rules,
        state_size=int(state.shape[-1]),
        target_size=target_size,
    )
    gain = inputs.get("gain", params.get("gain_init"))
    if gain is not None:
        gain_shape = getattr(jnp.asarray(gain), "shape", ())
        expected = (output_block_size, feature_dim_for_rules(feature_rules))
        if len(gain_shape) < 2 or tuple(gain_shape[-2:]) != expected:
            raise ValueError(
                "AffineValueComposer gain prototype must end with "
                f"(output_block_size, feature_dim)={expected}; got {gain_shape}"
            )
    return {"value": base}


def affine_composer_output_prototype(
    params: Mapping[str, Any], inputs: Mapping[str, Any]
) -> dict[str, Any]:
    for port in ("base", "state"):
        _required_input(inputs, port, component="AffineValueComposer")
    return affine_value_composer_output_prototype(params, inputs)


class _AffineValueComposerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    output_block_size: int = ConfigField(
        default=1, ge=1
    )
    feature_rules: Any = ConfigField(
        default=[{"kind": "identity", "state_slice": [0, 1]}],
        description="Ordered identity and target-relative difference rules.",
        json_schema_extra={"feedbax": {"type": "object"}},
    )
    gain_init: Any = ConfigField(default=[[0.0]], json_schema_extra={"feedbax": {"type": "array"}})
    bias_init: Any = ConfigField(default=[0.0], json_schema_extra={"feedbax": {"type": "array"}})
    use_bias: bool = ConfigField(default=True)


class AffineValueComposer(Component):
    """Compose ``base + gain @ features(state, target) + bias``.

    Feature rules are declarative and evaluated in order. ``identity`` appends
    ``state[state_slice]``. ``target_relative_difference`` appends
    ``state[state_slice] - target[target_slice]``. A downstream finite-policy
    epsilon composer can represent target-centered repeated state blocks by
    alternating target-relative position slices with identity slices for the
    remaining state coordinates, then choosing an ``output_block_size`` equal to
    the epsilon width and binding gain/bias through graph inputs or the
    component-parameter lane.
    """

    config: ClassVar[type[BaseModel]] = _AffineValueComposerConfig

    ports: ClassVar[Ports] = Ports(
        inputs={
            "base": PortType(dtype="vector"),
            "state": PortType(dtype="vector"),
            "target": PortType(dtype="vector"),
            "gain": PortType(dtype="array"),
            "bias": PortType(dtype="vector"),
        },
        outputs={"value": PortType(dtype="vector")},
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=affine_composer_output_prototype
    )
    config_bindings: ClassVar[tuple[ConfigBinding, ...]] = (
        ConfigBinding("gain", name="gain_init"),
        ConfigBinding("bias", name="bias_init"),
    )

    output_block_size: int = field(static=True)
    feature_rules: tuple[dict[str, Any], ...] = field(static=True)
    use_bias: bool = field(static=True)
    gain: Array
    bias: Array

    def __init__(
        self,
        *,
        output_block_size: int = _AffineValueComposerConfig.model_fields[
            "output_block_size"
        ].default,
        feature_rules: Sequence[Mapping[str, Any]] = _AffineValueComposerConfig.model_fields[
            "feature_rules"
        ].default,
        gain_init: PyTree | None = _AffineValueComposerConfig.model_fields["gain_init"].default,
        bias_init: PyTree | None = _AffineValueComposerConfig.model_fields["bias_init"].default,
        use_bias: bool = _AffineValueComposerConfig.model_fields["use_bias"].default,
    ) -> None:
        output_block_size = int(output_block_size)
        if output_block_size <= 0:
            raise ValueError("AffineValueComposer output_block_size must be positive")
        feature_rules = _normalize_feature_rules(feature_rules)
        feature_dim = feature_dim_for_rules(feature_rules)
        gain_shape = (output_block_size, feature_dim)
        bias_shape = (output_block_size,)
        if gain_init is None:
            gain_init = jnp.zeros(gain_shape, dtype=jnp.float32)
        if bias_init is None:
            bias_init = jnp.zeros(bias_shape, dtype=jnp.float32)
        self.output_block_size = output_block_size
        self.feature_rules = feature_rules
        self.use_bias = bool(use_bias)
        self.gain = _as_array_param(gain_init, gain_shape, "gain_init")
        self.bias = _as_array_param(bias_init, bias_shape, "bias_init")

    @property
    def feature_dim(self) -> int:
        return feature_dim_for_rules(self.feature_rules)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        del key
        for port in ("base", "state"):
            if port not in inputs:
                raise ValueError(f"AffineValueComposer requires input {port!r}")
        base = jnp.asarray(inputs["base"])
        state_value = jnp.asarray(inputs["state"])
        target_value = None if "target" not in inputs else jnp.asarray(inputs["target"])
        _require_vector(base, name="base")
        if int(base.shape[-1]) != self.output_block_size:
            raise ValueError(
                "AffineValueComposer base trailing dimension "
                f"{base.shape[-1]} does not match output_block_size {self.output_block_size}"
            )

        features = _extract_features(state_value, target_value, self.feature_rules)
        gain = inputs.get("gain")
        if gain is None:
            gain = self.gain
        delta = _affine_delta(features, jnp.asarray(gain), output_block_size=self.output_block_size)
        value = base + delta.astype(base.dtype)
        if self.use_bias:
            bias = inputs.get("bias")
            if bias is None:
                bias = self.bias
            try:
                jnp.broadcast_shapes(value.shape, jnp.asarray(bias).shape)
            except ValueError as exc:
                raise ValueError(
                    "AffineValueComposer bias cannot broadcast to value shape "
                    f"{value.shape}; got {jnp.asarray(bias).shape}"
                ) from exc
            value = value + jnp.asarray(bias, dtype=value.dtype)
        return {"value": value}, state
