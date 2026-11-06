"""Components that transform signal values."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import equinox as eqx
from equinox import field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.tree as jt
import feedbax._vendor.jax_cookbook.tree as jtree
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.runtime.graph import Component
from feedbax.runtime.noise import Normal


def _float_param(value):
    return jnp.asarray(value, dtype=jnp.float32)


def _bool_param(value):
    return jnp.asarray(value, dtype=bool)


def _merge_params_override(
    base_params: "TransformParams",
    override: "TransformParams",
) -> "TransformParams":
    """Merge override params into base, keeping base values where override has None.

    Override leaves take precedence. This allows per-step time-varying params
    to override the default State-stored params.
    """
    return jt.map(
        lambda o, b: o if o is not None else b,
        override,
        base_params,
        is_leaf=lambda x: x is None,
    )


class TransformParams(eqx.Module):
    scale: Array = field(default_factory=lambda: _float_param(1.0), converter=_float_param)
    active: Array = field(default_factory=lambda: _bool_param(True), converter=_bool_param)


def _strong_typed(params: TransformParams) -> TransformParams:
    """Convert params to strong-typed JAX arrays for StateIndex storage.

    State.set() enforces exact dtype/weak-type matching between old and new
    values.  Trial-specific params from JAX random functions are strong-typed.
    To ensure compatibility, the StateIndex initial values must also be
    strong-typed.
    """

    def _convert(x):
        if isinstance(x, jnp.ndarray):
            return jnp.asarray(x, dtype=jnp.result_type(x))
        if isinstance(x, bool | int | float):
            return jnp.asarray(x, dtype=jnp.result_type(x))
        return x

    return jt.map(_convert, params)


def _match_signal_dtype(signal, result):
    """Match the unchanged branch dtype for real floating signals."""
    dtype = getattr(signal, "dtype", None)
    if dtype is not None and dtype != result.dtype and jnp.issubdtype(dtype, jnp.floating):
        return result.astype(dtype)
    return result


class AddNoiseParams(TransformParams): ...


class NetworkValueParams(TransformParams):
    unit_spec: Optional[PyTree] = None


class ConstantInputParams(TransformParams):
    arrays: Optional[PyTree] = None


class AddNoise(Component):
    """Adds noise to a signal."""

    input_ports = ("input",)
    output_ports = ("output",)

    noise_func: Callable[[PRNGKeyArray, Array], Array] = field(default_factory=Normal)
    params_index: StateIndex

    def __init__(
        self,
        noise_func: Optional[Callable[[PRNGKeyArray, Array], Array]] = None,
        params: Optional[AddNoiseParams] = None,
    ):
        if params is None:
            params = AddNoiseParams(active=False)
        if noise_func is None:
            noise_func = Normal()
        self.noise_func = noise_func
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: AddNoiseParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply_noise():
            keys = jtree.random_split_like_tree(key, signal)
            noise = jt.map(lambda x, leaf_key: self.noise_func(leaf_key, x), signal, keys)
            return jt.map(
                lambda x, n: _match_signal_dtype(x, x + params.scale * n),
                signal,
                noise,
            )

        output = jax.lax.cond(params.active, apply_noise, lambda: signal)
        return {"output": output}, state


class NetworkClamp(Component):
    """Clamps units to specified values (NaN means unchanged)."""

    input_ports = ("input",)
    output_ports = ("output",)

    params_index: StateIndex

    def __init__(
        self,
        params: Optional[NetworkValueParams] = None,
    ):
        if params is None:
            params = NetworkValueParams(active=False)
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: NetworkValueParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply():
            if params.unit_spec is None:
                return signal
            return jt.map(
                lambda x, y: _match_signal_dtype(x, jnp.where(jnp.isnan(y), x, y)),
                signal,
                params.unit_spec,
            )

        output = jax.lax.cond(params.active, apply, lambda: signal)
        return {"output": output}, state


class NetworkConstantInput(Component):
    """Adds a constant input to a signal."""

    input_ports = ("input",)
    output_ports = ("output",)

    params_index: StateIndex

    def __init__(
        self,
        params: Optional[NetworkValueParams] = None,
    ):
        if params is None:
            params = NetworkValueParams(active=False)
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: NetworkValueParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply():
            if params.unit_spec is None:
                return signal
            return jt.map(
                lambda x, y: _match_signal_dtype(x, x + params.scale * jnp.nan_to_num(y)),
                signal,
                params.unit_spec,
            )

        output = jax.lax.cond(params.active, apply, lambda: signal)
        return {"output": output}, state


class ConstantInput(Component):
    """Adds a constant array to a signal."""

    input_ports = ("input",)
    output_ports = ("output",)

    params_index: StateIndex

    def __init__(
        self,
        params: Optional[ConstantInputParams] = None,
    ):
        if params is None:
            params = ConstantInputParams(active=False)
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: ConstantInputParams = state.get(self.params_index)
        return {"output": self._apply(inputs["input"], params)}, state

    @staticmethod
    def _apply(signal: PyTree, params: ConstantInputParams) -> PyTree:
        def apply():
            if params.arrays is None:
                return signal
            return jt.map(
                lambda x, y: _match_signal_dtype(x, x + params.scale * y),
                signal,
                params.arrays,
            )

        return jax.lax.cond(params.active, apply, lambda: signal)
