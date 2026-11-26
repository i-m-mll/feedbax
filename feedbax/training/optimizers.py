"""Optimizer and schedule registries derived from Optax, and the optimizers they build.

Each Optax optimizer alias and schedule becomes a settings class whose ``type`` names the Optax
function and whose fields are the function's parameters that JSON can carry. A parameter Optax types as
"scalar or schedule" takes a schedule class, so a schedule is chosen and parameterized like any other
class. Parameters JSON cannot carry (masks, dtypes, callables) keep Optax's defaults.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, Literal, Union

import jax
import jax.numpy as jnp
import jax.typing
import optax
from optax._src import alias
from pydantic import BaseModel, ConfigDict, Field, create_model

from feedbax.contracts.canonical_json import canonical_json_v2_bytes

type _Kind = Literal["schedule", "number", "static"]

# Values for parameters Optax requires, so that every class has a complete default.
_REQUIRED_DEFAULTS: Mapping[str, Any] = {
    "learning_rate": 1e-3,
    "value": 1e-3,
    "init_value": 1e-3,
    "peak_value": 1e-3,
    "end_value": 0.0,
    "power": 1.0,
    "decay_rate": 0.5,
    "decay_steps": 1000,
    "transition_steps": 1000,
    "warmup_steps": 100,
    "interpolate_type": "linear",
}
# Defaults for JSON-visible optional parameters. The contract test compares these to Optax.
_SIGNATURE_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "constant_schedule": {},
    "cosine_decay_schedule": {"alpha": 0.0, "exponent": 1.0},
    "cosine_onecycle_schedule": {"pct_start": 0.3, "div_factor": 25.0, "final_div_factor": 10000.0},
    "exponential_decay": {"transition_begin": 0, "staircase": False, "end_value": None},
    "linear_onecycle_schedule": {
        "pct_start": 0.3, "pct_final": 0.85, "div_factor": 25.0, "final_div_factor": 10000.0,
    },
    "linear_schedule": {"transition_begin": 0},
    "piecewise_constant_schedule": {"boundaries_and_scales": None},
    "piecewise_interpolate_schedule": {"boundaries_and_scales": None},
    "polynomial_schedule": {"transition_begin": 0},
    "warmup_constant_schedule": {},
    "warmup_cosine_decay_schedule": {"end_value": 0.0, "exponent": 1.0},
    "warmup_exponential_decay_schedule": {"transition_begin": 0, "staircase": False, "end_value": None},
    "adabelief": {"b1": 0.9, "b2": 0.999, "eps": 1e-16, "eps_root": 1e-16, "nesterov": False},
    "adadelta": {"learning_rate": None, "rho": 0.9, "eps": 1e-06},
    "adafactor": {
        "learning_rate": None, "min_dim_size_to_factor": 128, "decay_rate": 0.8, "decay_offset": 0,
        "multiply_by_parameter_scale": True, "clipping_threshold": 1.0, "momentum": None,
        "weight_decay_rate": None, "eps": 1e-30, "factored": True,
    },
    "adagrad": {"initial_accumulator_value": 0.1, "eps": 1e-07},
    "adam": {"b1": 0.9, "b2": 0.999, "eps": 1e-08, "eps_root": 0.0, "nesterov": False},
    "adamax": {"b1": 0.9, "b2": 0.999, "eps": 1e-08},
    "adamaxw": {"b1": 0.9, "b2": 0.999, "eps": 1e-08, "weight_decay": 0.0001},
    "adamw": {
        "b1": 0.9, "b2": 0.999, "eps": 1e-08, "eps_root": 0.0, "weight_decay": 0.0001,
        "nesterov": False,
    },
    "adan": {"b1": 0.98, "b2": 0.92, "b3": 0.99, "eps": 1e-08, "eps_root": 1e-08, "weight_decay": 0.0},
    "amsgrad": {
        "b1": 0.9, "b2": 0.999, "eps": 1e-08, "eps_root": 0.0, "bias_correction_mu": True,
        "bias_correction_nu": True,
    },
    "fromage": {"min_norm": 1e-06},
    "lamb": {"b1": 0.9, "b2": 0.999, "eps": 1e-06, "eps_root": 0.0, "weight_decay": 0.0},
    "lars": {
        "weight_decay": 0.0, "trust_coefficient": 0.001, "eps": 0.0, "momentum": 0.9,
        "nesterov": False,
    },
    "lion": {"b1": 0.9, "b2": 0.99, "weight_decay": 0.001},
    "noisy_sgd": {"eta": 0.01, "gamma": 0.55, "seed": None},
    "novograd": {"b1": 0.9, "b2": 0.25, "eps": 1e-06, "eps_root": 0.0, "weight_decay": 0.0},
    "optimistic_adam_v2": {
        "alpha": 1.0, "beta": 1.0, "b1": 0.9, "b2": 0.999, "eps": 1e-08, "eps_root": 0.0,
        "nesterov": True,
    },
    "optimistic_gradient_descent": {"alpha": 1.0, "beta": 1.0},
    "radam": {"b1": 0.9, "b2": 0.999, "eps": 1e-08, "eps_root": 0.0, "threshold": 5.0, "nesterov": False},
    "rmsprop": {
        "decay": 0.9, "eps": 1e-08, "initial_scale": 0.0, "eps_in_sqrt": True, "centered": False,
        "momentum": None, "nesterov": False, "bias_correction": False,
    },
    "rprop": {"eta_minus": 0.5, "eta_plus": 1.2, "min_step_size": 1e-06, "max_step_size": 50.0},
    "sgd": {"momentum": None, "nesterov": False},
    "sign_sgd": {},
    "signum": {"beta": 0.9},
    "sm3": {"momentum": 0.9},
    "yogi": {"b1": 0.9, "b2": 0.999, "eps": 0.001},
}

# Optax types noisy_sgd's PRNG key as a number.
_UNREPRESENTED_PARAMETERS = frozenset({"key"})
# Optax 0.2.9 exposed these controls. Keep the established transform and
# checkpoint state by passing the equivalent upstream defaults as static arguments.
_UPSTREAM_ONLY_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "adabelief": {"weight_decay": None, "weight_decay_mask": None},
    "sm3": {"eps": 1e-08},
}
# lbfgs and polyak_sgd read the loss value through update's extra arguments, which the training step does
# not pass; Optax deprecates optimistic_adam for optimistic_adam_v2.
_EXCLUDED_OPTIMIZERS = frozenset({"lbfgs", "polyak_sgd", "optimistic_adam"})
_TITLES: Mapping[str, str] = {
    "adabelief": "AdaBelief", "adamaxw": "AdamaxW", "adamw": "AdamW", "amsgrad": "AMSGrad", "lamb": "LAMB",
    "lars": "LARS", "noisy_sgd": "Noisy SGD", "novograd": "NovoGrad", "optimistic_adam_v2": "Optimistic Adam v2",
    "radam": "RAdam", "rmsprop": "RMSProp", "sgd": "SGD", "sign_sgd": "Sign SGD", "sm3": "SM3",
    "cosine_onecycle_schedule": "Cosine One-Cycle", "linear_onecycle_schedule": "Linear One-Cycle",
}


@dataclass(frozen=True)
class RegistryEntry:
    """One Optax function and the settings class that parameterizes it."""

    key: str
    factory: Callable[..., Any]
    model: type[BaseModel]
    kinds: Mapping[str, _Kind]
    upstream_defaults: Mapping[str, Any]

    @property
    def label(self) -> str:
        return self.model.model_config["title"]


def _title(key: str) -> str:
    return _TITLES.get(key) or re.sub(r"_schedule$", "", key).replace("_", " ").title()


def _class_name(key: str) -> str:
    return "".join(part.title() for part in key.split("_"))


def _strip_none(hint: Any) -> tuple[Any, bool]:
    if typing.get_origin(hint) in (Union, types.UnionType):
        members = [member for member in typing.get_args(hint) if member is not type(None)]
        if len(members) < len(typing.get_args(hint)):
            return Union[tuple(members)], True
    return hint, False


def _field_type(hint: Any, schedule: Any) -> tuple[Any, _Kind] | None:
    base, optional = _strip_none(hint)
    if base == optax.ScalarOrSchedule and schedule is not None:
        annotation, kind = schedule, "schedule"
    elif base == jax.typing.ArrayLike or base is float:
        annotation, kind = float, "number"
    elif base in (int, bool, str) or base == dict[int, float]:
        annotation, kind = base, "static"
    else:
        return None
    return (annotation | None if optional else annotation), kind


def _entry(key: str, factory: Callable[..., Any], schedule: Any, constant: type[BaseModel] | None) -> RegistryEntry | None:
    fields: dict[str, Any] = {"type": (Literal[key], key)}
    kinds: dict[str, _Kind] = {}
    parameters = inspect.signature(factory).parameters
    upstream_defaults = {name: value for name, value in _UPSTREAM_ONLY_DEFAULTS.get(key, {}).items() if name in parameters}
    for name, parameter in parameters.items():
        if name in upstream_defaults:
            continue
        derived = None if name in _UNREPRESENTED_PARAMETERS else _field_type(parameter.annotation, schedule)
        required = parameter.default is inspect.Parameter.empty
        if derived is None:
            if required:
                return None
            continue
        annotation, kinds[name] = derived
        default = _REQUIRED_DEFAULTS[name] if required else _SIGNATURE_DEFAULTS[key][name]
        if kinds[name] == "schedule" and default is not None:
            assert constant is not None
            default = constant(value=default)
        fields[name] = (annotation, default)
    model = create_model(
        _class_name(key), __config__=ConfigDict(extra="forbid", title=_title(key)), __module__=__name__, **fields,
    )
    return RegistryEntry(key, factory, model, kinds, upstream_defaults)


def _registry(entries: list[RegistryEntry]) -> tuple[dict[str, RegistryEntry], Any]:
    registry = {entry.key: entry for entry in entries}
    union = Annotated[Union[tuple(entry.model for entry in entries)], Field(discriminator="type")]
    return registry, union


def _schedule_functions() -> list[tuple[str, Callable[..., Any]]]:
    return [
        (name, function) for name, function in inspect.getmembers(optax.schedules, inspect.isfunction)
        if inspect.signature(function).return_annotation == optax.Schedule
    ]


def _optimizer_functions() -> list[tuple[str, Callable[..., Any]]]:
    return [
        (name, function) for name, function in inspect.getmembers(alias, inspect.isfunction)
        if function.__module__ == alias.__name__ and not name.startswith("_")
        and getattr(optax, name, None) is function and name not in _EXCLUDED_OPTIMIZERS
    ]


_constant = _entry("constant_schedule", optax.constant_schedule, None, None)
assert _constant is not None
SCHEDULES, ScheduleSpec = _registry([
    entry for name, function in _schedule_functions()
    if (entry := _constant if name == _constant.key else _entry(name, function, None, None)) is not None
])
OPTIMIZERS, OptimizerSpec = _registry([
    entry for name, function in _optimizer_functions()
    if (entry := _entry(name, function, ScheduleSpec, _constant.model)) is not None
])
"""``OptimizerSpec`` is the tagged union of the optimizer classes; ``ScheduleSpec`` that of the schedules."""

DEFAULT_OPTIMIZER = OPTIMIZERS["adam"].model()
_ENTRIES = {entry.model: entry for registry in (SCHEDULES, OPTIMIZERS) for entry in registry.values()}


def _arguments(spec: BaseModel) -> tuple[RegistryEntry, dict[str, Any]]:
    entry = _ENTRIES[type(spec)]
    return entry, {name: getattr(spec, name) for name in entry.kinds}


def build_schedule(spec: BaseModel) -> optax.Schedule:
    """Return the Optax schedule a schedule class declares."""
    entry, arguments = _arguments(spec)
    return entry.factory(**arguments)


@dataclass(frozen=True)
class _StepSchedule:
    rate: optax.Schedule
    dtype: jax.typing.DTypeLike

    def init(self) -> tuple[()]:
        return ()

    def update(self, state: tuple[()], **_extra_args) -> tuple[()]:
        return state

    def __call__(self, _state: tuple[()], *, schedule_step: int | jax.Array = 0, **_extra_args):
        # Optax derives the arithmetic precision of nonconstant schedules from the clock.
        if self.dtype == jnp.float64:
            schedule_step = jnp.asarray(schedule_step, dtype=jnp.int64)
        return jnp.asarray(self.rate(schedule_step), dtype=self.dtype)


def build_step_optimizer(
    spec: BaseModel, *, gradient_clip: float | None = None,
) -> optax.GradientTransformationExtraArgs:
    """Build an optimizer whose scheduled parameters follow an explicit update clock."""
    entry, arguments = _arguments(spec)
    static = list(entry.upstream_defaults)
    schedules = {}
    for name, kind in entry.kinds.items():
        value = arguments[name]
        if kind == "schedule" and value is not None:
            schedules[name] = build_schedule(value)
        elif kind == "static" or value is None:
            static.append(name)
    arguments.update(entry.upstream_defaults)
    inject = optax.inject_hyperparams(entry.factory, static_args=static)
    if schedules:
        rate_name = next(iter(schedules))

        @cache
        def with_dtype(dtype):
            scheduled = {name: _StepSchedule(rate, dtype) for name, rate in schedules.items()}
            return inject(**(arguments | scheduled))

        def init(params):
            return with_dtype(optax.tree.dtype(params, "highest")).init(params)

        def update(updates, state, params=None, **extra_args):
            return with_dtype(state.hyperparams[rate_name].dtype).update(updates, state, params, **extra_args)

        optimizer = optax.GradientTransformationExtraArgs(init, update)
    else:
        optimizer = inject(**arguments)
    if gradient_clip is None:
        return optax.chain(optimizer)
    return optax.chain(optax.clip_by_global_norm(float(gradient_clip)), optimizer)


def optimizer_configuration(spec: BaseModel) -> tuple[str, bytes]:
    """Identify the optimizer state's meaning: its class and every parameter that is not on the update clock."""
    entry, _ = _arguments(spec)
    fixed = [name for name, kind in entry.kinds.items() if kind != "schedule"]
    return spec.type, canonical_json_v2_bytes(spec.model_dump(mode="json", include=set(fixed)))
