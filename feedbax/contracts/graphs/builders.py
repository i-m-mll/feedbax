from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from feedbax.models.feedback import FeedbackChannels
from feedbax.runtime.channel import Channel, ChannelSpec
from feedbax.runtime.components import (
    Constant,
    Damper,
    DelayLine,
    Demux,
    ElementwiseAffineModulator,
    GRU,
    Gain,
    LSTM,
    Linear,
    MLP,
    Multiply,
    Mux,
    Noise,
    Pulse,
    Ramp,
    Ravel,
    Saturation,
    Sine,
    Spring,
    Sum,
)
from feedbax.runtime.affine_composer import (
    AFFINE_VALUE_COMPOSER_SCHEMA_VERSION,
    AffineValueComposer,
)
from feedbax.control.affine import build_affine_feedback_controller
from feedbax.runtime.filters import FirstOrderFilter
from feedbax.runtime.graph import Component
from feedbax.intervene.intervene import (
    AddNoise,
    AddNoiseParams,
    ConstantInput,
    ConstantInputParams,
    Copy,
    CurlField,
    CurlFieldParams,
    DynamicsMatrixPerturb,
    DynamicsMatrixPerturbParams,
    FixedField,
    FixedFieldParams,
    NetworkClamp,
    NetworkConstantInput,
    NetworkIntervenorParams,
)
from feedbax.objectives.loss import CompositeLoss
from feedbax.mechanics.linear_state_space import LinearStateSpace
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.muscles.relu_muscle import ReluMuscle
from feedbax.mechanics.muscles.thelen_muscle import RigidTendonHillMuscleThelen
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.models.networks import (
    LeakyRNNCell,
    SimpleStagedNetwork,
    VanillaRNN,
    population_structure_from_spec,
)
from feedbax.models.support import identity_func
from feedbax.runtime.noise import Multiplicative, Normal
from feedbax.components.penzai import (
    PENZAI_AVAILABLE,
    build_penzai_subgraph,
)
from feedbax.contracts.graphs.prototypes import array_proto_from_shape
from feedbax.runtime.state_feedback import build_state_feedback_selector
from feedbax.tasks import DelayedReaches, SimpleReaches, Stabilization, TaskComponent
from feedbax.tasks.presets import (
    apply_delayed_reaches_preset,
    delayed_reaches_n_steps_from_params,
)


_HIDDEN_TYPES: dict[str, Callable[..., eqx.Module]] = {
    "GRUCell": eqx.nn.GRUCell,
    "LSTMCell": eqx.nn.LSTMCell,
    "Linear": eqx.nn.Linear,
    "GRU": eqx.nn.GRUCell,
    "LSTM": eqx.nn.LSTMCell,
    "VanillaRNN": LeakyRNNCell,
    "VanillaRNNCell": LeakyRNNCell,
    "RNN": LeakyRNNCell,
    "RNNCell": LeakyRNNCell,
    "LeakyRNNCell": LeakyRNNCell,
}
_NONLINEARITIES: dict[str, Callable[[jax.Array], jax.Array]] = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "softmax": jax.nn.softmax,
    "identity": identity_func,
}


def resolve_nonlinearity(name: str | None) -> Callable[[jax.Array], jax.Array]:
    if not name:
        return _NONLINEARITIES["identity"]
    if name not in _NONLINEARITIES:
        raise ValueError(f"Unknown nonlinearity: {name!r}. Valid options: {list(_NONLINEARITIES)}")
    return _NONLINEARITIES[name]


def nonlinearity_name(fn: Callable[[jax.Array], jax.Array]) -> str:
    for name, func in _NONLINEARITIES.items():
        if fn is func:
            return name
    name = getattr(fn, "__name__", "")
    return name if name in _NONLINEARITIES else "identity"


def _hidden_type_vocabulary() -> str:
    return ", ".join(_HIDDEN_TYPES)


def _resolve_hidden_type(name: object, *, path: str) -> Callable[..., eqx.Module]:
    hidden_type_name = str(name)
    try:
        return _HIDDEN_TYPES[hidden_type_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown hidden_type at {path}: {hidden_type_name!r}. "
            f"Supported values: {_hidden_type_vocabulary()}"
        ) from exc


def _compat_dtype(params: Mapping[str, Any]) -> object:
    if "dtype" in params:
        return params["dtype"]
    return jnp.asarray(0.0).dtype


def _build_network(params: Mapping[str, Any]) -> SimpleStagedNetwork:
    hidden_type = _resolve_hidden_type(
        params.get("hidden_type", "GRUCell"),
        path="Network.params.hidden_type",
    )
    hidden_nonlinearity = resolve_nonlinearity(str(params.get("hidden_nonlinearity", "tanh")))
    out_nonlinearity = resolve_nonlinearity(str(params.get("out_nonlinearity", "tanh")))
    if hidden_type is LeakyRNNCell:
        hidden_type = partial(LeakyRNNCell, nonlinearity=hidden_nonlinearity)
        hidden_nonlinearity = identity_func
    encoding_size = int(params.get("encoding_size", 0) or 0)
    encoding_size = encoding_size if encoding_size > 0 else None
    out_size = params.get("out_size", params.get("output_size"))
    out_size = int(out_size) if out_size not in (None, "") else None
    if out_size is not None and out_size <= 0:
        out_size = None
    hidden_noise_std = params.get("hidden_noise_std", 0.0)
    if hidden_noise_std in (None, 0, 0.0):
        hidden_noise_std = None
    population_structure = None
    raw_population_structure = params.get("population_structure")
    if isinstance(raw_population_structure, Mapping):
        population_structure = population_structure_from_spec(
            int(params.get("hidden_size", 0)),
            raw_population_structure,
        )
    return SimpleStagedNetwork(
        input_size=int(params.get("input_size", 0)),
        hidden_size=int(params.get("hidden_size", 0)),
        out_size=out_size,
        encoding_size=encoding_size,
        hidden_type=hidden_type,
        hidden_nonlinearity=hidden_nonlinearity,
        out_nonlinearity=out_nonlinearity,
        hidden_noise_std=hidden_noise_std,
        population_structure=population_structure,
        population_mask_mode=str(params.get("population_mask_mode", "legacy_masked")),
        dtype=_compat_dtype(params),
        key=jr.PRNGKey(0),
    )


def _build_mechanics(params: Mapping[str, Any]) -> Mechanics:
    plant_type = params.get("plant_type", "TwoLinkArm")
    if plant_type == "TwoLinkArm":
        plant = DirectForceInput(
            TwoLinkArm(l=params.get("link_lengths", (0.30, 0.33)))
        )
    elif plant_type == "PointMass":
        plant = DirectForceInput(
            PointMass(
                mass=float(params.get("mass", 1.0)),
                damping=float(params.get("damping", 0.0)),
            )
        )
    else:
        raise ValueError(f"Unsupported plant_type '{plant_type}'")
    return Mechanics(plant=plant, dt=float(params.get("dt", 0.01)))


def _build_linear_state_space(params: Mapping[str, Any]) -> LinearStateSpace:
    return LinearStateSpace(
        A=jnp.asarray(params["A"]),
        B=jnp.asarray(params["B"]),
        B_w=None if params.get("B_w") is None else jnp.asarray(params["B_w"]),
        dt=float(params.get("dt", 1.0)),
        initial_state=None
        if params.get("initial_state") is None
        else jnp.asarray(params["initial_state"]),
        pos_slice=tuple(params.get("pos_slice", [0, 2])),
        vel_slice=tuple(params.get("vel_slice", [2, 4])),
    )


def _noise_params(params: Mapping[str, Any]) -> dict[str, Any]:
    nested = params.get("noise")
    noise_params = dict(nested) if isinstance(nested, Mapping) else {}
    model = str(
        noise_params.get(
            "model",
            noise_params.get("type", params.get("noise_model", "additive_gaussian")),
        )
    )
    if model in {"gaussian", "additive", "normal"}:
        model = "additive_gaussian"
    elif model in {"signal_dependent", "multiplicative_gaussian"}:
        model = "signal_dependent_gaussian"
    elif model in {"command", "motor_command"}:
        model = "signal_dependent_plus_additive"

    noise_std = noise_params.get("std", params.get("noise_std", 0.0))
    additive_std = noise_params.get(
        "additive_std",
        noise_params.get("additive_noise_std", params.get("additive_noise_std", noise_std)),
    )
    if model == "additive_gaussian" and noise_std not in (None, 0, 0.0):
        additive_std = noise_std
    signal_dependent_std = noise_params.get(
        "signal_dependent_std",
        noise_params.get(
            "signal_dependent_noise_std",
            params.get("signal_dependent_noise_std", 0.0),
        ),
    )
    add_noise_default = model != "none"
    add_noise = bool(noise_params.get("add_noise", params.get("add_noise", add_noise_default)))
    return {
        "model": model,
        "add_noise": add_noise,
        "additive_std": float(additive_std or 0.0),
        "signal_dependent_std": float(signal_dependent_std or 0.0),
        "role": noise_params.get("role", params.get("noise_role")),
        "timing": noise_params.get("timing", params.get("noise_timing")),
    }


def _build_noise_func(noise: Mapping[str, Any]):
    model = str(noise["model"])
    add_noise = bool(noise["add_noise"])
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
    if model == "none" or not add_noise:
        return None
    if model == "additive_gaussian":
        return Normal(std=additive_std) if additive_std != 0.0 else None
    if model == "signal_dependent_gaussian":
        return (
            Multiplicative(Normal(std=signal_dependent_std))
            if signal_dependent_std != 0.0
            else None
        )
    if model == "signal_dependent_plus_additive":
        terms = []
        if signal_dependent_std != 0.0:
            terms.append(Multiplicative(Normal(std=signal_dependent_std)))
        if additive_std != 0.0:
            terms.append(Normal(std=additive_std))
        if not terms:
            return None
        noise_func = terms[0]
        for term in terms[1:]:
            noise_func = noise_func + term
        return noise_func
    raise AssertionError(f"Unhandled Channel noise_model {model!r}")


def _build_channel(params: Mapping[str, Any]) -> Channel:
    delay = int(params.get("delay", 0))
    noise = _noise_params(params)
    noise_func = _build_noise_func(noise)
    input_proto = None
    input_shape = params.get("input_shape")
    if isinstance(input_shape, (list, tuple)):
        input_proto = jnp.zeros(tuple(int(dim) for dim in input_shape))
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


def _path_selector(paths: tuple[str, ...]):
    def _select(mechanics_state):
        values = []
        for path in paths:
            value = mechanics_state
            for part in path.split("."):
                if not hasattr(value, part):
                    raise AttributeError(
                        f"Mechanics feedback selector path {path!r} has no {part!r}"
                    )
                value = getattr(value, part)
            values.append(value)
        return values[0] if len(values) == 1 else tuple(values)

    _select._feedbax_feedback_selector = "paths"  # type: ignore[attr-defined]
    _select._feedbax_feedback_paths = paths  # type: ignore[attr-defined]
    return _select


def _feedback_selector(params: Mapping[str, Any]):
    selector = str(params.get("selector", "point_mass_pos_vel"))
    raw_paths = params.get("paths")
    if raw_paths is not None and selector == "paths":
        if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
            raise ValueError("FeedbackChannels 'paths' must be a non-empty list of attribute paths")
        return "paths", tuple(str(path) for path in raw_paths)

    if selector in {
        "point_mass_pos_vel",
        "point_mass_state",
        "mechanics.plant.skeleton.pos_vel",
    }:
        return "point_mass_pos_vel", ("plant.skeleton.pos", "plant.skeleton.vel")
    if selector in {"effector_pos_vel", "mechanics.effector.pos_vel"}:
        return "effector_pos_vel", ("effector.pos", "effector.vel")
    if selector in {"plant_skeleton", "mechanics.plant.skeleton"}:
        return "plant_skeleton", ("plant.skeleton",)
    raise ValueError(
        f"Unsupported FeedbackChannels selector {selector!r}; expected point_mass_pos_vel, "
        "effector_pos_vel, plant_skeleton, or explicit paths"
    )


def _proto_from_shape_spec(shape: Any):
    if not isinstance(shape, (list, tuple)):
        return None
    if shape and all(isinstance(item, (list, tuple)) for item in shape):
        return tuple(jnp.zeros(tuple(int(dim) for dim in item)) for item in shape)
    return jnp.zeros(tuple(int(dim) for dim in shape))


def _feedback_input_proto(params: Mapping[str, Any], paths: tuple[str, ...]):
    proto = _proto_from_shape_spec(params.get("input_shape"))
    if proto is not None:
        return proto
    if paths == ("plant.skeleton.pos", "plant.skeleton.vel") or paths == (
        "effector.pos",
        "effector.vel",
    ):
        return (jnp.zeros(2), jnp.zeros(2))
    return jnp.zeros(1)


def _build_feedback_channels(params: Mapping[str, Any]) -> FeedbackChannels:
    delay = int(params.get("delay", 0))
    selector, paths = _feedback_selector(params)
    noise = _noise_params(params)
    noise_func = _build_noise_func(noise)
    where = _path_selector(paths)
    where._feedbax_feedback_selector = selector  # type: ignore[attr-defined]
    channel = Channel(
        delay=delay,
        noise_func=noise_func,
        add_noise=bool(noise["add_noise"]),
        input_proto=_feedback_input_proto(params, paths),
        init_value=float(params.get("init_value", 0.0)),
        noise_model=str(noise["model"]),
        noise_role=None if noise["role"] is None else str(noise["role"]),
        noise_timing=None if noise["timing"] is None else str(noise["timing"]),
    )
    spec = ChannelSpec(where=where, delay=delay, noise_func=noise_func)
    return FeedbackChannels(channel, spec)


def _build_filter(params: Mapping[str, Any]) -> FirstOrderFilter:
    input_proto = array_proto_from_shape(params.get("input_shape"))
    return FirstOrderFilter(
        tau_rise=float(params.get("tau_rise", 0.05)),
        tau_decay=float(params.get("tau_decay", 0.05)),
        dt=float(params.get("dt", 0.001)),
        input_proto=input_proto,
        init_value=float(params.get("init_value", 0.0)),
    )


def _build_task_component(task_type: str, params: Mapping[str, Any]) -> TaskComponent:
    loss_func = CompositeLoss(())
    if task_type == "SimpleReaches":
        task = SimpleReaches(
            loss_func=loss_func,
            n_steps=int(params.get("n_steps", 200)),
            workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
            eval_n_directions=int(params.get("eval_n_directions", 7)),
            eval_reach_length=float(params.get("eval_reach_length", 0.5)),
            eval_grid_n=int(params.get("eval_grid_n", 1)),
        )
    elif task_type == "DelayedReaches":
        params = apply_delayed_reaches_preset(params)
        task = DelayedReaches(
            loss_func=loss_func,
            n_steps=delayed_reaches_n_steps_from_params(params),
            workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
            preset=params.get("preset", None),
            train_endpoint_mode=str(params.get("train_endpoint_mode", "workspace")),
            epoch_len_ranges=tuple(
                tuple(int(value) for value in item)
                for item in params.get("epoch_len_ranges", [[5, 15], [10, 20]])
            ),
            epoch_names=tuple(str(value) for value in params.get("epoch_names", []))
            or ("hold", "target_on", "movement"),
            target_on_epochs=tuple(int(value) for value in params.get("target_on_epochs", [1, 2])),
            hold_epochs=tuple(int(value) for value in params.get("hold_epochs", [0, 1])),
            move_epochs=tuple(int(value) for value in params.get("move_epochs", [2])),
            p_catch_trial=float(params.get("p_catch_trial", 0.5)),
            target_visible_from_start=bool(params.get("target_visible_from_start", False)),
            go_cue_event_name=params.get("go_cue_event_name", None),
            catch_metadata_policy=str(params.get("catch_metadata_policy", "none")),
            eval_n_directions=int(params.get("eval_n_directions", 7)),
            eval_reach_length=float(params.get("eval_reach_length", 0.5)),
            eval_grid_n=int(params.get("eval_grid_n", 1)),
        )
    elif task_type == "Stabilization":
        task = Stabilization(
            loss_func=loss_func,
            n_steps=int(params.get("n_steps", 200)),
            workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
        )
    else:
        raise ValueError(f"Unsupported task type '{task_type}'")

    mode = params.get("mode", "open_loop")
    if mode != "open_loop":
        raise ValueError("Only open_loop TaskComponent is supported from GraphSpec")

    trial_spec = task.get_train_trial_with_intervenor_params(key=jr.PRNGKey(0))
    return TaskComponent(task=task, trial_spec=trial_spec, mode="open_loop")


def _build_penzai_adapter(params: Mapping[str, Any]) -> Component:
    builder_name = str(params.get("builder_name", ""))
    if not builder_name:
        raise ValueError("PenzaiAdapter requires 'builder_name' parameter")
    if not PENZAI_AVAILABLE:
        raise ImportError(
            "penzai is required to instantiate PenzaiAdapter. Install with: pip install penzai"
        )
    builder_params = {
        key: value
        for key, value in params.items()
        if key not in ("builder_name", "input_port", "output_port")
    }
    return build_penzai_subgraph(
        builder_name=builder_name,
        params=builder_params,
        input_port=str(params.get("input_port", "input")),
        output_port=str(params.get("output_port", "output")),
    )


def _build_gain(params: Mapping[str, Any]) -> Gain:
    return Gain(gain=float(params.get("gain", 1.0)))


def _build_sum(params: Mapping[str, Any]) -> Sum:
    return Sum()


def _build_multiply(params: Mapping[str, Any]) -> Multiply:
    return Multiply()


def _build_elementwise_affine_modulator(
    params: Mapping[str, Any],
) -> ElementwiseAffineModulator:
    signal_shape = params.get("signal_shape")
    if not isinstance(signal_shape, (list, tuple)):
        raise ValueError("ElementwiseAffineModulator requires array parameter 'signal_shape'")
    return ElementwiseAffineModulator(
        signal_shape=signal_shape,
        baseline=params.get("baseline", 1.0),
        gain_init=params.get("gain_init", 0.0),
        bias_init=params.get("bias_init", 0.0),
    )


def _build_constant(params: Mapping[str, Any]) -> Constant:
    return Constant(value=params.get("value", 0.0))


def _build_ramp(params: Mapping[str, Any]) -> Ramp:
    return Ramp(
        slope=params.get("slope", 1.0),
        intercept=params.get("intercept", 0.0),
        dt=float(params.get("dt", 0.01)),
    )


def _build_sine(params: Mapping[str, Any]) -> Sine:
    return Sine(
        amplitude=params.get("amplitude", 1.0),
        frequency=float(params.get("frequency", 1.0)),
        phase=float(params.get("phase", 0.0)),
        offset=params.get("offset", 0.0),
        dt=float(params.get("dt", 0.01)),
    )


def _build_pulse(params: Mapping[str, Any]) -> Pulse:
    return Pulse(
        amplitude=params.get("amplitude", 1.0),
        period=float(params.get("period", 1.0)),
        duty_cycle=float(params.get("duty_cycle", 0.5)),
        offset=params.get("offset", 0.0),
        dt=float(params.get("dt", 0.01)),
    )


def _build_noise(params: Mapping[str, Any]) -> Noise:
    shape = params.get("shape", [1])
    if not isinstance(shape, (list, tuple)):
        shape = [int(shape)]
    return Noise(
        mean=float(params.get("mean", 0.0)),
        std=float(params.get("std", 1.0)),
        shape=shape,
    )


def _build_saturation(params: Mapping[str, Any]) -> Saturation:
    return Saturation(
        min_val=float(params.get("min_val", -1.0)),
        max_val=float(params.get("max_val", 1.0)),
    )


def _build_delay_line(params: Mapping[str, Any]) -> DelayLine:
    input_proto = array_proto_from_shape(params.get("input_shape"))
    return DelayLine(
        delay=int(params.get("delay", 1)),
        init_value=float(params.get("init_value", 0.0)),
        input_proto=input_proto,
    )


def _build_mlp(params: Mapping[str, Any]) -> MLP:
    hidden_sizes = params.get("hidden_sizes", [64])
    if not isinstance(hidden_sizes, (list, tuple)):
        hidden_sizes = [int(hidden_sizes)]
    return MLP(
        input_size=int(params.get("input_size", 1)),
        output_size=int(params.get("output_size", 1)),
        hidden_sizes=hidden_sizes,
        activation=str(params.get("activation", "relu")),
        final_activation=str(params.get("final_activation", "identity")),
        dtype=_compat_dtype(params),
        key=jr.PRNGKey(0),
    )


def _build_linear(params: Mapping[str, Any]) -> Linear:
    return Linear(
        input_size=int(params.get("input_size", 1)),
        output_size=int(params.get("output_size", 1)),
        use_bias=bool(params.get("use_bias", True)),
        activation=str(params.get("activation", "identity")),
        dtype=_compat_dtype(params),
        key=jr.PRNGKey(0),
    )


def _build_mux(params: Mapping[str, Any]) -> Mux:
    return Mux(n_inputs=int(params.get("n_inputs", 2)))


def _build_demux(params: Mapping[str, Any]) -> Demux:
    sizes = params.get("sizes")
    if not isinstance(sizes, (list, tuple)):
        raise ValueError("Demux requires 'sizes' as a non-empty list of positive integers")
    return Demux(sizes=sizes)


def _build_gru(params: Mapping[str, Any]) -> GRU:
    return GRU(
        input_size=int(params.get("input_size", 1)),
        hidden_size=int(params.get("hidden_size", 1)),
        dtype=_compat_dtype(params),
        key=jr.PRNGKey(0),
    )


def _build_vanilla_rnn(params: Mapping[str, Any]) -> VanillaRNN:
    activation_name = str(params.get("activation", params.get("nonlinearity", "tanh")))
    dtype = params.get("dtype", jnp.float32) or jnp.float32
    return VanillaRNN(
        input_size=int(params.get("input_size", 1)),
        hidden_size=int(params.get("hidden_size", 1)),
        activation_name=activation_name,
        nonlinearity=resolve_nonlinearity(activation_name),
        use_bias=bool(params.get("use_bias", True)),
        dtype=dtype,
        key=jr.PRNGKey(0),
    )


def _build_lstm(params: Mapping[str, Any]) -> LSTM:
    return LSTM(
        input_size=int(params.get("input_size", 1)),
        hidden_size=int(params.get("hidden_size", 1)),
        dtype=_compat_dtype(params),
        key=jr.PRNGKey(0),
    )


def _build_spring(params: Mapping[str, Any]) -> Spring:
    return Spring(stiffness=float(params.get("stiffness", 1.0)))


def _build_damper(params: Mapping[str, Any]) -> Damper:
    return Damper(damping=float(params.get("damping", 1.0)))


def _build_relu_muscle(params: Mapping[str, Any]) -> ReluMuscle:
    return ReluMuscle(
        max_isometric_force=float(params.get("max_isometric_force", 500.0)),
        tau_activation=float(params.get("tau_activation", 0.015)),
        tau_deactivation=float(params.get("tau_deactivation", 0.05)),
        min_activation=float(params.get("min_activation", 0.0)),
        dt=float(params.get("dt", 0.01)),
        initial_activation=float(params.get("initial_activation", 0.0)),
    )


def _build_rigid_tendon_hill_muscle_thelen(
    params: Mapping[str, Any],
) -> RigidTendonHillMuscleThelen:
    return RigidTendonHillMuscleThelen(
        max_isometric_force=float(params.get("max_isometric_force", 500.0)),
        optimal_muscle_length=float(params.get("optimal_muscle_length", 0.1)),
        tendon_slack_length=float(params.get("tendon_slack_length", 0.1)),
        vmax_factor=float(params.get("vmax_factor", 10.0)),
        min_activation=float(params.get("min_activation", 0.001)),
        tau_activation=float(params.get("tau_activation", 0.015)),
        tau_deactivation=float(params.get("tau_deactivation", 0.05)),
        dt=float(params.get("dt", 0.01)),
        initial_activation=float(params.get("initial_activation", 0.001)),
    )


def _build_curl_field(params: Mapping[str, Any]) -> CurlField:
    return CurlField(
        params=CurlFieldParams(
            scale=float(params.get("scale", 1.0)),
            amplitude=float(params.get("amplitude", 1.0)),
            active=bool(params.get("active", False)),
        ),
        label=str(params.get("label", "curl_field")),
    )


def _build_fixed_field(params: Mapping[str, Any]) -> FixedField:
    return FixedField(
        params=FixedFieldParams(
            scale=float(params.get("scale", 1.0)),
            amplitude=float(params.get("amplitude", 1.0)),
            field=jnp.asarray(params.get("field", [0.0, 0.0])),
            active=bool(params.get("active", False)),
        ),
        label=str(params.get("label", "fixed_field")),
    )


def _build_dynamics_matrix_perturb(params: Mapping[str, Any]) -> DynamicsMatrixPerturb:
    if "mass" not in params or params["mass"] is None:
        raise ValueError(
            "DynamicsMatrixPerturb requires explicit mass matching the wired plant"
        )
    delta_A = jnp.asarray(params.get("delta_A", [[0.0, 0.0, 0.0, 0.0]] * 2))
    if delta_A.ndim != 2:
        raise ValueError("DynamicsMatrixPerturb delta_A must be a rank-2 array")
    if delta_A.shape[1] != 2 * delta_A.shape[0]:
        raise ValueError("DynamicsMatrixPerturb delta_A must have shape (n_dim, 2 * n_dim)")
    return DynamicsMatrixPerturb(
        params=DynamicsMatrixPerturbParams(
            scale=float(params.get("scale", 1.0)),
            active=bool(params.get("active", False)),
            delta_A=delta_A,
        ),
        label=str(params.get("label", "dynamics_matrix_perturb")),
        mass=float(params["mass"]),
    )


def _build_affine_value_composer(params: Mapping[str, Any]) -> AffineValueComposer:
    schema_version = str(params.get("schema_version", AFFINE_VALUE_COMPOSER_SCHEMA_VERSION))
    if schema_version != AFFINE_VALUE_COMPOSER_SCHEMA_VERSION:
        raise ValueError(
            "AffineValueComposer unsupported schema_version "
            f"{schema_version!r}; expected {AFFINE_VALUE_COMPOSER_SCHEMA_VERSION!r}"
        )
    return AffineValueComposer(
        output_block_size=int(params.get("output_block_size", 1)),
        feature_rules=params.get(
            "feature_rules",
            [{"kind": "identity", "state_slice": [0, 1]}],
        ),
        gain_init=params.get("gain_init"),
        bias_init=params.get("bias_init"),
        use_bias=bool(params.get("use_bias", True)),
        label=str(params.get("label", "affine_value_composer")),
    )


_BUILDERS: dict[str, Callable[[Mapping[str, Any]], Component]] = {
    "Gain": _build_gain,
    "Sum": _build_sum,
    "Multiply": _build_multiply,
    "ElementwiseAffineModulator": _build_elementwise_affine_modulator,
    "Constant": _build_constant,
    "Ravel": lambda params: Ravel(),
    "Ramp": _build_ramp,
    "Sine": _build_sine,
    "Pulse": _build_pulse,
    "Noise": _build_noise,
    "Saturation": _build_saturation,
    "DelayLine": _build_delay_line,
    "Linear": _build_linear,
    "MLP": _build_mlp,
    "Mux": _build_mux,
    "Demux": _build_demux,
    "GRU": _build_gru,
    "VanillaRNN": _build_vanilla_rnn,
    "LSTM": _build_lstm,
    "Spring": _build_spring,
    "Damper": _build_damper,
    "ReluMuscle": _build_relu_muscle,
    "RigidTendonHillMuscleThelen": _build_rigid_tendon_hill_muscle_thelen,
    "LinearStateSpace": _build_linear_state_space,
    "StateFeedbackSelector": build_state_feedback_selector,
    "AffineFeedbackController": build_affine_feedback_controller,
    "Channel": _build_channel,
    "FeedbackChannels": _build_feedback_channels,
    "FirstOrderFilter": _build_filter,
    "CurlField": _build_curl_field,
    "FixedField": _build_fixed_field,
    "DynamicsMatrixPerturb": _build_dynamics_matrix_perturb,
    "AffineValueComposer": _build_affine_value_composer,
    "AddNoise": lambda params: AddNoise(
        params=AddNoiseParams(
            scale=float(params.get("scale", 1.0)),
            active=bool(params.get("active", False)),
        )
    ),
    "NetworkClamp": lambda params: NetworkClamp(
        params=NetworkIntervenorParams(
            scale=float(params.get("scale", 1.0)),
            active=bool(params.get("active", False)),
        )
    ),
    "NetworkConstantInput": lambda params: NetworkConstantInput(
        params=NetworkIntervenorParams(
            scale=float(params.get("scale", 1.0)),
            active=bool(params.get("active", False)),
        )
    ),
    "ConstantInput": lambda params: ConstantInput(
        params=ConstantInputParams(
            scale=float(params.get("scale", 1.0)),
            active=bool(params.get("active", False)),
        )
    ),
    "Copy": lambda params: Copy(),
    "TwoLinkArm": lambda params: _build_mechanics({**dict(params), "plant_type": "TwoLinkArm"}),
    "PointMass": lambda params: _build_mechanics({**dict(params), "plant_type": "PointMass"}),
    "SimpleReaches": lambda params: _build_task_component("SimpleReaches", params),
    "DelayedReaches": lambda params: _build_task_component("DelayedReaches", params),
    "Stabilization": lambda params: _build_task_component("Stabilization", params),
    "PenzaiAdapter": _build_penzai_adapter,
}


_DISPLAY_ONLY_MESSAGES: dict[str, str] = {
    "MomentArmProjection": (
        "MomentArmProjection node {node_name!r} has no Python builder yet. "
        "It is a display-only abstraction used in composite subgraph templates."
    ),
    "RadialForceProjection": (
        "RadialForceProjection node {node_name!r} has no Python builder yet. "
        "It is a display-only abstraction used in composite subgraph templates."
    ),
    "Arm6MuscleRigidTendon": (
        "Arm6MuscleRigidTendon node {node_name!r} requires musculoskeletal plant data "
        "(body presets, muscle topology) not stored in GraphSpec. This node type is "
        "supported in the training worker but not in local graph instantiation. "
        "For testing, use TwoLinkArm instead."
    ),
}


def _unsupported_component_builder(component_type: str) -> Callable[[Mapping[str, Any]], Component]:
    message = _DISPLAY_ONLY_MESSAGES.get(
        component_type,
        f"Component type {component_type!r} is registered for metadata but has no "
        "executable builder.",
    )

    def _builder(params: Mapping[str, Any]) -> Component:
        del params
        raise NotImplementedError(message.format(node_name="<unknown>"))

    _builder._feedbax_unsupported_builder = True  # type: ignore[attr-defined]
    _builder._feedbax_unsupported_builder_message = message  # type: ignore[attr-defined]
    return _builder


_UNREGISTERED_TEMPLATE_MESSAGES: dict[str, str] = {
    "Input": (
        "CDE template primitive {node_type!r} at node {node_name!r} is not executable. "
        "Graph inputs must be represented by GraphSpec input_bindings, not display-only "
        "placeholder nodes. CDE templates are currently display-only and fail closed; "
        "real subgraph-to-builder construction is future work tracked by issue 2f8dd61."
    ),
    "Subtract": (
        "CDE template primitive {node_type!r} at node {node_name!r} is not executable. "
        "CDE templates are currently display-only and fail closed; real subgraph-to-builder "
        "construction is future work tracked by issue 2f8dd61."
    ),
    "Reshape": (
        "CDE template primitive {node_type!r} at node {node_name!r} is not executable. "
        "CDE templates are currently display-only and fail closed; real subgraph-to-builder "
        "construction is future work tracked by issue 2f8dd61."
    ),
    "MatMul": (
        "CDE template primitive {node_type!r} at node {node_name!r} is not executable. "
        "CDE templates are currently display-only and fail closed; real subgraph-to-builder "
        "construction is future work tracked by issue 2f8dd61."
    ),
    "Scale": (
        "CDE template primitive {node_type!r} at node {node_name!r} is not executable. "
        "CDE templates are currently display-only and fail closed; real subgraph-to-builder "
        "construction is future work tracked by issue 2f8dd61."
    ),
    "Sigmoid": (
        "CDE template primitive {node_type!r} at node {node_name!r} is not executable. "
        "CDE templates are currently display-only and fail closed; real subgraph-to-builder "
        "construction is future work tracked by issue 2f8dd61."
    ),
}


def _template_builder_error(meta: Any, component_registry: Any) -> str | None:
    if getattr(meta, "template_graph", None) is None:
        return None
    template_builder_issues = getattr(component_registry, "template_builder_issues", None)
    if not callable(template_builder_issues):
        return None
    issues = template_builder_issues(meta)
    if not issues:
        return None
    details = "; ".join(issue.summary for issue in issues[:6])
    if len(issues) > 6:
        details += f"; and {len(issues) - 6} more"
    message = (
        f"Component template {meta.name!r} is not executable because its template graph "
        f"contains node types without registered builders: {details}"
    )
    template_id = getattr(meta, "template_id", "") or ""
    if template_id.startswith("feedbax.templates.cde_"):
        message += (
            ". CDE templates are currently display-only and fail closed; real "
            "subgraph-to-builder construction is future work tracked by issue 2f8dd61."
        )
    return message


def register_builtin_component_builders(registry: Any) -> None:
    for name, builder in _BUILDERS.items():
        registry.register_builder(name, builder, provenance="feedbax")
    for name in registry.names():
        meta = registry.get(name)
        if meta is not None and meta.builder is None:
            registry.register_builder(
                name,
                _unsupported_component_builder(name),
                provenance="feedbax",
            )


def build_component(
    node_name: str,
    node_type: str,
    params: Mapping[str, Any],
    *,
    component_registry: Any = None,
) -> Component:
    """Build a leaf component from a GraphSpec node."""

    if component_registry is None or not hasattr(component_registry, "names"):
        from feedbax.component_registry import get_component_registry

        component_registry = get_component_registry()
    meta = component_registry.get(node_type)
    if meta is None:
        message = _UNREGISTERED_TEMPLATE_MESSAGES.get(node_type)
        if message is not None:
            raise NotImplementedError(
                message.format(node_type=node_type, node_name=node_name)
            )
        known = ", ".join(component_registry.names())
        raise ValueError(
            f"Unsupported component type {node_type!r} for node {node_name!r}. "
            f"Known component types: {known}"
        )
    unsupported_message = (
        None
        if meta.builder is None
        else getattr(meta.builder, "_feedbax_unsupported_builder_message", None)
    )
    if meta.builder is None or unsupported_message is not None:
        message = (
            _template_builder_error(meta, component_registry)
            or unsupported_message
            or _DISPLAY_ONLY_MESSAGES.get(
                node_type,
                f"Component type {node_type!r} for node {node_name!r} is registered "
                "for metadata but has no executable builder.",
            )
        )
        raise NotImplementedError(message.format(node_name=node_name))
    try:
        return meta.builder(params)
    except ValueError as exc:
        if node_type == "PenzaiAdapter" and "builder_name" in str(exc):
            raise ValueError(
                f"PenzaiAdapter node {node_name!r} requires 'builder_name' parameter"
            ) from exc
        raise
