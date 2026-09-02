from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping

import equinox as eqx
import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr

from feedbax.models.feedback import FeedbackChannels
from feedbax.runtime.channel import Channel, ChannelSpec
from feedbax.runtime.graph import Component
from feedbax.intervene.intervene import (
    ThresholdLatchedForce,
    ThresholdLatchedForceParams,
)
from feedbax.objectives.loss import CompositeLoss
from feedbax.mechanics.linear_state_space import StructuralLinearStateSpace
from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant
from feedbax.mechanics.backend import DiffraxBackend
from feedbax.mechanics.body import BodyPreset, default_2link_bounds
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.model_builder import ChainConfig
from feedbax.mechanics.muscle_config import (
    default_6muscle_2link_segment_lengths,
    default_6muscle_2link_topology,
)
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.models.networks import (
    LeakyRNNCell,
    SimpleStagedNetwork,
    population_structure_from_spec,
)
from feedbax.models.support import identity_func
from feedbax.runtime.noise import Multiplicative, Normal
from feedbax.components.penzai import (
    PENZAI_AVAILABLE,
    build_penzai_subgraph,
    get_penzai_builder,
    penzai_state_variable_paths,
)
from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
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
        plant = DirectForceInput(TwoLinkArm(l=params.get("link_lengths", (0.30, 0.33))))
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


class _ExcitationMechanics(Mechanics):
    input_ports = ("excitation",)

    def __call__(self, inputs, state, *, key):
        excitation = inputs.get("excitation", inputs.get("force"))
        return super().__call__({"force": excitation}, state, key=key)


def _default_analytical_body_preset(params: Mapping[str, Any]) -> BodyPreset:
    bounds = default_2link_bounds()
    tau_act = float(params.get("tau_act", 0.01))
    tau_deact = float(params.get("tau_deact", 0.04))
    return BodyPreset(
        segment_lengths=default_6muscle_2link_segment_lengths(),
        segment_masses=(bounds.segment_masses_min + bounds.segment_masses_max) / 2.0,
        joint_damping=(bounds.joint_damping_min + bounds.joint_damping_max) / 2.0,
        joint_stiffness=(bounds.joint_stiffness_min + bounds.joint_stiffness_max) / 2.0,
        muscle_pcsa=(bounds.muscle_pcsa_min + bounds.muscle_pcsa_max) / 2.0,
        muscle_optimal_fiber_length=(
            bounds.muscle_optimal_fiber_length_min + bounds.muscle_optimal_fiber_length_max
        )
        / 2.0,
        muscle_tendon_slack_length=(
            bounds.muscle_tendon_slack_length_min + bounds.muscle_tendon_slack_length_max
        )
        / 2.0,
        muscle_moment_arm_magnitudes=(
            bounds.muscle_moment_arm_magnitudes_min + bounds.muscle_moment_arm_magnitudes_max
        )
        / 2.0,
        tau_act=tau_act,
        tau_deact=tau_deact,
    )


def _build_analytical_musculoskeletal_plant(params: Mapping[str, Any]) -> Mechanics:
    dt = float(params.get("dt", 0.01))
    n_steps = int(params.get("n_steps", 1))
    if n_steps < 1:
        raise ValueError("AnalyticalMusculoskeletalPlant n_steps must be >= 1")
    chain_config = ChainConfig(
        n_joints=2,
        muscle_topology=default_6muscle_2link_topology(),
    )
    plant = AnalyticalMusculoskeletalPlant.from_body_preset(
        _default_analytical_body_preset(params),
        chain_config,
        clip_states=bool(params.get("clip_states", True)),
        key=jr.PRNGKey(0),
    )
    backend = DiffraxBackend(control_dt=dt, sub_dt=dt / n_steps, solver=dfx.Euler())
    return _ExcitationMechanics(
        plant=plant,
        dt=dt,
        backend=backend,
        key=jr.PRNGKey(0),
    )


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
        "signal_dependent_std": float(signal_dependent_std or 0.0),
        "role": params.get("noise_role"),
        "timing": params.get("noise_timing"),
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
        if params.get("n_control_stages") is not None:
            params = {
                **params,
                "n_steps": int(params["n_control_stages"]) + 1,
            }
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
    builder_info = get_penzai_builder(builder_name)
    if builder_info is None:
        raise ValueError(f"Unknown Penzai builder {builder_name!r}")
    builder_fn, default_params = builder_info
    pz_model = builder_fn({**default_params, **builder_params})
    state_paths = penzai_state_variable_paths(pz_model)
    if state_paths:
        raise ValueError(
            "penzai.stateful_unsupported: PenzaiAdapter node uses unsupported "
            f"StateVariable leaves at {', '.join(state_paths)}"
        )
    return build_penzai_subgraph(
        builder_name=builder_name,
        params=builder_params,
        input_port=str(params.get("input_port", "input")),
        output_port=str(params.get("output_port", "output")),
    )


def _build_threshold_latched_force(params: Mapping[str, Any]) -> ThresholdLatchedForce:
    return ThresholdLatchedForce(
        state_selector=params["state_selector"],
        direction=str(params["direction"]),
        dt=float(params["dt"]),
        params=ThresholdLatchedForceParams(
            scale=float(params.get("scale", 1.0)),
            active=bool(params.get("active", False)),
            threshold=float(params["threshold"]),
            force=jnp.asarray(params["force"]),
            lateral_force=float(params.get("lateral_force", 0.0)),
            ramp_duration=float(params.get("ramp_duration", 0.0)),
        ),
        label=str(params.get("label", "threshold_latched_force")),
    )


def _build_structural_linear_state_space(
    params: Mapping[str, Any],
) -> StructuralLinearStateSpace:
    return StructuralLinearStateSpace(
        A=jnp.asarray(params["A"]),
        B=jnp.asarray(params["B"]),
        delta_A=params["delta_A"],
        authored_delta_A_value_spec=params.get("_authored_delta_A_value_spec"),
        B_w=None if params.get("B_w") is None else jnp.asarray(params["B_w"]),
        dt=float(params.get("dt", 1.0)),
        initial_state=(
            None if params.get("initial_state") is None else jnp.asarray(params["initial_state"])
        ),
        pos_slice=tuple(params.get("pos_slice", (0, 2))),
        vel_slice=tuple(params.get("vel_slice", (2, 4))),
        scale=float(params.get("scale", 1.0)),
        active=bool(params.get("active", False)),
        label=str(params.get("label", "structural_linear_dynamics")),
    )


def _build_two_link_arm(params: Mapping[str, Any]) -> Mechanics:
    return _build_mechanics({**dict(params), "plant_type": "TwoLinkArm"})


def _build_point_mass(params: Mapping[str, Any]) -> Mechanics:
    return _build_mechanics({**dict(params), "plant_type": "PointMass"})


def _build_simple_reaches(params: Mapping[str, Any]) -> TaskComponent:
    return _build_task_component("SimpleReaches", params)


def _build_delayed_reaches(params: Mapping[str, Any]) -> TaskComponent:
    return _build_task_component("DelayedReaches", params)


_DISPLAY_ONLY_MESSAGES: dict[str, str] = {
    "MomentArmProjection": (
        "MomentArmProjection node {node_name!r} has no Python builder yet. "
        "It is a display-only abstraction used in composite subgraph templates."
    ),
    "RadialForceProjection": (
        "RadialForceProjection node {node_name!r} has no Python builder yet. "
        "It is a display-only abstraction used in composite subgraph templates."
    ),
    # C1 issue 196bed0: projection helpers remain display-only until C2 turns
    # muscle paths and multibody geometry into real acausal mechanics interiors.
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


_UNREGISTERED_TEMPLATE_MESSAGES: dict[str, str] = {}


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
    return message


def _unsupported_component_message(
    node_name: str,
    node_type: str,
    component_registry: Any,
) -> str | None:
    meta = component_registry.get(node_type)
    if meta is None:
        return None
    unsupported_message = (
        None
        if meta.builder is None
        else getattr(meta.builder, "_feedbax_unsupported_builder_message", None)
    )
    if meta.builder is not None and unsupported_message is None:
        return None
    message = (
        _template_builder_error(meta, component_registry)
        or unsupported_message
        or _DISPLAY_ONLY_MESSAGES.get(
            node_type,
            f"Component type {node_type!r} for node {node_name!r} is registered "
            "for metadata but has no executable builder.",
        )
    )
    return message.format(node_name=node_name)


def build_component(
    node_name: str,
    node_type: str,
    params: Mapping[str, Any],
    *,
    component_registry: Any,
) -> Component:
    """Build a leaf component from a GraphSpec node."""
    meta = component_registry.get(node_type)
    if meta is None:
        message = _UNREGISTERED_TEMPLATE_MESSAGES.get(node_type)
        if message is not None:
            raise NotImplementedError(message.format(node_type=node_type, node_name=node_name))
        known = ", ".join(component_registry.names())
        raise ValueError(
            f"Unsupported component type {node_type!r} for node {node_name!r}. "
            f"Known component types: {known}"
        )
    if meta.domain != CAUSAL_DOMAIN_ID:
        raise ValueError(
            f"Component type {node_type!r} for node {node_name!r} belongs to "
            f"domain {meta.domain!r} and cannot be built by the causal component builder."
        )
    unsupported_message = _unsupported_component_message(
        node_name,
        node_type,
        component_registry,
    )
    if unsupported_message is not None:
        raise NotImplementedError(unsupported_message)
    try:
        return meta.builder(params)
    except ValueError as exc:
        if node_type == "PenzaiAdapter" and "builder_name" in str(exc):
            raise ValueError(
                f"PenzaiAdapter node {node_name!r} requires 'builder_name' parameter"
            ) from exc
        raise
