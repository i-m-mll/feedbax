from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Any, Protocol

import jax.numpy as jnp
import jax.tree as jt

from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID, PENZAI_DOMAIN_ID
from feedbax.contracts.graphs.mechanics_templates import point_mass_template_graph
from feedbax.contracts.graph import ParamSchema
from feedbax.contracts.graphs.penzai_compiler import penzai_builder_options
from feedbax.contracts.representation import RepresentationSpec
from feedbax.control.affine import affine_feedback_output_prototype
from feedbax.mechanics.muscle_config import default_6muscle_2link_attachment_paths
from feedbax.runtime.affine_composer import (
    AFFINE_VALUE_COMPOSER_SCHEMA_VERSION,
    affine_value_composer_output_prototype,
)
from feedbax.runtime.state import CartesianState
from feedbax.runtime.state_feedback import state_feedback_output_prototype

from .acausal_adapters import register_acausal_components
from .meta import ComponentMeta, MissingPrototypeInput
from .templates import register_builtin_graph_templates


class _Registry(Protocol):
    def register(self, meta: ComponentMeta) -> None: ...


def _missing_input(port: str, *, component: str) -> MissingPrototypeInput:
    return MissingPrototypeInput(f"{component} output prototype requires input prototype {port!r}")


def _required_input(inputs: Mapping[str, Any], port: str, *, component: str) -> Any:
    if port not in inputs:
        raise _missing_input(port, component=component)
    return inputs[port]


def _shape_from_single_array(proto: Any, *, component: str, port: str) -> tuple[int, ...]:
    leaves = jt.leaves(proto)
    if len(leaves) != 1 or not hasattr(leaves[0], "shape"):
        raise ValueError(
            f"{component} output prototype requires {port!r} to be a single array prototype"
        )
    return tuple(int(dim) for dim in leaves[0].shape)


def _array_proto_from_shape(shape: Any) -> Any | None:
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        return jnp.zeros(tuple(int(dim) for dim in shape))
    except (TypeError, ValueError):
        return None


def _proto_from_value(value: Any) -> Any:
    return jt.map(lambda x: jnp.zeros_like(jnp.asarray(x)), value)


def _positive_sizes(params: Mapping[str, Any], *, component: str) -> tuple[int, ...]:
    sizes = params.get("sizes")
    if not isinstance(sizes, (list, tuple)):
        raise ValueError(f"{component} requires 'sizes' as a list of positive integers")
    parsed = tuple(int(size) for size in sizes)
    if not parsed:
        raise ValueError(f"{component} sizes must contain at least one output size")
    if any(size <= 0 for size in parsed):
        raise ValueError(f"{component} sizes must be positive integers")
    return parsed


def _n_dims_proto(params: Mapping[str, Any], *, key: str = "n_dims") -> Any:
    return jnp.zeros((int(params.get(key, 1)),))


def _representation(data: Mapping[str, Any]) -> RepresentationSpec:
    return RepresentationSpec.model_validate(data)


def _selector_binding(
    namespace: str,
    compact: str,
    *,
    target_id: str | None = None,
    path: str | None = None,
    anchor_subpath: str | None = None,
    dim: int | None = None,
    role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selector: dict[str, Any] = {
        "namespace": namespace,
        "compact": compact,
    }
    if target_id is not None:
        selector["target_id"] = target_id
    if path is not None:
        selector["path"] = path
    if role is not None:
        selector["role"] = role
    if metadata:
        selector["metadata"] = dict(metadata)
    binding: dict[str, Any] = {
        "kind": "selector",
        "selector": selector,
    }
    if anchor_subpath is not None:
        binding["anchor_subpath"] = anchor_subpath
    if dim is not None:
        binding["dim"] = dim
    return binding


def _literal_binding(value: Any, *, dim: int | None = None) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "literal", "value": value}
    if dim is not None:
        binding["dim"] = dim
    return binding


def _param_binding(
    path: str,
    *,
    expected_type: str | None = None,
    dim: int | None = None,
) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "param_path", "path": path}
    if expected_type is not None:
        binding["expected_type"] = expected_type
    if dim is not None:
        binding["dim"] = dim
    return binding


def _trial_binding(path: str, *, dim: int | None = None) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "trial_spec_path", "path": path}
    if dim is not None:
        binding["dim"] = dim
    return binding


def _compose_children_metadata() -> dict[str, Any]:
    return {
        "composition_rule": {
            "kind": "subgraph_children",
            "allow_outer_geometry_fallback": False,
        }
    }


def _point_mass_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "center",
                    "semantic_role": "center",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Effector",
                    "binding": _selector_binding(
                        "mechanics_object",
                        "output:effector",
                        target_id="effector",
                        path="pos",
                        anchor_subpath="position",
                        dim=2,
                    ),
                    "units": "m",
                    "dim": 2,
                }
            ],
            "elements": [
                {
                    "id": "point-body",
                    "archetype": "point_body",
                    "anchors": ["center"],
                    "bindings": {
                        "position": _selector_binding(
                            "mechanics_object",
                            "output:effector",
                            target_id="effector",
                            path="pos",
                            anchor_subpath="position",
                            dim=2,
                        )
                    },
                    "style": [
                        {"channel": "radius", "value": 5},
                        {"channel": "fill", "value": "var(--workspace-body-fill)"},
                    ],
                    "metadata": {
                        "body_kind": "point_mass",
                        "source_output_port": "effector",
                    },
                }
            ],
        }
    )


def _two_link_arm_representation() -> RepresentationSpec:
    link_lengths = _param_binding("link_lengths", expected_type="array", dim=2)
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "shoulder",
                    "semantic_role": "joint",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Shoulder",
                    "binding": _literal_binding([0.0, 0.0], dim=2),
                    "units": "m",
                    "dim": 2,
                    "metadata": {"joint_index": 0},
                },
                {
                    "id": "elbow",
                    "semantic_role": "joint",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Elbow",
                    "units": "m",
                    "dim": 2,
                    "metadata": {
                        "computed_from": "joint_angles_and_link_lengths",
                        "joint_index": 1,
                    },
                },
                {
                    "id": "effector",
                    "semantic_role": "endpoint",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Effector",
                    "binding": _selector_binding(
                        "mechanics_object",
                        "output:effector",
                        target_id="effector",
                        path="pos",
                        anchor_subpath="position",
                        dim=2,
                    ),
                    "units": "m",
                    "dim": 2,
                },
            ],
            "elements": [
                {
                    "id": "links",
                    "archetype": "planar_chain",
                    "anchors": ["shoulder", "elbow", "effector"],
                    "bindings": {
                        "joint_angles": _selector_binding(
                            "mechanics_object",
                            "output:state",
                            target_id="state",
                            path="skeleton.angle",
                            anchor_subpath="orientation",
                            dim=2,
                        ),
                        "link_lengths": link_lengths,
                    },
                    "style": [
                        {"channel": "stroke", "value": "var(--workspace-body-stroke)"},
                        {"channel": "line_width", "value": 3},
                    ],
                    "metadata": {
                        "chain_kind": "two_link_arm",
                        "kinematics": "feedbax.mechanics.skeleton.arm.TwoLinkArm.forward_kinematics",
                        "link_length_param": "link_lengths",
                    },
                    "planar_chain": {
                        "frame_ids": ["world", "link0", "link1"],
                        "pose_fallback": "zero",
                    },
                }
            ],
            "reachability": {
                "kind": "radial",
                "origin_anchor": "shoulder",
                "radius_binding": link_lengths,
                "radius_transform": "sum_abs",
                "label": "arm reach",
                "units": "m",
            },
        }
    )


def _default_two_link_muscle_path_geometry() -> dict[str, Any]:
    """Expose canonical provider-owned body-local attachment topology."""
    paths = default_6muscle_2link_attachment_paths()
    return {
        "paths": [
            {
                "id": f"muscle-{index}",
                "points": [
                    {
                        "frame": path.origin.body,
                        "position": [float(path.origin.pos[0]), float(path.origin.pos[1])],
                    },
                    {
                        "frame": path.insertion.body,
                        "position": [
                            float(path.insertion.pos[0]),
                            float(path.insertion.pos[1]),
                        ],
                    },
                ],
            }
            for index, path in enumerate(paths)
        ],
    }


def _two_link_muscle_representation(
    *,
    source: str,
    composite: bool,
    muscle_path_geometry: Mapping[str, Any] | None = None,
    frame_input_port: str | None = None,
) -> RepresentationSpec:
    metadata = {
        "geometry_source": source,
        "chain_source": "feedbax.mechanics.skeleton.arm.TwoLinkArm",
        "muscle_count": 6,
        **(_compose_children_metadata() if composite else {}),
    }
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "metadata": metadata,
            **(
                {"muscle_path_geometry": dict(muscle_path_geometry)}
                if muscle_path_geometry is not None
                else {}
            ),
            "anchors": [
                {
                    "id": "shoulder",
                    "semantic_role": "joint",
                    "label": "Shoulder",
                    "binding": _literal_binding([0.0, 0.0], dim=2),
                    "units": "m",
                    "dim": 2,
                    "metadata": {"joint_index": 0},
                },
                {
                    "id": "elbow",
                    "semantic_role": "joint",
                    "label": "Elbow",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"computed_from": "joint_angles_and_link_lengths"},
                },
                {
                    "id": "effector",
                    "semantic_role": "endpoint",
                    "label": "Effector",
                    "binding": _selector_binding(
                        "mechanics_object",
                        "output:effector",
                        target_id="effector",
                        path="pos",
                        anchor_subpath="position",
                        dim=2,
                    ),
                    "units": "m",
                    "dim": 2,
                },
                {
                    "id": "muscle-origins",
                    "semantic_role": "origin",
                    "label": "Muscle origins",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"collection": "muscle_origins"},
                },
                {
                    "id": "muscle-insertions",
                    "semantic_role": "insertion",
                    "label": "Muscle insertions",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"collection": "muscle_insertions"},
                },
            ],
            "elements": [
                {
                    "id": "links",
                    "archetype": "planar_chain",
                    "anchors": ["shoulder", "elbow", "effector"],
                    "bindings": {
                        "joint_angles": _selector_binding(
                            "mechanics_object",
                            "output:state",
                            target_id="state",
                            path="skeleton.angle",
                            anchor_subpath="orientation",
                            dim=2,
                        )
                    },
                    "metadata": {"chain_kind": "two_link_arm"},
                },
                {
                    "id": "muscle-paths",
                    "archetype": "muscle_path",
                    "anchors": ["muscle-origins", "muscle-insertions"],
                    "style": [
                        {
                            "channel": "opacity",
                            "binding": _selector_binding(
                                "biomechanics_object",
                                "output:activations",
                                target_id="activations",
                                role="activation",
                                metadata={"style_binding": "per_muscle"},
                            ),
                            "metadata": {
                                "maps": "muscle_activation",
                                "range": [0.15, 1.0],
                            },
                        },
                        {"channel": "stroke", "value": "var(--workspace-muscle-stroke)"},
                    ],
                    **(
                        {
                            "frame_provider": {
                                "kind": "from_input_port",
                                "input_port": frame_input_port,
                            }
                        }
                        if frame_input_port is not None
                        else {}
                    ),
                    "metadata": {
                        "geometry_source": source,
                        "runtime_frame_paths": "resolver_required",
                        "activation_channel": "opacity",
                    },
                },
            ],
        }
    )


def _point_mass_muscle_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "metadata": {
                "geometry_source": "feedbax.mechanics.geometry.PointMassRadialGeometry",
                "muscle_count_param": "n_pairs",
                **_compose_children_metadata(),
            },
            "anchors": [
                {
                    "id": "center",
                    "semantic_role": "center",
                    "label": "Point mass",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"source_output_port": "force_2d"},
                },
                {
                    "id": "radial-muscle-paths",
                    "semantic_role": "path",
                    "label": "Radial muscle paths",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"collection": "radial_muscle_paths"},
                },
            ],
            "elements": [
                {
                    "id": "point-body",
                    "archetype": "point_body",
                    "anchors": ["center"],
                    "metadata": {"body_kind": "point_mass"},
                },
                {
                    "id": "radial-muscles",
                    "archetype": "muscle_path",
                    "anchors": ["radial-muscle-paths"],
                    "bindings": {
                        "n_pairs": _param_binding("n_pairs", expected_type="int"),
                    },
                    "style": [
                        {
                            "channel": "opacity",
                            "binding": _selector_binding(
                                "biomechanics_object",
                                "output:activations",
                                target_id="activations",
                                role="activation",
                                metadata={"style_binding": "per_muscle"},
                            ),
                            "metadata": {
                                "maps": "muscle_activation",
                                "range": [0.15, 1.0],
                            },
                        }
                    ],
                    "metadata": {
                        "geometry_source": "feedbax.mechanics.geometry.PointMassRadialGeometry",
                        "runtime_frame_paths": "resolver_required",
                    },
                },
            ],
        }
    )


def _reach_task_representation(*, delayed: bool) -> RepresentationSpec:
    temporality = (
        {
            "kind": "scheduled",
            "target_on_epochs_param": "target_on_epochs",
            "move_epochs_param": "move_epochs",
            "target_visible_from_start_param": "target_visible_from_start",
        }
        if delayed
        else {"kind": "static"}
    )
    endpoint_mode = (
        {"kind": "param", "param": "train_endpoint_mode", "center_out_value": "center_out"}
        if delayed
        else {"kind": "validation_center_out", "training": "workspace_uniform"}
    )
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "metadata": {
                "schematic": True,
                "temporality": temporality,
                "canonical_goal_anchor": "goal",
                "endpoint_mode": endpoint_mode,
            },
            "anchors": [
                {
                    "id": "start",
                    "semantic_role": "origin",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Start",
                    "binding": _trial_binding("inits.effector.pos", dim=2),
                    "units": "m",
                    "dim": 2,
                    "metadata": {"task_role": "initial_effector_position"},
                },
                {
                    "id": "goal",
                    "semantic_role": "target",
                    "interaction_roles": ["selectable", "hoverable", "editable"],
                    "label": "Goal",
                    "binding": _trial_binding("targets.effector.pos", dim=2),
                    "units": "m",
                    "dim": 2,
                    "metadata": {
                        "canonical_goal": True,
                        "task_role": "effector_target",
                    },
                },
            ],
            "elements": [
                {
                    "id": "workspace",
                    "archetype": "region",
                    "bindings": {
                        "bounds": _param_binding("workspace", expected_type="bounds2d", dim=2)
                    },
                    "style": [
                        {"channel": "fill", "value": "var(--workspace-region-fill)"},
                        {"channel": "opacity", "value": 0.08},
                    ],
                    "metadata": {"region_kind": "workspace_bounds"},
                },
                {
                    "id": "start-marker",
                    "archetype": "marker",
                    "anchors": ["start"],
                    "style": [{"channel": "glyph", "value": "start"}],
                },
                {
                    "id": "goal-marker",
                    "archetype": "marker",
                    "anchors": ["goal"],
                    "style": [{"channel": "glyph", "value": "target"}],
                    "metadata": {"canonical_goal": True},
                },
                {
                    "id": "reach-distribution",
                    "archetype": "distribution_glyph",
                    "anchors": ["start", "goal"],
                    "bindings": {
                        "workspace": _param_binding("workspace", expected_type="bounds2d", dim=2),
                        "eval_reach_length": _param_binding(
                            "eval_reach_length",
                            expected_type="float",
                        ),
                        "eval_n_directions": _param_binding(
                            "eval_n_directions",
                            expected_type="int",
                        ),
                        "eval_grid_n": _param_binding("eval_grid_n", expected_type="int"),
                    },
                    "metadata": {
                        "distribution": "center_out" if delayed else "workspace_uniform",
                        "canonical_objective": "goal",
                    },
                },
                {
                    "id": "objective",
                    "archetype": "objective_link",
                    "anchors": ["start", "goal"],
                    "metadata": {
                        "canonical_goal": "goal",
                        "objective": "effector_position",
                    },
                },
            ],
        }
    )


def _ravel_width(proto: Any, *, component: str, port: str) -> int:
    leaves = jt.leaves(proto)
    if not leaves or any(not hasattr(leaf, "shape") for leaf in leaves):
        raise ValueError(f"{component} output prototype requires {port!r} to be array-shaped")
    return sum(prod(tuple(int(dim) for dim in leaf.shape)) for leaf in leaves)


def input_passthrough_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the incoming ``input`` prototype as ``output``."""

    return {"output": _required_input(inputs, "input", component="input passthrough")}


def force_passthrough_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the incoming force prototype for force-passthrough interventions."""

    return {"force": _required_input(inputs, "force", component="force passthrough")}


def source_value_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a source prototype from the configured scalar/array value."""

    return {"output": _proto_from_value(params.get("value", 0.0))}


def step_source_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a ramp/sine/pulse prototype from amplitude or slope params."""

    return {"output": _proto_from_value(params.get("amplitude", params.get("slope", 1.0)))}


def noise_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a random-noise sample prototype from the configured shape."""

    return {"output": jnp.zeros(tuple(int(dim) for dim in params.get("shape", [1])))}


def binary_elementwise_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the first available binary elementwise input prototype."""

    proto = inputs.get("a")
    if proto is None:
        proto = inputs.get("b")
    if proto is None:
        raise MissingPrototypeInput(
            "binary elementwise output prototype requires input prototype 'a' or 'b'"
        )
    return {"output": proto}


def input_node_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an input-node prototype under its configured output port."""

    output_port = str(params.get("output_port", "output"))
    return {output_port: _required_input(inputs, "input", component="Input")}


def subtract_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the first available subtract input prototype as ``out``."""

    del params
    proto = inputs.get("a")
    if proto is None:
        proto = inputs.get("b")
    if proto is None:
        raise MissingPrototypeInput(
            "Subtract output prototype requires input prototype 'a' or 'b'"
        )
    return {"out": proto}


def sum_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the first available sum input prototype as ``output``."""

    del params
    for port in ("a", "b", "c", "d"):
        proto = inputs.get(port)
        if proto is not None:
            return {"output": proto}
    raise MissingPrototypeInput(
        "Sum output prototype requires at least one input prototype"
    )


def reshape_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a reshaped array prototype."""

    proto = _required_input(inputs, "input", component="Reshape")
    leaves = jt.leaves(proto)
    if len(leaves) != 1:
        raise ValueError("Reshape output prototype requires a single array input")
    shape = params.get("shape", params.get("output_shape", [1, 1]))
    if not isinstance(shape, (list, tuple)):
        raise ValueError("Reshape output prototype requires 'shape' as a list")
    return {"output": jnp.zeros(tuple(int(dim) for dim in shape), dtype=leaves[0].dtype)}


def matmul_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a prototype for ``a @ b``."""

    del params
    a = _required_input(inputs, "a", component="MatMul")
    b = _required_input(inputs, "b", component="MatMul")
    return {"out": jnp.matmul(jnp.zeros_like(a), jnp.zeros_like(b))}


def ravel_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a one-dimensional flattened prototype for an arbitrary PyTree input."""

    proto = _required_input(inputs, "input", component="Ravel")
    return {"output": jnp.zeros((_ravel_width(proto, component="Ravel", port="input"),))}


def stateful_passthrough_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return input-shaped output, falling back to configured ``input_shape``."""

    proto = inputs.get("input")
    if proto is None:
        proto = _array_proto_from_shape(params.get("input_shape"))
    if proto is None:
        raise _missing_input("input", component="stateful passthrough")
    return {"output": proto}


def linear_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a dense-layer output prototype from ``output_size``."""

    return {"output": jnp.zeros((int(params.get("output_size", 1)),))}


def recurrent_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return GRU output and hidden prototypes from ``hidden_size``."""

    hidden = jnp.zeros((int(params.get("hidden_size", 1)),))
    return {"output": hidden, "hidden": hidden}


def lstm_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return LSTM output, hidden, and cell prototypes from ``hidden_size``."""

    hidden = jnp.zeros((int(params.get("hidden_size", 1)),))
    return {"output": hidden, "hidden": hidden, "cell": hidden}


def mux_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a concatenated vector prototype for all declared mux inputs."""

    n_inputs = int(params.get("n_inputs", 2))
    ports = [f"in_{index}" for index in range(n_inputs)]
    extra_dynamic_ports = [
        port
        for port in inputs
        if port.startswith("in_") and port.removeprefix("in_").isdigit() and port not in ports
    ]
    if extra_dynamic_ports:
        raise ValueError(
            f"Mux output prototype received ports {extra_dynamic_ports!r} outside "
            f"declared n_inputs={n_inputs}"
        )
    missing = [port for port in ports if port not in inputs]
    if missing:
        raise MissingPrototypeInput(
            "Mux output prototype requires input prototypes for ports "
            + ", ".join(repr(port) for port in missing)
        )
    widths = [
        _ravel_width(inputs[port], component="Mux", port=port)
        for port in ports
    ]
    return {"output": jnp.zeros((sum(widths),))}


def demux_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return per-output prototypes by slicing the input final dimension."""

    proto = _required_input(inputs, "input", component="Demux")
    shape = _shape_from_single_array(proto, component="Demux", port="input")
    if not shape:
        raise ValueError("Demux input prototype must have at least one dimension")
    sizes = _positive_sizes(params, component="Demux")
    width = shape[-1]
    total = sum(sizes)
    if width != total:
        raise ValueError(
            f"Demux input final dimension {width} does not match sum(sizes) {total}"
        )
    prefix = shape[:-1]
    return {
        f"out_{index}": jnp.zeros((*prefix, size))
        for index, size in enumerate(sizes)
    }


def switch_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the routed-signal prototype when both switch arms agree."""

    true_proto = _required_input(inputs, "true_input", component="Switch")
    false_proto = _required_input(inputs, "false_input", component="Switch")
    true_shape = _shape_from_single_array(true_proto, component="Switch", port="true_input")
    false_shape = _shape_from_single_array(false_proto, component="Switch", port="false_input")
    if true_shape != false_shape:
        raise ValueError(
            "Switch output prototype requires true_input and false_input "
            f"to have matching shapes, got {true_shape!r} and {false_shape!r}"
        )
    return {"output": true_proto}


def elementwise_affine_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the signal prototype for elementwise affine modulation."""

    signal = inputs.get("signal")
    if signal is not None:
        return {"output": signal}
    proto = _array_proto_from_shape(params.get("signal_shape"))
    if proto is None:
        raise _missing_input("signal", component="ElementwiseAffineModulator")
    return {"output": proto}


def feedback_channels_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the configured feedback-channel state prototype."""

    input_shape = params.get("input_shape")
    if isinstance(input_shape, (list, tuple)) and input_shape and all(
        isinstance(item, (list, tuple)) for item in input_shape
    ):
        return {"feedback": tuple(jnp.zeros(tuple(int(dim) for dim in item)) for item in input_shape)}
    proto = _array_proto_from_shape(input_shape)
    if proto is None:
        proto = (jnp.zeros(2), jnp.zeros(2))
    return {"feedback": proto}


def spring_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return spring force with displacement shape."""

    return {"force": _required_input(inputs, "displacement", component="Spring")}


def damper_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return damper force with velocity shape."""

    return {"force": _required_input(inputs, "velocity", component="Damper")}


def mechanics_state_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the default Cartesian state prototype used by mechanics plants."""

    effector = CartesianState()
    return {"effector": effector, "state": effector}


def linear_state_space_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return LinearStateSpace effector/state prototypes from matrices and slices."""

    a_matrix = jnp.asarray(params["A"])
    b_matrix = jnp.asarray(params["B"])
    state_dim = int(a_matrix.shape[0])
    pos_start, pos_stop = (int(value) for value in params.get("pos_slice", [0, 2]))
    vel_start, vel_stop = (int(value) for value in params.get("vel_slice", [2, 4]))
    force_dim = int(b_matrix.shape[1]) if b_matrix.ndim == 2 else 0
    return {
        "effector": CartesianState(
            pos=jnp.zeros((pos_stop - pos_start,), dtype=a_matrix.dtype),
            vel=jnp.zeros((vel_stop - vel_start,), dtype=a_matrix.dtype),
            force=jnp.zeros((force_dim,), dtype=a_matrix.dtype),
        ),
        "state": jnp.zeros((state_dim,), dtype=a_matrix.dtype),
    }


def scalar_muscle_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return scalar muscle force and activation prototypes."""

    proto = jnp.zeros(())
    return {"force": proto, "activation": proto}


def input_or_n_dims_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the input prototype, or an ``n_dims`` vector when unconnected."""

    proto = inputs.get("input")
    if proto is not None:
        return {"output": proto}
    return {"output": _n_dims_proto(params)}


def error_or_n_dims_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the error prototype, or an ``n_dims`` vector when unconnected."""

    proto = inputs.get("error")
    if proto is not None:
        return {"output": proto}
    return {"output": _n_dims_proto(params)}


def rate_limiter_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return input-shaped rate limiter output, or ``n_dims`` when unconnected."""

    return input_or_n_dims_output_prototype(params, inputs)


def register_builtin_components(registry: _Registry) -> None:
    registry.register(
        ComponentMeta(
            name='Subgraph',
            category='Structure',
            description='Nested graph container.',
            param_schema=[],
            input_ports=[],
            output_ports=[],
            icon='Layers',
            is_composite=True,
        )
    )
    register_builtin_graph_templates(registry)
    penzai_builders = penzai_builder_options()
    registry.register(
        ComponentMeta(
            name='PenzaiAdapter',
            category='Structure',
            description='Penzai neural network adapter (leaf). Wraps a trained Penzai model for inference.',
            param_schema=[
                ParamSchema(
                    name='builder_name',
                    type='enum',
                    options=[builder["name"] for builder in penzai_builders],
                    option_descriptions={
                        builder["name"]: str(builder["description"])
                        for builder in penzai_builders
                    },
                    default=penzai_builders[0]["name"] if penzai_builders else '',
                    description="Registered Penzai model builder.",
                    required=True,
                ),
                ParamSchema(name='input_port', type='str', default='input', required=False),
                ParamSchema(name='output_port', type='str', default='output', required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Hexagon',
            interior_domain=PENZAI_DOMAIN_ID,
            is_composite=True,
            template_kind='display',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Gain',
            category='Math',
            description='Multiply input by constant.',
            param_schema=[
                ParamSchema(name='gain', type='float', default=1.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='SlidersHorizontal',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=input_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Sum',
            category='Math',
            description='Add present inputs.',
            param_schema=[],
            input_ports=['a', 'b', 'c', 'd'],
            output_ports=['output'],
            icon='Sigma',
            port_types=PortTypeSpec(
                inputs={
                    'a': PortType(dtype='any'),
                    'b': PortType(dtype='any'),
                    'c': PortType(dtype='any'),
                    'd': PortType(dtype='any'),
                },
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=sum_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Input',
            category='Sources',
            description='Pass an external graph input through a source-like node.',
            param_schema=[
                ParamSchema(name='output_port', type='str', default='output', required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='LogIn',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=input_node_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Subtract',
            category='Math',
            description='Subtract input b from input a.',
            param_schema=[],
            input_ports=['a', 'b'],
            output_ports=['out'],
            icon='Minus',
            port_types=PortTypeSpec(
                inputs={'a': PortType(dtype='any'), 'b': PortType(dtype='any')},
                outputs={'out': PortType(dtype='any')},
            ),
            output_prototype_fn=subtract_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Multiply',
            category='Math',
            description='Element-wise product.',
            param_schema=[],
            input_ports=['a', 'b'],
            output_ports=['output'],
            icon='X',
            port_types=PortTypeSpec(
                inputs={'a': PortType(dtype='any'), 'b': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=binary_elementwise_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Reshape',
            category='Math',
            description='Reshape an array to a fixed output shape.',
            param_schema=[
                ParamSchema(name='shape', type='array', default=[1, 1], required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Rows3',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='array')},
            ),
            output_prototype_fn=reshape_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='MatMul',
            category='Math',
            description='Matrix multiply a and b.',
            param_schema=[],
            input_ports=['a', 'b'],
            output_ports=['out'],
            icon='Table2',
            port_types=PortTypeSpec(
                inputs={'a': PortType(dtype='array'), 'b': PortType(dtype='array')},
                outputs={'out': PortType(dtype='array')},
            ),
            output_prototype_fn=matmul_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Scale',
            category='Math',
            description='Multiply input by a scalar or broadcastable scale.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Scaling',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=input_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Sigmoid',
            category='Math',
            description='Apply a logistic sigmoid elementwise.',
            param_schema=[],
            input_ports=['input'],
            output_ports=['output'],
            icon='Activity',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=input_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='ElementwiseAffineModulator',
            category='Math',
            description=(
                'Per-element affine modulation: '
                'signal * (baseline + gain * modulator) + bias * modulator.'
            ),
            param_schema=[
                ParamSchema(name='signal_shape', type='array', default=[1], required=True),
                ParamSchema(name='baseline', type='array', default=1.0, required=False),
                ParamSchema(name='gain_init', type='array', default=0.0, required=False),
                ParamSchema(name='bias_init', type='array', default=0.0, required=False),
                ParamSchema(name='trainable', type='bool', default=True, required=False),
            ],
            input_ports=['signal', 'modulator', 'scale', 'bias'],
            output_ports=['output'],
            icon='SlidersHorizontal',
            port_types=PortTypeSpec(
                inputs={
                    'signal': PortType(dtype='vector'),
                    'modulator': PortType(dtype='vector'),
                    'scale': PortType(dtype='vector'),
                    'bias': PortType(dtype='vector'),
                },
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=elementwise_affine_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Constant',
            category='Sources',
            description='Constant value output.',
            param_schema=[
                ParamSchema(name='value', type='float', default=0.0, required=True),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='Circle',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=source_value_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Ramp',
            category='Sources',
            description='Linear ramp over time.',
            param_schema=[
                ParamSchema(name='slope', type='float', default=1.0, required=True),
                ParamSchema(name='intercept', type='float', default=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.01, required=True),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='TrendingUp',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=step_source_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Sine',
            category='Sources',
            description='Sinusoidal signal.',
            param_schema=[
                ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                ParamSchema(name='frequency', type='float', default=1.0, required=True),
                ParamSchema(name='phase', type='float', default=0.0, required=False),
                ParamSchema(name='offset', type='float', default=0.0, required=False),
                ParamSchema(name='dt', type='float', default=0.01, required=True),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='AudioWaveform',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=step_source_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Pulse',
            category='Sources',
            description='Pulse/square wave.',
            param_schema=[
                ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                ParamSchema(name='period', type='float', default=1.0, required=True),
                ParamSchema(name='duty_cycle', type='float', default=0.5, required=True),
                ParamSchema(name='offset', type='float', default=0.0, required=False),
                ParamSchema(name='dt', type='float', default=0.01, required=True),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='Activity',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=step_source_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Noise',
            category='Signal Processing',
            description='Random noise source.',
            param_schema=[
                ParamSchema(name='mean', type='float', default=0.0, required=False),
                ParamSchema(name='std', type='float', default=1.0, required=True),
                ParamSchema(name='shape', type='array', default=[1], required=False),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='Sparkles',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=noise_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Saturation',
            category='Signal Processing',
            description='Clamp to min/max range.',
            param_schema=[
                ParamSchema(name='min_val', type='float', default=-1.0, required=True),
                ParamSchema(name='max_val', type='float', default=1.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='SlidersHorizontal',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=input_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='DelayLine',
            category='Signal Processing',
            description='Discrete delay buffer.',
            param_schema=[
                ParamSchema(name='delay', type='int', default=1, min=0, required=True),
                ParamSchema(name='init_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Clock',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=stateful_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='MLP',
            category='Neural Networks',
            description='Multi-layer perceptron.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=4, min=1, required=True),
                ParamSchema(name='output_size', type='int', default=2, min=1, required=True),
                ParamSchema(name='hidden_sizes', type='array', default=[64], required=False),
                ParamSchema(
                    name='activation',
                    type='enum',
                    options=['relu', 'tanh', 'identity'],
                    default='relu',
                    required=False,
                ),
                ParamSchema(
                    name='final_activation',
                    type='enum',
                    options=['identity', 'tanh', 'relu'],
                    default='identity',
                    required=False,
                ),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Brain',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=linear_output_prototype,
            trainable_by_default=True,
        )
    )
    registry.register(
        ComponentMeta(
            name='Linear',
            category='Neural Networks',
            description='Linear layer.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=1, min=1, required=True),
                ParamSchema(name='output_size', type='int', default=1, min=1, required=True),
                ParamSchema(name='use_bias', type='bool', default=True, required=False),
                ParamSchema(
                    name='activation',
                    type='enum',
                    options=['identity', 'tanh', 'relu', 'sigmoid'],
                    default='identity',
                    required=False,
                ),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Minus',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=linear_output_prototype,
            trainable_by_default=True,
        )
    )
    registry.register(
        ComponentMeta(
            name='GRU',
            category='Neural Networks',
            description='GRU cell.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=4, min=1, required=True),
                ParamSchema(name='hidden_size', type='int', default=4, min=1, required=True),
            ],
            input_ports=['input', 'hidden'],
            output_ports=['output', 'hidden'],
            icon='BrainCircuit',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector'), 'hidden': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector'), 'hidden': PortType(dtype='vector')},
            ),
            output_prototype_fn=recurrent_output_prototype,
            trainable_by_default=True,
        )
    )
    registry.register(
        ComponentMeta(
            name='VanillaRNN',
            category='Neural Networks',
            description='Vanilla recurrent cell.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=4, min=1, required=True),
                ParamSchema(name='hidden_size', type='int', default=4, min=1, required=True),
                ParamSchema(name='use_bias', type='bool', default=True, required=False),
                ParamSchema(
                    name='activation',
                    type='enum',
                    options=['tanh', 'relu', 'identity'],
                    default='tanh',
                    required=False,
                ),
            ],
            input_ports=['input', 'hidden'],
            output_ports=['output', 'hidden'],
            icon='BrainCircuit',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector'), 'hidden': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector'), 'hidden': PortType(dtype='vector')},
            ),
            output_prototype_fn=recurrent_output_prototype,
            trainable_by_default=True,
        )
    )
    registry.register(
        ComponentMeta(
            name='LSTM',
            category='Neural Networks',
            description='LSTM cell.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=4, min=1, required=True),
                ParamSchema(name='hidden_size', type='int', default=4, min=1, required=True),
            ],
            input_ports=['input', 'hidden', 'cell'],
            output_ports=['output', 'hidden', 'cell'],
            icon='BrainCircuit',
            port_types=PortTypeSpec(
                inputs={
                    'input': PortType(dtype='vector'),
                    'hidden': PortType(dtype='vector'),
                    'cell': PortType(dtype='vector'),
                },
                outputs={
                    'output': PortType(dtype='vector'),
                    'hidden': PortType(dtype='vector'),
                    'cell': PortType(dtype='vector'),
                },
            ),
            output_prototype_fn=lstm_output_prototype,
            trainable_by_default=True,
        )
    )
    registry.register(
        ComponentMeta(
            name='GRUOracle',
            category='Neural Networks',
            description='GRU-based oracle/policy network that maps observations to muscle excitations.',
            param_schema=[
                ParamSchema(
                    name='hidden_size', type='int',
                    default=128, min=1, required=True,
                ),
                ParamSchema(
                    name='n_layers', type='int',
                    default=1, min=1, required=False,
                ),
                ParamSchema(
                    name='out_size', type='int',
                    default=6, min=1, required=True,
                ),
            ],
            input_ports=['input'],
            output_ports=['output', 'hidden'],
            icon='BrainCircuit',
            port_types=PortTypeSpec(
                inputs={
                    'input': PortType(dtype='vector'),
                },
                outputs={
                    'output': PortType(dtype='vector'),
                    'hidden': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Spring',
            category='Mechanics',
            description='Linear spring.',
            param_schema=[
                ParamSchema(name='stiffness', type='float', default=1.0, required=True),
            ],
            input_ports=['displacement'],
            output_ports=['force'],
            icon='Move',
            port_types=PortTypeSpec(
                inputs={'displacement': PortType(dtype='vector')},
                outputs={'force': PortType(dtype='vector')},
            ),
            output_prototype_fn=spring_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Damper',
            category='Mechanics',
            description='Viscous damper.',
            param_schema=[
                ParamSchema(name='damping', type='float', default=1.0, required=True),
            ],
            input_ports=['velocity'],
            output_ports=['force'],
            icon='Move',
            port_types=PortTypeSpec(
                inputs={'velocity': PortType(dtype='vector')},
                outputs={'force': PortType(dtype='vector')},
            ),
            output_prototype_fn=damper_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='TwoLinkArm',
            category='Mechanics',
            description='Two-link arm plant with direct force input.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.001, required=True),
                ParamSchema(
                    name='link_lengths',
                    type='array',
                    default=[0.30, 0.33],
                    required=False,
                ),
            ],
            input_ports=['force'],
            output_ports=['effector', 'state'],
            icon='Activity',
            port_types=PortTypeSpec(
                inputs={'force': PortType(dtype='vector')},
                outputs={
                    'effector': PortType(dtype='state'),
                    'state': PortType(dtype='state'),
                },
            ),
            output_prototype_fn=mechanics_state_output_prototype,
            representation=_two_link_arm_representation(),
        )
    )
    registry.register(
        ComponentMeta(
            name='PointMass',
            category='Mechanics',
            description='Point-mass plant with direct force input.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.001, required=True),
                ParamSchema(name='mass', type='float', default=1.0, min=0.0, required=False),
                ParamSchema(name='damping', type='float', default=0.0, min=0.0, required=False),
            ],
            input_ports=['force'],
            output_ports=['effector', 'state'],
            icon='Activity',
            port_types=PortTypeSpec(
                inputs={'force': PortType(dtype='vector')},
                outputs={
                    'effector': PortType(dtype='state'),
                    'state': PortType(dtype='state'),
                },
            ),
            output_prototype_fn=mechanics_state_output_prototype,
            representation=_point_mass_representation(),
            template_graph=point_mass_template_graph(),
            template_id="feedbax.templates.mechanics.point_mass",
            template_kind="executable",
        )
    )
    registry.register(
        ComponentMeta(
            name='LinearStateSpace',
            category='Mechanics',
            description='Discrete linear state-space mechanics.',
            param_schema=[
                ParamSchema(
                    name='A',
                    type='array',
                    default=[
                        [1.0, 0.0, 0.01, 0.0],
                        [0.0, 1.0, 0.0, 0.01],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    required=True,
                ),
                ParamSchema(
                    name='B',
                    type='array',
                    default=[
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.01, 0.0],
                        [0.0, 0.01],
                    ],
                    required=True,
                ),
                ParamSchema(name='B_w', type='array', default=None, required=False),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=False),
                ParamSchema(
                    name='initial_state',
                    type='array',
                    default=[0.0, 0.0, 0.0, 0.0],
                    required=False,
                ),
                ParamSchema(name='pos_slice', type='array', default=[0, 2], required=False),
                ParamSchema(name='vel_slice', type='array', default=[2, 4], required=False),
            ],
            input_ports=['force', 'epsilon'],
            output_ports=['effector', 'state'],
            icon='Grid3x3',
            port_types=PortTypeSpec(
                inputs={
                    'force': PortType(dtype='vector'),
                    'epsilon': PortType(dtype='vector'),
                },
                outputs={
                    'effector': PortType(dtype='state'),
                    'state': PortType(dtype='vector'),
                },
            ),
            output_prototype_fn=linear_state_space_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='StateFeedbackSelector',
            category='Mechanics',
            description='Select named state-vector slices and optional target-relative feedback.',
            param_schema=[
                ParamSchema(
                    name='state_slices',
                    type='object',
                    default={
                        'position': {'start': 0, 'stop': 2},
                        'velocity': {'start': 2, 'stop': 4},
                    },
                    required=False,
                ),
                ParamSchema(
                    name='channels',
                    type='array',
                    default=[
                        {'slice': 'position', 'transform': 'identity'},
                        {'slice': 'velocity', 'transform': 'identity'},
                    ],
                    required=False,
                ),
                ParamSchema(name='expected_state_dim', type='int', default=None, required=False),
                ParamSchema(name='output_size', type='int', default=None, required=False),
            ],
            input_ports=['state', 'target'],
            output_ports=['feedback'],
            icon='Route',
            port_types=PortTypeSpec(
                inputs={
                    'state': PortType(dtype='vector'),
                    'target': PortType(dtype='vector'),
                },
                outputs={'feedback': PortType(dtype='vector')},
            ),
            output_prototype_fn=state_feedback_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='MomentArmProjection',
            category='Mechanics',
            description='Projects muscle forces to joint torques via moment arm matrix (R^T @ forces). Also computes musculotendon lengths and velocities from joint kinematics.',
            param_schema=[
                ParamSchema(name='n_muscles', type='int', default=6, min=1, required=True),
                ParamSchema(name='n_joints', type='int', default=2, min=1, required=True),
            ],
            input_ports=['forces', 'angles', 'angular_velocities'],
            output_ports=['torques', 'musculotendon_lengths', 'musculotendon_velocities'],
            icon='Ruler',
            port_types=PortTypeSpec(
                inputs={
                    'forces': PortType(dtype='vector'),
                    'angles': PortType(dtype='vector'),
                    'angular_velocities': PortType(dtype='vector'),
                },
                outputs={
                    'torques': PortType(dtype='vector'),
                    'musculotendon_lengths': PortType(dtype='vector'),
                    'musculotendon_velocities': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='RadialForceProjection',
            category='Mechanics',
            description='Projects radially-arranged muscle forces to a 2D net force vector. Muscles are arranged in evenly-spaced antagonist pairs.',
            param_schema=[
                ParamSchema(name='n_muscles', type='int', default=8, min=2, required=True),
            ],
            input_ports=['forces'],
            output_ports=['force_2d'],
            icon='Compass',
            port_types=PortTypeSpec(
                inputs={'forces': PortType(dtype='vector')},
                outputs={'force_2d': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='AcausalSystem',
            category='Mechanics',
            description='Assembled acausal mechanical system (mass-spring-damper etc.).',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.001, min=0.0001, required=True),
                ParamSchema(
                    name='domain',
                    type='enum',
                    options=['translational', 'rotational'],
                    default='translational',
                    required=False,
                ),
            ],
            input_ports=['input'],
            output_ports=['state'],
            icon='Cog',
            interior_domain=ACAUSAL_DOMAIN_ID,
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'state': PortType(dtype='state')},
            ),
            is_composite=True,
        )
    )
    register_acausal_components(registry)
    registry.register(
        ComponentMeta(
            name='Channel',
            category='Channels',
            description='Delay and noise for a signal.',
            param_schema=[
                ParamSchema(name='delay', type='int', default=5, min=0, required=True),
                ParamSchema(
                    name='noise_model',
                    type='enum',
                    options=[
                        'none',
                        'additive_gaussian',
                        'signal_dependent_gaussian',
                        'signal_dependent_plus_additive',
                    ],
                    default='additive_gaussian',
                    required=False,
                ),
                ParamSchema(name='noise_std', type='float', default=0.01, min=0, required=False),
                ParamSchema(
                    name='additive_noise_std',
                    type='float',
                    default=0.0,
                    min=0,
                    required=False,
                ),
                ParamSchema(
                    name='signal_dependent_noise_std',
                    type='float',
                    default=0.0,
                    min=0,
                    required=False,
                ),
                ParamSchema(name='add_noise', type='bool', default=True, required=False),
                ParamSchema(name='noise_role', type='str', default=None, required=False),
                ParamSchema(name='noise_timing', type='str', default=None, required=False),
                ParamSchema(name='input_shape', type='array', default=[1], required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Signal',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=stateful_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='FeedbackChannels',
            category='Channels',
            description='Mechanics feedback selector followed by delay/noise channels.',
            param_schema=[
                ParamSchema(name='delay', type='int', default=0, min=0, required=False),
                ParamSchema(
                    name='selector',
                    type='enum',
                    options=['point_mass_pos_vel', 'effector_pos_vel', 'plant_skeleton', 'paths'],
                    default='point_mass_pos_vel',
                    required=False,
                ),
                ParamSchema(
                    name='paths',
                    type='array',
                    default=['plant.skeleton.pos', 'plant.skeleton.vel'],
                    required=False,
                ),
                ParamSchema(
                    name='noise_model',
                    type='enum',
                    options=[
                        'none',
                        'additive_gaussian',
                        'signal_dependent_gaussian',
                        'signal_dependent_plus_additive',
                    ],
                    default='additive_gaussian',
                    required=False,
                ),
                ParamSchema(name='noise_std', type='float', default=0.0, min=0, required=False),
                ParamSchema(name='add_noise', type='bool', default=False, required=False),
                ParamSchema(name='noise_role', type='str', default='sensory_feedback', required=False),
                ParamSchema(name='noise_timing', type='str', default='pre_controller', required=False),
                ParamSchema(name='input_shape', type='array', default=[[2], [2]], required=False),
            ],
            input_ports=['mechanics'],
            output_ports=['feedback'],
            icon='Radio',
            port_types=PortTypeSpec(
                inputs={'mechanics': PortType(dtype='state')},
                outputs={'feedback': PortType(dtype='state')},
            ),
            output_prototype_fn=feedback_channels_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='FirstOrderFilter',
            category='Channels',
            description='First-order low-pass filter.',
            param_schema=[
                ParamSchema(name='tau_rise', type='float', default=0.05, min=0.0, required=True),
                ParamSchema(name='tau_decay', type='float', default=0.05, min=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.001, min=0.0, required=True),
                ParamSchema(name='init_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Filter',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=stateful_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='CurlField',
            category='Interventions',
            description='Velocity-dependent curl field.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
                ParamSchema(name='label', type='str', default='curl_field', required=False),
            ],
            input_ports=['effector', 'force', 'params_override'],
            output_ports=['force'],
            icon='Wind',
            port_types=PortTypeSpec(
                inputs={
                    'effector': PortType(dtype='state'),
                    'force': PortType(dtype='vector'),
                    'params_override': PortType(dtype='object'),
                },
                outputs={'force': PortType(dtype='vector')},
            ),
            output_prototype_fn=force_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='FixedField',
            category='Interventions',
            description='Fixed force field.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                ParamSchema(name='field', type='array', default=[0.0, 0.0], required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
                ParamSchema(name='label', type='str', default='fixed_field', required=False),
            ],
            input_ports=['force', 'params_override'],
            output_ports=['force'],
            icon='Flag',
            port_types=PortTypeSpec(
                inputs={
                    'force': PortType(dtype='vector'),
                    'params_override': PortType(dtype='object'),
                },
                outputs={'force': PortType(dtype='vector')},
            ),
            output_prototype_fn=force_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='DynamicsMatrixPerturb',
            category='Interventions',
            description='State-feedback dynamics-matrix perturbation in the force channel.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(
                    name='delta_A',
                    type='array',
                    default=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    required=True,
                ),
                ParamSchema(name='active', type='bool', default=False, required=False),
                ParamSchema(
                    name='label',
                    type='str',
                    default='dynamics_matrix_perturb',
                    required=False,
                ),
                ParamSchema(
                    name='mass',
                    type='float',
                    default=None,
                    min=0.0,
                    required=True,
                    description='Effector mass in kg; must match the wired plant.',
                ),
            ],
            input_ports=['effector', 'force', 'params_override'],
            output_ports=['force'],
            icon='Activity',
            port_types=PortTypeSpec(
                inputs={
                    'effector': PortType(dtype='state'),
                    'force': PortType(dtype='vector'),
                    'params_override': PortType(dtype='object'),
                },
                outputs={'force': PortType(dtype='vector')},
            ),
            output_prototype_fn=force_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='AffineValueComposer',
            category='Interventions',
            description='State/target-conditioned affine value composer.',
            param_schema=[
                ParamSchema(
                    name='schema_version',
                    type='str',
                    default=AFFINE_VALUE_COMPOSER_SCHEMA_VERSION,
                    required=True,
                ),
                ParamSchema(name='output_block_size', type='int', default=1, min=1, required=True),
                ParamSchema(
                    name='feature_rules',
                    type='object',
                    default=[{'kind': 'identity', 'state_slice': [0, 1]}],
                    required=True,
                    description=(
                        "Ordered rules. Supported kinds: identity and "
                        "target_relative_difference."
                    ),
                ),
                ParamSchema(name='gain_init', type='array', default=[[0.0]], required=False),
                ParamSchema(name='bias_init', type='array', default=[0.0], required=False),
                ParamSchema(name='use_bias', type='bool', default=True, required=False),
                ParamSchema(
                    name='label',
                    type='str',
                    default='affine_value_composer',
                    required=False,
                ),
            ],
            input_ports=['base', 'state', 'target', 'gain', 'bias'],
            output_ports=['value'],
            icon='Blend',
            port_types=PortTypeSpec(
                inputs={
                    'base': PortType(dtype='vector'),
                    'state': PortType(dtype='vector'),
                    'target': PortType(dtype='vector'),
                    'gain': PortType(dtype='array'),
                    'bias': PortType(dtype='vector'),
                },
                outputs={'value': PortType(dtype='vector')},
            ),
            output_prototype_fn=affine_value_composer_output_prototype,
            param_schema_version=AFFINE_VALUE_COMPOSER_SCHEMA_VERSION,
            supported_param_schema_versions=[AFFINE_VALUE_COMPOSER_SCHEMA_VERSION],
        )
    )
    registry.register(
        ComponentMeta(
            name='AddNoise',
            category='Interventions',
            description='Add noise to a signal.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Sparkles',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=input_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='NetworkClamp',
            category='Interventions',
            description='Clamp network unit activity.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Pin',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_passthrough_output_prototype,
        )
    )
    # --- Muscles ---
    registry.register(
        ComponentMeta(
            name='ReluMuscle',
            category='Muscles',
            description='Simple muscle: force = activation * F_max.',
            param_schema=[
                ParamSchema(
                    name='max_isometric_force', type='float',
                    default=500.0, min=0.0, required=True,
                ),
                ParamSchema(
                    name='tau_activation', type='float',
                    default=0.015, min=0.001, required=False,
                ),
                ParamSchema(
                    name='tau_deactivation', type='float',
                    default=0.05, min=0.001, required=False,
                ),
                ParamSchema(
                    name='min_activation', type='float',
                    default=0.0, min=0.0, required=False,
                ),
                ParamSchema(
                    name='dt', type='float',
                    default=0.01, min=0.001, required=True,
                ),
                ParamSchema(
                    name='initial_activation', type='float',
                    default=0.0, min=0.0, required=False,
                ),
            ],
            input_ports=['excitation'],
            output_ports=['force', 'activation'],
            icon='Zap',
            port_types=PortTypeSpec(
                inputs={'excitation': PortType(dtype='scalar')},
                outputs={
                    'force': PortType(dtype='scalar'),
                    'activation': PortType(dtype='scalar'),
                },
            ),
            output_prototype_fn=scalar_muscle_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='RigidTendonHillMuscleThelen',
            category='Muscles',
            description='Hill-type muscle with rigid tendon assumption. Vectorized for multiple muscles.',
            param_schema=[
                ParamSchema(name='n_muscles', type='int', default=6, min=1, required=True),
                ParamSchema(name='dt', type='float', default=0.01, min=0.001, required=True),
                ParamSchema(name='tau_activation', type='float', default=0.015, min=0.001, required=False),
                ParamSchema(name='tau_deactivation', type='float', default=0.05, min=0.001, required=False),
                ParamSchema(name='max_isometric_force', type='float', default=500.0, min=0.0, required=False),
                ParamSchema(name='optimal_muscle_length', type='float', default=0.1, min=0.001, required=False),
                ParamSchema(name='tendon_slack_length', type='float', default=0.2, min=0.001, required=False),
                ParamSchema(name='initial_activation', type='float', default=0.001, min=0.0, required=False),
            ],
            input_ports=['excitation', 'musculotendon_length', 'musculotendon_velocity'],
            output_ports=['force', 'activation', 'fiber_length', 'fiber_velocity'],
            icon='Dumbbell',
            port_types=PortTypeSpec(
                inputs={
                    'excitation': PortType(dtype='vector'),
                    'musculotendon_length': PortType(dtype='vector'),
                    'musculotendon_velocity': PortType(dtype='vector'),
                },
                outputs={
                    'force': PortType(dtype='vector'),
                    'activation': PortType(dtype='vector'),
                    'fiber_length': PortType(dtype='vector'),
                    'fiber_velocity': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Arm6MuscleRigidTendon',
            category='Mechanics',
            description='6-muscle arm with Thelen rigid tendon.',
            param_schema=[
                ParamSchema(
                    name='dt', type='float',
                    default=0.01, min=0.001, required=True,
                ),
                ParamSchema(
                    name='max_isometric_force', type='float',
                    default=500.0, min=0.0, required=False,
                ),
                ParamSchema(
                    name='optimal_muscle_length', type='float',
                    default=0.1, min=0.001, required=False,
                ),
                ParamSchema(
                    name='tendon_slack_length', type='float',
                    default=0.1, min=0.0, required=False,
                ),
            ],
            input_ports=['excitation', 'angles', 'angular_velocities'],
            output_ports=['torques', 'forces', 'activations'],
            icon='Activity',
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={
                    'excitation': PortType(dtype='vector'),
                    'angles': PortType(dtype='vector'),
                    'angular_velocities': PortType(dtype='vector'),
                },
                outputs={
                    'torques': PortType(dtype='vector'),
                    'forces': PortType(dtype='vector'),
                    'activations': PortType(dtype='vector'),
                },
            ),
            representation=_two_link_muscle_representation(
                source="feedbax.mechanics.geometry.TwoLinkArmMuscleGeometry.default_six_muscle",
                composite=True,
                muscle_path_geometry=_default_two_link_muscle_path_geometry(),
                frame_input_port="angles",
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='PointMass8MuscleRelu',
            category='Mechanics',
            description='8-muscle point mass with ReLU actuators.',
            param_schema=[
                ParamSchema(
                    name='n_pairs', type='int',
                    default=4, min=1, required=False,
                ),
                ParamSchema(
                    name='max_isometric_force', type='float',
                    default=500.0, min=0.0, required=False,
                ),
                ParamSchema(
                    name='dt', type='float',
                    default=0.01, min=0.001, required=True,
                ),
            ],
            input_ports=['excitation'],
            output_ports=['force_2d', 'forces', 'activations'],
            icon='Activity',
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={'excitation': PortType(dtype='vector')},
                outputs={
                    'force_2d': PortType(dtype='vector'),
                    'forces': PortType(dtype='vector'),
                    'activations': PortType(dtype='vector'),
                },
            ),
            representation=_point_mass_muscle_representation(),
        )
    )
    registry.register(
        ComponentMeta(
            name='AnalyticalMusculoskeletalPlant',
            category='Mechanics',
            description=(
                'Two-link arm musculoskeletal plant with pure JAX Lagrangian '
                'dynamics and Hill-type rigid-tendon muscles. Fully '
                'differentiable; no MuJoCo dependency. ODE state: 2 joint '
                'angles + 2 angular velocities + 6 muscle activations.'
            ),
            param_schema=[
                ParamSchema(
                    name='dt', type='float',
                    default=0.01, min=0.0001, required=True,
                ),
                ParamSchema(
                    name='n_steps', type='int',
                    default=1, min=1, required=False,
                ),
                ParamSchema(
                    name='tau_act', type='float',
                    default=0.01, min=0.001, required=False,
                ),
                ParamSchema(
                    name='tau_deact', type='float',
                    default=0.04, min=0.001, required=False,
                ),
                ParamSchema(
                    name='clip_states', type='bool',
                    default=True, required=False,
                ),
            ],
            input_ports=['excitation'],
            output_ports=['effector', 'state'],
            icon='Activity',
            is_composite=False,
            port_types=PortTypeSpec(
                inputs={
                    'excitation': PortType(dtype='vector'),
                },
                outputs={
                    'effector': PortType(dtype='vector'),
                    'state': PortType(dtype='state'),
                },
            ),
            representation=_two_link_muscle_representation(
                source=(
                    "feedbax.mechanics.muscle_config."
                    "default_6muscle_2link_muscled_arm_parameters"
                ),
                composite=False,
                muscle_path_geometry=_default_two_link_muscle_path_geometry(),
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='NetworkConstantInput',
            category='Interventions',
            description='Add constant input to network units.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Asterisk',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='ConstantInput',
            category='Interventions',
            description='Add a constant input to a signal.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Minus',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='SimpleReaches',
            category='Tasks',
            description='Random reach endpoints in a workspace.',
            param_schema=[
                ParamSchema(name='n_steps', type='int', default=200, min=1, required=True),
                ParamSchema(
                    name='workspace',
                    type='bounds2d',
                    default=[[-1.0, -1.0], [1.0, 1.0]],
                    required=True,
                ),
                ParamSchema(name='eval_n_directions', type='int', default=7, min=1, required=False),
                ParamSchema(name='eval_reach_length', type='float', default=0.5, required=False),
                ParamSchema(name='eval_grid_n', type='int', default=1, min=1, required=False),
            ],
            input_ports=[],
            output_ports=['inputs', 'targets', 'inits', 'intervene'],
            icon='Target',
            port_types=PortTypeSpec(
                inputs={},
                outputs={
                    'inputs': PortType(dtype='any'),
                    'targets': PortType(dtype='state'),
                    'inits': PortType(dtype='state'),
                    'intervene': PortType(dtype='any'),
                },
            ),
            representation=_reach_task_representation(delayed=False),
        )
    )
    registry.register(
        ComponentMeta(
            name='DelayedReaches',
            category='Tasks',
            description='Reaches with a delay period before movement.',
            param_schema=[
                ParamSchema(name='n_steps', type='int', default=140, min=1, required=True),
                ParamSchema(
                    name='n_control_stages',
                    type='int',
                    default=None,
                    min=1,
                    required=False,
                ),
                ParamSchema(
                    name='workspace',
                    type='bounds2d',
                    default=[[-1.0, -1.0], [1.0, 1.0]],
                    required=True,
                ),
                ParamSchema(
                    name='preset',
                    type='enum',
                    options=['default', 'delayed_center_out'],
                    default='default',
                    required=False,
                ),
                ParamSchema(
                    name='train_endpoint_mode',
                    type='enum',
                    options=['workspace', 'center_out'],
                    default='workspace',
                    required=False,
                ),
                ParamSchema(
                    name='epoch_len_ranges',
                    type='array',
                    default=[[5, 15], [10, 20]],
                    required=False,
                ),
                ParamSchema(
                    name='epoch_names',
                    type='array',
                    default=['hold', 'target_on', 'movement'],
                    required=False,
                ),
                ParamSchema(name='target_on_epochs', type='array', default=[1, 2], required=False),
                ParamSchema(name='hold_epochs', type='array', default=[0, 1], required=False),
                ParamSchema(name='move_epochs', type='array', default=[2], required=False),
                ParamSchema(
                    name='target_visible_from_start',
                    type='bool',
                    default=False,
                    required=False,
                ),
                ParamSchema(name='go_cue_event_name', type='str', default=None, required=False),
                ParamSchema(
                    name='p_catch_trial',
                    type='float',
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    required=False,
                ),
                ParamSchema(
                    name='catch_metadata_policy',
                    type='enum',
                    options=['none', 'flag'],
                    default='none',
                    required=False,
                ),
                ParamSchema(name='eval_n_directions', type='int', default=7, min=1, required=False),
                ParamSchema(name='eval_reach_length', type='float', default=0.5, required=False),
                ParamSchema(name='eval_grid_n', type='int', default=1, min=1, required=False),
            ],
            input_ports=[],
            output_ports=['inputs', 'targets', 'inits', 'intervene'],
            icon='Timer',
            port_types=PortTypeSpec(
                inputs={},
                outputs={
                    'inputs': PortType(dtype='any'),
                    'targets': PortType(dtype='state'),
                    'inits': PortType(dtype='state'),
                    'intervene': PortType(dtype='any'),
                },
            ),
            representation=_reach_task_representation(delayed=True),
        )
    )
    # --- Control components ---
    registry.register(
        ComponentMeta(
            name='Integrator',
            category='Control',
            description='Continuous-time integrator (Euler).',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Integral',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_or_n_dims_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Derivative',
            category='Control',
            description='Finite-difference derivative.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='TrendingUp',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_or_n_dims_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='StateSpace',
            category='Control',
            description='Continuous LTI state-space (Euler).',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Grid3x3',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='TransferFunction',
            category='Control',
            description='Transfer function H(s)=num/den.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='FunctionSquare',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='PID',
            category='Control',
            description='Continuous PID with anti-windup.',
            param_schema=[
                ParamSchema(name='Kp', type='float', default=1.0, required=True),
                ParamSchema(name='Ki', type='float', default=0.0, required=False),
                ParamSchema(name='Kd', type='float', default=0.0, required=False),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='integral_limit', type='float', default=1000.0, required=False),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
            ],
            input_ports=['error'],
            output_ports=['output'],
            icon='Gauge',
            port_types=PortTypeSpec(
                inputs={'error': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=error_or_n_dims_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='PIDDiscrete',
            category='Control',
            description='Discrete PID (velocity form).',
            param_schema=[
                ParamSchema(name='Kp', type='float', default=1.0, required=True),
                ParamSchema(name='Ki', type='float', default=0.0, required=False),
                ParamSchema(name='Kd', type='float', default=0.0, required=False),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='output_limit', type='float', default=1000.0, required=False),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
            ],
            input_ports=['error'],
            output_ports=['output'],
            icon='Gauge',
            port_types=PortTypeSpec(
                inputs={'error': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=error_or_n_dims_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='AffineFeedbackController',
            category='Control',
            description='Time-varying affine feedback controller with optional feedforward.',
            param_schema=[
                ParamSchema(name='gain', type='array', default=[[[1.0]]], required=True),
                ParamSchema(name='bias', type='array', default=None, required=False),
                ParamSchema(name='feedforward', type='array', default=None, required=False),
                ParamSchema(
                    name='schedule_policy',
                    type='enum',
                    options=['hold', 'error'],
                    default='hold',
                    required=False,
                ),
            ],
            input_ports=['feedback', 'reference', 'feedforward'],
            output_ports=['command'],
            icon='GitBranch',
            port_types=PortTypeSpec(
                inputs={
                    'feedback': PortType(dtype='vector'),
                    'reference': PortType(dtype='vector'),
                    'feedforward': PortType(dtype='vector'),
                },
                outputs={'command': PortType(dtype='vector')},
            ),
            output_prototype_fn=affine_feedback_output_prototype,
        )
    )
    # --- Discrete components ---
    registry.register(
        ComponentMeta(
            name='IntegratorDiscrete',
            category='Discrete',
            description='Discrete-time accumulator.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=1.0, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='PlusSquare',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_or_n_dims_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='UnitDelay',
            category='Discrete',
            description='Unit delay (z^-1).',
            param_schema=[
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Clock',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_or_n_dims_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='ZeroOrderHold',
            category='Discrete',
            description='Sample and hold every N steps.',
            param_schema=[
                ParamSchema(name='hold_steps', type='int', default=1, min=1, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Pause',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_or_n_dims_output_prototype,
        )
    )
    # --- Signal processing components ---
    registry.register(
        ComponentMeta(
            name='Mux',
            category='Signal Processing',
            description='Concatenate inputs into single vector.',
            param_schema=[
                ParamSchema(name='n_inputs', type='int', default=2, min=1, required=True),
            ],
            input_ports=['in_0', 'in_1'],
            output_ports=['output'],
            icon='GitMerge',
            port_types=PortTypeSpec(
                inputs={
                    'in_0': PortType(dtype='vector'),
                    'in_1': PortType(dtype='vector'),
                },
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=mux_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Ravel',
            category='Signal Processing',
            description='Flatten a PyTree value into a vector.',
            param_schema=[],
            input_ports=['input'],
            output_ports=['output'],
            icon='Layers',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=ravel_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Demux',
            category='Signal Processing',
            description='Split vector into multiple outputs.',
            param_schema=[
                ParamSchema(name='sizes', type='array', default=[1, 1], required=True),
            ],
            input_ports=['input'],
            output_ports=['out_0', 'out_1'],
            icon='GitBranch',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={
                    'out_0': PortType(dtype='vector'),
                    'out_1': PortType(dtype='vector'),
                },
            ),
            output_prototype_fn=demux_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Switch',
            category='Signal Processing',
            description='Route signal by threshold condition.',
            param_schema=[
                ParamSchema(name='threshold', type='float', default=0.0, required=True),
            ],
            input_ports=['condition', 'true_input', 'false_input'],
            output_ports=['output'],
            icon='GitCompare',
            port_types=PortTypeSpec(
                inputs={
                    'condition': PortType(dtype='scalar'),
                    'true_input': PortType(dtype='any'),
                    'false_input': PortType(dtype='any'),
                },
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=switch_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='DeadZone',
            category='Signal Processing',
            description='Zero output for small inputs.',
            param_schema=[
                ParamSchema(name='threshold', type='float', default=0.1, min=0.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='MinusSquare',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
            output_prototype_fn=input_passthrough_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='RateLimiter',
            category='Signal Processing',
            description='Limit rate of change of signal.',
            param_schema=[
                ParamSchema(name='max_rate', type='float', default=1.0, min=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Gauge',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=rate_limiter_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='HighPassFilter',
            category='Signal Processing',
            description='High-pass filter (input - lowpass).',
            param_schema=[
                ParamSchema(name='tau', type='float', default=0.1, min=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Filter',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_or_n_dims_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='BandPassFilter',
            category='Signal Processing',
            description='Band-pass: high-pass then low-pass.',
            param_schema=[
                ParamSchema(name='tau_low', type='float', default=0.1, min=0.0, required=True),
                ParamSchema(name='tau_high', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Filter',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
            output_prototype_fn=input_or_n_dims_output_prototype,
        )
    )
    registry.register(
        ComponentMeta(
            name='Stabilization',
            category='Tasks',
            description='Hold position against perturbations.',
            param_schema=[
                ParamSchema(name='n_steps', type='int', default=200, min=1, required=True),
                ParamSchema(
                    name='workspace',
                    type='bounds2d',
                    default=[[-1.0, -1.0], [1.0, 1.0]],
                    required=True,
                ),
            ],
            input_ports=[],
            output_ports=['inputs', 'targets', 'inits', 'intervene'],
            icon='Anchor',
            port_types=PortTypeSpec(
                inputs={},
                outputs={
                    'inputs': PortType(dtype='any'),
                    'targets': PortType(dtype='state'),
                    'inits': PortType(dtype='state'),
                    'intervene': PortType(dtype='any'),
                },
            ),
        )
    )
