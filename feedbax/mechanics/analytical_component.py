"""Graph component factory for the analytical musculoskeletal plant."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import jax.random as jr
from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from feedbax.compiler.solver_config import (
    DEFAULT_PLANAR_SOLVER,
    MIN_PLANAR_DT,
    PlanarSolverType,
    planar_solver_name,
    resolve_acausal_solver,
)
from feedbax.component_registry.declarations import DeclaredComponent, declare_component
from feedbax.component_registry.discovery import source_path
from feedbax.contracts.acausal_interface import MUSCLE_EXCITATION_PAYLOAD
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant
from feedbax.mechanics.backend import DiffraxBackend
from feedbax.mechanics.body import BodyPreset, default_2link_bounds
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.model_builder import ChainConfig
from feedbax.mechanics.muscle_config import (
    default_6muscle_2link_segment_lengths,
    default_6muscle_2link_topology,
)


class _AnalyticalPlantParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dt: float = Field(
        default=0.01, ge=MIN_PLANAR_DT, json_schema_extra={"feedbax": {"required": True}}
    )
    n_steps: int = Field(default=1, ge=1)
    solver_type: PlanarSolverType = DEFAULT_PLANAR_SOLVER
    tau_act: float = Field(default=0.01, ge=0.001)
    tau_deact: float = Field(default=0.04, ge=0.001)
    clip_states: bool = True

    @model_serializer(mode="wrap", when_used="json")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        params = handler(self)
        if self.solver_type == DEFAULT_PLANAR_SOLVER:
            params.pop("solver_type", None)
        return params


def _extract_analytical_plant_params(component: Any, model: Any) -> Any:
    from feedbax.mechanics.backend import DiffraxBackend, _mechanics_control_dt

    backend = component.backend
    if not isinstance(backend, DiffraxBackend):
        raise ValueError(
            "AnalyticalMusculoskeletalPlant serialization requires DiffraxBackend "
            "to recover the authored substep count"
        )
    return type(model).model_validate(
        {
            "dt": _mechanics_control_dt(component.dt, backend),
            "n_steps": int(backend.n_substeps),
            "solver_type": planar_solver_name(backend.solver),
            **component.plant.to_params(),
        }
    )


class _ExcitationMechanics(Mechanics):
    input_ports = ("excitation",)

    def __call__(self, inputs, state, *, key):
        excitation = inputs.get("excitation", inputs.get("force"))
        if excitation is None:
            raise ValueError("Missing required acausal source input 'excitation'")
        value = jnp.asarray(excitation)
        MUSCLE_EXCITATION_PAYLOAD.validate_shape("excitation", value.shape)
        return super().__call__({"force": value}, state, key=key)


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


def _build_analytical_musculoskeletal_plant(params: Mapping[str, Any]) -> _ExcitationMechanics:
    dt = float(params.get("dt", 0.01))
    solver = resolve_acausal_solver(
        params.get("solver_type", DEFAULT_PLANAR_SOLVER), dt, physical_domain="planar_multibody"
    )
    n_steps = int(params.get("n_steps", 1))
    if n_steps < 1:
        raise ValueError("AnalyticalMusculoskeletalPlant n_steps must be >= 1")
    backend = DiffraxBackend(control_dt=dt, sub_dt=dt / n_steps, solver=solver)
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
    return _ExcitationMechanics(
        plant=plant,
        backend=backend,
        key=jr.PRNGKey(0),
    )


def analytical_musculoskeletal_plant_definition() -> DeclaredComponent:
    from feedbax.mechanics.muscle_representation import (
        _default_two_link_muscle_path_geometry,
        _two_link_muscle_representation,
    )

    definition = declare_component(
        name="AnalyticalMusculoskeletalPlant",
        param_model=_AnalyticalPlantParams,
        builder=_build_analytical_musculoskeletal_plant,
        extractor=_extract_analytical_plant_params,
        override_reason=(
            "constructor aggregates body-preset parameters into an analytical plant "
            "and Diffrax backend"
        ),
        category="Mechanics",
        description=(
            "Two-link arm musculoskeletal plant with pure JAX Lagrangian "
            "dynamics and Hill-type rigid-tendon muscles. Fully "
            "differentiable; no MuJoCo dependency. ODE state: 2 joint "
            "angles + 2 angular velocities + 6 muscle activations."
        ),
        input_ports=_ExcitationMechanics.input_ports,
        output_ports=_ExcitationMechanics.output_ports,
        icon="Activity",
        is_composite=False,
        port_types=PortTypeSpec(
            inputs={
                "excitation": PortType(dtype="vector"),
            },
            outputs={
                "effector": PortType(dtype="vector"),
                "state": PortType(dtype="state"),
            },
        ),
        representation=_two_link_muscle_representation(
            source=("feedbax.mechanics.muscle_config.default_6muscle_2link_muscled_arm_parameters"),
            composite=False,
            muscle_path_geometry=_default_two_link_muscle_path_geometry(),
            frame_element_id="links",
            self_link_lengths=[float(value) for value in default_6muscle_2link_segment_lengths()],
        ),
    )
    return replace(
        definition,
        source_paths=(
            *definition.source_paths,
            source_path(AnalyticalMusculoskeletalPlant),
            source_path(_two_link_muscle_representation),
            source_path(default_6muscle_2link_segment_lengths),
        ),
    )
