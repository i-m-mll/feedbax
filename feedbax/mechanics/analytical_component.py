"""Portable analytical musculoskeletal mechanics owner."""

from __future__ import annotations
from typing import Annotated, Any
from collections.abc import Mapping
import jax.numpy as jnp
import jax.random as jr
from pydantic import Field
from feedbax.contracts.acausal_interface import MUSCLE_EXCITATION_PAYLOAD
from feedbax.contracts.component import PortType
from feedbax.contracts.portable import Ports, ConfigBinding
from feedbax.runtime.graph import Component
from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant as AnalyticalPlant
from feedbax.mechanics.backend import DiffraxBackend
from feedbax.mechanics.body import BodyPreset, default_2link_bounds
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.model_builder import ChainConfig
from feedbax.mechanics.muscle_config import (
    default_6muscle_2link_segment_lengths,
    default_6muscle_2link_topology,
)
from feedbax.mechanics.solver_config import (
    DEFAULT_PLANAR_SOLVER,
    MIN_PLANAR_DT,
    PlanarSolverType,
    planar_solver_name,
    resolve_acausal_solver,
)


def _default_analytical_body_preset(params: Mapping[str, Any]) -> BodyPreset:
    bounds = default_2link_bounds()
    tau_act = params["tau_act"]
    tau_deact = params["tau_deact"]
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


class AnalyticalMusculoskeletalPlant(Mechanics, Component):
    """Two-link musculoskeletal arm with analytical dynamics and rigid-tendon muscles."""

    ports = Ports(
        inputs={"excitation": PortType(dtype="vector")},
        outputs={"effector": PortType(dtype="vector"), "state": PortType(dtype="state")},
    )
    config_bindings = (
        ConfigBinding(path="backend.n_substeps", name="n_steps"),
        ConfigBinding(path="plant.tau_act", name="tau_act"),
        ConfigBinding(path="plant.tau_deact", name="tau_deact"),
        ConfigBinding(path="plant.clip_states", name="clip_states"),
    )

    def __init__(
        self,
        dt: Annotated[float, Field(ge=MIN_PLANAR_DT)] = 0.01,
        n_steps: Annotated[int, Field(ge=1)] = 1,
        solver_type: PlanarSolverType = DEFAULT_PLANAR_SOLVER,
        tau_act: Annotated[float, Field(ge=0.001)] = 0.01,
        tau_deact: Annotated[float, Field(ge=0.001)] = 0.04,
        clip_states: bool = True,
    ):
        solver = resolve_acausal_solver(solver_type, dt, physical_domain="planar_multibody")
        backend = DiffraxBackend(control_dt=dt, sub_dt=dt / n_steps, solver=solver)
        chain_config = ChainConfig(n_joints=2, muscle_topology=default_6muscle_2link_topology())
        plant = AnalyticalPlant.from_body_preset(
            _default_analytical_body_preset({"tau_act": tau_act, "tau_deact": tau_deact}),
            chain_config,
            clip_states=clip_states,
            key=jr.PRNGKey(0),
        )
        super().__init__(plant=plant, backend=backend, key=jr.PRNGKey(0))

    @property
    def solver_type(self):
        return planar_solver_name(self.backend.solver)

    def __call__(self, inputs, state, *, key):
        excitation = inputs.get("excitation")
        if excitation is None:
            raise ValueError("Missing required acausal source input 'excitation'")
        value = jnp.asarray(excitation)
        MUSCLE_EXCITATION_PAYLOAD.validate_shape("excitation", value.shape)
        return super().__call__({"force": value}, state, key=key)
