"""Couple authored muscle laws and MJX tendon transmissions at each stage."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import equinox as eqx
import jax
from equinox.nn import StateIndex
from jaxtyping import Array, PRNGKeyArray, Scalar

from feedbax.mechanics.compiled_model import MusclePathBinding
from feedbax.mechanics.hill_muscles import BoundedRigidTendonHillLaw, HillMuscleParams
from feedbax.mechanics.muscle_law import MuscleLaw, MuscleLawBank
from feedbax.mechanics.plant import AbstractPlant, PlantState
from feedbax.mechanics.skeleton.mjx_skeleton import MJXSkeleton
from feedbax.mechanics.units import DEFAULT_MUSCLE_VMAX, require_positive_finite

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry


LEGACY_RIGID_LAW = "feedbax.RigidTendonHillMuscle"


def _finite_range(name: str, value: object, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} requires a real number")
    if not isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} requires a finite value in [{low}, {high}]")
    return float(value)


def _bounded_rigid_law(parameters: dict) -> BoundedRigidTendonHillLaw:
    supported = {
        "max_isometric_force", "optimal_muscle_length", "tendon_slack_length",
        "max_contraction_velocity", "tau_act", "tau_deact", "initial_activation",
        "reference_length",
    }
    if unknown := parameters.keys() - supported:
        raise ValueError(f"Unsupported rigid Hill law parameters: {sorted(unknown)}")
    vmax = parameters.get("max_contraction_velocity")
    return BoundedRigidTendonHillLaw(
        params=HillMuscleParams(
            max_isometric_force=_finite_range(
                "max_isometric_force", parameters.get("max_isometric_force", 1.0),
                0.0, float("inf"),
            ),
            optimal_fiber_length=require_positive_finite(
                "optimal_muscle_length", parameters.get("optimal_muscle_length", 0.1),
            ),
            tendon_slack_length=_finite_range(
                "tendon_slack_length", parameters.get("tendon_slack_length", 0.1),
                0.0, float("inf"),
            ),
            tau_activation=require_positive_finite("tau_act", parameters.get("tau_act", 0.01)),
            tau_deactivation=require_positive_finite("tau_deact", parameters.get("tau_deact", 0.04)),
            vmax=require_positive_finite(
                "max_contraction_velocity", DEFAULT_MUSCLE_VMAX if vmax is None else vmax,
            ),
        ),
        initial_activation=_finite_range(
            "initial_activation", parameters.get("initial_activation", 0.0), 0.0, 1.0,
        ),
    )


def build_muscle_law(
    binding: MusclePathBinding, registry: ComponentRegistry,
) -> tuple[MuscleLaw, tuple[str, int]]:
    """Resolve one authored law through the same registry as the graph."""
    parameters = dict(binding.parameters)
    if binding.law == LEGACY_RIGID_LAW:
        if binding.law_revision not in (None, 1):
            raise ValueError(f"Unsupported {LEGACY_RIGID_LAW} revision {binding.law_revision}")
        law = _bounded_rigid_law(parameters)
        identity = (binding.law, 1)
    else:
        resolved = registry.resolve_component_spec(
            binding.law, parameters, revision=binding.law_revision,
        )
        component = resolved.meta.params.build_resolved(resolved.params)
        factory = getattr(component, "as_muscle_law", None)
        law = component if factory is None else factory()
        identity = (resolved.type_id, resolved.revision)
    if not isinstance(law, eqx.Module) or not isinstance(law, MuscleLaw):
        raise ValueError(f"Muscle path {binding.name!r} requires a pure JAX muscle law")
    if any(isinstance(leaf, StateIndex) for leaf in jax.tree.leaves(
        law, is_leaf=lambda value: isinstance(value, StateIndex),
    )):
        raise ValueError(f"Muscle path {binding.name!r} law has an independent StateIndex")
    if law.state_size != binding.state_size:
        raise ValueError(
            f"Muscle path {binding.name!r} declares state width {binding.state_size}, "
            f"but law {binding.law!r} requires {law.state_size}"
        )
    return law, identity


class ArticulatedMusclePlant(AbstractPlant):
    """The muscle and skeleton share one state and continuous integration clock."""

    skeleton: MJXSkeleton
    law_bank: MuscleLawBank
    law_identities: tuple[tuple[str, int], ...] = eqx.field(static=True)
    clip_states: bool = eqx.field(static=True, default=False)

    def __init__(self, skeleton: MJXSkeleton, registry: ComponentRegistry):
        descriptor = skeleton.descriptor
        if descriptor is None or not descriptor.paths:
            raise ValueError("Articulated muscle excitation requires authored muscle paths")
        self.skeleton = skeleton
        bound = tuple(build_muscle_law(path, registry) for path in descriptor.paths)
        laws, self.law_identities = zip(*bound, strict=True)
        self.law_bank = MuscleLawBank(laws)

    @property
    def laws(self) -> tuple[MuscleLaw, ...]:
        return self.law_bank.laws

    @property
    def input_size(self) -> int:
        return len(self.laws)

    def init(self, *, key: PRNGKeyArray) -> PlantState:
        skeleton = self.skeleton.init(key=key)
        stage = self.skeleton.continuous_stage(skeleton)
        geometry = self.skeleton.muscle_path_kinematics(stage)
        states = self.law_bank.init_state(geometry.length, geometry.velocity)
        return PlantState(skeleton=skeleton, muscles=states)

    def vector_field(self, t: Scalar, state: PlantState, input: Array) -> PlantState:
        stage = self.skeleton.continuous_stage(state.skeleton)
        geometry = self.skeleton.muscle_path_kinematics(stage)
        derivatives, tension = self.law_bank.evaluate(
            state.muscles, input, geometry.length, geometry.velocity,
        )
        control = self.skeleton.tendon_controls(tension)
        skeleton = self.skeleton.continuous_derivative(state.skeleton, stage, control)
        return PlantState(skeleton=skeleton, muscles=derivatives)
