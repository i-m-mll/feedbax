"""MuJoCo/MJX physics as a feedbax AbstractPlant.

MJXPlant wraps an MJXSkeleton and exposes MuJoCo's monolithic dynamics
through the standard ``vector_field()`` interface for Diffrax integration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from feedbax.mechanics.plant import AbstractPlant, DynamicsComponent, PlantState
from feedbax.mechanics.skeleton.mjx_skeleton import MJXSkeleton, MJXSkeletonState
from feedbax.state import StateBounds


class MJXPlant(AbstractPlant):
    """MuJoCo/MJX plant model implementing feedbax's AbstractPlant interface.

    MJX dynamics are monolithic — the ``vector_field()`` is overridden
    directly rather than composed via ``dynamics_spec``.

    Attributes:
        skeleton: The MJXSkeleton providing physics computations.
        clip_states: Whether to clip states to bounds after integration.
    """

    skeleton: MJXSkeleton
    clip_states: bool = eqx.field(static=True)

    def __init__(
        self,
        skeleton: MJXSkeleton,
        clip_states: bool = True,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        self.skeleton = skeleton
        self.clip_states = clip_states

    @classmethod
    def from_body_preset(
        cls,
        preset,
        chain_config,
        sim_config,
        *,
        clip_states: bool = True,
        key: Optional[PRNGKeyArray] = None,
    ) -> MJXPlant:
        """Construct an MJXPlant from a body preset and configurations.

        Args:
            preset: BodyPreset with physical parameters.
            chain_config: ChainConfig with topology.
            sim_config: SimConfig with timing.
            clip_states: Whether to clip states to joint limits.
            key: Optional PRNG key.

        Returns:
            Configured MJXPlant ready for simulation.
        """
        from feedbax.mechanics.model_builder import (
            build_model,
            get_body_id,
            get_site_id,
            to_mjx,
        )

        mj_model = build_model(preset, chain_config, sim_config)
        mjx_model = to_mjx(mj_model)

        effector_site_id = get_site_id(mj_model, "effector")
        effector_body_id = get_body_id(
            mj_model, f"link{chain_config.n_joints - 1}"
        )

        skeleton = MJXSkeleton(
            mjx_model=mjx_model,
            effector_site_id=effector_site_id,
            effector_body_id=effector_body_id,
        )

        return cls(skeleton=skeleton, clip_states=clip_states, key=key)

    @jax.named_scope("fbx.MJXPlant.vector_field")
    def vector_field(
        self, t: Scalar, state: PlantState, input: PyTree[Array]
    ) -> PlantState:
        """Compute time derivatives via MJX forward dynamics.

        Overrides the base ``AbstractPlant.vector_field`` because MJX
        dynamics are monolithic (no skeleton + muscle decomposition).

        Args:
            t: Time (unused).
            state: Current plant state.
            input: Actuator controls.

        Returns:
            PlantState with time derivatives.
        """
        d_skeleton = self.skeleton.vector_field(t, state.skeleton, input)
        return PlantState(skeleton=d_skeleton, muscles=None)

    @property
    def dynamics_spec(self) -> Mapping[str, DynamicsComponent[PlantState]]:
        """Empty — MJXPlant overrides ``vector_field`` directly."""
        return {}

    def kinematics_update(
        self,
        input: PyTree[Array],
        state: PlantState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PlantState:
        """Apply instantaneous updates (state clipping if enabled).

        MuJoCo handles muscle dynamics internally, so this is minimal.

        Args:
            input: Actuator controls.
            state: Current plant state.
            key: Unused.

        Returns:
            Optionally clipped plant state.
        """
        if not self.clip_states:
            return state
        return eqx.tree_at(
            lambda s: s.skeleton,
            state,
            self._clip_state(self.bounds.skeleton, state.skeleton),
        )

    def init(self, *, key: PRNGKeyArray) -> PlantState:
        """Initialize the plant state.

        Args:
            key: PRNG key for random initialization.

        Returns:
            Initial PlantState with MJXSkeletonState.
        """
        return PlantState(
            skeleton=self.skeleton.init(key=key),
            muscles=None,
        )

    @property
    def input_size(self) -> int:
        """Number of actuator inputs."""
        return self.skeleton.input_size
