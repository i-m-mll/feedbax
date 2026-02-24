"""MuJoCo/MJX physics as a feedbax AbstractPlant.

MJXPlant wraps an MJXSkeleton and exposes MuJoCo's monolithic dynamics
through the standard ``vector_field()`` interface for Diffrax integration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
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
        segment_lengths: Segment lengths from the body preset, shape
            ``(n_joints,)``. Used for FK-based reachable target sampling.
            ``None`` if constructed without a body preset.
    """

    skeleton: MJXSkeleton
    clip_states: bool = eqx.field(static=True)
    segment_lengths: Array | None

    def __init__(
        self,
        skeleton: MJXSkeleton,
        clip_states: bool = True,
        segment_lengths: Array | None = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        self.skeleton = skeleton
        self.clip_states = clip_states
        self.segment_lengths = segment_lengths

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

        return cls(
            skeleton=skeleton,
            clip_states=clip_states,
            segment_lengths=jnp.asarray(preset.segment_lengths),
            key=key,
        )

    @classmethod
    def build_batch(
        cls,
        presets: Sequence,
        chain_config,
        sim_config,
        *,
        clip_states: bool = True,
    ) -> MJXPlant:
        """Build N MJXPlants from presets, stack into vmappable batch.

        All presets must produce the same model topology (same nq, nv, nbody,
        nu). Only numeric parameters (masses, lengths, damping, etc.) may
        differ.

        Args:
            presets: Sequence of BodyPreset instances.
            chain_config: ChainConfig with topology (shared).
            sim_config: SimConfig with timing (shared).
            clip_states: Whether to clip states to joint limits.

        Returns:
            MJXPlant with leading ``(B,)`` dim on all array leaves.
        """
        plants = [
            cls.from_body_preset(p, chain_config, sim_config,
                                 clip_states=clip_states)
            for p in presets
        ]

        ref = plants[0].skeleton
        for i, p in enumerate(plants[1:], 1):
            assert p.skeleton.nq == ref.nq, (
                f"Body {i}: nq={p.skeleton.nq} != {ref.nq}"
            )
            assert p.skeleton.nv == ref.nv, (
                f"Body {i}: nv={p.skeleton.nv} != {ref.nv}"
            )
            assert p.skeleton.nbody == ref.nbody, (
                f"Body {i}: nbody={p.skeleton.nbody} != {ref.nbody}"
            )

        # MJX Model puts geometry-dependent arrays (geom_rbound_hfield etc.)
        # into the pytree auxiliary data, so different bodies have different
        # treedefs. We flatten each plant, stack the array leaves, and
        # unflatten with the first plant's treedef (the auxiliary data only
        # affects collision detection, not forward dynamics).
        all_flat = [jt.flatten(p) for p in plants]
        template_treedef = all_flat[0][1]
        stacked_leaves = [
            jnp.stack(leaves)
            for leaves in zip(*[f[0] for f in all_flat])
        ]
        return jt.unflatten(template_treedef, stacked_leaves)

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
