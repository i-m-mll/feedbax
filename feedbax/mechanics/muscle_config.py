"""Muscle attachment configuration for articulated planar chains.

Defines how antagonist muscle pairs attach to body segments,
including origin/insertion geometry and moment arms.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from feedbax.mechanics.body import BodyPreset
from feedbax.mechanics.model_builder import ChainConfig


class MuscleConfig(eqx.Module):
    """Attachment geometry for antagonist muscle pairs per joint.

    Attributes:
        origin_body: Name of the body each muscle originates from.
        insertion_body: Name of the body each muscle inserts into.
        origin_pos: Origin attachment position in local body frame, shape ``(n_muscles, 3)``.
        insertion_pos: Insertion attachment position in local body frame, shape ``(n_muscles, 3)``.
        moment_arm: Signed moment arm for each muscle, shape ``(n_muscles,)``.
        joint_index: Index of the joint each muscle spans, shape ``(n_muscles,)``.
    """

    origin_body: tuple[str, ...]
    insertion_body: tuple[str, ...]
    origin_pos: Float[Array, "n_muscles 3"]
    insertion_pos: Float[Array, "n_muscles 3"]
    moment_arm: Float[Array, " n_muscles"]
    joint_index: Int[Array, " n_muscles"]

    @property
    def n_muscles(self) -> int:
        return self.origin_pos.shape[0]


def default_muscle_config(preset: BodyPreset, chain_config: ChainConfig) -> MuscleConfig:
    """Create a simple flexor/extensor attachment layout.

    Muscles span each joint with small lateral offsets to create opposing
    moment arms. Positions are defined in the local body frames.

    Args:
        preset: Body parameters (segment lengths used for attachment positions).
        chain_config: Chain topology configuration.

    Returns:
        MuscleConfig with antagonist pairs for each joint.
    """
    n_joints = chain_config.n_joints
    n_muscles = chain_config.n_muscles
    lengths = preset.segment_lengths

    origin_body: list[str] = []
    insertion_body: list[str] = []
    origin_pos = jnp.zeros((n_muscles, 3))
    insertion_pos = jnp.zeros((n_muscles, 3))
    moment_arm = jnp.zeros((n_muscles,))
    joint_index = jnp.zeros((n_muscles,), dtype=jnp.int32)

    y_offset = 0.02
    for joint in range(n_joints):
        parent_name = "world" if joint == 0 else f"link{joint - 1}"
        child_name = f"link{joint}"
        parent_len = float(lengths[joint - 1]) if joint > 0 else 0.0
        child_len = float(lengths[joint])

        for k, sign in enumerate((1.0, -1.0)):
            idx = joint * chain_config.muscles_per_joint + k
            origin_body.append(parent_name)
            insertion_body.append(child_name)
            origin_pos = origin_pos.at[idx].set(
                jnp.array([0.1 * parent_len, sign * y_offset, 0.0])
            )
            insertion_pos = insertion_pos.at[idx].set(
                jnp.array([0.1 * child_len, sign * y_offset, 0.0])
            )
            moment_arm = moment_arm.at[idx].set(sign * y_offset)
            joint_index = joint_index.at[idx].set(joint)

    return MuscleConfig(
        origin_body=tuple(origin_body),
        insertion_body=tuple(insertion_body),
        origin_pos=origin_pos,
        insertion_pos=insertion_pos,
        moment_arm=moment_arm,
        joint_index=joint_index,
    )
