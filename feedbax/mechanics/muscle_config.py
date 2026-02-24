"""Muscle topology and attachment configuration for articulated planar chains.

Defines static muscle-joint connectivity (MuscleTopology) and per-body
attachment geometry (MuscleConfig) including moment arm matrices that
support biarticular muscles spanning multiple joints.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from feedbax.mechanics.body import BodyPreset


# ---------------------------------------------------------------------------
# MuscleTopology — static connectivity (shared across a vmap batch)
# ---------------------------------------------------------------------------


class MuscleTopology(eqx.Module):
    """Static muscle-joint connectivity shared across a body batch.

    Stored in the pytree treedef (``static=True``) so that all bodies
    in a vmapped batch share the same topology without duplicating
    these arrays on the leading batch dimension.

    Attributes:
        routing: Boolean mask indicating which muscles span which joints,
            shape ``(n_muscles, n_joints)``.
        sign: Signed direction per muscle-joint pair (+1 flexion, -1 extension,
            0 not connected), shape ``(n_muscles, n_joints)``.
    """

    routing: Bool[Array, "n_muscles n_joints"] = eqx.field(static=True)
    sign: Int[Array, "n_muscles n_joints"] = eqx.field(static=True)

    @property
    def n_muscles(self) -> int:
        """Number of muscles."""
        return self.routing.shape[0]

    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return self.routing.shape[1]


def default_6muscle_2link_topology() -> MuscleTopology:
    """4 monoarticular + 2 biarticular muscles for a 2-joint arm.

    Muscle layout:
        0: shoulder flexor      (mono, joint 0)
        1: shoulder extensor    (mono, joint 0)
        2: elbow flexor         (mono, joint 1)
        3: elbow extensor       (mono, joint 1)
        4: biarticular flexor   (joints 0+1)
        5: biarticular extensor (joints 0+1)

    Returns:
        MuscleTopology with ``(6, 2)`` routing and sign arrays.
    """
    routing = jnp.array([
        [True, False],   # shoulder flexor
        [True, False],   # shoulder extensor
        [False, True],   # elbow flexor
        [False, True],   # elbow extensor
        [True, True],    # biarticular flexor
        [True, True],    # biarticular extensor
    ])
    sign = jnp.array([
        [+1, 0],    # shoulder flexor
        [-1, 0],    # shoulder extensor
        [0, +1],    # elbow flexor
        [0, -1],    # elbow extensor
        [+1, +1],   # biarticular flexor
        [-1, -1],   # biarticular extensor
    ], dtype=jnp.int32)
    return MuscleTopology(routing=routing, sign=sign)


def default_monoarticular_topology(
    n_joints: int, muscles_per_joint: int = 2,
) -> MuscleTopology:
    """Backward-compatible monoarticular topology.

    Creates ``n_joints * muscles_per_joint`` muscles, each spanning
    exactly one joint.  Within each joint group the first muscle is a
    flexor (+1) and the second an extensor (-1), alternating for any
    additional muscles.

    Args:
        n_joints: Number of joints.
        muscles_per_joint: Muscles per joint (default 2 for antagonist pairs).

    Returns:
        MuscleTopology with ``(n_muscles, n_joints)`` arrays.
    """
    n_muscles = n_joints * muscles_per_joint
    routing = jnp.zeros((n_muscles, n_joints), dtype=bool)
    sign = jnp.zeros((n_muscles, n_joints), dtype=jnp.int32)
    for j in range(n_joints):
        for m in range(muscles_per_joint):
            idx = j * muscles_per_joint + m
            routing = routing.at[idx, j].set(True)
            sign = sign.at[idx, j].set(1 if m % 2 == 0 else -1)
    return MuscleTopology(routing=routing, sign=sign)


# ---------------------------------------------------------------------------
# MuscleConfig — per-body attachment geometry + moment arm matrix
# ---------------------------------------------------------------------------


class MuscleConfig(eqx.Module):
    """Attachment geometry and moment arm matrix for muscles.

    The moment arm matrix encodes how each muscle's force contributes
    to joint torques: ``joint_torques = moment_arms.T @ muscle_forces``.
    Biarticular muscles have non-zero entries in multiple columns.

    Attributes:
        origin_body: Name of the body each muscle originates from.
        insertion_body: Name of the body each muscle inserts into.
        origin_pos: Origin attachment position in local body frame,
            shape ``(n_muscles, 3)``.
        insertion_pos: Insertion attachment position in local body frame,
            shape ``(n_muscles, 3)``.
        moment_arms: Signed moment arm matrix, shape ``(n_muscles, n_joints)``.
            Computed as ``magnitudes * topology.sign``.
        topology: Static muscle-joint connectivity (in treedef).
    """

    origin_body: tuple[str, ...]
    insertion_body: tuple[str, ...]
    origin_pos: Float[Array, "n_muscles 3"]
    insertion_pos: Float[Array, "n_muscles 3"]
    moment_arms: Float[Array, "n_muscles n_joints"]
    topology: MuscleTopology

    @property
    def n_muscles(self) -> int:
        """Number of muscles."""
        return self.origin_pos.shape[0]

    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return self.moment_arms.shape[1]


def default_muscle_config(
    preset: BodyPreset,
    chain_config,
    topology: MuscleTopology | None = None,
) -> MuscleConfig:
    """Create a muscle attachment layout from topology.

    For monoarticular topologies, muscles span each joint with small
    lateral offsets to create opposing moment arms.  For topologies with
    biarticular muscles (e.g. ``default_6muscle_2link_topology``), the
    moment arm matrix is built from the preset's
    ``muscle_moment_arm_magnitudes`` if available, otherwise from a
    default lateral offset.

    Args:
        preset: Body parameters (segment lengths used for attachment positions).
        chain_config: Chain topology configuration (provides ``muscle_topology``).
        topology: Explicit topology override.  If ``None``, uses
            ``chain_config.muscle_topology``.

    Returns:
        MuscleConfig with moment arm matrix and attachment sites.
    """
    if topology is None:
        topology = chain_config.muscle_topology

    n_joints = topology.n_joints
    n_muscles = topology.n_muscles
    lengths = preset.segment_lengths

    origin_body: list[str] = []
    insertion_body: list[str] = []
    origin_pos = jnp.zeros((n_muscles, 3))
    insertion_pos = jnp.zeros((n_muscles, 3))

    y_offset = 0.02

    # Build attachment sites for each muscle based on topology routing.
    for i in range(n_muscles):
        # Find the joints this muscle spans.
        spanned = [j for j in range(n_joints) if bool(topology.routing[i, j])]
        if not spanned:
            # Defensive — muscle spans no joint (should not happen).
            origin_body.append("world")
            insertion_body.append("link0")
            continue

        # Origin: parent of the first spanned joint.
        first_joint = spanned[0]
        parent_name = "world" if first_joint == 0 else f"link{first_joint - 1}"
        parent_len = float(lengths[first_joint - 1]) if first_joint > 0 else 0.0

        # Insertion: child of the last spanned joint.
        last_joint = spanned[-1]
        child_name = f"link{last_joint}"
        child_len = float(lengths[last_joint])

        # Sign for lateral offset: flexor → +y, extensor → -y.
        # Use the sign at the first spanned joint to choose offset direction.
        sign_val = float(topology.sign[i, first_joint])
        lateral = sign_val * y_offset if sign_val != 0 else y_offset

        origin_body.append(parent_name)
        insertion_body.append(child_name)
        origin_pos = origin_pos.at[i].set(
            jnp.array([0.1 * parent_len, lateral, 0.0])
        )
        insertion_pos = insertion_pos.at[i].set(
            jnp.array([0.1 * child_len, lateral, 0.0])
        )

    # Build the signed moment arm matrix.
    if hasattr(preset, "muscle_moment_arm_magnitudes"):
        moment_arms = preset.muscle_moment_arm_magnitudes * topology.sign
    else:
        # Legacy fallback: uniform magnitude from lateral offset.
        moment_arms = jnp.full(
            (n_muscles, n_joints), y_offset,
        ) * topology.sign
    # Zero out entries where muscle does not span the joint.
    moment_arms = jnp.where(topology.routing, moment_arms, 0.0)

    return MuscleConfig(
        origin_body=tuple(origin_body),
        insertion_body=tuple(insertion_body),
        origin_pos=origin_pos,
        insertion_pos=insertion_pos,
        moment_arms=moment_arms,
        topology=topology,
    )
