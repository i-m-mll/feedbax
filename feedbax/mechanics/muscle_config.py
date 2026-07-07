"""Muscle topology and attachment configuration for articulated planar chains.

Defines static muscle-joint connectivity (MuscleTopology) and per-body
attachment geometry (MuscleConfig) including moment arm matrices that
support biarticular muscles spanning multiple joints.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from feedbax.mechanics.body import BodyPreset


_DEFAULT_6MUSCLE_2LINK_ROUTING: tuple[tuple[bool, ...], ...] = (
    (True, False),   # shoulder flexor
    (True, False),   # shoulder extensor
    (False, True),   # elbow flexor
    (False, True),   # elbow extensor
    (True, True),    # biarticular flexor
    (True, True),    # biarticular extensor
)
_DEFAULT_6MUSCLE_2LINK_SIGN: tuple[tuple[int, ...], ...] = (
    (+1, 0),    # shoulder flexor
    (-1, 0),    # shoulder extensor
    (0, +1),    # elbow flexor
    (0, -1),    # elbow extensor
    (+1, +1),   # biarticular flexor
    (-1, -1),   # biarticular extensor
)
_DEFAULT_6MUSCLE_2LINK_MOMENT_ARM_MAGNITUDES: tuple[tuple[float, ...], ...] = (
    (0.04, 0.0),      # shoulder flexor
    (0.04, 0.0),      # shoulder extensor
    (0.0, 0.025),     # elbow flexor
    (0.0, 0.025),     # elbow extensor
    (0.035, 0.022),   # biarticular flexor
    (0.035, 0.022),   # biarticular extensor
)
_DEFAULT_6MUSCLE_2LINK_REFERENCE_LENGTHS: tuple[float, ...] = (
    0.2,
    0.2,
    0.2,
    0.2,
    0.3,
    0.3,
)


class MuscleAttachmentFrame(NamedTuple):
    """A local muscle attachment frame.

    Attributes:
        body: Body name that owns the attachment frame.
        pos: Attachment position in that body's local frame, shape ``(3,)`` [m].
    """

    body: str
    pos: Float[Array, "3"]


# ---------------------------------------------------------------------------
# MuscleTopology — static connectivity (shared across a vmap batch)
# ---------------------------------------------------------------------------


def _to_nested_tuple(arr: np.ndarray) -> tuple[tuple, ...]:
    """Convert a 2D numpy array to a nested tuple for hashable static storage."""
    return tuple(tuple(row) for row in arr)


class MuscleTopology(eqx.Module):
    """Static muscle-joint connectivity shared across a body batch.

    Stored in the pytree treedef (``static=True``) so that all bodies
    in a vmapped batch share the same topology without duplicating
    these arrays on the leading batch dimension.

    The ``routing`` and ``sign`` fields are stored as nested tuples (not
    arrays) to avoid equinox warnings about JAX/numpy arrays in static
    fields.  Use the ``routing_array`` and ``sign_array`` properties for
    JAX-compatible array access.

    Attributes:
        routing: Nested tuple of booleans, shape ``(n_muscles, n_joints)``.
        sign: Nested tuple of ints (+1/-1/0), shape ``(n_muscles, n_joints)``.
    """

    routing: tuple[tuple[bool, ...], ...] = eqx.field(static=True)
    sign: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    @property
    def n_muscles(self) -> int:
        """Number of muscles."""
        return len(self.routing)

    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return len(self.routing[0]) if self.routing else 0

    @property
    def routing_array(self) -> Bool[Array, "n_muscles n_joints"]:
        """Boolean routing mask as a JAX array."""
        return jnp.array(self.routing, dtype=bool)

    @property
    def sign_array(self) -> Int[Array, "n_muscles n_joints"]:
        """Signed direction matrix as a JAX array."""
        return jnp.array(self.sign, dtype=jnp.int32)


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
    return MuscleTopology(
        routing=_DEFAULT_6MUSCLE_2LINK_ROUTING,
        sign=_DEFAULT_6MUSCLE_2LINK_SIGN,
    )


def default_6muscle_2link_moment_arm_magnitudes() -> Float[Array, "6 2"]:
    """Canonical unsigned 6-muscle, 2-link moment-arm magnitudes [m].

    Rows follow ``default_6muscle_2link_topology``.  Non-spanned joints are
    encoded as zero so callers can use this matrix directly for presets and
    sampling bounds.
    """
    return jnp.array(_DEFAULT_6MUSCLE_2LINK_MOMENT_ARM_MAGNITUDES)


def default_6muscle_2link_reference_lengths() -> Float[Array, "6"]:
    """Canonical musculotendon reference lengths at zero angles [m]."""
    return jnp.array(_DEFAULT_6MUSCLE_2LINK_REFERENCE_LENGTHS)


def signed_moment_arm_matrix(
    magnitudes: Array,
    topology: MuscleTopology,
) -> Float[Array, "n_muscles n_joints"]:
    """Apply topology signs and routing to unsigned moment-arm magnitudes.

    Args:
        magnitudes: Unsigned moment-arm magnitudes, shape
            ``(n_muscles, n_joints)`` [m].
        topology: Static muscle-joint routing and signs.

    Returns:
        Signed moment-arm matrix with non-routed entries forced to zero,
        shape ``(n_muscles, n_joints)`` [m].

    Raises:
        ValueError: If ``magnitudes`` does not match ``topology``.
    """
    magnitudes = jnp.asarray(magnitudes)
    expected_shape = (topology.n_muscles, topology.n_joints)
    if magnitudes.shape != expected_shape:
        raise ValueError(
            "moment arm magnitudes must match topology shape: "
            f"{magnitudes.shape} != {expected_shape}"
        )
    return jnp.where(
        topology.routing_array,
        magnitudes * topology.sign_array,
        0.0,
    )


def default_6muscle_2link_moment_arms() -> Float[Array, "6 2"]:
    """Canonical signed 6-muscle, 2-link moment-arm matrix [m]."""
    return signed_moment_arm_matrix(
        default_6muscle_2link_moment_arm_magnitudes(),
        default_6muscle_2link_topology(),
    )


def default_6muscle_2link_muscled_arm_parameters() -> dict[str, Array]:
    """Canonical parameters for legacy ``MuscledArm`` defaults.

    ``MuscledArm`` stores moment arms as ``(n_joints, n_muscles)`` and uses
    normalized muscle lengths.  The returned ``l0`` values are the canonical
    musculotendon reference lengths, so its normalized length model matches
    ``TwoLinkArmMuscleGeometry.default_six_muscle()`` divided by ``l0``.
    """
    moment_arms = default_6muscle_2link_moment_arms().T
    l0 = default_6muscle_2link_reference_lengths()
    return {
        "moment_arms": moment_arms,
        "theta0": jnp.zeros_like(moment_arms),
        "l0": l0,
        "f0": jnp.ones_like(l0),
    }


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
    routing_list = [[False] * n_joints for _ in range(n_muscles)]
    sign_list = [[0] * n_joints for _ in range(n_muscles)]
    for j in range(n_joints):
        for m in range(muscles_per_joint):
            idx = j * muscles_per_joint + m
            routing_list[idx][j] = True
            sign_list[idx][j] = 1 if m % 2 == 0 else -1
    return MuscleTopology(
        routing=tuple(tuple(row) for row in routing_list),
        sign=tuple(tuple(row) for row in sign_list),
    )


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

    @property
    def origin_frames(self) -> tuple[MuscleAttachmentFrame, ...]:
        """Origin attachment frames for each muscle.

        Returns:
            One frame per muscle.  Each frame contains the body name and local
            attachment position, shape ``(3,)`` [m].
        """
        return tuple(
            MuscleAttachmentFrame(body=body, pos=self.origin_pos[i])
            for i, body in enumerate(self.origin_body)
        )

    @property
    def insertion_frames(self) -> tuple[MuscleAttachmentFrame, ...]:
        """Insertion attachment frames for each muscle.

        Returns:
            One frame per muscle.  Each frame contains the body name and local
            attachment position, shape ``(3,)`` [m].
        """
        return tuple(
            MuscleAttachmentFrame(body=body, pos=self.insertion_pos[i])
            for i, body in enumerate(self.insertion_body)
        )


def default_muscle_config(
    preset: BodyPreset,
    chain_config,
    topology: MuscleTopology | None = None,
) -> MuscleConfig:
    """Create a muscle attachment layout from topology.

    For monoarticular topologies, muscles span each joint with small
    lateral offsets to create opposing attachment sites.  The moment arm
    matrix is built from the preset's ``muscle_moment_arm_magnitudes`` and
    the topology's sign/routing matrix.

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
        # Find the joints this muscle spans (tuple indexing).
        spanned = [j for j in range(n_joints) if topology.routing[i][j]]
        if not spanned:
            # Defensive -- muscle spans no joint (should not happen).
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

        # Sign for lateral offset: flexor -> +y, extensor -> -y.
        sign_val = topology.sign[i][first_joint]
        lateral = sign_val * y_offset if sign_val != 0 else y_offset

        origin_body.append(parent_name)
        insertion_body.append(child_name)
        origin_pos = origin_pos.at[i].set(
            jnp.array([0.1 * parent_len, lateral, 0.0])
        )
        insertion_pos = insertion_pos.at[i].set(
            jnp.array([0.1 * child_len, lateral, 0.0])
        )

    if not hasattr(preset, "muscle_moment_arm_magnitudes"):
        raise ValueError(
            "default_muscle_config requires preset.muscle_moment_arm_magnitudes; "
            "construct a BodyPreset with explicit magnitudes."
        )
    moment_arms = signed_moment_arm_matrix(
        preset.muscle_moment_arm_magnitudes,
        topology,
    )

    return MuscleConfig(
        origin_body=tuple(origin_body),
        insertion_body=tuple(insertion_body),
        origin_pos=origin_pos,
        insertion_pos=insertion_pos,
        moment_arms=moment_arms,
        topology=topology,
    )
