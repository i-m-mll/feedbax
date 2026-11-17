"""Project planar samples onto aligned axes."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def project_onto_direction(
    var: Float[Array, "*batch conditions time xy=2"],
    direction_vec: Float[Array, "*batch conditions xy=2"],
):
    """Projects components of arbitrary variables into components parallel and orthogonal to a given direction.

    Arguments:
        var: Data with x-y components to be projected.
        direction_vec: Direction vectors.

    Returns:
        projected: Projected components (parallel and lateral).
    """
    # Normalize the line vector
    direction_vec_norm = direction_vec / jnp.linalg.norm(direction_vec, axis=-1, keepdims=True)

    # Broadcast line_vec_norm to match velocity's shape
    direction_vec_norm = direction_vec_norm[..., None, :]

    # Calculate forward component (dot product)
    parallel = jnp.sum(var * direction_vec_norm, axis=-1)

    # Calculate lateral component (cross product)
    lateral = jnp.cross(direction_vec_norm, var)

    return jnp.stack([parallel, lateral], axis=-1)
