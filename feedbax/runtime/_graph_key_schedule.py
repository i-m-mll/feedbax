"""PRNG schedule for graph transitions and recurrent carry initialization."""

from collections.abc import Iterator

import jax.random as jr
from jaxtyping import PRNGKeyArray

KEY_SCHEDULE_ID = "feedbax.graph_key_schedule.canonical_topological_split.v3"
_RESET_KEY_SALT = 0xFFFF_FFFF


def keyed_nodes(
    execution_order: tuple[str, ...], key: PRNGKeyArray | None
) -> Iterator[tuple[str, PRNGKeyArray]]:
    """Assign split keys in canonical topological order within one graph."""
    if not execution_order:
        return iter(())
    if key is None:
        key = jr.PRNGKey(0)
    return zip(execution_order, jr.split(key, len(execution_order)))


def reset_key(key: PRNGKeyArray) -> PRNGKeyArray:
    """Derive a reset stream from the rollout key or a nested graph's node key."""
    return jr.fold_in(key, _RESET_KEY_SALT)
