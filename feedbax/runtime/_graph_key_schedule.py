"""Versioned PRNG derivation for graph execution and task sampling."""

from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache
from hashlib import sha256
from numbers import Integral
from struct import unpack

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

KEY_SCHEDULE_ID = "feedbax.graph_key_schedule.scoped_occurrence.v4"


@lru_cache(maxsize=4096)
def _scope_words(scope: tuple[str, ...], indexed: bool) -> tuple[int, ...]:
    digest = sha256()
    for token in (KEY_SCHEDULE_ID, *scope):
        encoded = token.encode("utf-8")
        digest.update(b"s")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    if indexed:
        digest.update(b"i")
    return unpack(">4I", digest.digest()[:16])


def _fold_scope(key: PRNGKeyArray | None, words: Iterable[int | jax.Array]) -> PRNGKeyArray:
    if key is None:
        key = jr.PRNGKey(0)
    for word in words:
        key = jr.fold_in(key, word)
    return key


def derive_key(
    key: PRNGKeyArray | None,
    *scope: str,
    index: int | jax.Array | None = None,
) -> PRNGKeyArray:
    """Derive a stream from framed UTF-8 identities and an optional uint32 index.

    The numeric position is a scalar integer in [0, 2**32). Traced positions
    must stay in range. Static scope tokens may contain arbitrary delimiters.
    """
    if index is not None:
        scalar_integer = isinstance(index, Integral) and not isinstance(index, bool)
        if hasattr(index, "dtype"):
            scalar_integer = index.ndim == 0 and jnp.issubdtype(index.dtype, jnp.integer)
        if not scalar_integer:
            raise ValueError("Random stream index must be a scalar integer")
        if not isinstance(index, jax.core.Tracer) and not 0 <= int(index) < 2**32:
            raise ValueError("Random stream index must fit in uint32")
    scoped_key = _fold_scope(key, _scope_words(scope, index is not None))
    return scoped_key if index is None else jr.fold_in(scoped_key, index)


def _derive_many(
    derive: Callable[[PRNGKeyArray, jax.Array], PRNGKeyArray],
    key: PRNGKeyArray | None,
    values: jax.Array,
) -> PRNGKeyArray:
    if key is None:
        key = jr.PRNGKey(0)

    def derive_one(value):
        return derive(key, value)

    # The unsafe RBG implementation derives batched keys from the first key
    # in each batch. Map scalar folds to keep its streams independent.
    if jr.key_impl(key) == "unsafe_rbg":
        return jax.lax.map(derive_one, values)
    return jax.vmap(derive_one)(values)


def step_key(key: PRNGKeyArray, t: int | jax.Array) -> PRNGKeyArray:
    return derive_key(key, "step", index=t)


def step_keys(key: PRNGKeyArray, n_steps: int) -> PRNGKeyArray:
    return _derive_many(step_key, key, jnp.arange(n_steps, dtype=jnp.uint32))


def sample_steps(
    sample: Callable[[PRNGKeyArray], jax.Array], key: PRNGKeyArray, n_steps: int
) -> jax.Array:
    """Draw one sample per timestep from its independent key."""
    keys = step_keys(key, n_steps)
    # Both RBG samplers use the first key of a vmapped batch for all draws.
    if jr.key_impl(key) in ("rbg", "unsafe_rbg"):
        return jax.lax.map(sample, keys)
    return jax.vmap(sample)(keys)


def keyed_nodes(
    execution_order: Iterable[str], key: PRNGKeyArray | None
) -> Iterator[tuple[str, PRNGKeyArray]]:
    """Traverse in execution order, selecting each stream by occurrence ID."""
    names = tuple(execution_order)
    if not names:
        return iter(())
    words = jnp.asarray(
        [_scope_words(("occurrence", name), False) for name in names], dtype=jnp.uint32
    )
    keys = _derive_many(_fold_scope, key, words)
    return zip(names, keys, strict=True)


def reset_key(key: PRNGKeyArray) -> PRNGKeyArray:
    """Derive a trial reset stream independently of every transition."""
    return derive_key(key, "reset")
