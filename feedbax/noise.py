"""

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
from collections.abc import Callable
from functools import reduce
import logging
from typing import Any, Optional

import equinox as eqx
from equinox import AbstractVar
from equinox._pretty_print import tree_pp, bracketed
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax._src.pretty_printer as pp
from jaxtyping import Array, PRNGKeyArray, Shaped

from feedbax.misc import _simple_module_pprint


logger = logging.getLogger(__name__)


class AbstractNoise(eqx.Module):

    @abstractmethod
    def __call__(
        self, key: PRNGKeyArray, x: Shaped[Array, "*dims"]
    ) -> Shaped[Array, "*dims"]:
        ...

    def __add__(self, other):
        return CompositeNoise(terms=(self, other))


class ZeroNoise(AbstractNoise):
    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        return jnp.zeros_like(x)


class CompositeNoise(AbstractNoise):
    terms: tuple[AbstractNoise, ...]

    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        keys = jr.split(key, len(self.terms))
        return reduce(jnp.add, [
            term(key, x)
            for term, key in zip(self.terms, keys)
        ])

    def __getitem__(self, idx):
        return self.terms[idx]

    # def __tree_pp__(self, **kwargs):
    #     _term_sep = pp.concat([pp.brk(), pp.text("+ ")])
    #     return bracketed(
    #         None,
    #         kwargs['indent'],
    #         [pp.join(_term_sep, [tree_pp(term, **kwargs) for term in self.terms])],
    #         '(',
    #         ')',
    #     )


class Normal(AbstractNoise):
    std: float = 1.0
    mean: float = 0.0

    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        return self.std * jr.normal(key, x.shape, x.dtype) + self.mean


class HalfNormal2Vector(AbstractNoise):
    """2-vectors with half-normally-distributed lengths, and uniform directions.

    NOTE: This only makes sense when `x.shape[-1] == 2`
    """
    std: float = 1.0

    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        key_lengths, key_angles = jr.split(key)
        lengths = self.std * jr.normal(key_lengths, x.shape[:-1], x.dtype)
        angles = jr.uniform(key_angles, x.shape[:-1], minval=0, maxval=2*jnp.pi)
        return lengths * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)


class Multiplicative(AbstractNoise):
    """Scales the output of another noise term by the magnitude of the input signal.

    Arguments:
        noise_func: The noise function to multiplicatively scale.
        scale_func: Applied to the input signal to produce the scaling factor. For example,
            if the input is a vector, we may want to scale the noise sample by the vector
            length, in which case we could pass `lambda x: jnp.linalg.norm(x, axis=-1)`.
    """
    noise_func: AbstractNoise | Callable[[PRNGKeyArray, Array], Array]
    scale_func: Callable[[Array], Array] = lambda x: x

    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        return self.scale_func(x) * self.noise_func(key, x)

    def __tree_pp__(self, **kwargs):
        return _simple_module_pprint("Multiplicative", self.noise_func, **kwargs)


def replace_noise(tree, replace_fn: Callable = lambda _: None):
    """Replaces all `AbstractNoise` leaves of a PyTree, by default with `None`."""
    # This is kind of annoying, but use `tree_map` instead of (say) a list comprehension
    get_noise_terms = lambda tree: jt.leaves(jt.map(
        lambda x: x if isinstance(x, AbstractNoise) else None,
        tree,
        is_leaf=lambda x: isinstance(x, AbstractNoise),
    ), is_leaf=lambda x: isinstance(x, AbstractNoise))

    return eqx.tree_at(
        get_noise_terms,
        tree,
        replace_fn=replace_fn,
    )