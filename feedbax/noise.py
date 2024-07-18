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
    broadcast: bool = False

    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        shape = (1,) if self.broadcast else x.shape
        return self.std * jr.normal(key, shape, x.dtype) + self.mean



class Multiplicative(AbstractNoise):
    noise_func: Callable[[PRNGKeyArray, Array], Array]

    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        return x * self.noise_func(key, x)

    # def __getattr__(self, name):
    #     return getattr(self.noise_func, name)

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