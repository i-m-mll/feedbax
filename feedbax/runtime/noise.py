"""

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
from collections.abc import Callable
from functools import reduce

import equinox as eqx
# from equinox._pretty_print import tree_pp, bracketed
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Shaped

class AbstractNoise(eqx.Module):

    @abstractmethod
    def __call__(
        self, key: PRNGKeyArray, x: Shaped[Array, "*dims"]
    ) -> Shaped[Array, "*dims"]:
        ...

    def __add__(self, other):
        return CompositeNoise(terms=(self, other))


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

    # def __tree_pp__(self, **kwargs):
    #     return _simple_module_pprint("Multiplicative", self.noise_func, **kwargs)
