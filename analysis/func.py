from ast import TypeVar
from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Optional, ParamSpec, TypeAlias

import equinox as eqx
import jax.tree as jt
import jax_cookbook.tree as jtree
from jax_cookbook import map_rich
from jax_cookbook.misc import construct_tuple_like
from jaxtyping import PyTree

from feedbax_experiments.analysis.analysis import (
    AbstractAnalysis,
    AbstractAnalysisPorts,
    InputOf,
    SinglePort,
)
from feedbax_experiments.types import AnalysisInputData


class ApplyFns(AbstractAnalysis[SinglePort[PyTree[Any]]]):
    """Apply a PyTree of callables to a PyTree of data.

    The functions are stored in the instance field `funcs` and are applied to
    the leaves of the `data` PyTree (customizable via `is_data_leaf`).

    - By default, callable/equinox-`Measure` leaves are treated as function
      leaves; `Responses` or arrays are treated as data leaves.
    """

    Ports = SinglePort[PyTree[Any]]
    inputs: SinglePort[PyTree] = eqx.field(
        default_factory=SinglePort[PyTree], converter=SinglePort[PyTree].converter
    )

    fns: PyTree[Callable] = eqx.field(kw_only=True)  # required at runtime
    is_leaf: Optional[Callable[[Any], bool]] = None

    def _apply_fn(self, func, subdata):
        return jt.map(lambda leaf: func(leaf), subdata, is_leaf=self.is_leaf)

    def compute(self, data: AnalysisInputData, *, input: PyTree, **kwargs):
        result = jt.map(
            lambda func: jt.map(
                lambda x: func(x),
                input,
                is_leaf=self.is_leaf,
            ),
            self.fns,
            is_leaf=callable,
        )
        return result


class CallerPorts(AbstractAnalysisPorts):
    """Input ports for analyses which call other functions with positional arguments."""

    fns: InputOf[Callable]
    fn_args: tuple[InputOf[Any], ...]


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

# A functional takes (func, *args) and returns T.
Functional = Callable[Concatenate[Callable[P, R], P], T]


class ApplyFunctional(AbstractAnalysis[CallerPorts]):
    """Apply a user-supplied functional(func, *args) per (func, *args) leaf."""

    Ports = CallerPorts
    inputs: CallerPorts = eqx.field(default_factory=CallerPorts, converter=CallerPorts.converter)
    # Keep the callable static so we don't try to treat it as an array.
    functional: Functional = eqx.field(kw_only=True)
    # Optionally JIT each per-leaf application
    jit_per_leaf: bool = True
    is_leaf: Optional[Callable[[Any], bool]] = callable

    def compute(self, data, *, fns, fn_args, **kwargs):
        tuple_type = type(fn_args)

        def per_leaf(func, *args):
            # The functional itself should only use JAX ops; then this whole thing is JIT-able.
            args_tuple = construct_tuple_like(tuple_type, args)
            return self.functional(func, args_tuple)

        if self.jit_per_leaf:
            per_leaf = eqx.filter_jit(per_leaf)

        # Apply per leaf: funcs and each item of fn_args must be matching PyTrees
        return map_rich(per_leaf, fns, *fn_args, is_leaf=self.is_leaf, description="")


def _canon_argnums(argnums: Optional[int | Sequence[int]], nargs: int) -> tuple[int, ...]:
    if argnums is None:
        return tuple(range(nargs))
    if isinstance(argnums, int):
        argnums = (argnums,)
    out: list[int] = []
    for a in argnums:
        i = a % nargs
        if i not in out:
            out.append(i)
    return tuple(out)


ArgwisePer: TypeAlias = Callable[[Callable[..., Any], tuple, int], Any]


def make_argwise_functional(
    *,
    argnums: Optional[int | Sequence[int]] = None,
    per: ArgwisePer,
):
    """Return functional(func, args) -> container like args with per-arg results.
    The per-object may include scalarization and reducers; non-selected slots are None.
    """

    def functional(func, args):
        nargs = len(args)
        sel = _canon_argnums(argnums, nargs)
        if not sel:
            return construct_tuple_like(type(args), [None] * nargs)
        picked = tuple(per(func, args, i) for i in sel)
        out = [None] * nargs
        for k, i in enumerate(sel):
            out[i] = picked[k]
        return construct_tuple_like(type(args), out)

    return functional
