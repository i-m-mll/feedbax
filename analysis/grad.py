"""
Compute derivatives of functions.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Optional, Self, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import jax_cookbook.tree as jtree
from jax.flatten_util import ravel_pytree
from jax_cookbook import is_module, is_none
from jax_cookbook.misc import construct_tuple_like
from jaxtyping import Array, PyTree

from feedbax_experiments.analysis.analysis import AbstractAnalysis, InputOf
from feedbax_experiments.analysis.func import CallerPorts, make_argwise_functional
from feedbax_experiments.types import AnalysisInputData

_Tuple = jtree.make_named_tuple_subclass("Tuple")


T = TypeVar("T")


def _compute_grads(
    grad_func: Callable,
    funcs: PyTree[Callable, "T"],  # type: ignore
    func_args: tuple[PyTree[Any, "T ..."], ...],  # type: ignore
    argnums: Optional[int | Sequence[int]],
) -> tuple:
    if argnums is None:
        argnums = tuple(range(len(func_args)))
    elif isinstance(argnums, int):
        argnums = (argnums,)

    grads_raw = jt.map(
        lambda func, *args: _Tuple(grad_func(func, *args, argnums=argnums)),
        funcs,
        *func_args,
        is_leaf=callable,
    )

    grads_by_argnum = jtree.unzip(grads_raw, tuple_cls=_Tuple)

    grads_expanded: list = [
        grads_by_argnum[argnums.index(i)] if i in argnums else None for i in range(len(func_args))
    ]

    return construct_tuple_like(type(func_args), grads_expanded)


#! TODO: `Jacobians` and `Hessians` seem like good candidates for
#! refactoring by a simpler `AbstractAnalysis` functional constructor
class Jacobians(AbstractAnalysis[CallerPorts]):
    Ports = CallerPorts
    inputs: CallerPorts = eqx.field(default_factory=CallerPorts, converter=CallerPorts.converter)

    variant: Optional[str] = "full"

    argnums: Optional[int | Sequence[int]] = None

    def compute(
        self,
        data: AnalysisInputData,
        *,
        funcs,
        func_args: tuple,
        **kwargs,
    ) -> tuple:
        return _compute_grads(
            lambda func, *args, argnums: jax.jacobian(func, argnums=argnums)(*args),
            funcs,
            func_args,
            self.argnums,
        )


class Hessians(AbstractAnalysis[CallerPorts]):
    Ports = CallerPorts
    inputs: CallerPorts = eqx.field(default_factory=CallerPorts, converter=CallerPorts.converter)

    variant: Optional[str] = "full"

    argnums: Optional[Sequence[int]] = None
    diag_only: bool = (
        True  # Whether to compute only the diagonal Hessians (i.e. no cross-input terms)
    )

    def compute(
        self,
        data: AnalysisInputData,
        *,
        funcs,
        func_args,
        **kwargs,
    ):
        def get_hessians(func, *args, argnums):
            if self.diag_only:
                return [jax.hessian(func, argnums=i)(*args) for i in argnums]
            else:
                return jax.hessian(func, argnums=argnums)(*args)

        return _compute_grads(get_hessians, funcs, func_args, self.argnums)


# =============================================================================
# New API, for use with `make_argwise_functional` and `ApplyFunctional`.
# =============================================================================


def matricize_block_in(block: Any, in_like: Any):
    """Reshape a dense J/H block with trailing input axes into 2D (out, in).
    Collapses all non-input axes into the leading dimension.
    """
    flat_in, _ = ravel_pytree(in_like)
    in_dim = flat_in.shape[0]
    return block.reshape((-1, in_dim))


# Scalarizers (compose at input side)


class IdentityScalarizer(eqx.Module):
    """No-op: assumes func is already scalar if a scalar Hessian is desired."""

    def __call__(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return f


class DotUScalarizer(eqx.Module):
    """phi_u(x) = <u, f(x)> where shapes of u and f(x) match (vdot)."""

    u: Any = eqx.field(kw_only=True)

    def __call__(self, f: Callable[..., Any]) -> Callable[..., Any]:
        u = self.u

        def phi(*a):
            y = f(*a)
            return jnp.vdot(u, y)

        return phi


class BatchDotUScalarizer(eqx.Module):
    """Batch of cotangents U with leading batch axis: (B, *y_shape).
    Produces phi_U(x) in R^B; `jax.hessian(phi_U, ...)` then stacks B Hessians.
    """

    U: Any = eqx.field(kw_only=True)

    def __call__(self, f: Callable[..., Any]) -> Callable[..., Any]:
        U = self.U

        def one(u, *a):
            return jnp.vdot(u, f(*a))

        def phi(*a):
            return jax.vmap(lambda u: one(u, *a))(U)

        return phi


# ABC for per-funcs


class ArgwisePer(eqx.Module):
    """Abstract per-arg computation with optional scalarization, flattening, reducer.

    Subclasses must implement:
      - per_fn(self, func, args, i) -> block or tuple of blocks
      - flatten(self, value, args, i) -> 2D block(s) (only used if reducers present)

    Fields:
      - scalarizer: Optional callable that takes func and returns a scalarized func.
      - reducer:
    """

    scalarizer: Optional[Callable[[Callable[..., Any]], Callable[..., Any]]] = None
    reducer: Optional[Callable[[Array], Any] | tuple[Callable[[Array], Any], ...]] = None

    def per_fn(self, func: Callable[..., Any], args: tuple, i: int):
        raise NotImplementedError

    def flatten(self, value: Any, args: tuple, i: int):
        return value

    def _reducer_for(self, args: tuple, i: int) -> Optional[Callable[[Any], Any]]:
        spec = self.reducer
        if spec is None:
            return None
        if callable(spec):
            return spec
        # tuple or namedtuple-like
        if isinstance(spec, tuple) and len(spec) == len(args):
            return spec[i]
        # dict mapping index -> reducer
        if isinstance(spec, dict):
            return spec.get(i, None)
        # namedtuple subclass fallback (has _fields)
        if hasattr(spec, "_fields") and hasattr(spec, "__getitem__"):
            try:
                return spec[i]
            except Exception as e:
                raise ValueError("namedtuple reducer spec must index by arg position") from e
        raise TypeError("Unsupported reducer spec; use callable, tuple/namedtuple, or dict.")

    def _apply_reducer(self, x: Any, reducer_fn: Callable[[Any], Any]):
        if isinstance(x, tuple):
            return tuple(self._apply_reducer(xi, reducer_fn) for xi in x)
        return reducer_fn(x)

    def __call__(self, func: Callable[..., Any], args: tuple, i: int):
        f = self.scalarizer(func) if self.scalarizer is not None else func
        value = self.per_fn(f, args, i)
        reducer_fn = self._reducer_for(args, i)
        if reducer_fn is not None:
            value = self.flatten(value, args, i)  # flatten once
            value = self._apply_reducer(value, reducer_fn)  # map recursively over tuples
        return value

    # immutable mutators
    def with_scalarizer(self, scalarizer) -> "ArgwisePer":
        return eqx.tree_at(lambda m: m.scalarizer, self, scalarizer)

    def with_reducer(self, reducer: Callable[[Any], Any]) -> "ArgwisePer":
        return eqx.tree_at(lambda m: m.reducer, self, reducer, is_leaf=is_none)


# Per-arg producers (subclasses)


class PerJacobianBlock(ArgwisePer):
    """J_i per arg. Raw: y ⊗ x_i; Flattened: (out × in_i)."""

    def per_fn(self, func, args: tuple, i: int):
        return jax.jacobian(func, argnums=i)(*args)

    def flatten(self, value, args: tuple, i: int):
        return matricize_block_in(value, args[i])


class PerHessianDiag(ArgwisePer):
    """H_{i,i}. Uses `jax.hessian` on (possibly scalarized) `func`.

    Raw (no reducers):
      * scalar-output or scalarized: x_i ⊗ x_i
      * vector-output without scalarization: y ⊗ x_i ⊗ x_i
    Flattened: (out × in_i), where `out` collapses leading non-input axes.
    """

    def per_fn(self, func, args: tuple, i: int):
        return jax.hessian(func, argnums=i)(*args)

    def flatten(self, value, args: tuple, i: int):
        return matricize_block_in(value, args[i])


class PerHessianRow(ArgwisePer):
    """Row (H_{i,j}) for j in `idxs`.

    Raw (no reducers): tuple of blocks; each block has shape
      * scalar-output or scalarized: x_i ⊗ x_j
      * vector-output without scalarization: y ⊗ x_i ⊗ x_j
    Flattened: tuple of 2D blocks with shape (out × in_j), collapsing non-input axes.
    """

    idxs: tuple[int, ...] = eqx.field(kw_only=True)

    def per_fn(self, func, args: tuple, i: int):
        g_i = jax.grad(func, argnums=i)
        return jax.jacobian(lambda *a: g_i(*a), argnums=self.idxs)(*args)

    def flatten(self, value, args: tuple, i: int):
        # value is a tuple aligned with self.idxs
        return tuple(matricize_block_in(value[k], args[j]) for k, j in enumerate(self.idxs))


#! TODO: Move to `misc` or something
# Reducers (operate on 2D or batched …×2D matrices)


def spectral_norm(A):
    s = jnp.linalg.svd(A, compute_uv=False)
    return s.max(axis=-1)  # works for 2D or batched


def frobenius_norm(A):
    return jnp.sqrt(jnp.sum(jnp.square(A), axis=(-2, -1)))


def trace_square(A):
    return jnp.trace(A, axis1=-2, axis2=-1)


# =============================================================================
# Usage Examples
# =============================================================================
# Assume a function with signature (x0, x1) -> y
# and example args `args_ex = (x0_ex, x1_ex)`.
# -----------------------------------------------------------------------------
# 1) Full Jacobian blocks (raw) per selected arg
# per = PerJacobianBlock()                             # no reducers → raw y⊗x_i
# jac_fn = make_argwise_functional(argnums=(0,1), per=per)
# result = jac_fn(func, args_ex)                       # (J_0_raw, J_1_raw)
# -----------------------------------------------------------------------------
# 2) Argwise Jacobian operator norms
# per = PerJacobianBlock().with_reducer(spectral_norm) # reducers → flattened → reduced
# jac_opnorm_fn = make_argwise_functional(argnums=(0,1), per=per)
# opnorms = jac_opnorm_fn(func, args_ex)               # (||J_0||_2, ||J_1||_2)
# -----------------------------------------------------------------------------
# 3) Full Hessian diagonal blocks (vector-output allowed, no scalarization)
# per = PerHessianDiag()                               # no reducers → raw blocks
# hdiag_fn = make_argwise_functional(argnums=(0,1), per=per)
# hdiag = hdiag_fn(func, args_ex)                      # (H_00_raw, H_11_raw)
#   where shapes are y⊗x_i⊗x_i if y is vector; x_i⊗x_i if y is scalar
# -----------------------------------------------------------------------------
# 4) Hessian diagonal operator norms along a direction u in output space
# per = PerHessianDiag().with_scalarizer(DotUScalarizer(u)).with_reducer(spectral_norm)
# hdiag_opnorm_fn = make_argwise_functional(argnums=(0,1), per=per)
# hdiag_opnorms = hdiag_opnorm_fn(func, args_ex)       # (||H_00^{(u)}||_2, ||H_11^{(u)}||_2)
# -----------------------------------------------------------------------------
# 5) Hessian row of mixed blocks (raw) for j in idxs
# per = PerHessianRow(idxs=(0,1))                      # no reducers → row of raw blocks
# hrow_fn = make_argwise_functional(argnums=(0,1), per=per)
# hrow = hrow_fn(func, args_ex)                        # each slot: (H_i0_raw, H_i1_raw)
# -----------------------------------------------------------------------------
# 6) Hessian row operator norms (vector-output, scalarized by u)
# per = PerHessianRow(idxs=(0,1)).with_scalarizer(DotUScalarizer(u)).with_reducer(spectral_norm)
# hrow_opnorm_fn = make_argwise_functional(argnums=(0,1), per=per)
# hrow_opnorms = hrow_opnorm_fn(func, args_ex)         # each slot: (||H_i0^{(u)}||, ||H_i1^{(u)}||)
# -----------------------------------------------------------------------------
# 7) Multiple directions U (batch). Reducers handle batched matrices.
# per = PerHessianDiag().with_scalarizer(BatchDotUScalarizer(U)).with_reducer(spectral_norm)
# hdiag_opnorms_batch_fn = make_argwise_functional(argnums=(0,1), per=per)
# # returns per-arg arrays of shape (B,) with op-norms per u in U.
# hdiag_opnorms_batch = hdiag_opnorms_batch_fn(func, args_ex)


# =============================================================================
# Sketch of linear operator based computation of operator norms, traces, etc.
# Will revisit if there are performance issues with storing full Jacobians/Hessians for a
# single leaf (e.g. for larger neural networks).
# =============================================================================


def _zeros_like_tree(x):
    return jt.map(jnp.zeros_like, x)


def _ones_like_tree(x):
    return jt.map(jnp.ones_like, x)


def _tree_dot(x, y):
    xf, _ = ravel_pytree(x)
    yf, _ = ravel_pytree(y)
    # use vdot to support complex; take real part
    return jnp.vdot(xf, yf).real


def _tree_norm(x):
    return jnp.sqrt(_tree_dot(x, x))


def _tree_normalize(x):
    n = _tree_norm(x)
    return jt.map(lambda a: a / n, x)


# Linear operators


def jacobian_linop_per_arg(
    func: Callable[..., Any],
    args: tuple,
    i: int,
    *,
    y_like: Optional[Any] = None,
):
    """Return (mv, mtv, like_v, like_u) for J_i."""
    if y_like is None:
        y_like = jt.map(jnp.zeros_like, func(*args))

    def mv(v_i):
        tangents = [_zeros_like_tree(a) for a in args]
        tangents[i] = v_i
        return jax.jvp(lambda *a: func(*a), tuple(args), tuple(tangents))[1]

    def mtv(u):
        _, pb = jax.vjp(lambda *a: func(*a), *args)
        return pb(u)[i]

    return mv, mtv, args[i], y_like


def jacobian_linop_selected(
    func: Callable[..., Any],
    args: tuple,
    idxs: tuple[int, ...],
    *,
    y_like: Optional[Any] = None,
):
    """Return (mv, mtv, like_v, like_u) for J restricted to selected arg indices."""
    if y_like is None:
        y_like = jt.map(jnp.zeros_like, func(*args))

    def mv(v_sel):  # v_sel is a tuple aligned with idxs
        tangents = [_zeros_like_tree(a) for a in args]
        for k, i in enumerate(idxs):
            tangents[i] = v_sel[k]
        return jax.jvp(lambda *a: func(*a), tuple(args), tuple(tangents))[1]

    def mtv(u):
        _, pb = jax.vjp(lambda *a: func(*a), *args)
        grads_all = pb(u)  # tuple over all args
        return tuple(grads_all[i] for i in idxs)

    like_v = tuple(args[i] for i in idxs)
    like_u = y_like
    return mv, mtv, like_v, like_u


def hessian_linop_per_arg(
    func: Callable[..., Any],
    args: tuple,
    i: int,
    *,
    u: Optional[Any] = None,
):
    """
    Return (mv, mtv, like) for a Hessian block with respect to args[i].

    - If `u is None`: `func(*args)` must be scalar; mv(v_i) = H_{i,i} v_i (symmetric).
    - If `u` is provided (vector output): we scalarize with phi_u(x) = <u, func(x)>,
      and mv(v_i) = ∇^2 phi_u(x) applied to v_i (still symmetric for fixed u).

    In both cases, mtv == mv and `like` has the shape of args[i].
    """
    if u is None:
        # scalar-output path
        g_i = jax.grad(lambda *a: func(*a), argnums=i)

        def mv(v_i):
            tangents = [_zeros_like_tree(a) for a in args]
            tangents[i] = v_i
            return jax.jvp(lambda *a: g_i(*a), tuple(args), tuple(tangents))[1]
    else:
        # vector-output path via scalarization phi_u(x) = <u, func(x)>
        def phi(*a):
            y = func(*a)
            # vdot keeps complex-safe behavior; take real part in reducers
            return jnp.vdot(u, y)

        g_i = jax.grad(phi, argnums=i)

        def mv(v_i):
            tangents = [_zeros_like_tree(a) for a in args]
            tangents[i] = v_i
            return jax.jvp(lambda *a: g_i(*a), tuple(args), tuple(tangents))[1]

    like = args[i]
    return mv, mv, like  # symmetric: mtv == mv


def hessian_linop_selected(
    func: Callable[..., Any],
    args: tuple,
    idxs: tuple[int, ...],
    *,
    u: Optional[Any] = None,
):
    """
    Return (mv, mtv, like) for a joint Hessian over the selected indices.

    - If `u is None`: `func(*args)` must be scalar; mv(v_sel) yields the tuple
      (H_{i,j} v_j) aggregated over j for each i in `idxs` (symmetric).
    - If `u` is provided` (vector output): we scalarize with phi_u(x) = <u, func(x)>,
      and mv(v_sel) = ∇^2 phi_u(x) applied to the block vector over selected args.
      Still symmetric for fixed u.

    In both cases, mtv == mv and `like` is a tuple with the shapes of args[i] for i in idxs.
    """
    if u is None:
        g_sel = jax.grad(lambda *a: func(*a), argnums=idxs)

        def mv(v_sel):
            tangents = [_zeros_like_tree(a) for a in args]
            for k, i in enumerate(idxs):
                tangents[i] = v_sel[k]
            return jax.jvp(lambda *a: g_sel(*a), tuple(args), tuple(tangents))[1]
    else:

        def phi(*a):
            y = func(*a)
            return jnp.vdot(u, y)

        g_sel = jax.grad(phi, argnums=idxs)

        def mv(v_sel):
            tangents = [_zeros_like_tree(a) for a in args]
            for k, i in enumerate(idxs):
                tangents[i] = v_sel[k]
            return jax.jvp(lambda *a: g_sel(*a), tuple(args), tuple(tangents))[1]

    like = tuple(args[i] for i in idxs)
    return mv, mv, like


def hessian_block_linop_ij(
    func: Callable[..., Any],
    args: tuple,
    idxs: tuple[int, ...],
    u: Any,
    i: int,
    j: int,
):
    """Linear operator for the (i,j) block of the scalarized Hessian ∇²⟨u,F(x)⟩.
    Domain is args[j], codomain is args[i]. Returns (mv, mtv, like_v, like_u).
    Uses symmetry: mtv(u_i) = H_{j,i} u_i.
    """
    pos = {idx: k for k, idx in enumerate(idxs)}

    def phi(*a):
        y = func(*a)
        return jnp.vdot(u, y)

    g_sel = jax.grad(phi, argnums=idxs)

    def mv(vj):
        tangents = [_zeros_like_tree(a) for a in args]
        tangents[j] = vj
        res = jax.jvp(lambda *a: g_sel(*a), tuple(args), tuple(tangents))[1]
        return res[pos[i]]

    def mtv(ui):
        tangents = [_zeros_like_tree(a) for a in args]
        tangents[i] = ui
        res = jax.jvp(lambda *a: g_sel(*a), tuple(args), tuple(tangents))[1]
        return res[pos[j]]

    like_v = args[j]
    like_u = args[i]
    return mv, mtv, like_v, like_u


# Reducers for operator norms, traces, etc.


def reducer_opnorm_power_bidiag(*, iters: int = 40, tol: float = 1e-6):
    """Return a reducer (mv, mtv, like_v, like_u) -> scalar ≈ ||A||_2.
    Works for J (rectangular) and also for symmetric H by passing mtv==mv and like_u==like_v.
    """

    def reduce(mv, mtv, like_v, like_u):
        v = _tree_normalize(_ones_like_tree(like_v))

        def one_step(v):
            Av = mv(v)
            sigma = jnp.sqrt(_tree_dot(Av, Av))
            u = jt.map(lambda a: a / sigma, Av)
            v_new = _tree_normalize(mtv(u))
            return v_new, sigma

        v, sigma = one_step(v)

        def cond(carry):
            v_prev, sigma_prev, k = carry
            v_next, sigma_new = one_step(v_prev)
            return jnp.logical_and(k < iters, jnp.abs(sigma_new - sigma_prev) > tol)

        def body(carry):
            v_prev, sigma_prev, k = carry
            v_next, sigma_new = one_step(v_prev)
            return (v_next, sigma_new, k + 1)

        _, sigma, _ = jax.lax.while_loop(cond, body, (v, sigma, 0))
        return sigma

    return reduce


def _rademacher_like(like, key):
    flat, unravel = ravel_pytree(like)
    r = jax.random.rademacher(key, shape=flat.shape, dtype=flat.dtype)
    return unravel(r)


def reducer_frobenius_hutchinson(*, samples: int = 32, key=jax.random.PRNGKey(0)):
    """Reducer (mv, like_v) -> scalar ≈ ||A||_F.
    For Jacobian A=J this uses only mv; for symmetric H it also works.
    """

    def reduce(mv, like_v):
        keys = jax.random.split(key, samples)

        def one(k):
            r = _rademacher_like(like_v, k)
            Ar = mv(r)
            return _tree_dot(Ar, Ar)

        vals = jax.vmap(one)(keys)
        return jnp.sqrt(vals.mean())

    return reduce


def reducer_trace_hutchinson(*, samples: int = 32, key=jax.random.PRNGKey(0)):
    """Reducer (mv, like) -> scalar ≈ tr(H). Assumes symmetry."""

    def reduce(mv, like):
        keys = jax.random.split(key, samples)

        def one(k):
            r = _rademacher_like(like, k)
            Hr = mv(r)
            return _tree_dot(r, Hr)

        vals = jax.vmap(one)(keys)
        return vals.mean()

    return reduce


# =============================================================================
# Example compositions
# =============================================================================


# -- Argwise operator norms, per-arg, ready to be passed as `per_fn` to `make_argwise_functional`
def jacobian_opnorm_per_fn(func: Callable[..., Any], args: tuple, i: int, *, iters: int = 40):
    # Acquire output template once; JVP will re-evaluate f during iterations anyway
    y_like = jt.map(jnp.zeros_like, func(*args))
    mv, mtv, like_v, like_u = jacobian_linop_per_arg(func, args, i, y_like=y_like)
    reducer = reducer_opnorm_power_bidiag(iters=iters)
    return reducer(mv, mtv, like_v, like_u)


def hessian_opnorm_per_fn(func: Callable[..., Any], args: tuple, i: int, *, iters: int = 40):
    mv, mtv, like = hessian_linop_per_arg(func, args, i)
    reducer = reducer_opnorm_power_bidiag(iters=iters)
    return reducer(mv, mtv, like, like)  # mtv==mv, like_u==like_v


# -- Joint block operator norms; return a single scalar aggregating over selected indices
def jacobian_opnorm_all_fn(
    func: Callable[..., Any], args: tuple, idxs: tuple[int, ...], *, iters: int = 40
):
    y_like = jt.map(jnp.zeros_like, func(*args))
    mv, mtv, like_v, like_u = jacobian_linop_selected(func, args, idxs, y_like=y_like)
    reducer = reducer_opnorm_power_bidiag(iters=iters)
    return reducer(mv, mtv, like_v, like_u)


def hessian_opnorm_all_fn(
    func: Callable[..., Any], args: tuple, idxs: tuple[int, ...], *, iters: int = 40
):
    mv, mtv, like = hessian_linop_selected(func, args, idxs)
    reducer = reducer_opnorm_power_bidiag(iters=iters)
    return reducer(mv, mtv, like, like)


# -- Block operator norms for a given u; returns a k×k tuple-of-tuples of norms for
def hessian_block_opnorm_matrix_u(
    func: Callable[..., Any],
    args: tuple,
    idxs: tuple[int, ...],
    u: Any,
    *,
    iters: int = 40,
):
    """Return a k×k tuple-of-tuples of block operator norms ||H_{i,j}^{(u)}||_2 for i,j in idxs."""
    reducer = reducer_opnorm_power_bidiag(iters=iters)
    rows = []
    for i in idxs:
        row = []
        for j in idxs:
            mv, mtv, like_v, like_u = hessian_block_linop_ij(func, args, idxs, u, i, j)
            row.append(reducer(mv, mtv, like_v, like_u))
        rows.append(tuple(row))
    return tuple(rows)


# # Can use the last in a couple of ways (I think):
# # Given: idxs = (0, 1) for (input, hidden)
# u = None  # some hidden direction to probe the curvature
# functional = lambda f, args: hessian_block_opnorm_matrix_u(f, args, idxs=(0, 1), u=u, iters=40)

# # Each selected slot i gets row i of the block op-norm matrix; others are None.
# # The only difference here is that the outer level of the result will have the same type as the
# # `args` passed to `functional`.
# functional = make_argwise_functional(
#     argnums=(0, 1),  # e.g., (input, hidden)
#     all_fn=lambda f, args, idxs: hessian_block_opnorm_matrix_u(f, args, idxs, u=u, iters=40),
# )
