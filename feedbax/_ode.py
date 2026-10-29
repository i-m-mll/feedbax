"""Numerical boundary between stored physical state and Diffrax integration.

Compile at this boundary so changing state and inputs remain dynamic arguments
inside Diffrax's nested traces, including when JAX's persistent cache is enabled.
"""

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PyTree


def restore_storage_dtypes(y: PyTree[Array], storage: PyTree[Array]) -> PyTree[Array]:
    """Apply a physical storage contract after state preprocessing or integration."""
    return jt.map(lambda value, previous: value.astype(previous.dtype), y, storage)


def _working_state(y: PyTree[Array], dt, storage):
    dtype = jnp.result_type(*(leaf.dtype for leaf in jt.leaves(storage)))
    return jt.map(lambda leaf: leaf.astype(dtype), y), jnp.asarray(dt, dtype=dtype)


@eqx.filter_jit
def ode_init(solver: dfx.AbstractSolver, term: dfx.ODETerm, dt, y: PyTree[Array], args):
    """Initialize native solver auxiliary state using the integration representation."""
    working_y, t1 = _working_state(y, dt, y)
    return solver.init(term, jnp.zeros_like(t1), t1, working_y, args)


@eqx.filter_jit
def ode_step(
    solver: dfx.AbstractSolver,
    term: dfx.ODETerm,
    dt,
    y: PyTree[Array],
    args,
    solver_state,
    *,
    storage_like=None,
):
    """Integrate a sampled interval and restore each physical leaf's storage dtype.

    Diffrax Runge–Kutta tableaux promote across the whole state tree, so all
    numerical leaves must share a working dtype. Auxiliary solver state retains
    its native representation. ``storage_like`` supplies the stored state before
    any kinematics or force injection. Each call may change the input or physical state,
    invalidating any derivative cached by a stateful solver.
    """
    storage = y if storage_like is None else storage_like
    working_y, t1 = _working_state(y, dt, storage)
    next_y, _, _, next_solver_state, _ = solver.step(
        term,
        jnp.zeros_like(t1),
        t1,
        working_y,
        args,
        solver_state,
        made_jump=True,
    )
    stored_y = restore_storage_dtypes(next_y, storage)
    return stored_y, next_solver_state
