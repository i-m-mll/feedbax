# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %%
import math
from types import SimpleNamespace
from typing import Optional 

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax 
from tqdm import tqdm


# %% [markdown]
# Adapted from [this example](https://docs.kidger.site/diffrax/examples/kalman_filter/)

# %%
class LTISystem(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray


# %%
def interpolate_us(ts, us, B):
    if us is None:
        m = B.shape[-1]
        u_t = SimpleNamespace(evaluate=lambda t: jnp.zeros((m,)))
    else:
        u_t = dfx.LinearInterpolation(ts=ts, ys=us)
    return u_t


# %%
def diffeqsolve(
    rhs,
    ts: jnp.ndarray,
    y0: jnp.ndarray,
    solver: dfx.AbstractSolver = dfx.Dopri5(),
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
    dt0: float = 0.01,
) -> jnp.ndarray:
    return dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        solver=solver,
        stepsize_controller=stepsize_controller,
        t0=ts[0],
        t1=ts[-1],
        y0=y0,
        dt0=dt0,
        saveat=dfx.SaveAt(ts=ts),
    ).ys


# %%
def simulate_lti_system(
    sys: LTISystem,
    y0: jnp.ndarray,
    ts: jnp.ndarray,
    us: Optional[jnp.ndarray] = None,
    std_measurement_noise: float = 0.0,
    key=jr.PRNGKey(
        1,
    ),
):
    u_t = interpolate_us(ts, us, sys.B)

    def rhs(t, y, args):
        return sys.A @ y + sys.B @ u_t.evaluate(t)

    xs = diffeqsolve(rhs, ts, y0)
    # noisy measurements
    ys = xs @ sys.C.transpose()
    ys = ys + jr.normal(key, shape=ys.shape) * std_measurement_noise
    return xs, ys


# %% [markdown]
# Not sure how to do this exactly. I think the Riccati equation needs to be solved until convergence. I'm not sure what this corresponds to in terms of time endpoints.
# Maybe check out [this](https://docs.kidger.site/diffrax/examples/steady_state/) on solving to reach a steady state.
#
# OTOH the diffrax Kalman filter example uses a single step of the Riccati equation to update the error covariance... See eq 3.22 [here](https://lewisgroup.uta.edu/ee5322/lectures/CTKalmanFilterNew.pdf)
#
# We can use the solution for P to calculate the control gain K, which gives the optimal controls.

# %%
class Riccati(eqx.Module):
    
    sys: LTISystem
    P: jnp.ndarray
    R: jnp.ndarray
    Q: jnp.ndarray
    
    def __call__(self, t, y, args):
        A, B, C = self.sys.A, self.sys.B, self.sys.C
        
        self.P = y
                
        dPdt = (
            -A.transpose() @ self.P - self.P @ A + self.P @ B @ self.invR @ B.transpose() @ self.P - self.Q
        )
        
        return dPdt
        
    
    @property
    def invR(self):
        return jnp.linalg.inv(self.R)


# %%
def loss(model):
    term = dfx.ODETerm(model)
    controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
    event = dfx.SteadyStateEvent()
    adjoint = dfx.ImplicitAdjoint()
    
    sol = dfx.diffeqsolve(
        term,
        solver=dfx.Tsit5(),
        t0=0,
        t1=jnp.inf,
        dt0=0.1,
        y0=1.0,
        max_steps=None,
        #stepsize_controller=controller,
        discrete_terminating_event=event,
        adjoint=adjoint,
    )
    
    return sol


# %% [markdown]
# Solve the Riccati recursion for a simple point mass system.
#
# TODO: probably need to partition `Riccati` to avoid optimizing Q, R, and sys? see the kalman filter [diffrax example](https://docs.kidger.site/diffrax/examples/kalman_filter/)

# %%
def exp_taylor(x: float, n: int):
    """First `n` terms of the Taylor series for `exp` at the origin.
    """
    return [(x ** i) / math.factorial(i) for i in range(n)]


def point_mass(order: int = 2, n_dim: int = 2, dt: float = 0.1) -> LTISystem:
    A = sum([jnp.diagflat(jnp.array([t] * (order - i) * n_dim), i * n_dim)
             for i, t in enumerate(exp_taylor(dt, order))])
    B_terms = exp_taylor(dt, 3)[::-1] + [0] * (order - 3)
    B = jnp.concatenate([term * jnp.eye(n_dim) for term in B_terms[:order]], 
                        axis=0)
    C = jnp.array(1)
    return LTISystem(A, B, C)


# %%
sys = point_mass(order=2, n_dim=2, dt=0.1)
model = Riccati(sys=sys, P=jnp.eye(4), R=jnp.eye(2), Q=jnp.eye(4))

optim = optax.sgd(1e-2, momentum=0.7, nesterov=True)
opt_state = optim.init(model)

@eqx.filter_jit 
def make_step(model, opt_state):
    grads = eqx.filter_grad(loss)(model)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state 

for step in range(100):
    model, opt_state = make_step(model, opt_state)
    print(f"Step: {step} P: {model.P}")


# %%
