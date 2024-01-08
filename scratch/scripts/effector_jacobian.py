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
import jax 
import jax.numpy as jnp
import jax.random as jr

from feedbax.mechanics.skeleton.arm import TwoLink, TwoLinkState

# %%
arm = TwoLink()
angle = jnp.array([0.1, 0.2])


# %% [markdown]
# Compare the exact effector Jacobian with the Jacobian computed by JAX from the forward kinematics of the arm

# %%
def effector_jac_exact(angle):
    J = jnp.array([
        [-arm.l[0] * jnp.sin(angle[0]) - arm.l[1] * jnp.sin(angle[0] + angle[1]), -arm.l[1] * jnp.sin(angle[0] + angle[1])],
        [arm.l[0] * jnp.cos(angle[0]) + arm.l[1] * jnp.cos(angle[0] + angle[1]), arm.l[1] * jnp.cos(angle[0] + angle[1])]
    ])
    return J


# %%
effector_jac_exact(angle)


# %%
# %timeit effector_jac_exact(angle)

# %%
def forward_kin_wrapped(angle):
    state = TwoLinkState(angle=angle, d_angle=jnp.zeros(2))
    return arm.forward_kinematics(state).pos 


# %%
# compilation time
# %time jax.jacfwd(forward_kin_wrapped)(angle)[-1]

# %%
jax.jacfwd(forward_kin_wrapped)(angle)[-1]

# %%
jnp.allclose(
    effector_jac_exact(angle),
    jax.jacfwd(forward_kin_wrapped)(angle)[-1],
)

# %%
# %timeit jax.jacfwd(forward_kin_wrapped)(angle)[-1]

# %% [markdown]
# So the JAX `jacfwd` is, aside from compilation time, about 50% slower than the exact Jacobian. 
#
# Note that `jacfwd` is computing something a bit larger. Since `forward_kin_wrapped` takes two joint angles and returns a Cartesian state for both joints (i.e. 2x2 matrix), we get 2x(2x2) derivatives.
#
# I'm not sure how to interpret the first row...

# %%
jax.jacfwd(forward_kin_wrapped)(angle)

# %%
angle.shape

# %%
forward_kin_wrapped(angle).shape


# %% [markdown]
# Separate out the relevant part of `arm.forward_kinematics` and repeat

# %%
def forward_kin(angle):
    angle_sum = jnp.cumsum(angle)  # links
    length_components = arm.l * jnp.array([jnp.cos(angle_sum),
                                           jnp.sin(angle_sum)])  # xy, links
    xy_position = jnp.cumsum(length_components, axis=1)  # xy, links
    return xy_position.T



# %%
jax.jacfwd(forward_kin)(angle)[-1]

# %%
# %timeit effector_jac_exact(angle)
# %timeit jax.jacfwd(forward_kin)(angle)[-1]

# %% [markdown]
# Now it's only slightly worse.

# %% [markdown]
# Verify vmap behaviour

# %%
thetas = jnp.abs(jr.normal(jr.PRNGKey(0), shape=(5, 2)))

# %%
jax.vmap(effector_jac_exact)(thetas)

# %%
effector_jac_jax = lambda x: jax.jacfwd(forward_kin_wrapped)(x)[-1]
jax.vmap(effector_jac_jax)(thetas)

# %%
effector_jac_jax2 = lambda x: jax.jacfwd(forward_kin)(x)[-1]

# %%
# %timeit jax.vmap(effector_jac_exact)(thetas)
# %timeit jax.vmap(effector_jac_jax)(thetas)
# %timeit jax.vmap(effector_jac_jax2)(thetas)

# %% [markdown]
# Interestingly, the JAX version is faster in this case. I guess this is because the vmap compilation works better when we're not manually constructing the Jacobian matrix element-by-element.

# %% [markdown]
# I added the appropriate methods to `TwoLink`, so let's try them out

# %%
arm._effector_jac(angle)

# %% [markdown]
#
