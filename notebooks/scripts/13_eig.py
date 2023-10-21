# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %%
import jax.numpy as jnp
import jax.random as jr    
import matplotlib.pyplot as plt 

# %% [markdown]
# Generate some overlapping Gaussian data

# %%
samples = 1000
features = 10

seed = 0
key = jr.PRNGKey(seed)

key1, key2, key3 = jr.split(key, 3)

means = jnp.arange(features)
stds = jr.uniform(key1, (features,)) * 2

x = jr.normal(key2, (samples, features)) * stds + means

# %%
ax = plt.subplot(111)

for i in range(features):
    ax.hist(x[:,i], bins=25, alpha=0.5)


# %% [markdown]
# These shouldn't have much off-diagonal covariance

# %%
def cov(x):
    # mean subtract
    X = x - jnp.mean(x, axis=0)
    C = (X.T @ X) / (x.shape[0] - 1)
    return C

C = cov(x)

# %%
plt.imshow(C)

# %% [markdown]
# Add some covariance by randomly rolling the columns and adding them back into the data with some random scaling factor

# %%
n_mix = 4
mix_scale = 0.5
max_shift = 4  # features - 1

key = jr.PRNGKey(seed + 5)
keys = jr.split(key, n_mix)

x_mixed = jnp.copy(x)

for i in range(n_mix):
    key1, key2 = jr.split(keys[i])
    scale = mix_scale * jr.uniform(key1, (1,)).item() 
    x_rolled = jnp.roll(
        x, 
        jr.randint(key2, (1,), 0, max_shift).item(), 
        axis=1
    )
    x_mixed = x_mixed + scale * x_rolled

# %%
plt.imshow(cov(x_mixed))


# %% [markdown]
# Compare different methods for eigendecomposition

# %%
def eig_svd(x):
    x = x - jnp.mean(x, axis=0)
    U, S, Vt = jnp.linalg.svd(x, full_matrices=False)
    L = S ** 2 / (x.shape[0] - 1)
    return L, Vt , U


# %%
L_svd, Vt_svd, U_svd = eig_svd(x_mixed)

# %%
L_jax, V_jax = jnp.linalg.eigh(cov(x_mixed))

# %%
L_jax / L_svd[::-1]

# %% [markdown]
# We also have to reverse the columns of the eigenvector matrix to get them to match

# %%
Vt_svd.T[:,::-1] / V_jax

# %%
