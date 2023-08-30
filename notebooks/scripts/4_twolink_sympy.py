# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
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
import sympy 

# %% [markdown]
# Differentiation of angular-to-Cartesian transformation for n-link arm

# %%
N_LINKS = 3

t = sympy.symbols('t')
ls = sympy.symbols(f'l:{N_LINKS}')
ths = [sympy.Function(f'theta_{i}') for i in range(N_LINKS)]

xs = [0] * N_LINKS
ys = [0] * N_LINKS
for i in range(1, N_LINKS + 1):
    xs[i - 1] = xs[i - 2] + ls[i - 1] * sympy.cos(sum([ths[j](t) for j in range(i)]))
    ys[i - 1] = xs[i - 2] + ls[i - 1] * sympy.sin(sum([ths[j](t) for j in range(i)]))

# %%
sympy.diff(xs[0], t)

# %%
sympy.diff(xs[0], t, 2)

# %%
sympy.diff(xs[1], t)

# %%
sympy.diff(xs[1], t, 2)

# %%
