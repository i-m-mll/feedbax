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
# %load_ext autoreload
# %autoreload 2

# %%
import feedbax.loss as fbl

# %%
composite = fbl.CompositeLoss(
    terms=[
        fbl.EffectorFinalVelocityLoss(),
        fbl.EffectorFixationLoss(),
    ],
    weights=[3.0, 2.0],
)

composite

# %%
simple = fbl.EffectorPositionLoss()

# %%
complex = fbl.CompositeLoss(
    terms=[
        composite,
        simple,
    ],
    weights=[4.0, 1.0],
)

complex

# %%
simple_weighted = 7.4 * simple
simple_weighted

# %%
simple_weighted + complex

# %%
simple_weighted - complex

# %%
complex | simple_weighted

# %%
complex = fbl.CompositeLoss(
    terms=dict(
        c=composite,
        d=composite,
        simple=simple,
    ),
    weights=dict(
        c=4.0, 
        d=5.0,
        simple=1.0
    ),
)

# %%
complex

# %%
loss_func = fbl.EffectorFinalVelocityLoss() + 1e-4 * fbl.EffectorFixationLoss()

loss_func
