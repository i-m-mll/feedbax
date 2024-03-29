"""

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import TYPE_CHECKING

import jax
import jax.tree_util as jtu
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import polars as pl

if TYPE_CHECKING:
    from feedbax.train import TaskTrainerHistory


logger = logging.getLogger(__name__)


def color_add_alpha(rgb_str: str, alpha: float):
    return f"rgba{rgb_str[3:-1]}, {alpha})"


def plot_loss_mean_history(
    train_history: "TaskTrainerHistory", 
    colors: list[str] = px.colors.qualitative.Set1,
    error_bars_alpha: float = 0.3,
    n_std_plot: int = 2,
):
    fig = go.Figure()
    
    losses = jax.tree_map(
        lambda x: np.array(x),
        train_history.loss,
    )
    
    timesteps = pl.DataFrame({"timestep": range(train_history.loss.total.shape[0])})
    
    dfs = jax.tree_map(
        lambda losses: timesteps.hstack(pl.DataFrame(losses)),
        dict(losses) | {"Total": losses.total},
    )
    
    loss_statistics = jax.tree_map(
        lambda df: df.select(
            timestep=pl.col("timestep"),
            mean=pl.concat_list(pl.col('*').exclude("timestep")).list.mean(),
            std=pl.concat_list(pl.col('*').exclude("timestep")).list.std(),
        ),
        dfs,
    )

    error_bars_bounds = jax.tree_map(
        lambda df: pl.DataFrame(dict(
            timestep=df['timestep'],
            lb=df['mean'] - n_std_plot * df['std'],
            ub=df['mean'] + n_std_plot * df['std'],
        )),
        loss_statistics,
    )
    
    colors_dict = {
        label: 'rgb(0,0,0)' if label == 'Total' else color
        for label, color in zip(dfs, colors)
    }

    for i, label in enumerate(dfs):
        # Mean
        fig.add_trace(go.Scatter(
            name=label,
            x=loss_statistics[label]["timestep"],
            y=loss_statistics[label]['mean'],
            line=dict(color=colors_dict[label])
        ))
        
        # Error bars
        fig.add_trace(go.Scatter(
            name="Upper bound",
            x=error_bars_bounds[label]["timestep"],
            y=error_bars_bounds[label]["ub"],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
        ))
        
        fig.add_trace(go.Scatter(
            name="Lower bound",
            x=error_bars_bounds[label]["timestep"],
            y=error_bars_bounds[label]["lb"],
            line=dict(color='rgba(255,255,255,0)'),
            fill="tonexty",
            fillcolor=color_add_alpha(colors_dict[label], error_bars_alpha),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    
    return fig