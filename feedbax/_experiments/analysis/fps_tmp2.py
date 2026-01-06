"""Temporary module for reach-specific fixed-point analyses.

This is a refactoring of the logic in `notebooks/markdown/part2__fps_reach.md`
into the declarative analysis framework.
"""

from collections.abc import Sequence
from typing import Any, Optional

import equinox as eqx
import feedbax.plot as fbp
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import plotly.graph_objects as go
from jax_cookbook import is_module
from jax_cookbook.tree import first
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax._experiments.analysis.analysis import (
    AbstractAnalysis,
    AbstractAnalysisPorts,
    Data,
    InputOf,
)
from feedbax._experiments.analysis.fps import get_simple_reach_first_fps
from feedbax._experiments.analysis.pca import StatesPCA
from feedbax._experiments.analysis.state_utils import exclude_bad_replicates
from feedbax._experiments.tree_utils import first
from feedbax._experiments.types import AnalysisInputData, LDict, TreeNamespace

# ########################################################################## #
# Helper functions from the notebook
# ########################################################################## #


def get_endpoint_positions(task):
    trials = task.validation_trials
    return (
        trials.inits[lambda s: s.mechanics.effector].pos,
        trials.targets[lambda s: s.mechanics.effector.pos].value[:, -1],
    )


def get_initial_network_inputs(task):
    trials = task.validation_trials
    return jnp.concatenate(
        [
            trials.inputs["sisu"][..., None],
            trials.inputs["effector_target"].pos,
            trials.inputs["effector_target"].vel,
        ],
        axis=-1,
    )[:, 0, :]  # Index the first time step


def readout_vector_traces(
    readout_weights_pc: Float[Array, "out=2 pcs=3"],
    vector_start_pc: Optional[Float[Array, "out=2 pcs=3"]] = None,
    colors: tuple = ("#FF0000", "#0000FF"),
    scale: float = 0.25,
):
    """Create traces for readout weight vectors in PC space."""
    traces = []

    if vector_start_pc is None:
        vector_start_pc = np.zeros_like(readout_weights_pc)

    for j, readout_label in enumerate(("x", "y")):
        start = vector_start_pc[j]
        end = vector_start_pc[j] + scale * readout_weights_pc[j]

        # Interleave start and end points with None for multiple disconnected lines
        x = np.column_stack((start[..., 0], end[..., 0], np.full_like(start[..., 0], None))).ravel()
        y = np.column_stack((start[..., 1], end[..., 1], np.full_like(start[..., 1], None))).ravel()
        z = np.column_stack((start[..., 2], end[..., 2], np.full_like(start[..., 2], None))).ravel()

        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(width=10),
                showlegend=True,
                name=readout_label,
                line_color=colors[j],
                legend="legend2",
            )
        )

    return traces


def plot_hidden_and_fp_trajectories_3D(
    fp_trajs_pc: Float[Array, "curves index pcs=3"],
    hidden_pc: Float[Array, "curves index pcs=3"],
    colors: Sequence,  # len curves
    fp_alpha: float = 0.4,
    stride_plot: int = 1,
    marker_size: float = 2,
    axis_labels: tuple[str, str, str] = ("PC1", "PC2", "PC3"),
):
    """Plot hidden state trajectories and fixed points in 3D PC space."""
    fig = go.Figure(
        layout=dict(
            width=800,
            height=600,
            margin=dict(l=10, r=10, t=0, b=10),
            legend=dict(
                yanchor="top",
                y=0.9,
                xanchor="right",
            ),
            scene_aspectmode="data",
        ),
    )
    fig = fbp.trajectories_3D(
        fp_trajs_pc[::stride_plot],
        colors=colors[::stride_plot],
        mode="markers",
        marker_size=marker_size,
        marker_opacity=fp_alpha,
        endpoint_symbol="square-open",
        name="Local FP",
        axis_labels=axis_labels,
        fig=fig,
    )
    fig = fbp.trajectories_3D(
        hidden_pc[::stride_plot],
        colors=colors[::stride_plot],
        mode="lines",
        line_width=2,
        endpoint_symbol="diamond-open",
        name="Reach trajectory",
        axis_labels=axis_labels,
        fig=fig,
    )

    return fig


# ########################################################################## #
# Analysis classes
# ########################################################################## #


class ReachFPsInPCSpacePorts(AbstractAnalysisPorts):
    """Input ports for ReachFPsInPCSpace analysis."""

    fps_results: InputOf[ReachFPs]
    pca_results: InputOf[StatesPCA]


class ReachFPsInPCSpace(AbstractAnalysis[ReachFPsInPCSpacePorts]):
    """Plot reach fixed points in PC space."""

    Ports = ReachFPsInPCSpacePorts
    inputs: ReachFPsInPCSpacePorts = eqx.field(
        default_factory=ReachFPsInPCSpacePorts, converter=ReachFPsInPCSpacePorts.converter
    )
    variant: Optional[str] = "full"

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        result: PyTree,
        fps_results: TreeNamespace,
        pca_results: TreeNamespace,
        colors: PyTree,
        replicate_info: PyTree,
        hps_common: TreeNamespace,
        **kwargs,
    ):
        # Get the best replicate for visualization
        from feedbax._experiments.analysis.state_utils import get_best_replicate

        fps = jt.map(
            lambda fps_dict: jtree.stack(list(fps_dict.values())),
            fps_results.fps,
            is_leaf=LDict.is_of("sisu"),
        )

        hidden = jt.map(
            lambda states_dict: jtree.stack(list(states_dict.values())),
            jt.map(
                lambda states: jt.map(lambda s: s.net.hidden, states, is_leaf=is_module),
                fps_results.states,
                is_leaf=LDict.is_of("sisu"),
            ),
            is_leaf=LDict.is_of("sisu"),
        )

        # Get best replicate
        fps_best = get_best_replicate(fps, replicate_info=replicate_info, axis=0, keep_axis=True)
        hidden_best = get_best_replicate(
            hidden, replicate_info=replicate_info, axis=0, keep_axis=True
        )

        # Transform to PC space
        fps_pc = pca_results.batch_transform(fps_best)
        hidden_pc = pca_results.batch_transform(hidden_best)

        # Get readout weights in PC space
        all_readout_weights = exclude_bad_replicates(
            jt.map(
                lambda model: model.step.net.readout.weight,
                jt.map(first, data.models, is_leaf=LDict.is_of("sisu")),
                is_leaf=is_module,
            ),
            replicate_info=replicate_info,
            axis=0,
        )
        readout_weights_pc = pca_results.batch_transform(all_readout_weights)

        def plot_goals_vs_inits_fps(fps_pc, readout_weights_pc, context_idx=0):
            """Plot goals-goals vs inits-goals FPs across context inputs."""
            pcs_plot = slice(0, 3)
            fp_alpha = 0.4
            stride_plot = 8

            fig = go.Figure(
                layout=dict(
                    width=800,
                    height=600,
                    margin=dict(l=10, r=10, t=0, b=10),
                    legend=dict(
                        yanchor="top",
                        y=0.9,
                        xanchor="right",
                    ),
                ),
            )

            # Goals-goals FPs
            fig = fbp.trajectories_3D(
                fps_pc["goals-goals"][..., pcs_plot],
                colors=list(colors["sisu"].dark.values()),
                mode="markers",
                marker_size=4,
                endpoint_symbol=None,
                name="Goals-goals FPs",
                axis_labels=("PC1", "PC2", "PC3"),
                fig=fig,
                marker_symbol="circle-open",
            )

            # Inits-goals FPs
            fig = fbp.trajectories_3D(
                fps_pc["inits-goals"][..., pcs_plot],
                colors=list(colors["sisu"].dark.values()),
                mode="markers",
                marker_size=4,
                endpoint_symbol=None,
                name="Inits-goals FPs",
                axis_labels=("PC1", "PC2", "PC3"),
                fig=fig,
                marker_symbol="circle",
            )

            # Add readout vectors
            fig.add_traces(
                readout_vector_traces(
                    readout_weights_pc[..., pcs_plot],
                    jnp.tile(hidden_pc[context_idx, 0, 0, pcs_plot], (2, 1)),
                ),
            )

            return fig

        return jt.map(
            plot_goals_vs_inits_fps,
            fps_pc,
            readout_weights_pc,
            is_leaf=LDict.is_of("sisu"),
        )


class ReachTrajectoriesInPCSpacePorts(AbstractAnalysisPorts):
    """Input ports for ReachTrajectoriesInPCSpace analysis."""

    fps_results: InputOf[ReachFPs]
    pca_results: InputOf[StatesPCA]


class ReachTrajectoriesInPCSpace(AbstractAnalysis[ReachTrajectoriesInPCSpacePorts]):
    """Plot reach trajectories and fixed points in PC space."""

    Ports = ReachTrajectoriesInPCSpacePorts
    inputs: ReachTrajectoriesInPCSpacePorts = eqx.field(
        default_factory=ReachTrajectoriesInPCSpacePorts,
        converter=ReachTrajectoriesInPCSpacePorts.converter,
    )
    variant: Optional[str] = "full"

    stride_plot: int = 6
    fp_alpha: float = 0.4
    marker_size: float = 2

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        result: PyTree,
        fps_results: TreeNamespace,
        pca_results: TreeNamespace,
        colors: PyTree,
        replicate_info: PyTree,
        hps_common: TreeNamespace,
        **kwargs,
    ):
        from feedbax._experiments.analysis.state_utils import get_best_replicate

        fps = jt.map(
            lambda fps_dict: jtree.stack(list(fps_dict.values())),
            fps_results.fps,
            is_leaf=LDict.is_of("sisu"),
        )

        hidden = jt.map(
            lambda states_dict: jtree.stack(list(states_dict.values())),
            jt.map(
                lambda states: jt.map(lambda s: s.net.hidden, states, is_leaf=is_module),
                fps_results.states,
                is_leaf=LDict.is_of("sisu"),
            ),
            is_leaf=LDict.is_of("sisu"),
        )

        # Get best replicate
        fps_best = get_best_replicate(fps, replicate_info=replicate_info, axis=0, keep_axis=True)
        hidden_best = get_best_replicate(
            hidden, replicate_info=replicate_info, axis=0, keep_axis=True
        )

        # Transform to PC space
        fps_pc = pca_results.batch_transform(fps_best)
        hidden_pc = pca_results.batch_transform(hidden_best)

        # Get readout weights in PC space
        all_readout_weights = exclude_bad_replicates(
            jt.map(
                lambda model: model.step.net.readout.weight,
                jt.map(first, data.models, is_leaf=LDict.is_of("sisu")),
                is_leaf=is_module,
            ),
            replicate_info=replicate_info,
            axis=0,
        )
        readout_weights_pc = pca_results.batch_transform(all_readout_weights)

        def plot_single_context_trajectories(fps_pc, hidden_pc, readout_weights_pc, context_idx=0):
            """Plot hidden and FP trajectories for a single context input."""
            pcs_plot = slice(0, 3)

            fig = plot_hidden_and_fp_trajectories_3D(
                fps_pc["states"][context_idx, ..., pcs_plot],
                hidden_pc[context_idx, ..., pcs_plot],
                colors=list(colors["sisu"].dark.values()),
                stride_plot=self.stride_plot,
                fp_alpha=self.fp_alpha,
                marker_size=self.marker_size,
            )

            fig.add_traces(
                readout_vector_traces(
                    readout_weights_pc[..., pcs_plot],
                    jnp.tile(hidden_pc[context_idx, 0, 0, pcs_plot], (2, 1)),
                ),
            )

            return fig

        return jt.map(
            plot_single_context_trajectories,
            fps_pc,
            hidden_pc,
            readout_weights_pc,
            is_leaf=LDict.is_of("sisu"),
        )


class ReachDirectionTrajectoriesPorts(AbstractAnalysisPorts):
    """Input ports for ReachDirectionTrajectories analysis."""

    fps_results: InputOf[ReachFPs]
    pca_results: InputOf[StatesPCA]


class ReachDirectionTrajectories(AbstractAnalysis[ReachDirectionTrajectoriesPorts]):
    """Plot fixed point trajectories for a single reach direction across context inputs."""

    Ports = ReachDirectionTrajectoriesPorts
    inputs: ReachDirectionTrajectoriesPorts = eqx.field(
        default_factory=ReachDirectionTrajectoriesPorts,
        converter=ReachDirectionTrajectoriesPorts.converter,
    )
    variant: Optional[str] = "full"

    direction_idx: int = 0
    stride_plot: int = 1
    fp_alpha: float = 0.4
    marker_size: float = 2

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        result: PyTree,
        fps_results: TreeNamespace,
        pca_results: TreeNamespace,
        colors: PyTree,
        replicate_info: PyTree,
        hps_common: TreeNamespace,
        **kwargs,
    ):
        from feedbax._experiments.analysis.state_utils import get_best_replicate

        fps = jt.map(
            lambda fps_dict: jtree.stack(list(fps_dict.values())),
            fps_results.fps,
            is_leaf=LDict.is_of("sisu"),
        )

        hidden = jt.map(
            lambda states_dict: jtree.stack(list(states_dict.values())),
            jt.map(
                lambda states: jt.map(lambda s: s.net.hidden, states, is_leaf=is_module),
                fps_results.states,
                is_leaf=LDict.is_of("sisu"),
            ),
            is_leaf=LDict.is_of("sisu"),
        )

        # Get best replicate
        fps_best = get_best_replicate(fps, replicate_info=replicate_info, axis=0, keep_axis=True)
        hidden_best = get_best_replicate(
            hidden, replicate_info=replicate_info, axis=0, keep_axis=True
        )

        # Transform to PC space
        fps_pc = pca_results.batch_transform(fps_best)
        hidden_pc = pca_results.batch_transform(hidden_best)

        # Get readout weights in PC space
        all_readout_weights = exclude_bad_replicates(
            jt.map(
                lambda model: model.step.net.readout.weight,
                jt.map(first, data.models, is_leaf=LDict.is_of("sisu")),
                is_leaf=is_module,
            ),
            replicate_info=replicate_info,
            axis=0,
        )
        readout_weights_pc = pca_results.batch_transform(all_readout_weights)

        def plot_direction_across_contexts(fps_pc, hidden_pc, readout_weights_pc):
            """Plot a single reach direction's FP trajectory across context inputs."""
            pcs_plot = slice(0, 3)

            fig = plot_hidden_and_fp_trajectories_3D(
                fps_pc["states"][:, self.direction_idx, ..., pcs_plot],
                hidden_pc[:, self.direction_idx, ..., pcs_plot],
                colors=list(colors["sisu"].dark.values()),
                stride_plot=self.stride_plot,
                fp_alpha=self.fp_alpha,
                marker_size=self.marker_size,
            )

            fig.add_traces(
                readout_vector_traces(
                    readout_weights_pc[..., pcs_plot],
                    jnp.tile(hidden_pc[0, self.direction_idx, 0, pcs_plot], (2, 1)),
                ),
            )

            return fig

        return jt.map(
            plot_direction_across_contexts,
            fps_pc,
            hidden_pc,
            readout_weights_pc,
            is_leaf=LDict.is_of("sisu"),
        )
