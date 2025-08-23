"""
Analysis of fixed point (FP) structure of the network at steady state.

Includes eigendecomposition of the linearization (Jacobian).

Steady state FPs are the "goal-goal" FPs; i.e. the point mass is at the goal, so the
network will stabilize on some hidden state that outputs a constant force that does not change the
position of the point mass on average.

This contrasts with non-steady-state FPs, which correspond to network outputs which should
cause the point mass to move, and thus the network's feedback input (and thus FP) to
change.
"""

from collections import namedtuple
from collections.abc import Callable, Sequence
from functools import partial, wraps
from operator import is_
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Mapping, Optional

from arrow import get
import equinox as eqx
from equinox import Module
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from matplotlib.pylab import svd
import numpy as np
import plotly.graph_objects as go

from feedbax.task import AbstractTask
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rlrmp.analysis import AbstractAnalysis
from rlrmp.analysis.analysis import (
    AbstractAnalysisPorts,
    ExpandTo,
    FigIterCtx, 
    InputOf, 
    LiteralInput, 
    Data, 

    Transformed,
)
from rlrmp.analysis.eig import SVD, Eig, complex_to_polar_abs_angle
from rlrmp.analysis.fp_finder import FPFilteredResults, take_top_fps
from rlrmp.analysis.fps import FixedPoints, PlotInPCSpace
from rlrmp.analysis.grad import Jacobians, Hessians
from rlrmp.analysis.pca import StatesPCA
from rlrmp.analysis.plot import ScatterN3D
from rlrmp.analysis.violins import Violins
from rlrmp.misc import create_arr_df, get_constant_input_fn, take_non_nan
from rlrmp.analysis.state_utils import get_best_model_replicate, vmap_eval_ensemble
from rlrmp.plot import plot_eigvals_df
from rlrmp.tree_utils import first_shape, take_replicate, tree_level_labels
from rlrmp.types import AnalysisInputData, TreeNamespace
from rlrmp.types import LDict
from rlrmp.analysis.execution import AnalysisModuleTransformSpec


N_PCA = 50
PCA_START_STEP = 0
PCA_END_STEP = 100
STRIDE_FP_CANDIDATES = 16


COLOR_FUNCS: dict[str, Callable[[TreeNamespace], Sequence]] = dict(
)


def setup_eval_tasks_and_models(
    task_base: Module,
    models_base: LDict[float, Module],
    hps: TreeNamespace
):
    """Define a task where the point mass is already at the goal.

    This is similar to `feedback_perts`, but without an impulse.
    """
    all_tasks, all_models, all_hps = jtree.unzip(
        LDict.of('sisu')({
            sisu: (
                task_base.add_input(
                    name="sisu",
                    input_fn=get_constant_input_fn(
                        sisu, hps.model.n_steps, task_base.n_validation_trials,
                    ),
                ),
                models_base,  
                hps | dict(sisu=sisu),
            )
            for sisu in hps.sisu
        })
    )
    # Provides any additional data needed for the analysis
    extras = SimpleNamespace()  
    
    return all_tasks, all_models, all_hps, extras


# Unlike `feedback_perts`, we don't need to vmap over impulse amplitude 
eval_func: Callable = vmap_eval_ensemble


class EigvalsPlotPorts(AbstractAnalysisPorts):
    eigvals: InputOf[Array]


class EigvalsPlot(AbstractAnalysis[EigvalsPlotPorts]):
    """
    Innermost LDict level is the legend level.
    """
    non_data_trace_names = ['zerolines', 'boundary_circle', 'boundary_line']
    Ports = EigvalsPlotPorts
    inputs: EigvalsPlotPorts = eqx.field(
        default_factory=EigvalsPlotPorts, converter=EigvalsPlotPorts.converter
    )

    axis_names: Optional[Sequence[str]] = None
    reverse_traces: bool = False
    hide_histograms: bool = True
    
    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        eigvals: PyTree[LDict],
        hps_common: TreeNamespace,
        colors: PyTree,
        **kwargs
    ) -> PyTree[go.Figure]:
        level_labels = tree_level_labels(eigvals)
        legend_var_label = level_labels[-1]
        plot_labels = list(jt.leaves(eigvals, is_leaf=LDict.is_of(legend_var_label))[0].keys())
        
        if self.axis_names is not None:
            col_names = [legend_var_label, self.axis_names]
        else: 
            col_names = [
                legend_var_label, 
                *[f'var_{i}' for i in range(1, len(first_shape(eigvals)))],
                "eigvals",
            ]
            
        eigval_dfs = jt.map(
            lambda arr: create_arr_df(
                arr, 
                col_names=col_names,
            ).astype({legend_var_label: 'str'}),
            jtree.stack_subtrees(eigvals, axis=0, is_subtree=LDict.is_of(legend_var_label)),
        )

        plot_func_partial = partial(
            plot_eigvals_df,
            marginals='box',
            color=legend_var_label,
            marginal_boundary_lines=not self.hide_histograms,
            trace_kws=dict(marker_size=2.5),
            scatter_kws=dict(opacity=1),
            layout_kws=dict(
                legend_title=legend_var_label,
                legend_itemsizing='constant',
                xaxis_title='Re',
                yaxis_title='Im',
            ),
        )

        figs = jt.map(
            lambda df: plot_func_partial(
                df, color_discrete_sequence=list(colors[legend_var_label].dark.values())
            ),
            eigval_dfs
        )
        
        if self.reverse_traces:
            jt.map(
                lambda fig: setattr(fig, 'data', fig.data[::-1]),
                figs,
                is_leaf=is_type(go.Figure),
            )

        def _update_trace_name(trace):
            if trace.name is not None and trace.name not in self.non_data_trace_names:
                return trace.update(name=plot_labels[int(trace.name)])
            else:
                return trace

        jt.map(
            lambda fig: fig.for_each_trace(_update_trace_name),
            figs,
            is_leaf=is_type(go.Figure),
        )
        
        # Set the axis limits of all figures to be equal,
        # and determined only by the scatter data
        # def trace_selector(trace):
        #     return (
        #         trace.type.startswith('scatter') 
        #         and trace.name not in non_data_trace_names
        #     )

        # figs = set_axes_bounds_equal(
        #     figs,
        #     trace_selector=trace_selector,
        #     padding_factor=padding_factor,
        # )
        
        if self.hide_histograms:
            def _hide_trace(trace):
                if trace.type == 'box' or trace.name == 'boundary_line':
                    trace.update(visible=False)
                return trace
            
            jt.map(
                lambda fig: fig.for_each_trace(_hide_trace),
                figs,
                is_leaf=is_type(go.Figure),
            )

        return figs
    

#! TODO: Refactor out the common parts, e.g. meets criteria, top fps, etc.
def process_fps(all_fps: PyTree[FPFilteredResults], n_keep: int = 6) -> TreeNamespace:
    """Only keep FPs/replicates that meet criteria."""
    
    n_fps_meeting_criteria = jt.map(
        lambda fps: fps.counts['meets_all_criteria'],
        all_fps,
        is_leaf=is_type(FPFilteredResults),
    )

    # satisfactory_replicates = jt.map(
    #     lambda n_matching_fps_by_sisu: jnp.all(
    #         jnp.stack(jt.leaves(n_matching_fps_by_sisu), axis=0),
    #         axis=0,
    #     ),
    #     n_fps_meeting_criteria,
    #     is_leaf=LDict.is_of('sisu'),
    # )

    all_top_fps = take_top_fps(all_fps, n_keep=n_keep)

    # Average over the top fixed points, to get a single one for each included replicate and SISU
    fps_final = jt.map(
        lambda top_fps_by_sisu: jt.map(
            lambda fps: jnp.nanmean(fps, axis=-2),
            top_fps_by_sisu,
            is_leaf=is_type(FPFilteredResults),
        ),
        all_top_fps,
        is_leaf=LDict.is_of('sisu'),
    )

    return TreeNamespace(
        fps=fps_final,
        all_top_fps=all_top_fps,
        n_fps_meeting_criteria=n_fps_meeting_criteria,
        # satisfactory_replicates=satisfactory_replicates,
    )


def get_ss_rnn_input(sisu: float, pos: Float[Array, "2"]):
    vel = jnp.zeros_like(pos)
    return jnp.array([sisu, *pos, *vel, *pos, *vel])


def get_ss_rnn_func(rnn_cell: Callable[[Array, Array], Array]):
    def rnn_func(sisu, pos, h):
        input_star = get_ss_rnn_input(sisu, pos)
        return rnn_cell(input_star, h)

    return rnn_func
    
    
def reshape_candidates(states: Array) -> Array:
    """Reshape the initial FP candidates to (positions, candidates, state)."""
    positions_first = jnp.moveaxis(states, 1, 0)
    candidates = jnp.reshape(
        positions_first, 
        (positions_first.shape[0], -1, positions_first.shape[-1]),
    )
    return candidates  # Take every STRIDE_FP_CANDIDATES candidate


"""Apply global transformations to reduce computation.
Since this module always uses get_best_replicate, we apply it globally
before evaluation to save computational resources."""
TRANSFORMS = AnalysisModuleTransformSpec(
    pre_setup=dict(models=get_best_model_replicate),
    # `get_best_model_replicate` leaves in the singleton replicate axis -- so, remove it 
    post_eval=dict(
        states=lambda states: jt.map(lambda x: x[:, 0], states),
        models=partial(take_replicate, 0),
    )
)


ss_rnn_funcs = Data.models(where=lambda model: get_ss_rnn_func(model.step.net.hidden))
sisu = Data.hps(where=lambda hps: hps.sisu, is_leaf=is_type(TreeNamespace))
positions = Data.tasks(where=lambda task: (
    task.validation_trials.targets["mechanics.effector.pos"].value[:, -1]
))
steady_state_fps = Transformed(
    "steady_state_fp_results",
    lambda fp_results: fp_results.fps,  # Just the top first FP for each steady state
)


# Model PyTree structure: ['sisu', 'train__pert__std']
# State batch shape: (eval, replicate, condition)
DEPENDENCIES = {
    "states_pca": (
        StatesPCA(
            n_components=N_PCA, 
            where_states=lambda states: states.net.hidden,
            aggregate_over_labels=('sisu',),
        )
        .after_indexing(-2, np.arange(PCA_START_STEP, PCA_END_STEP), axis_label="timestep")
    ),
    "steady_state_fp_results": (
        FixedPoints(
            stride_candidates=STRIDE_FP_CANDIDATES,
            inputs=FixedPoints.Ports(
                funcs=ss_rnn_funcs,
                func_args=ExpandTo.map(  
                    "funcs", 
                    (sisu, positions), 
                    is_leaf_prefix=(is_type(TreeNamespace), is_type(AbstractTask)),
                ),
                # We can reshape_candidates here since get_best_replicate is applied globally
                candidates=Data.states(
                    # FP initial conditions <- full hidden state trajectories
                    where=lambda states: reshape_candidates(states.net.hidden),
                ),
            ),
        )
        # over the steady state workspace positions & corresponding candidates
        .vmap(in_axes={'func_args': (None, 0), 'candidates': 0})
        .then_transform_result(process_fps)
    ),
}


GradArgs = namedtuple("GradArgs", ["sisu", "pos", "h"])


def jac_eigval_violin_params_fn(fig_params, ctx: FigIterCtx):
    if ctx.key == 'angle':
        yaxis_title = 'Eigenvalue angle (rad)'
        yaxis_range = [0, jnp.pi]
    elif ctx.key == 'magnitude':
        yaxis_title = 'Eigenvalue magnitude'
        yaxis_range = [0, 1.1]
    else:
        raise ValueError(f"Unknown component {ctx.key}")
    return fig_params | dict(
        yaxis_title=yaxis_title,
        yaxis_range=yaxis_range,
    )


# State PyTree structure: ['sisu', 'train__pert__std']
# Array batch shape: (evals, replicates, reach conditions)
ANALYSES = {
    "plot--fps_pc": (
        PlotInPCSpace(
            inputs=PlotInPCSpace.Ports(
                pca_results="states_pca",
                plot_data="steady_state_fp_results",
            ),
            spread_label='sisu',
        )
        .after_transform(lambda fp_results: fp_results.fps, dependency_names="plot_data")
    ),
    "plot--fps_pc": (
        ScatterN3D(
            inputs=PlotInPCSpace.Ports(
                pca_results="states_pca",
                plot_data="steady_state_fp_results",
            ),
            spread_label='sisu',
        )
        .after_transform(lambda fp_results: fp_results.fps, dependency_names="plot_data")
    ),
    #! Maybe it would make sense to just make a single class, `Grads` with `grad_func`?
    #! Though it wouldn't change the verbosity here unless we made `grad_func` a `Port`
    #! and returned a PyTree containing both the Jacobians and the Hessians
    **{  # "steady_state_jacobians" and "steady_state_hessians"
        f"steady_state_{cls.__name__.lower()}": (
            cls(
                inputs=cls.Ports(
                    funcs=ss_rnn_funcs,  
                    func_args=ExpandTo.map(
                        "funcs",  # signature: (sisu, pos, h)
                        GradArgs(sisu, positions, steady_state_fps),
                        is_leaf=is_module,
                        is_leaf_prefix=GradArgs(is_type(TreeNamespace), is_module, None),
                    ),
                ),
            )
            # over the steady state workspace positions
            .vmap(in_axes={'func_args': GradArgs(None, 0, 0)})
        )
        for cls in (Jacobians,)#!, Hessians)
    },
    
    "steady_state_jac-x_eigs": (
        Eig(
            inputs=Eig.Ports(
                # Process only the square (state) Jacobians
                matrices=Transformed("steady_state_jacobians", lambda jacs: jacs.h),
            ),
        )
        # .vmap(in_axes={'matrices': 0}) #? Should be unnecessary since `eig` works on batches
    ),
    "steady_state_jac-u_svd": (
        SVD(
            inputs=SVD.Ports(
                # Note the SVD for `jacs.sisu`, a vector, will just be its Euclidean norm 
                matrices=Transformed("steady_state_jacobians", lambda jacs: jacs.pos),
            ),
        )
    ),
    
    "plot--steady_state_jac-x_eigvals": (
        EigvalsPlot(
            hide_histograms=False,
            inputs=EigvalsPlot.Ports(
                eigvals=Transformed("steady_state_jac-x_eigs", lambda eigs: eigs.eigvals),
            ),
        )
        .after_level_to_bottom('sisu', dependency_name="eigvals")
    ),
    
    #! This will not work together with "plot--steady_state_jac-x_eigvals" because they have the 
    #! same `md5_str`; need to make `md5_str` depend on `self.inputs`!
    #! TODO: Don't use `EigvalsPlot` anyway, since singular values are real-valued and only 
    #! represent input gains (wrt network state), and *not* dynamics!
    # "plot--steady_state_jac-u_singvals": (
    #     EigvalsPlot(
    #         hide_histograms=False,
    #         inputs=EigvalsPlot.Ports(
    #             eigvals=Transformed("steady_state_jac-u_svd", lambda svd: svd.singvals),
    #         ),
    #     )
    #     .after_level_to_bottom('sisu', dependency_name="eigvals")
    # ),
    
    "plot--jac_x_eigval-violins": (
        Violins(
            inputs=Violins.Ports(
                input=Transformed.map(
                    source="steady_state_jac-x_eigs",
                    transform=complex_to_polar_abs_angle,
                )
            ),
        )
        # .after_subdict_at_level('sisu', keys=SISUS_TO_PLOT_EIGVALS)
        .after_rearrange_levels(
            [..., 'component', 'sisu', 'train__pert__std'], 
            dependency_name="input",
        )
        .map_figs_at_level('component', fig_params_fn=jac_eigval_violin_params_fn)
    ),
    
    "plot--jac_u_singval-violins": (
        Violins(
            inputs=Violins.Ports(
                input="steady_state_jac-u_svd",
            ),
        )
        # .after_subdict_at_level('sisu', keys=SISUS_TO_PLOT_EIGVALS)
        .after_rearrange_levels(
            [..., 'sisu', 'train__pert__std'], 
            dependency_name="input",
        )
    )
}


#! The following should either be discarded or refactored into a function for plotting readout
#! weights. If we refactor, it will be necessary to implement figure combination across `ANALYSES`
#! entries. 
# all_readout_weights = exclude_bad_replicates(
#     jt.map(
#         lambda model: model.step.net.readout.weight,
#         # Weights do not depend on SISU, take first
#         jt.map(first, data.models, is_leaf=LDict.is_of('sisu')),
#         is_leaf=is_module,
#     ),
#     replicate_info=replicate_info,
#     axis=0,
# )

# all_readout_weights_pc = pca_results.batch_transform(all_readout_weights)

# if readout_weights_pc is not None:
#     fig.update_layout(
#         legend2=dict(title='Readout components', itemsizing='constant', y=0.45),
#     )
#     mean_base_fp_pc = jnp.mean(fp_pcs_by_sisu[min(hps_common.sisu)], axis=1)
#     traces = []
#     k = 0.25
#     for j in range(readout_weights_pc.shape[-2]):
#         start, end = mean_base_fp_pc, mean_base_fp_pc + k * readout_weights_pc[..., j, :]
#         x = np.column_stack((start[..., 0], end[..., 0], np.full_like(start[..., 0], None))).ravel()
#         y = np.column_stack((start[..., 1], end[..., 1], np.full_like(start[..., 1], None))).ravel()
#         z = np.column_stack((start[..., 2], end[..., 2], np.full_like(start[..., 2], None))).ravel()
#         traces.append(go.Scatter3d(
#             x=x, y=y, z=z, mode='lines', line=dict(width=10),
#             showlegend=True, name=j, legend="legend2",
#         ))
#     fig.add_traces(traces)


"""
If we decide to aggregate over multiple replicates rather than taking only the best one,
it will be necessary to exclude the NaN replicates prior to performing sklearn PCA.
The following are the original functions I used for that purpose.
"""
def exclude_nan_replicates(tree, replicate_info, exclude_underperformers_by='best_total_loss'):
    return jt.map(
        take_replicate,
        replicate_info['included_replicates'][exclude_underperformers_by],
        tree,
    )

