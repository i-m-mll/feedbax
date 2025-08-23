"""What happens if we change the network's sisu input, at steady state or during a reach?
"""

import equinox as eqx

from feedbax.intervene import schedule_intervenor
import jax_cookbook.tree as jtree

from rlrmp.analysis.activity import NetworkActivity_SampleUnits
from rlrmp.analysis.aligned import get_aligned_trajectories_node
from rlrmp.analysis.analysis import FigIterCtx
from rlrmp.analysis.effector import EffectorTrajectories
from rlrmp.colors import ColorscaleSpec
from rlrmp.analysis.disturbance import PLANT_PERT_FUNCS, get_pert_amp_vmap_eval_func
from rlrmp.analysis.profiles import Profiles
from rlrmp.analysis.state_utils import get_best_replicate, get_step_task_input_fn
from rlrmp.analysis.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.types import LDict


COLOR_FUNCS = dict(
    # pert__amp=lambda hps: [final - hps.pert.sisu.init for final in hps.pert.sisu.final],
    # sisu=lambda hps: [final - hps.pert.sisu.init for final in hps.pert.sisu.final],
    pert__sisu__amp=ColorscaleSpec(
        sequence_func=lambda hps: [final - hps.pert.sisu.init for final in hps.pert.sisu.final],
        colorscale="thermal",
    ),
)


def setup_eval_tasks_and_models(task_base, models_base, hps):
    """Modify the task so that SISU varies over trials.
    
    Note that this is a bit different to how we perturb state variables; normally we'd use an intervenor 
    but since the SISU is supplied by the task, we can just change the way that's defined.
    """
    plant_disturbance = PLANT_PERT_FUNCS[hps.pert.plant.type]
    
    # Add placeholder for plant perturbations
    task_base, models_base = schedule_intervenor(
        task_base, models_base, 
        lambda model: model.step.mechanics,
        plant_disturbance(0),
        default_active=False,
        label=PLANT_INTERVENOR_LABEL,
    )
    
    #! Neither `pert__amp` nor `sisu` are entirely valid as labels here, I think
    tasks, models, hps = jtree.unzip(LDict.of("pert__sisu__amp")({
        sisu_final: (
            task_base.add_input(
                name="sisu",
                input_fn=get_step_task_input_fn(
                    hps.pert.sisu.init, 
                    sisu_final,
                    hps.pert.sisu.step,  
                    hps.model.n_steps - 1, 
                    task_base.n_validation_trials,
                ),
            ),
            models_base, 
            hps | dict(pert=dict(amp=sisu_final - hps.pert.sisu.init)),
        )
        for sisu_final in hps.pert.sisu.final
    }))
    
    return tasks, models, hps, None


eval_func = get_pert_amp_vmap_eval_func(lambda hps: hps.pert.plant.amp, PLANT_INTERVENOR_LABEL)


PLANT_PERT_LABELS = {0: "no curl", 1: "curl"}
PLANT_PERT_STYLES = dict(line_dash={0: "dot", 1: "solid"})


def dashed_fig_params_fn(fig_params, ctx: FigIterCtx):
    return fig_params | dict(
        scatter_kws=dict(
            line_dash=PLANT_PERT_STYLES['line_dash'][ctx.idx],
            legendgroup=PLANT_PERT_LABELS[ctx.idx],
            legendgrouptitle_text=PLANT_PERT_LABELS[ctx.idx].capitalize(),
        ),
    )



ANALYSES = {
    "effector_trajectories_steady": (
        # -- Steady-state --
        # 0. Show that SISU perturbation does not cause a significant change in force output at steady-state.
        EffectorTrajectories(
            variant="steady",
            pos_endpoints=False,
            straight_guides=False,
            colorscale_axis=1, 
            colorscale_key="reach_condition",
        )
            .after_transform(get_best_replicate)  # By default has `axis=1` for replicates
            .with_fig_params(
                mean_exclude_axes=(-3,),  # Average over all extra batch axes *except* reach direction/condition
                legend_title="SISU<br>pert. amp.",
            )
    ),

    #! TODO: Not displaying; debug pytree structure
    #! Also only one of the two legendgroup titles is displayed, even though the respective values/labels appear to be properly passed
    # "profiles_steady": (
    #     Profiles(variant="steady")
    #         .after_level_to_top('train__pert__std')
    #         .combine_figs_by_axis(
    #             axis=2,     
    #             fig_params_fn=lambda fig_params, i, item: dict(
    #                 scatter_kws=dict(
    #                     line_dash=PLANT_PERT_STYLES['line_dash'][i],
    #                     legendgroup=PLANT_PERT_LABELS[i],
    #                     legendgrouptitle_text=PLANT_PERT_LABELS[i].capitalize(),
    #                 ),
    #             ),
    #         )
    # ),

    "network_activity_steady": (
        # 1. Activity of sample units, to show they change when SISU does
        NetworkActivity_SampleUnits(variant="steady")
        .after_transform(get_best_replicate)
        .after_level_to_top('train__pert__std')
        .with_fig_params(
            legend_title="SISU pert. amp.",  #! No effect
        )
    ),

    "aligned_trajectories_reach": (
        # -- Reaching --
        # 2. Plot aligned vars for reaching +/- plant pert, +/- SISU pert on same plot
        # (It only makes sense to do this for reaches (not ss), at least for curl fields.)
        # Hide individual trials for this plot, since they make it hard to distinguish the means;
        # the variability should be clear from other plots. 
        get_aligned_trajectories_node(colorscale_key="pert__sisu__amp")
        .after_getitem_at_level("task_variant", "reach")
        .with_fig_params(
            legend_title="Final SISU",
            scatter_kws=dict(line_width=0),  # Hide individual trials
            layout_kws=dict(
                legend_title_font_weight="bold",
                #! TODO: Nested dict update so we don't need to pass these redundantly
                width=900, 
                height=300,
                legend_tracegroupgap=1, 
                margin_t=50,
                margin_b=20,
            ),
        )
        .combine_figs_by_axis(
            axis=3,  # Not 2, because of the prior stacking
            fig_params_fn=dashed_fig_params_fn,
        )
    ),

    "profiles_reach": (
        #! Only one of the two legendgroup titles is displayed, even though the respective values/labels appear to be properly passed.
        #! I'm not sure why this is different from `AlignedEffectorTrajectories`, where the legend
        #! is displayed correctly (2025-08-20: Is it still different from `ScatterN2D` now that 
        #! `AlignedEffectorTrajectories` is gone?)
        
        Profiles(variant="reach")
            .after_level_to_top('train__pert__std')
            .combine_figs_by_axis(
                axis=2,     
                fig_params_fn=dashed_fig_params_fn,
            ),
    ),

    # "network_activity_project_pca": (
    #     # 4. Perform PCA wrt baseline `reach` variant, and project `steady` variant into that space
    #     # (To show that SISU causally varies the network activity in a null direction)
    #     NetworkActivity_ProjectPCA(
    #         variant="steady", 
    #         variant_pca="reach_pca",
    #     )
    # ),
}
