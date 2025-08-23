from collections.abc import Callable, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Literal as L, Optional

import jax.numpy as jnp
import jax.tree as jt

import feedbax.plotly as fbp
from jax_cookbook import is_module
import jax_cookbook.tree as jtree

from rlrmp.analysis.analysis import AbstractAnalysis, NoPorts
from rlrmp.analysis.state_utils import vmap_eval_ensemble
from rlrmp.types import AnalysisInputData, LDict


COLOR_FUNCS = dict()


def setup_eval_tasks_and_models(task_base, models_base, hps):
    """Use the base task, where the usual Gaussian noise in the feedback channels suffices for the frequency analysis."""
    return task_base, models_base, hps, None


eval_func = vmap_eval_ensemble


INPUT_WHERE = lambda state, idx: state.feedback.noise[idx]
OUTPUT_WHERE = lambda state: state.net.output


class FrequencyResponse(AbstractAnalysis[NoPorts]):
    variant: Optional[str] = "full"
    
    def compute(self, data: AnalysisInputData, **kwargs):
        all_freqs, all_gains, all_phases = jtree.unzip(jt.map(
            lambda fb_idx: jt.map(
                lambda states: frequency_analysis(
                    INPUT_WHERE(states, fb_idx),
                    OUTPUT_WHERE(states),
                    data.hps[self.variant].model.dt, 
                ),
                data.states[self.variant],
                is_leaf=is_module,
            ),
            dict(fb_pos=0, fb_vel=1),    
        ))
        
        return SimpleNamespace(
            freqs=all_freqs,
            gains=all_gains,
            phases=all_phases,
        )

    def make_figs(self, data: AnalysisInputData, *, result, colors, **kwargs):
        gains_plot, phases_plot = jt.map(
            lambda arr: jnp.moveaxis(
                arr, -1, 0
            ),
            (result.gains, result.phases),
        )

        gain_figs = LDict.of("fb_var")({
            fb_var: jt.map(
                lambda xy_idx: fbp.profiles(
                    jtree.take(gains_plot[fb_var], xy_idx),
                    keep_axis=None,
                    mode='std',
                    varname="Gain (dB)",
                    colors=list(colors["train__pert__std"].dark.values()),
                    # labels=disturbance_stds_load,
                    layout_kws=dict(
                        legend_title="Train<br>field std.",
                        width=600,
                        height=400,
                        legend_tracegroupgap=1,
                        yaxis_type="log",
                        xaxis_title="Frequency",
                    )
                ),
                LDict.of("coord")(dict(x=0, y=1)),
            )
            for fb_var in result.freqs
        })

        phase_figs = LDict.of("fb_var")({
            fb_var: jt.map(
                lambda xy_idx: fbp.profiles(
                    jtree.take(phases_plot[fb_var], xy_idx),
                    keep_axis=None,
                    mode='std',
                    varname="Phase (rad)",
                    colors=list(colors[self.variant]["train__pert__std"]["dark"].values()),
                    # labels=disturbance_stds_load,
                    layout_kws=dict(
                        legend_title="Train<br>field std.",
                        width=600,
                        height=400,
                        legend_tracegroupgap=1,
                        # yaxis_type="log",
                        xaxis_title="Frequency",
                    )
                ),
                LDict.of("coord")(dict(x=0, y=1)),
            )
            for fb_var in result.freqs
        })
        
        # Wrap in LabelDict so we get (e.g.) `label="gain"` in the database, for gain plots
        # (Alternatively, we could use two different named subclasses of `AbstractAnalysis` for gains and phases)
        return LDict.of("label")(dict(
            gain=gain_figs,
            phase=phase_figs,
        ))
        

def frequency_analysis(input_, output, dt):
    # input and output have shape (..., timesteps, 2)
    n_timesteps = input_.shape[-2]
    
    # (j)np.fft.fft handles batch dims automatically
    f_input = jnp.fft.fft(input_, axis=-2)  # shape (..., timesteps, 2)
    f_output = jnp.fft.fft(output, axis=-2)  # shape (..., timesteps, 2)
    freqs = jnp.fft.fftfreq(n_timesteps, dt)  # shape (timesteps,)
    
    # Compute gain and phase
    gain = jnp.abs(f_output / (f_input + 1e-8))
    phase = jnp.angle(f_output / (f_input + 1e-8))
    
    # We are only interested in real signals, so exclude the negative frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    gain = gain[..., pos_mask, :]
    phase = phase[..., pos_mask, :]
    
    return freqs, gain, phase


ANALYSES = {
    "frequency_response": FrequencyResponse(),
}