"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .analytical_component import AnalyticalMusculoskeletalPlant


def _matches_AnalyticalMusculoskeletalPlant_v1(params):
    return set(params) <= {"tau_act", "dt", "tau_deact", "clip_states", "n_steps", "solver_type"}


HISTORIES = {
    AnalyticalMusculoskeletalPlant: History(
        legacy=(
            LegacyEndpoint(
                "AnalyticalMusculoskeletalPlant", "1", 1, _matches_AnalyticalMusculoskeletalPlant_v1
            ),
        )
    ),
}
