"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .reach_components import SimpleReaches, DelayedReaches, Stabilization


def _matches_SimpleReaches_v1(params):
    return set(params) <= {
        "workspace",
        "n_steps",
        "eval_n_directions",
        "eval_grid_n",
        "eval_reach_length",
    }


def _matches_Stabilization_v1(params):
    return set(params) <= {"workspace", "n_steps"}


def _matches_DelayedReaches_v1(params):
    return set(params) <= {
        "hold_epochs",
        "move_epochs",
        "n_control_stages",
        "preset",
        "eval_reach_length",
        "epoch_names",
        "workspace",
        "n_steps",
        "p_catch_trial",
        "eval_n_directions",
        "target_on_epochs",
        "epoch_len_ranges",
        "go_cue_event_name",
        "train_endpoint_mode",
        "eval_grid_n",
        "catch_metadata_policy",
        "target_visible_from_start",
    }


HISTORIES = {
    SimpleReaches: History(
        legacy=(LegacyEndpoint("SimpleReaches", "1", 1, _matches_SimpleReaches_v1),)
    ),
    DelayedReaches: History(
        legacy=(LegacyEndpoint("DelayedReaches", "1", 1, _matches_DelayedReaches_v1),)
    ),
    Stabilization: History(
        legacy=(LegacyEndpoint("Stabilization", "1", 1, _matches_Stabilization_v1),)
    ),
}
