"""Task definitions, timelines, trial preparation, and task presets."""

from importlib import import_module

__all__ = [
    "AbstractTask",
    "DELAYED_CENTER_OUT_PRESET",
    "DelayedReaches",
    "DelayedReachTaskInputs",
    "PreparedTrial",
    "SimpleReaches",
    "SimpleReachTaskInputs",
    "Stabilization",
    "TaskComponent",
    "TaskTrialSpec",
    "TrialSpecDependency",
    "TrialTimeline",
    "WhereDict",
    "_forceless_task_inputs",
    "_pos_only_states",
    "apply_delayed_reaches_preset",
    "centreout_endpoints",
    "delayed_center_out_reaches_params",
    "eval_ensemble_on_trials",
    "gen_epoch_lengths",
    "get_masks",
    "get_masked_seqs",
    "get_scalar_epoch_seq",
    "infer_n_steps",
    "internal_grid_points",
    "prepare_inputs",
    "prepare_trial",
    "safe_state_set",
    "set_state_by_path",
    "set_state_matching_dtypes",
    "where_key_to_path",
]


_PRESET_EXPORTS = frozenset({
    "DELAYED_CENTER_OUT_PRESET",
    "apply_delayed_reaches_preset",
    "delayed_center_out_reaches_params",
})


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = ".presets" if name in _PRESET_EXPORTS else ".task"
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
