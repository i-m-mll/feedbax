import importlib.util

import feedbax
import feedbax.tasks as task_api
from feedbax.config.mapping import WhereDict


def test_generic_tree_and_io_helpers_are_not_package_root_exports() -> None:
    for name in ("load", "load_with_hyperparameters", "save", "tree_take", "tree_labels"):
        assert not hasattr(feedbax, name)


def test_root_support_helper_modules_are_removed() -> None:
    assert importlib.util.find_spec("feedbax.tree_utils") is None
    assert importlib.util.find_spec("feedbax.setup_utils") is None
    assert importlib.util.find_spec("feedbax.config.tree") is not None
    assert importlib.util.find_spec("feedbax.analysis.setup") is not None


def test_task_exports_are_available_from_package_root() -> None:
    assert feedbax.AbstractTask is task_api.AbstractTask
    assert feedbax.DelayedReaches is task_api.DelayedReaches
    assert feedbax.DelayedReachTaskInputs is task_api.DelayedReachTaskInputs
    assert feedbax.SimpleReaches is task_api.SimpleReaches
    assert feedbax.TaskTrialSpec is task_api.TaskTrialSpec
    assert feedbax.TrialTimeline is task_api.TrialTimeline
    assert feedbax.WhereDict is WhereDict
    assert feedbax.centreout_endpoints is task_api.centreout_endpoints
    assert feedbax.forceless_task_inputs is task_api._forceless_task_inputs
    assert feedbax.gen_epoch_lengths is task_api.gen_epoch_lengths
    assert feedbax.get_masks is task_api.get_masks
    assert feedbax.get_masked_seqs is task_api.get_masked_seqs
    assert feedbax.get_scalar_epoch_seq is task_api.get_scalar_epoch_seq
    assert feedbax.pos_only_states is task_api._pos_only_states
    assert feedbax.prepare_trial is task_api.prepare_trial
