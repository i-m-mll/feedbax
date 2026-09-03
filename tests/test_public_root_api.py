import importlib.util
import subprocess
import sys

import pytest

import feedbax
import feedbax.tasks as task_api
from feedbax.runtime.mapping import WhereDict


def test_generic_tree_and_io_helpers_are_not_package_root_exports() -> None:
    for name in ("load", "load_with_hyperparameters", "save", "tree_take", "tree_labels"):
        assert not hasattr(feedbax, name)


def test_root_support_helper_modules_are_removed() -> None:
    assert importlib.util.find_spec("feedbax.tree_utils") is None
    assert importlib.util.find_spec("feedbax.setup_utils") is None
    assert importlib.util.find_spec("feedbax.config.tree") is not None
    assert importlib.util.find_spec("feedbax.analysis.setup") is None


def test_task_exports_are_available_from_package_root() -> None:
    assert feedbax.AbstractTask is task_api.AbstractTask
    assert feedbax.DelayedReaches is task_api.DelayedReaches
    assert feedbax.DelayedReachTaskInputs is task_api.DelayedReachTaskInputs
    assert feedbax.SimpleReaches is task_api.SimpleReaches
    assert feedbax.TaskTrialSpec is task_api.TaskTrialSpec
    assert feedbax.TrialTimeline is task_api.TrialTimeline
    assert feedbax.WhereDict is WhereDict
    assert feedbax.centreout_endpoints is task_api.centreout_endpoints
    assert feedbax.eval_ensemble_on_trials is task_api.eval_ensemble_on_trials
    assert feedbax.forceless_task_inputs is task_api._forceless_task_inputs
    assert feedbax.gen_epoch_lengths is task_api.gen_epoch_lengths
    assert feedbax.get_masks is task_api.get_masks
    assert feedbax.get_masked_seqs is task_api.get_masked_seqs
    assert feedbax.get_scalar_epoch_seq is task_api.get_scalar_epoch_seq
    assert feedbax.pos_only_states is task_api._pos_only_states
    assert feedbax.prepare_trial is task_api.prepare_trial


def test_plain_package_import_does_not_import_plotly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import feedbax; raise SystemExit('plotly' in sys.modules)",
        ],
        check=False,
    )

    assert result.returncode == 0


@pytest.mark.parametrize(
    "module_name",
    ("feedbax.orchestration.events", "feedbax.orchestration.bundle"),
)
def test_orchestration_modules_import_in_a_fresh_process(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
