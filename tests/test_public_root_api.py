import importlib.util
import subprocess
import sys

import pytest

import feedbax
from feedbax.runtime.mapping import WhereDict


def test_generic_tree_and_io_helpers_are_not_package_root_exports() -> None:
    for name in ("load", "load_with_hyperparameters", "save", "tree_take", "tree_labels"):
        assert not hasattr(feedbax, name)


def test_root_support_helper_modules_are_removed() -> None:
    assert importlib.util.find_spec("feedbax.tree_utils") is None
    assert importlib.util.find_spec("feedbax.setup_utils") is None
    assert importlib.util.find_spec("feedbax.config.tree") is not None
    assert importlib.util.find_spec("feedbax.analysis.setup") is None


def test_runtime_mapping_is_available_from_package_root() -> None:
    assert feedbax.WhereDict is WhereDict


@pytest.mark.parametrize(
    "module_name",
    ("feedbax.training.metrics", "feedbax.orchestration.runpod"),
)
def test_execution_support_modules_import_in_a_fresh_process(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
