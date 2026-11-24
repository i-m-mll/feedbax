"""The Python task of `tests/support/plugins`, installed as a plugin for one test."""

from collections.abc import Iterator
import os
from pathlib import Path
import sys

import pytest

from feedbax.application.composition import load_application
from feedbax.tasks.registry import PythonTaskRegistry
from tests.support.plugins.reaching_task import REACHING_TASK as REACHING_TASK

PLUGINS = Path(__file__).parent / "plugins"
"""A folder of installed distributions: the reaching task's package beside its metadata. An e2e spec that runs the
task puts it on its Studio's path."""


@pytest.fixture
def reaching_python_task(monkeypatch) -> Iterator[PythonTaskRegistry]:
    """The Python tasks of this process and of the Hosts it starts, the reaching task among them, until the test ends.

    The shared Host was started before; a test that runs the task at a Host starts its own.
    """
    monkeypatch.syspath_prepend(str(PLUGINS))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(PLUGINS), os.environ.get("PYTHONPATH")))))
    yield load_application().bundle.python_tasks
    for name in [name for name in sys.modules if name == "reaching_task" or name.startswith("reaching_task.")]:
        del sys.modules[name]
