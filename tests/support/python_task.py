"""The Python task of `tests.support.reaching_task`, installed as a plugin for one test."""

import os
from pathlib import Path
import sys

import pytest

from feedbax.application.composition import load_application
from feedbax.tasks.registry import PythonTaskRegistry
from tests.support import reaching_task
from tests.support.reaching_task import REACHING_TASK as REACHING_TASK


@pytest.fixture
def reaching_python_task(tmp_path, monkeypatch) -> PythonTaskRegistry:
    """The Python tasks of this process and of the Hosts it starts, the reaching task among them, until the test ends.

    The shared Host was started before; a test that runs the task at a Host starts its own.
    """
    plugins = tmp_path / "python-task-plugins"
    package = plugins / "reaching_task"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(Path(reaching_task.__file__).read_text())
    metadata = plugins / "reaching_task-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: reaching-task\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text("[feedbax.plugins]\nreaching_task = reaching_task\n")
    monkeypatch.syspath_prepend(str(plugins))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(plugins), os.environ.get("PYTHONPATH")))))
    yield load_application().bundle.python_tasks
    for name in [name for name in sys.modules if name == "reaching_task" or name.startswith("reaching_task.")]:
        del sys.modules[name]
