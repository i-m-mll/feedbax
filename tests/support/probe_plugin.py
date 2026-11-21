"""The probe operations of `tests.support.map_probe`, installed as a plugin for one test."""

import os
from pathlib import Path
import sys

import pytest

from feedbax.plugins.composition import load_application
from tests.support import map_probe


@pytest.fixture
def probe_operations(tmp_path, monkeypatch):
    """The probe operations installed as a plugin of this process and the Hosts it starts, until the test ends."""
    package = tmp_path / "plugins" / "map_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(Path(map_probe.__file__).read_text())
    metadata = tmp_path / "plugins" / "map_probe-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: map-probe\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text("[feedbax.plugins]\nmap_probe = map_probe\n")
    monkeypatch.syspath_prepend(str(tmp_path / "plugins"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(tmp_path / "plugins"), os.environ.get("PYTHONPATH")))))
    yield load_application().bundle.operations
    for name in [name for name in sys.modules if name == "map_probe" or name.startswith("map_probe.")]:
        del sys.modules[name]
