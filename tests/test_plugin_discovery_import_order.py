"""Regression tests for the plugin-discovery / config import ordering.

These guard against the `feedbax.config` circular-import warning fixed under
issue ccfe63a:

- Importing `feedbax.plugins` must NOT run `discover_experiment_packages()` as
  an import-time side effect. Eager discovery at import time re-enters a
  partially initialized `feedbax.config` (and, transitively, a partially
  initialized `feedbax` root package) when an experiment package is installed,
  producing the `partially initialized module 'feedbax.config'` warning at CLI
  startup.
- After `import feedbax.config`, the global config namespaces must be populated
  from the base feedbax config, so consumers (e.g.
  `feedbax.config.hyperparams`, which reads `STRINGS.hps_level_label_sep` as a
  default argument value at import time) see real values rather than empty
  namespaces.
"""

import importlib
import os
from pathlib import Path
import subprocess
import sys


def test_importing_plugins_does_not_eagerly_discover(monkeypatch):
    """`import feedbax.plugins` must not call `discover_experiment_packages`.

    Re-imports `feedbax.plugins` from a clean `sys.modules` state with the
    discovery function patched on the package namespace, and asserts that the
    import itself does not invoke discovery — only first access of
    ``EXPERIMENT_REGISTRY`` does.
    """
    # Drop the cached plugins package so its module body re-runs on import.
    for name in list(sys.modules):
        if name == "feedbax.plugins" or name.startswith("feedbax.plugins."):
            del sys.modules[name]

    calls = {"count": 0}

    plugins = importlib.import_module("feedbax.plugins")

    real_fn = plugins.discover_experiment_packages

    def counting_discover(*args, **kwargs):
        calls["count"] += 1
        return real_fn(*args, **kwargs)

    # Patch the name bound in the package namespace (what `_get_experiment_registry`
    # actually calls) and reset the lazy cache so the next access runs discovery.
    monkeypatch.setattr(plugins, "discover_experiment_packages", counting_discover)
    monkeypatch.setattr(plugins, "_EXPERIMENT_REGISTRY", None)

    # Importing the package alone must not have triggered discovery.
    assert calls["count"] == 0, (
        "Importing feedbax.plugins ran plugin discovery as an import-time side "
        "effect; this re-enters a partially initialized feedbax.config "
        "(issue ccfe63a)."
    )

    # Accessing the name triggers discovery exactly once.
    _ = plugins.EXPERIMENT_REGISTRY
    assert calls["count"] == 1
    # Cached on subsequent access.
    _ = plugins.EXPERIMENT_REGISTRY
    assert calls["count"] == 1


def test_config_globals_populated_from_base_config():
    """`feedbax.config` globals are populated after import (not empty)."""
    config = importlib.import_module("feedbax.config")

    # `STRINGS.hps_level_label_sep` is read at import time as a default argument
    # value in `feedbax.config.hyperparams`; it must be populated.
    assert hasattr(config.STRINGS, "hps_level_label_sep")
    assert config.STRINGS.hps_level_label_sep == "__"

    # Base paths are populated too.
    assert hasattr(config.PATHS, "base")


def test_public_custody_loader_survives_downstream_plugin_import_order(tmp_path: Path):
    """A reentrant downstream import must retain the public custody loader.

    This reproduces RLRMP's startup shape in a clean interpreter: config plugin
    discovery imports a downstream training module which imports the public
    ``feedbax.training`` loader before config initialization has completed.
    The loader is a public contract and may not disappear merely because later,
    optional training-package exports are temporarily unavailable.
    """

    package_root = tmp_path / "rlrmp"
    train_root = package_root / "train"
    train_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (train_root / "__init__.py").write_text("", encoding="utf-8")
    (train_root / "cs_nominal_gru.py").write_text(
        "from feedbax.training import load_checkpoint_custody_documents\n"
        "assert callable(load_checkpoint_custody_documents)\n",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import sys

sys.path.insert(0, {str(tmp_path)!r})
from feedbax.plugins import discovery


class RlrmpEntryPoint:
    name = "rlrmp"

    def load(self):
        import rlrmp.train.cs_nominal_gru
        return lambda registry: None


discovery.feedbax_plugin_entry_points = lambda _group: [RlrmpEntryPoint()]
import feedbax.config
from feedbax.training import load_checkpoint_custody_documents

assert callable(load_checkpoint_custody_documents)
"""
    env = dict(os.environ)
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + previous if previous else "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
