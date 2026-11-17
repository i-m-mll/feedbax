from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.feedbax_contract


def _run_import_probe(source: str) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    pythonpath = str(repo_root)
    if env.get("PYTHONPATH"):
        pythonpath = os.pathsep.join([pythonpath, env["PYTHONPATH"]])
    env["PYTHONPATH"] = pythonpath

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_importing_feedbax_defers_web_and_plotly() -> None:
    payload = _run_import_probe(
        """
        import importlib
        import json
        import sys

        importlib.import_module("feedbax")
        web_modules = sorted(
            name for name in sys.modules
            if name == "feedbax.web" or name.startswith("feedbax.web.")
        )
        print(json.dumps({"web_modules": web_modules, "plotly_loaded": "plotly" in sys.modules}))
        """
    )

    assert payload == {"web_modules": [], "plotly_loaded": False}


def test_core_training_contract_imports_do_not_load_web_package() -> None:
    payload = _run_import_probe(
        """
        import importlib
        import json
        import sys

        for module_name in [
            "feedbax.contracts",
            "feedbax.contracts.artifact_schema",
            "feedbax.contracts.graph",
            "feedbax.contracts.manifest",
            "feedbax.contracts.migrations",
            "feedbax.contracts.retention_artifact_schema",
            "feedbax.contracts.training",
            "feedbax.builtins.templates.templates",
            "feedbax.integrations.provider",
            "feedbax.objectives.service",
            "feedbax.compiler.serialization",
            "feedbax.studio.protocol",
            "feedbax.studio.schema",
        ]:
            importlib.import_module(module_name)

        web_modules = sorted(
            name for name in sys.modules
            if name == "feedbax.web" or name.startswith("feedbax.web.")
        )
        print(json.dumps({"web_modules": web_modules}))
        """
    )

    assert payload["web_modules"] == []


def test_value_schema_contract_import_does_not_load_studio_package() -> None:
    payload = _run_import_probe(
        """
        import importlib.util
        import json
        import sys

        module_path = "feedbax/contracts/value_schema.py"
        spec = importlib.util.spec_from_file_location(
            "feedbax.contracts.value_schema",
            module_path,
        )
        value_schema = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(value_schema)

        studio_modules = sorted(
            name for name in sys.modules
            if name == "feedbax.studio" or name.startswith("feedbax.studio.")
        )
        print(json.dumps({
            "studio_modules": studio_modules,
        }))
        """
    )

    assert payload["studio_modules"] == []


def test_plugin_facade_import_defers_registry_and_analysis_subsystems() -> None:
    """The guaranteed downstream entry point must stay cheap to import."""
    payload = _run_import_probe(
        """
        import importlib
        import json
        import sys

        importlib.import_module("feedbax.plugins")

        feedbax_modules = sorted(
            name for name in sys.modules
            if name == "feedbax" or name.startswith("feedbax.")
        )
        sentinels = [
            "jax",
            "diffrax",
            "equinox",
            "plotly",
            "feedbax.plugins.application",
            "feedbax.component_registry",
            "feedbax.analysis",
            "feedbax.contracts",
            "feedbax.orchestration",
            "feedbax.plot",
            "feedbax.training",
        ]
        print(json.dumps({
            "feedbax_modules": feedbax_modules,
            "loaded_sentinels": [name for name in sentinels if name in sys.modules],
        }))
        """
    )

    assert payload["feedbax_modules"] == ["feedbax", "feedbax.plugins"]
    assert payload["loaded_sentinels"] == []
