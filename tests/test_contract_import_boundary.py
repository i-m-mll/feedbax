from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


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
    )

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_importing_feedbax_does_not_load_web_package() -> None:
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
        print(json.dumps({"web_modules": web_modules}))
        """
    )

    assert payload["web_modules"] == []


def test_core_training_contract_imports_do_not_load_web_package() -> None:
    payload = _run_import_probe(
        """
        import importlib
        import json
        import sys

        for module_name in [
            "feedbax.contracts",
            "feedbax.contracts.graph",
            "feedbax.contracts.training",
            "feedbax.graph_templates",
            "feedbax.loss_service",
            "feedbax.provider",
            "feedbax.serialization",
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


def test_obsolete_web_alias_modules_are_absent() -> None:
    payload = _run_import_probe(
        """
        import importlib
        import importlib.util
        import json

        canonical_modules = [
            "feedbax.graph_normalization",
            "feedbax.serialization",
            "feedbax.loss_service",
            "feedbax.component_registry",
        ]
        for module_name in canonical_modules:
            importlib.import_module(module_name)

        obsolete_aliases = [
            "feedbax.web.graph_normalization",
            "feedbax.web.serialization",
            "feedbax.web.services.loss_service",
            "feedbax.web.services.component_registry",
        ]
        alias_specs = {
            module_name: importlib.util.find_spec(module_name) is not None
            for module_name in obsolete_aliases
        }
        print(json.dumps({"alias_specs": alias_specs}))
        """
    )

    assert not any(payload["alias_specs"].values())
