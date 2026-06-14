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
            "feedbax.objectives.service",
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


def test_task_objective_training_boundaries_use_canonical_modules() -> None:
    payload = _run_import_probe(
        """
        import importlib
        import importlib.util
        import json

        canonical_modules = [
            "feedbax.tasks",
            "feedbax.tasks.presets",
            "feedbax.tasks.timeline_masks",
            "feedbax.objectives.loss",
            "feedbax.objectives.spec",
            "feedbax.objectives.streaming",
            "feedbax.objectives.service",
            "feedbax.training.trainer",
        ]
        old_root_modules = [
            "feedbax.task",
            "feedbax.task_presets",
            "feedbax.task_timeline_masks",
            "feedbax.loss",
            "feedbax.objective_spec",
            "feedbax.streaming_loss",
            "feedbax.loss_service",
            "feedbax.train",
        ]

        loaded = sorted(importlib.import_module(name).__name__ for name in canonical_modules)
        rejected = sorted(name for name in old_root_modules if importlib.util.find_spec(name) is None)

        print(json.dumps({"loaded": loaded, "rejected": sorted(rejected)}))
        """
    )

    assert payload["loaded"] == [
        "feedbax.objectives.loss",
        "feedbax.objectives.service",
        "feedbax.objectives.spec",
        "feedbax.objectives.streaming",
        "feedbax.tasks",
        "feedbax.tasks.presets",
        "feedbax.tasks.timeline_masks",
        "feedbax.training.trainer",
    ]
    assert payload["rejected"] == [
        "feedbax.loss",
        "feedbax.loss_service",
        "feedbax.objective_spec",
        "feedbax.streaming_loss",
        "feedbax.task",
        "feedbax.task_presets",
        "feedbax.task_timeline_masks",
        "feedbax.train",
    ]
