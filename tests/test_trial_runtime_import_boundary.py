"""Observe runtime imports from the production document and retained entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

from feedbax.host import HostClient
from feedbax.jobs import JobService
from feedbax.plugins.application import new_application_registry_bundle
from tests.support.managed_import_probe import retained_implementations
from tests.test_contract_import_boundary import _run_import_probe
from tests.test_training_run_executor import _synthetic_cli_run_spec


def test_entry_point_import_closures(tmp_path):
    service = JobService(HostClient(tmp_path / "host"), new_application_registry_bundle())
    launch = service._training_worker
    managed = _run_import_probe(
        f"""
        import importlib
        import json
        import sys
        from pathlib import Path

        for name in {launch.preload!r}:
            importlib.import_module(name)
        for reference in ({launch.warm_entrypoint!r}, "feedbax.training.evaluate_worker:main"):
            module, _, function = reference.partition(":")
            assert callable(getattr(importlib.import_module(module), function))

        from fastapi.testclient import TestClient
        from feedbax.web.app import create_app

        with TestClient(create_app(projects_dir=Path({str(tmp_path / 'projects')!r}))) as client:
            assert any(
                route.path == "/api/execution/task-trials/sample"
                for route in client.app.routes
            )
            print(json.dumps(sorted(name for name in sys.modules if name.startswith("feedbax."))))
        """
    )

    spec = tmp_path / "envelope.json"
    spec.write_text(_synthetic_cli_run_spec().model_dump_json())
    retained_roots = {
        "envelope": f"""
            from feedbax.cli import main
            assert main([
                "execute-training-run-spec", "--plugin", "tests.support.training_fixtures",
                {str(spec)!r}, "--manifest-root", {str(tmp_path / 'envelope-output')!r},
            ]) == 0
        """,
        "orchestrate": """
            from feedbax.bin.orchestrate import main
            # The empty status path still runs the actual retained bootstrap and dispatcher.
            assert main(["status", "--run-set", "unknown-run-set"]) != 0
        """,
        "direct": f"""
            from feedbax.training.executor import execute_training_run_spec
            from tests.test_training_run_executor import (
                test_explicit_numerical_registration_executes_zero_gradient_step,
            )
            assert callable(execute_training_run_spec)
            Path({str(tmp_path / "direct-output")!r}).mkdir()
            test_explicit_numerical_registration_executes_zero_gradient_step(
                Path({str(tmp_path / 'direct-output')!r})
            )
        """,
        "gradient": """
            from tests.test_parameter_constraints import test_nested_parameter_constraints_survive_gradient_updates
            test_nested_parameter_constraints_survive_gradient_updates("supervised")
        """,
    }
    retained = {}
    for name, source in retained_roots.items():
        report = tmp_path / f"{name}.json"
        script = (
            "import json, sys\nfrom pathlib import Path\n" + textwrap.dedent(source) +
            f"\nPath({str(report)!r}).write_text(json.dumps(sorted("
            "name for name in sys.modules if name.startswith('feedbax.'))))\n"
        )
        run = subprocess.run(
            [sys.executable, "-c", script],
            env=os.environ | {"FEEDBAX_ORCHESTRATION_ROOT": str(tmp_path / "runsets")},
            capture_output=True,
            text=True,
            timeout=25,
        )
        assert run.returncode == 0, f"{name}\n{run.stdout}\n{run.stderr}"
        retained[name] = set(json.loads(report.read_text()))

    retained_closure = set.union(*retained.values())
    assert "feedbax.training.retained_executor" in retained["envelope"] & retained["direct"]
    assert "feedbax.plugins.retained" in retained["envelope"] & retained["orchestrate"]
    assert "feedbax.training.trainer" in retained["gradient"]
    assert "feedbax.compiler.trial_execution" in managed

    retained_implementation = retained_implementations(retained_closure)
    assert not retained_implementation.intersection(managed)
    document_implementation = {
        name for name in managed
        if name.startswith((
            "feedbax.compiler.trial_execution",
            "feedbax.objectives.document",
            "feedbax.runtime.trial_execution",
            "feedbax.tasks.document",
            "feedbax.training.evaluate_worker",
            "feedbax.training.managed_studio",
        ))
    }
    assert not document_implementation.intersection(retained_closure)
