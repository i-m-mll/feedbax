"""Observe Feedbax modules imported in delivered Host processes."""

import json
import os
from dataclasses import replace
from pathlib import Path
from textwrap import dedent


_RETAINED_MODULES = (
    "feedbax.component_registry.retained",
    "feedbax.component_migrations.tasks",
    "feedbax.objectives.loss",
    "feedbax.objectives.service",
    "feedbax.objectives.streaming",
    "feedbax.plugins.retained",
    "feedbax.runtime.retained_observables",
    "feedbax.tasks.presets",
    "feedbax.tasks.reach_components",
    "feedbax.tasks.task",
    "feedbax.xabdeef.losses",
    "feedbax.training.invocation",
    "feedbax.training.retained_",
    "feedbax.training.trainer",
    "feedbax.web.worker.execution",
)


def retained_implementations(modules):
    return {module for module in modules if module.startswith(_RETAINED_MODULES)}


def probe_managed_workers(service, directory):
    directory.mkdir()
    (directory / "sitecustomize.py").write_text(dedent("""
        import atexit
        import json
        import os
        import sys
        from pathlib import Path

        @atexit.register
        def report_imports():
            if job := os.environ.get("FEEDBAX_HOST_JOB_ID"):
                report = Path(__file__).parent / f"{job}.json"
                report.write_text(json.dumps(sorted(
                    module for module in sys.modules if module.startswith("feedbax.")
                )))
    """))
    launch = service._training_worker
    repo = Path(__file__).resolve().parents[2]
    service._training_worker = replace(
        launch,
        env=launch.env | {
            "PYTHONPATH": os.pathsep.join((str(directory), str(repo))),
        },
    )


def check_managed_worker(directory, job_id, *, required):
    modules = set(json.loads((directory / f"{job_id}.json").read_text()))
    assert required in modules
    assert "feedbax.tasks.document" in modules
    assert "feedbax.objectives.document" in modules
    assert not (unexpected := retained_implementations(modules)), sorted(unexpected)
