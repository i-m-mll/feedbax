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
            "feedbax.contracts.artifact_schema",
            "feedbax.contracts.graph",
            "feedbax.contracts.manifest",
            "feedbax.contracts.migrations",
            "feedbax.contracts.retention_artifact_schema",
            "feedbax.contracts.training",
            "feedbax.graph_templates",
            "feedbax.integrations.provider",
            "feedbax.objectives.service",
            "feedbax.serialization",
            "feedbax.studio.execution",
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


def test_root_compatibility_wrappers_export_canonical_objects() -> None:
    payload = _run_import_probe(
        """
        import json

        from feedbax import artifact_schema as root_artifact_schema
        from feedbax import manifest as root_manifest
        from feedbax import migrations as root_migrations
        from feedbax import provider as root_provider
        from feedbax import retention_artifact_schema as root_retention_artifact_schema
        from feedbax import studio_execution as root_studio_execution
        from feedbax import studio_protocol as root_studio_protocol
        from feedbax import studio_schema as root_studio_schema
        from feedbax.contracts import artifact_schema
        from feedbax.contracts import manifest
        from feedbax.contracts import migrations
        from feedbax.contracts import retention_artifact_schema
        from feedbax.integrations import provider
        from feedbax.studio import execution as studio_execution
        from feedbax.studio import protocol as studio_protocol
        from feedbax.studio import schema as studio_schema

        checks = {
            "ArrayStorePayload": root_artifact_schema.ArrayStorePayload
            is artifact_schema.ArrayStorePayload,
            "ModelArtifactManifest": root_manifest.ModelArtifactManifest
            is manifest.ModelArtifactManifest,
            "SpecSchemaRegistry": root_migrations.SpecSchemaRegistry
            is migrations.SpecSchemaRegistry,
            "default_spec_registry": root_migrations.default_spec_registry
            is migrations.default_spec_registry,
            "provider_manifest": root_provider.provider_manifest
            is provider.provider_manifest,
            "RETENTION_PLAN_SCHEMA_ID": root_retention_artifact_schema.RETENTION_PLAN_SCHEMA_ID
            == retention_artifact_schema.RETENTION_PLAN_SCHEMA_ID,
            "prepare_studio_training_execution": (
                root_studio_execution.prepare_studio_training_execution
                is studio_execution.prepare_studio_training_execution
            ),
            "parse_positive_n_steps": root_studio_protocol.parse_positive_n_steps
            is studio_protocol.parse_positive_n_steps,
            "enumerate_studio_schema_registry": (
                root_studio_schema.enumerate_studio_schema_registry
                is studio_schema.enumerate_studio_schema_registry
            ),
        }
        print(json.dumps(checks, sort_keys=True))
        """
    )

    assert all(payload.values()), payload
