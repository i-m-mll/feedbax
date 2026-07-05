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
            "feedbax.contracts.graphs.templates",
            "feedbax.integrations.provider",
            "feedbax.objectives.service",
            "feedbax.contracts.graphs.serialization",
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
            "ValueSchema": value_schema.ValueSchema.__module__,
        }))
        """
    )

    assert payload["studio_modules"] == []
    assert payload["ValueSchema"] == "feedbax.contracts.value_schema"


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


def test_canonical_contract_studio_provider_and_execution_imports() -> None:
    payload = _run_import_probe(
        """
        import importlib
        import json

        from feedbax.contracts import artifact_schema
        from feedbax.contracts import manifest
        from feedbax.contracts import migrations
        from feedbax.contracts import retention_artifact_schema
        from feedbax.execution.backends import render_modal_app
        from feedbax.execution.models import ExecutionPlan, ExecutionSpec, LocalExecutionResult
        from feedbax.execution.planning import default_feedbax_sources, prepare_execution_plan
        from feedbax.integrations import provider
        from feedbax.execution.local import run_local_execution
        from feedbax.studio import execution as studio_execution
        from feedbax.studio import protocol as studio_protocol
        from feedbax.studio import schema as studio_schema

        canonical_modules = [
            "feedbax.contracts.artifact_schema",
            "feedbax.contracts.manifest",
            "feedbax.contracts.migrations",
            "feedbax.contracts.retention_artifact_schema",
            "feedbax.execution.backends",
            "feedbax.execution.models",
            "feedbax.execution.planning",
            "feedbax.execution.local",
            "feedbax.integrations.provider",
            "feedbax.studio.execution",
            "feedbax.studio.protocol",
            "feedbax.studio.schema",
        ]

        checks = {
            "modules": sorted(importlib.import_module(name).__name__ for name in canonical_modules),
            "ArrayStorePayload": artifact_schema.ArrayStorePayload.__module__,
            "ModelArtifactManifest": manifest.ModelArtifactManifest.__module__,
            "SpecSchemaRegistry": migrations.SpecSchemaRegistry.__module__,
            "default_spec_registry": migrations.default_spec_registry.__module__,
            "provider_manifest": provider.provider_manifest.__module__,
            "RETENTION_PLAN_SCHEMA_ID": retention_artifact_schema.RETENTION_PLAN_SCHEMA_ID,
            "prepare_studio_training_execution": (
                studio_execution.prepare_studio_training_execution.__module__
            ),
            "parse_positive_n_steps": studio_protocol.parse_positive_n_steps.__module__,
            "enumerate_studio_schema_registry": (
                studio_schema.enumerate_studio_schema_registry.__module__
            ),
            "ExecutionPlan": ExecutionPlan.__module__,
            "ExecutionSpec": ExecutionSpec.__module__,
            "LocalExecutionResult": LocalExecutionResult.__module__,
            "default_feedbax_sources": default_feedbax_sources.__module__,
            "prepare_execution_plan": prepare_execution_plan.__module__,
            "render_modal_app": render_modal_app.__module__,
            "run_local_execution": run_local_execution.__module__,
        }
        print(json.dumps(checks, sort_keys=True))
        """
    )

    assert payload["modules"] == [
        "feedbax.contracts.artifact_schema",
        "feedbax.contracts.manifest",
        "feedbax.contracts.migrations",
        "feedbax.contracts.retention_artifact_schema",
        "feedbax.execution.backends",
        "feedbax.execution.local",
        "feedbax.execution.models",
        "feedbax.execution.planning",
        "feedbax.integrations.provider",
        "feedbax.studio.execution",
        "feedbax.studio.protocol",
        "feedbax.studio.schema",
    ]
    assert payload["ArrayStorePayload"] == "feedbax.contracts.artifact_schema"
    assert payload["ModelArtifactManifest"] == "feedbax.contracts.manifest"
    assert payload["SpecSchemaRegistry"] == "feedbax.contracts.migrations"
    assert payload["default_spec_registry"] == "feedbax.contracts.migrations"
    assert payload["provider_manifest"] == "feedbax.integrations.provider"
    assert payload["RETENTION_PLAN_SCHEMA_ID"] == "feedbax.manifest.training.retention_plan"
    assert payload["prepare_studio_training_execution"] == "feedbax.studio.execution"
    assert payload["parse_positive_n_steps"] == "feedbax.studio.protocol"
    assert payload["enumerate_studio_schema_registry"] == "feedbax.studio.schema"
    assert payload["ExecutionPlan"] == "feedbax.execution.models"
    assert payload["ExecutionSpec"] == "feedbax.execution.models"
    assert payload["LocalExecutionResult"] == "feedbax.execution.models"
    assert payload["default_feedbax_sources"] == "feedbax.execution.planning"
    assert payload["prepare_execution_plan"] == "feedbax.execution.planning"
    assert payload["render_modal_app"] == "feedbax.execution.backends"
    assert payload["run_local_execution"] == "feedbax.execution.local"


def test_residual_root_compatibility_facades_are_absent() -> None:
    payload = _run_import_probe(
        """
        import importlib.util
        import json

        obsolete_facades = [
            "feedbax.artifact_schema",
            "feedbax.cloud_backends",
            "feedbax.execution_models",
            "feedbax.execution_plan",
            "feedbax.manifest",
            "feedbax.migrations",
            "feedbax.provider",
            "feedbax.retention_artifact_schema",
            "feedbax.studio_execution",
            "feedbax.studio_protocol",
            "feedbax.studio_schema",
            "feedbax.local_execution",
        ]
        facade_specs = {
            module_name: importlib.util.find_spec(module_name) is not None
            for module_name in obsolete_facades
        }
        print(json.dumps({"facade_specs": facade_specs}, sort_keys=True))
        """
    )

    assert not any(payload["facade_specs"].values())


def test_obsolete_web_alias_modules_are_absent() -> None:
    payload = _run_import_probe(
        """
        import importlib
        import importlib.util
        import json

        canonical_modules = [
            "feedbax.contracts.graphs.normalization",
            "feedbax.contracts.graphs.serialization",
            "feedbax.objectives.service",
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
