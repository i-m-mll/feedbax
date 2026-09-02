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
            "feedbax.compiler.templates",
            "feedbax.integrations.provider",
            "feedbax.objectives.service",
            "feedbax.compiler.serialization",
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
        from feedbax.execution.records import Invocation
        from feedbax.integrations import provider
        from feedbax.orchestration.realization import Attempt, BackendPlan
        from feedbax.studio import execution as studio_execution
        from feedbax.studio import protocol as studio_protocol
        from feedbax.studio import schema as studio_schema

        canonical_modules = [
            "feedbax.contracts.artifact_schema",
            "feedbax.contracts.manifest",
            "feedbax.contracts.migrations",
            "feedbax.contracts.retention_artifact_schema",
            "feedbax.execution.records",
            "feedbax.orchestration.realization",
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
            "Invocation": Invocation.__module__,
            "BackendPlan": BackendPlan.__module__,
            "Attempt": Attempt.__module__,
        }
        print(json.dumps(checks, sort_keys=True))
        """
    )

    assert payload["modules"] == [
        "feedbax.contracts.artifact_schema",
        "feedbax.contracts.manifest",
        "feedbax.contracts.migrations",
        "feedbax.contracts.retention_artifact_schema",
        "feedbax.execution.records",
        "feedbax.integrations.provider",
        "feedbax.orchestration.realization",
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
    assert payload["Invocation"] == "feedbax.execution.records"
    assert payload["BackendPlan"] == "feedbax.orchestration.realization"
    assert payload["Attempt"] == "feedbax.orchestration.realization"


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
            "feedbax.compiler.normalization",
            "feedbax.compiler.serialization",
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


def test_plugin_facade_lazy_exports_resolve_to_their_owning_modules() -> None:
    payload = _run_import_probe(
        """
        import importlib
        import json

        facade = importlib.import_module("feedbax.plugins")

        star_namespace: dict[str, object] = {}
        exec("from feedbax.plugins import *", star_namespace)

        resolved = {}
        for name in facade.__all__:
            attribute = getattr(facade, name)
            owner = facade._PUBLIC_ATTR_MODULES[name]
            owning_module = importlib.import_module(owner, "feedbax.plugins")
            resolved[name] = (
                getattr(owning_module, name) is attribute
                and star_namespace.get(name, object()) is attribute
            )

        submodules = {
            name: getattr(facade, name).__name__
            for name in facade._PUBLIC_SUBMODULES
        }

        try:
            facade.definitely_not_a_plugin_export
        except AttributeError as error:
            unknown_attribute_error = str(error)
        else:
            unknown_attribute_error = ""

        print(json.dumps({
            "unmapped": sorted(set(facade.__all__) - set(facade._PUBLIC_ATTR_MODULES)),
            "unexported": sorted(set(facade._PUBLIC_ATTR_MODULES) - set(facade.__all__)),
            "unresolved": sorted(name for name, ok in resolved.items() if not ok),
            "missing_from_dir": sorted(set(facade.__all__) - set(dir(facade))),
            "submodules": submodules,
            "unknown_attribute_error": unknown_attribute_error,
        }))
        """
    )

    assert payload["unmapped"] == []
    assert payload["unexported"] == []
    assert payload["unresolved"] == []
    assert payload["missing_from_dir"] == []
    assert payload["submodules"] == {
        "application": "feedbax.plugins.application",
        "bootstrap": "feedbax.plugins.bootstrap",
        "registry": "feedbax.plugins.registry",
    }
    assert payload["unknown_attribute_error"] == (
        "module 'feedbax.plugins' has no attribute 'definitely_not_a_plugin_export'"
    )
