"""Bounded public production lifecycle exercised from installed wheels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
    StudioTrainingIdentityAdapter,
)
from feedbax.orchestration import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    BudgetPolicy,
    CheckEntry,
    CheckRegistry,
    CompiledExecutionRow,
    CompiledRunSet,
    DeploymentPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    LocalOrchestrationDriver,
    RowLaunchSpec,
    RunAssemblyRequest,
    RunBundle,
    RunSetStateStore,
    SchemaArtifactRef,
    StageEngine,
)
from feedbax.orchestration.revision import resolve_feedbax_revision


_COMPILER_ID = "feedbax-external-conformance.local-lifecycle"
_COMPILER_VERSION = f"{_COMPILER_ID}.v1"
_RUN_SET_ID = "external-conformance-local"
_LOCAL_LIFECYCLE_SCRIPT = "print('feedbax external conformance lifecycle')"


@dataclass(frozen=True)
class _LocalLifecycleCompiler:
    """Incubated fixture compiler for one deterministic bounded local row."""

    def compile(
        self,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        del context
        payload = dict(authored)
        return CompiledRunSet(
            rows=[
                CompiledExecutionRow(
                    row_id=f"{run_set_id}-row",
                    payload=payload,
                    resolved_semantics=payload,
                    launch=RowLaunchSpec(
                        command=[
                            sys.executable,
                            "-c",
                            _LOCAL_LIFECYCLE_SCRIPT,
                        ],
                    ),
                )
            ]
        )


def _request(root: Path) -> tuple[RunAssemblyRequest, AssemblyContext, AssemblyCompilerRegistry]:
    authored = {
        "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
        "total_batches": 1,
        "training_config": {"fixture": "bounded-local-lifecycle-v1"},
    }
    authored_bytes = json.dumps(
        authored,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    authored_path = root / "authored.json"
    authored_path.write_bytes(authored_bytes)
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
            schema_version=STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
            artifact_id="fixture:bounded-local-lifecycle-v1",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler={
            "compiler_id": _COMPILER_ID,
            "compiler_version": _COMPILER_VERSION,
        },
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        environment=EnvironmentDeclaration(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}"
        ),
        launch_policy=LaunchPolicy(max_parallel_rows=1),
        budget=BudgetPolicy(max_wall_clock_seconds=10.0, max_spend_usd=0.0),
        orchestration_root=str(root / "orchestration"),
    )
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        compiler_id=_COMPILER_ID,
        compiler_version=_COMPILER_VERSION,
        compiler=_LocalLifecycleCompiler(),
        identity_adapter=StudioTrainingIdentityAdapter(),
    )
    return request, AssemblyContext(custody_root=root / "custody"), registry


def _driver(root: Path, _bundle: RunBundle) -> LocalOrchestrationDriver:
    return LocalOrchestrationDriver(
        cwd=root,
        python_executable=sys.executable,
    )


def check_public_lifecycle_recovery() -> bool:
    """Run and resume the installed-wheel local production lifecycle."""
    with tempfile.TemporaryDirectory(prefix="feedbax-external-conformance-") as temporary:
        root = Path(temporary).resolve()
        request, context, registry = _request(root)
        store = RunSetStateStore(root / "orchestration" / _RUN_SET_ID / "state.json")
        checks = CheckRegistry(
            {
                "external_fixture": lambda _row: CheckEntry(
                    check_id="external_fixture",
                    status="pass",
                )
            }
        )
        initial_engine = StageEngine.from_request(
            request,
            context=context,
            registry=registry,
            driver_factory=lambda bundle: _driver(root, bundle),
            run_set_id=_RUN_SET_ID,
            store=store,
            conformance_registry=checks,
        )
        first = initial_engine.run(stop_after_stage="PREFLIGHT")
        assert initial_engine.bundle is not None
        expected_command = [sys.executable, "-c", _LOCAL_LIFECYCLE_SCRIPT]
        if initial_engine.bundle.rows[0].launch.command != expected_command:
            raise AssertionError(
                "bounded lifecycle child command drifted from print-only execution"
            )
        revision_check = next(
            check
            for check in first.stage("PREFLIGHT").checks
            if check.name == "feedbax-revision-pin"
        )
        installed_revision = resolve_feedbax_revision()
        if revision_check.status != "pass" or revision_check.observed != installed_revision:
            raise AssertionError("installed-wheel revision gate did not authenticate PREFLIGHT")
        persisted = store.load()
        if persisted.stage("ASSEMBLE").attempts != 1 or persisted.stage("PREFLIGHT").attempts != 1:
            raise AssertionError(
                "public lifecycle state was not persisted at the recovery boundary"
            )

        recovered = StageEngine.from_request(
            request,
            context=context,
            registry=registry,
            driver_factory=lambda bundle: _driver(root, bundle),
            run_set_id=_RUN_SET_ID,
            store=store,
            conformance_registry=checks,
        ).run()
        if recovered.stage("ASSEMBLE").attempts != 1:
            raise AssertionError("recovery reran completed ASSEMBLE state")
        if recovered.stage("PREFLIGHT").attempts != 1:
            raise AssertionError("recovery reran completed PREFLIGHT state")
        if recovered.stage("LAUNCH").status != "completed":
            raise AssertionError("recovered lifecycle did not execute LAUNCH")
        if recovered.stage("REGISTER").status != "completed":
            raise AssertionError("recovered lifecycle did not finish registration")
        if recovered.rows[f"{_RUN_SET_ID}-row"].status != "completed":
            raise AssertionError("bounded local lifecycle row did not complete")
        if store.load() != recovered:
            raise AssertionError("recovered lifecycle result was not persisted")
    return True


__all__ = ["check_public_lifecycle_recovery"]
