from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest
from pydantic import ValidationError

from feedbax.orchestration import AuthorizedBatchStop, RowConformanceRuntimeInputs
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.manifest import ParentRef, TrainingRunManifest
from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowProvenance
from feedbax.contracts.spec_storage import (
    build_resolved_semantics_snapshot,
    training_run_execution_hash,
    training_spec_canonical_bytes,
    training_spec_sha256,
)
from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
    StudioTrainingAssemblySpec,
    StudioTrainingIdentityAdapter,
)
from feedbax.contracts.training import (
    TRAINING_RUN_SPEC_SCHEMA_ID,
    TRAINING_RUN_SPEC_SCHEMA_VERSION,
    TRAINING_RUN_SPEC_SCHEMA_VERSION_V3,
    LrScheduleSpec,
    LossTermSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    StandardSupervisedMethodPayload,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.orchestration import conformance, revision, schedule_eval, stages
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    AssemblyInputDeclaration,
    CompiledExecutionRow,
    CompiledRunSet,
    CompilerIdentity,
    RunAssemblyRequest,
    assemble_run_bundle,
)
from feedbax.orchestration.bundle import (
    CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_ID,
    CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_VERSION,
    CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
    RUN_BUNDLE_SCHEMA_ID,
    RUN_BUNDLE_SCHEMA_VERSION,
    RUN_BUNDLE_SCHEMA_VERSION_V1,
    RUN_BUNDLE_SCHEMA_VERSION_V2,
    RUN_BUNDLE_SCHEMA_VERSION_V3,
    RUN_BUNDLE_SCHEMA_VERSION_V4,
    RUN_BUNDLE_SCHEMA_VERSION_V5,
    RUN_BUNDLE_SCHEMA_VERSION_V6,
    BudgetPolicy,
    CheckpointCustodyArchiveMaterializer,
    DeploymentPolicy,
    DeploymentResourceRequest,
    EnvironmentDeclaration,
    ExecutionIdentityEnvelope,
    ImmutableInputArtifactRef,
    ImmutableInputIdentity,
    InputCustodySource,
    InputFormatIdentity,
    LaunchPolicy,
    RepoRevision,
    ResolvedAssemblyInput,
    RunBundle,
    RunRowSpec,
    RowLaunchSpec,
    SchemaArtifactRef,
    environment_declaration_identity_projection,
    execution_identity_projection,
)
from feedbax.orchestration.conformance import (
    CheckEntry,
    CheckRegistry,
    ConformanceRowArtifacts,
    build_default_check_registry,
    run_conformance_checks,
)
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.drivers.local import (
    LocalDriverError,
    LocalOrchestrationDriver,
    _canonicalize_dependency_inventory,
    compute_environment_fingerprint,
)
from feedbax.orchestration.drivers.runpod import (
    _run_command,
    project_runpod_provision_facts,
)
from feedbax.orchestration.stages import (
    STAGE_CERTIFY,
    STAGE_ORDER,
    STAGE_PREFLIGHT,
    STAGE_PROVISION,
    STAGE_REALIZE_ENV,
    STAGE_STAGE_INPUTS,
    STAGE_TEARDOWN,
    OrchestrationStageError,
    PreflightFailed,
    StageEngine,
    _ScopedSignalSupervisor,
    run_preflight_checks,
)
from feedbax.orchestration.state import (
    RUN_SET_STATE_SCHEMA_ID,
    RUN_SET_STATE_SCHEMA_VERSION,
    RUN_SET_STATE_SCHEMA_VERSION_V1,
    RowState,
    RunSetState,
    RunSetStateStore,
    StageState,
    StateLockError,
)
from feedbax.training.diagnostics import TRAINING_DIAGNOSTICS_SCHEMA_ID, TrainingDiagnostics
from feedbax.training.interruption import CancellationAction, CancellationDecision
from feedbax.training.manifest_preflight import preflight_training_run_manifest_payloads
from feedbax.training.spec_storage import TrainingRunIdentityAdapter


class FakeDriver:
    def __init__(self, *, fail: dict[str, int] | None = None) -> None:
        self.calls: list[str] = []
        self.fail = dict(fail or {})

    def _call(self, name: str) -> None:
        self.calls.append(name)
        remaining = self.fail.get(name, 0)
        if remaining > 0:
            self.fail[name] = remaining - 1
            raise RuntimeError(f"{name} failed")

    def provision(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
        self._call("provision")
        return {"driver": "local", "provisioned": True}

    def realize_env(self, bundle: RunBundle, state: RunSetState) -> str:
        self._call("realize_env")
        return "fake-fingerprint"

    def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
        self._call("stage_inputs")
        return {"inputs": True}

    def launch_row(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> dict[str, Any]:
        self._call(f"launch:{row.row_id}")
        return {"pid": 1000 + len(self.calls)}

    def probe(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> DriverRowProbe:
        self._call(f"probe:{row.row_id}")
        return DriverRowProbe(status="completed")

    def stop_row(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> dict[str, Any]:
        self._call(f"stop:{row.row_id}")
        return {"stopped": row.row_id}

    def collect(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> dict[str, str]:
        self._call(f"collect:{row.row_id}")
        return {"payload": str(bundle.run_set_dir / row.row_id / "payload.json")}

    def teardown(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
        self._call("teardown")
        return {"torn_down": True}


class BillingFakeDriver(FakeDriver):
    def __init__(
        self,
        *,
        provision_record: Mapping[str, Any],
        probe_status: str = "completed",
    ) -> None:
        super().__init__()
        self.provision_record = dict(provision_record)
        self.probe_status = probe_status

    def provision(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
        self._call("provision")
        return dict(self.provision_record)

    def realize_env(self, bundle: RunBundle, state: RunSetState) -> str:
        self._call("realize_env")
        return json.dumps(
            {
                "image_id": self.provision_record.get("immutable_image_id"),
                "runtime": {
                    "device_kind": self.provision_record.get("gpu_model"),
                    "device_count": self.provision_record.get("gpu_count"),
                },
            },
            sort_keys=True,
        )

    def probe(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> DriverRowProbe:
        self._call(f"probe:{row.row_id}")
        return DriverRowProbe(status=self.probe_status)

    def teardown(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
        self._call("teardown")
        return {
            "driver": "runpod",
            "final_pod_inventory": {
                "scope": "provider-account",
                "verified": True,
                "observed_at": "2026-07-18T20:00:00+00:00",
                "observation_basis": "runpodctl pod list --output json",
                "outcome": "empty",
                "pod_count": 0,
                "pod_ids": [],
            },
        }


def _remote_billing_record() -> dict[str, Any]:
    return {
        "driver": "runpod",
        "provider": "runpod",
        "gpu_model": "NVIDIA GeForce RTX 4090",
        "gpu_count": 1,
        "region": "CA-MTL-1",
        "immutable_image_id": "registry.example/feedbax@sha256:" + "a" * 64,
        "billing_started_at": "1970-01-01T00:00:00+00:00",
        "hourly_rate": 1.0,
        "currency": "USD",
    }


def _deployment_policy(driver: str = "local") -> DeploymentPolicy:
    return DeploymentPolicy(
        driver=driver,
        venue="local" if driver == "local" else "remote",
        cloud_authorized=driver == "runpod",
        review_required=False,
        review_authorized=False,
        resources=DeploymentResourceRequest(
            gpu_id="NVIDIA GeForce RTX 4090" if driver == "runpod" else None,
            regions=["CA-MTL-1", "US-OR-1"] if driver == "runpod" else [],
        ),
    )


class _IdentityFakeDriver(FakeDriver):
    """Executor fixture whose emitted identity is supplied independently of ASSEMBLE."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.manifest = dict(manifest)
        self.diagnostics = dict(diagnostics)

    def launch_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> dict[str, Any]:
        outputs = super().launch_row(bundle, row, state)
        events = bundle.run_set_dir / "events"
        events.mkdir(parents=True, exist_ok=True)
        (events / f"{row.row_id}.events.jsonl").write_text(
            json.dumps(
                {
                    "run_set_id": bundle.run_set_id,
                    "row_id": row.row_id,
                    "seq": 0,
                    "emitted_at_ms": 1,
                    "type": "complete",
                    "payload": {"status": "completed"},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return outputs

    def collect(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> dict[str, str]:
        self._call(f"collect:{row.row_id}")
        collected = bundle.run_set_dir / "collected" / row.row_id
        collected.mkdir(parents=True, exist_ok=True)
        manifest_path = collected / "training_manifest.json"
        diagnostics_path = collected / "training_diagnostics.json"
        manifest_path.write_text(
            json.dumps(self.manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        diagnostics_path.write_text(
            json.dumps(self.diagnostics, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_root = collected / "checkpoints"
        checkpoint_root.mkdir()
        (checkpoint_root / "latest.json").write_text(
            '{"transaction_id":"fixture-checkpoint"}\n', encoding="utf-8"
        )
        (checkpoint_root / "manifest.json").write_text(
            '{"coordinate":{"program_step":10}}\n', encoding="utf-8"
        )
        return {
            "manifest": str(manifest_path),
            "diagnostics": str(diagnostics_path),
            "checkpoint_custody": str(checkpoint_root),
        }


@dataclass(frozen=True)
class _FixtureCompiler:
    rows: tuple[CompiledExecutionRow, ...]
    expected_input_roles: tuple[str, ...] = ()

    def compile(
        self,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        assert tuple(item.role for item in context.resolved_inputs) == self.expected_input_roles
        del authored, run_set_id
        return CompiledRunSet(rows=list(self.rows))


def _compiled_row(
    row_id: str,
    *,
    command: list[str] | None = None,
    collect: list[str] | None = None,
    run_spec: dict[str, Any] | None = None,
    immutable_inputs: list[ImmutableInputIdentity] | None = None,
) -> CompiledExecutionRow:
    payload = (
        {
            "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
            "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
            "total_batches": 1,
            "training_config": {},
        }
        if run_spec is None
        else {
            "schema_id": TRAINING_RUN_SPEC_SCHEMA_ID,
            "schema_version": TRAINING_RUN_SPEC_SCHEMA_VERSION,
            **run_spec,
        }
    )
    return CompiledExecutionRow(
        row_id=row_id,
        payload=payload,
        resolved_semantics=payload,
        immutable_inputs=immutable_inputs or [],
        launch=RowLaunchSpec(
            command=command or [sys.executable, "-c", "pass"],
            collect=collect or [],
        ),
    )


def _assembly_parts(
    tmp_path: Path,
    *,
    rows: list[CompiledExecutionRow] | None = None,
    launch_policy: LaunchPolicy | None = None,
    max_wall_clock_seconds: float = 10.0,
    max_spend_usd: float | None = None,
    run_set_id: str = "2026-01-02-deadbeef",
    python_version: str | None = "3.12",
    driver: str = "local",
    expected_input_roles: tuple[str, ...] = (),
) -> tuple[RunAssemblyRequest, AssemblyContext, AssemblyCompilerRegistry]:
    authored = {
        "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
        "total_batches": 1,
    }
    authored_bytes = training_spec_canonical_bytes(authored)
    authored_path = tmp_path / "fixture-inputs" / run_set_id / "authored.json"
    authored_path.parent.mkdir(parents=True, exist_ok=True)
    authored_path.write_bytes(authored_bytes)
    compiler_id = "feedbax.tests.orchestration-fixture"
    compiler_version = "feedbax.tests.orchestration-fixture.v1"
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
            schema_version=STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
            artifact_id=f"fixture:{run_set_id}:authored",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=compiler_id,
            compiler_version=compiler_version,
        ),
        deployment_policy=_deployment_policy(driver),
        environment=EnvironmentDeclaration(python_version=python_version),
        launch_policy=launch_policy or LaunchPolicy(max_parallel_rows=2),
        budget=BudgetPolicy(
            max_wall_clock_seconds=max_wall_clock_seconds,
            max_spend_usd=max_spend_usd,
        ),
        orchestration_root=str(tmp_path),
    )
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        compiler_id=compiler_id,
        compiler_version=compiler_version,
        compiler=_FixtureCompiler(tuple(rows or [_compiled_row("row-a")]), expected_input_roles),
        identity_adapter=StudioTrainingIdentityAdapter(),
    )
    context = AssemblyContext(custody_root=tmp_path / "fixture-custody" / run_set_id)
    return request, context, registry


def _bundle(
    tmp_path: Path,
    *,
    rows: list[CompiledExecutionRow] | None = None,
    launch_policy: LaunchPolicy | None = None,
    max_wall_clock_seconds: float = 10.0,
    max_spend_usd: float | None = None,
    run_set_id: str = "2026-01-02-deadbeef",
    python_version: str | None = "3.12",
    driver: str = "local",
) -> RunBundle:
    request, context, registry = _assembly_parts(
        tmp_path,
        rows=rows,
        launch_policy=launch_policy,
        max_wall_clock_seconds=max_wall_clock_seconds,
        max_spend_usd=max_spend_usd,
        run_set_id=run_set_id,
        python_version=python_version,
        driver=driver,
    )
    return assemble_run_bundle(
        request,
        run_set_id=run_set_id,
        context=context,
        registry=registry,
    )


def test_environment_declaration_identity_projection_classifies_every_field() -> None:
    identity_fields = {
        "python_version",
        "repo_revisions",
        "lockfile_hashes",
        "overlay_steps",
        "image_id",
    }
    operational_fields = {"metadata"}
    assert set(EnvironmentDeclaration.model_fields) == identity_fields | operational_fields

    environment = EnvironmentDeclaration(
        python_version="3.12",
        repo_revisions=[RepoRevision(path="feedbax", revision="abc123")],
        lockfile_hashes={"uv.lock": "b" * 64, "requirements.lock": "a" * 64},
        overlay_steps=["uv sync --frozen"],
        image_id="registry.example/feedbax@sha256:" + "c" * 64,
        metadata={"operator_note": "not identity"},
    )

    assert environment_declaration_identity_projection(environment) == {
        "python_version": "3.12",
        "repo_revisions": [{"path": "feedbax", "revision": "abc123", "dirty_allowed": False}],
        "lockfile_hashes": {"requirements.lock": "a" * 64, "uv.lock": "b" * 64},
        "overlay_steps": ["uv sync --frozen"],
        "image_id": "registry.example/feedbax@sha256:" + "c" * 64,
    }


def test_execution_identity_projection_classifies_every_envelope_field(tmp_path: Path) -> None:
    projected_source_fields = {
        "authored_intent",
        "resolved_snapshot",
        "execution_capsule",
        "immutable_inputs",
    }
    nonprojected_fields = {"schema_id", "schema_version", "payload", "row_provenance"}
    assert set(ExecutionIdentityEnvelope.model_fields) == (
        projected_source_fields | nonprojected_fields
    )

    envelope = _bundle(tmp_path).rows[0].execution
    projection = execution_identity_projection(envelope)

    assert set(projection) == {
        "intent_hash",
        "resolved_semantics_root_hash",
        "execution_hash",
        "input_data_identities",
    }
    assert projection == {
        "intent_hash": envelope.authored_intent.intent_hash,
        "resolved_semantics_root_hash": envelope.resolved_snapshot.root_hash,
        "execution_hash": envelope.execution_capsule.execution_hash,
        "input_data_identities": [],
    }


def test_assembly_records_the_loaded_feedbax_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned_revision = "a" * 40
    monkeypatch.setattr(
        "feedbax.orchestration.assembly.resolve_feedbax_revision", lambda: pinned_revision
    )

    assert _bundle(tmp_path).feedbax_revision == pinned_revision


def test_resolve_feedbax_revision_uses_imported_package_source_and_disables_git_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_revision = "a" * 40
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=f"{resolved_revision}\n", stderr="")

    source = Path(revision.feedbax.__file__).resolve()
    monkeypatch.setattr(revision.subprocess, "run", fake_run)

    assert revision.resolve_feedbax_revision() == resolved_revision
    assert calls == [
        (
            ["git", "-C", str(source.parent), "rev-parse", "--verify", "HEAD^{commit}"],
            {
                "capture_output": True,
                "check": True,
                "env": {
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_OPTIONAL_LOCKS": "0",
                    "LC_ALL": "C",
                    "PATH": os.defpath,
                },
                "text": True,
            },
        )
    ]


def test_run_bundle_v7_requires_feedbax_revision_pin(tmp_path: Path) -> None:
    payload = _bundle(tmp_path).model_dump(mode="json")
    payload.pop("feedbax_revision")

    with pytest.raises(ValueError, match="feedbax_revision"):
        RunBundle.model_validate(payload)


def test_preflight_fails_closed_on_missing_or_mismatched_feedbax_revision(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    missing = bundle.model_copy(update={"feedbax_revision": ""})
    mismatch = bundle.model_copy(update={"feedbax_revision": "a" * 40})

    missing_check = {entry.name: entry for entry in run_preflight_checks(missing)}[
        "feedbax-revision-pin"
    ]
    mismatch_check = {entry.name: entry for entry in run_preflight_checks(mismatch)}[
        "feedbax-revision-pin"
    ]
    matching_check = {entry.name: entry for entry in run_preflight_checks(bundle)}[
        "feedbax-revision-pin"
    ]

    assert missing_check.status == "fail"
    assert "full lowercase Git commit" in missing_check.detail
    assert mismatch_check.status == "fail"
    assert "mismatch" in mismatch_check.detail
    assert matching_check.status == "pass"
    assert matching_check.observed == bundle.feedbax_revision


def test_launch_rechecks_feedbax_revision_before_calling_driver(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path).model_copy(update={"feedbax_revision": "a" * 40})
    driver = FakeDriver()
    engine = StageEngine(bundle=bundle, driver=driver)
    state = RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()})

    with pytest.raises(OrchestrationStageError, match="Feedbax revision pin mismatch"):
        engine._launch_one(bundle.rows[0], state)

    assert driver.calls == []


def test_launch_accepts_matching_feedbax_revision_pin(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    driver = FakeDriver()
    engine = StageEngine(bundle=bundle, driver=driver)
    state = RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()})

    launched = engine._launch_one(bundle.rows[0], state)

    assert driver.calls == ["launch:row-a"]
    assert launched.rows["row-a"].status == "launched"


def test_deployment_policy_does_not_change_assembled_scientific_identity(
    tmp_path: Path,
) -> None:
    request, context, registry = _assembly_parts(tmp_path)
    reviewed_policy = request.deployment_policy.model_copy(
        update={"review_required": True, "review_authorized": True}
    )
    reviewed_request = request.model_copy(update={"deployment_policy": reviewed_policy})

    ordinary = assemble_run_bundle(
        request,
        run_set_id="2026-01-02-policy-identity",
        context=context,
        registry=registry,
    )
    reviewed = assemble_run_bundle(
        reviewed_request,
        run_set_id="2026-01-02-policy-identity",
        context=context,
        registry=registry,
    )

    assert request.deployment_policy != reviewed_request.deployment_policy
    assert ordinary.deployment_policy != reviewed.deployment_policy
    assert ordinary.rows[0].execution == reviewed.rows[0].execution


def test_v3_policy_migration_evidence_survives_without_authorizing_launch(tmp_path: Path) -> None:
    source = _identity_training_payload()
    source["schema_version"] = TRAINING_RUN_SPEC_SCHEMA_VERSION_V3
    source["execution"] = {"mode": "remote", "allow_cloud": True, "require_review": False}
    migrated = default_spec_registry.migrate("TrainingRunSpec", source)
    source_bytes = training_spec_canonical_bytes(source)
    source_path = tmp_path / "training-v3.json"
    source_path.write_bytes(source_bytes)
    request, context, _ = _assembly_parts(tmp_path)
    request = request.model_copy(
        update={
            "authored": SchemaArtifactRef(
                schema_id=TRAINING_RUN_SPEC_SCHEMA_ID,
                schema_version=TRAINING_RUN_SPEC_SCHEMA_VERSION_V3,
                artifact_id="fixture:training-v3",
                sha256=hashlib.sha256(source_bytes).hexdigest(),
                uri=str(source_path),
            )
        }
    )
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=TRAINING_RUN_SPEC_SCHEMA_ID,
        compiler_id=request.compiler.compiler_id,
        compiler_version=request.compiler.compiler_version,
        compiler=_FixtureCompiler((_compiled_row("row-a", run_spec=migrated.payload),)),
        identity_adapter=TrainingRunIdentityAdapter(),
    )
    bundle = assemble_run_bundle(
        request, run_set_id="v3-evidence", context=context, registry=registry
    )
    record = bundle.migration_evidence[-1]

    assert record.metadata["removed_execution_policy"]["normalized_values"]["allow_cloud"]
    assert bundle.deployment_policy.cloud_authorized is False
    assert (
        bundle.model_copy(update={"migration_evidence": []}).rows[0].execution
        == bundle.rows[0].execution
    )
    manifest_payload = preflight_training_run_manifest_payloads(source).training_spec
    assert (
        manifest_payload.migration_records[-1].metadata["removed_execution_policy"]
        == record.metadata["removed_execution_policy"]
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"cloud_authorized": False}, "cloud_authorized=true"),
        ({"venue": "local"}, "requires venue='remote'"),
    ],
)
def test_deployment_policy_validation_fails_closed(updates: dict[str, Any], message: str) -> None:
    payload = {
        **_deployment_policy("runpod").model_dump(mode="json"),
        **updates,
    }
    with pytest.raises(ValueError, match=message):
        DeploymentPolicy.model_validate(payload)


def test_pending_review_is_durable_but_blocks_provider_until_authorized(
    tmp_path: Path,
) -> None:
    pending = DeploymentPolicy(
        driver="local",
        venue="local",
        cloud_authorized=False,
        review_required=True,
        review_authorized=False,
    )
    request, context, registry = _assembly_parts(tmp_path)
    request = request.model_copy(update={"deployment_policy": pending})
    request = RunAssemblyRequest.model_validate_json(request.model_dump_json())
    bundle = assemble_run_bundle(
        request, run_set_id="pending-review", context=context, registry=registry
    )
    bundle = RunBundle.model_validate_json(bundle.model_dump_json())
    driver = FakeDriver()

    with pytest.raises(PreflightFailed, match="deployment-policy"):
        StageEngine(bundle=bundle, driver=driver).run()
    assert driver.calls == []

    authorized = DeploymentPolicy.model_validate(
        {**pending.model_dump(mode="json"), "review_authorized": True}
    )
    rebound = bundle.model_copy(update={"deployment_policy": authorized})
    assert {check.name: check for check in run_preflight_checks(rebound)}[
        "deployment-policy"
    ].status == "pass"


def _resolved_checkpoint_input(
    *, role: str = "checkpoint", digest_character: str = "d"
) -> ResolvedAssemblyInput:
    digest = digest_character * 64
    return ResolvedAssemblyInput(
        identity=ImmutableInputIdentity(
            role=role,
            kind="checkpoint-custody-archive",
            identifier=f"checkpoint:tx-{role}",
            digest={"value": digest},
        ),
        custody=InputCustodySource(
            target_role=role,
            provider=ImmutableArtifactBlobProviderSpec(),
            provider_binding="checkpoint.inputs",
            artifact=ImmutableInputArtifactRef(
                artifact_id=f"artifact://sha256/{digest}",
                sha256=digest,
                size_bytes=123,
                media_type=CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
                storage_backend="feedbax-local",
            ),
            format=InputFormatIdentity(
                format_id=CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_ID,
                format_version=CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_VERSION,
                media_type=CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
            ),
            materializer=CheckpointCustodyArchiveMaterializer(
                expected_parent_ref=ParentRef(
                    kind="TrainingCheckpointTransactionManifest",
                    id=f"tx-{role}",
                    role="training_checkpoint_custody",
                    uri=f"transactions/tx-{role}/manifest.json",
                    metadata={"manifest_sha256": "a" * 64},
                ),
                expected_transaction_root_sha256="b" * 64,
            ),
        ),
    )


def test_resolved_input_role_cannot_collide_with_row_payload_filename(
    tmp_path: Path,
) -> None:
    resolved = _resolved_checkpoint_input(role="row-a.json")
    bundle = _bundle(tmp_path)
    execution = bundle.rows[0].execution
    execution_hash = training_run_execution_hash(
        execution.resolved_snapshot.root_hash,
        [resolved.identity.model_dump(mode="json", exclude_none=True)],
    )
    row = bundle.rows[0].model_copy(
        update={
            "execution": execution.model_copy(
                update={
                    "immutable_inputs": [resolved.identity],
                    "execution_capsule": execution.execution_capsule.model_copy(
                        update={"execution_hash": execution_hash}
                    ),
                }
            )
        }
    )
    colliding = bundle.model_copy(update={"rows": [row], "resolved_inputs": [resolved]})

    with pytest.raises(ValueError, match="collide with generated row payload filenames"):
        RunBundle.model_validate_json(colliding.model_dump_json())


def test_assemble_canonicalizes_resolved_input_records(tmp_path: Path) -> None:
    inputs = {
        "zeta": _resolved_checkpoint_input(role="zeta"),
        "alpha": _resolved_checkpoint_input(role="alpha", digest_character="e"),
    }
    request, context, registry = _assembly_parts(
        tmp_path,
        rows=[_compiled_row("row-a", immutable_inputs=[item.identity for item in inputs.values()])],
        expected_input_roles=("alpha", "zeta"),
    )
    request = request.model_copy(
        update={
            "inputs": [
                AssemblyInputDeclaration(
                    role=role,
                    kind="checkpoint-custody-archive",
                    locator=f"checkpoint:tx-{role}",
                )
                for role in ("zeta", "alpha")
            ]
        }
    )
    bundle = assemble_run_bundle(
        request,
        run_set_id="canonical-input-order",
        context=replace(context, input_resolver=lambda declaration: inputs[declaration.role]),
        registry=registry,
    )

    assert [item.identity.role for item in bundle.resolved_inputs] == ["alpha", "zeta"]


def test_conformance_records_declared_inapplicability() -> None:
    registry = CheckRegistry(
        {"lr_trace": lambda _row: CheckEntry(check_id="lr_trace", status="fail")}
    )

    certificate = run_conformance_checks(
        run_set_id="declared-inapplicable",
        rows=[ConformanceRowArtifacts(row_id="constant-rate")],
        registry=registry,
        declared_inapplicable={"lr_trace": "constant-rate rows have no schedule trace"},
    )

    checks = {entry.check_id: entry for entry in certificate.rows["constant-rate"].checks}
    result = checks["lr_trace"]
    assert certificate.overall == "fail"
    assert result.status == "skipped"
    assert result.detail == (
        "inapplicable-by-declaration: constant-rate rows have no schedule trace"
    )


def _scheduled_optimizer_payload() -> dict[str, Any]:
    return OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=0.1,
            total_steps=3500,
            constant_lr_iterations=500,
            warmup_init_fraction=0.1,
            cosine_annealing_alpha=0.2,
        ),
    ).model_dump(mode="json")


def _identity_training_payload() -> dict[str, Any]:
    method_payload = standard_supervised_method_payload()
    method_payload.payload = StandardSupervisedMethodPayload(
        optimizer=OptimizerSpec(
            type="adamw",
            params={"learning_rate": 0.001, "weight_decay": 0.0},
        )
    ).model_dump(mode="json")
    return TrainingRunSpec(
        graph={
            "inline": {
                "nodes": {
                    "gain": {
                        "type": "Gain",
                        "params": {"gain": 1.0},
                        "input_ports": ["input"],
                        "output_ports": ["output"],
                    }
                },
                "wires": [],
                "input_ports": ["input"],
                "output_ports": ["output"],
                "input_bindings": {"input": ("gain", "input")},
                "output_bindings": {"output": ("gain", "output")},
            }
        },
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=10, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=method_payload,
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
        checkpoint_progress={"checkpoint_interval": 5},
        metadata={"seeds": {"controller": 17}},
    ).model_dump(mode="json", exclude_none=True)


def _third_party_controller_training_payload(
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a strict TrainingRunSpec-shaped row with a third-party optimizer slot."""
    payload = _identity_training_payload()
    method_payload = payload["method_payload"]["payload"]
    method_payload["controller_optimizer"] = _scheduled_optimizer_payload()
    method_payload.pop("optimizer")
    payload["metadata"] = dict(metadata or {})
    return payload


def _schedule_context(
    *,
    schedule_origin_step: int,
    current_step: int,
    optimizer_count_at_current_step: int,
) -> dict[str, int]:
    return {
        "schedule_origin_step": schedule_origin_step,
        "current_step": current_step,
        "optimizer_count_at_current_step": optimizer_count_at_current_step,
    }


def _fixture_pass_registry() -> CheckRegistry:
    return CheckRegistry(
        {"fixture_pass": lambda _row: CheckEntry(check_id="fixture_pass", status="pass")}
    )


def _with_local_realized_proof(state: RunSetState) -> RunSetState:
    completed_at = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=1)
    started_at = completed_at - timedelta(seconds=1)
    rows = {
        row_id: row.model_copy(update={"started_at": started_at, "completed_at": completed_at})
        for row_id, row in state.rows.items()
    }
    provision_record = {"driver": "local"}
    provision = state.stage(STAGE_PROVISION).model_copy(
        update={
            "status": "completed",
            "completed_at": started_at - timedelta(seconds=1),
            "outputs": provision_record,
        }
    )

    realize_env = state.stage(STAGE_REALIZE_ENV).model_copy(
        update={
            "status": "completed",
            "completed_at": started_at,
            "outputs": {"environment_fingerprint": "fake-fingerprint"},
        }
    )
    stage_inputs = state.stage(STAGE_STAGE_INPUTS).model_copy(
        update={
            "status": "completed",
            "completed_at": started_at,
            "outputs": {
                "input_count": 0,
                "inputs": [],
                "payload_count": len(rows),
            },
        }
    )
    return state.model_copy(
        update={
            "rows": rows,
            "provision_record": provision_record,
            "environment_fingerprint": "fake-fingerprint",
            "stages": {
                **state.stages,
                STAGE_PROVISION: provision,
                STAGE_REALIZE_ENV: realize_env,
                STAGE_STAGE_INPUTS: stage_inputs,
            },
        }
    )


@pytest.mark.parametrize("tamper_surface", ["provision_record", "environment_fingerprint"])
def test_certify_rejects_resumed_top_level_observation_substitution(
    tmp_path: Path,
    tamper_surface: str,
) -> None:
    bundle = _bundle(tmp_path)
    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        conformance_registry=_fixture_pass_registry(),
    )
    state = _with_local_realized_proof(
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={"row-a": RowState(status="completed")},
        )
    )
    if tamper_surface == "provision_record":
        state = state.model_copy(update={"provision_record": {"driver": "substituted"}})
        message = "PROVISION outputs"
    else:
        state = state.model_copy(update={"environment_fingerprint": "substituted"})
        message = "REALIZE_ENV outputs"

    with pytest.raises(OrchestrationStageError, match=message):
        engine._stage_certify(state)


def test_certify_rejects_resumed_state_without_completed_stage_inputs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        conformance_registry=_fixture_pass_registry(),
    )
    state = _with_local_realized_proof(
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={"row-a": RowState(status="completed")},
        )
    )
    state = state.model_copy(
        update={
            "stages": {
                stage_id: stage
                for stage_id, stage in state.stages.items()
                if stage_id != STAGE_STAGE_INPUTS
            }
        }
    )

    with pytest.raises(OrchestrationStageError, match="completed STAGE_INPUTS authority"):
        engine._stage_certify(state)


def test_state_atomic_write_locking_and_schema_registration(tmp_path: Path) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    old = RunSetState(run_set_id="set", rows={"row": RowState(status="pending")})
    store.save(old)

    crashed_tmp = store.save(
        old.model_copy(update={"rows": {"row": RowState(status="completed")}}),
        crash_before_replace=True,
    )

    assert crashed_tmp.exists()
    assert store.load().rows["row"].status == "pending"

    with store.lock():
        with pytest.raises(StateLockError, match="active"):
            with store.lock():
                pass

    store.lock_path.write_text(json.dumps({"pid": 999999999}), encoding="utf-8")
    with pytest.raises(StateLockError, match="stale"):
        with store.lock():
            pass
    with store.lock(break_stale=True):
        assert store.lock_path.exists()

    assert default_spec_registry.resolve("RunBundle").identity == RUN_BUNDLE_SCHEMA_ID
    assert default_spec_registry.resolve("RunBundle").current_version == RUN_BUNDLE_SCHEMA_VERSION
    assert default_spec_registry.resolve("RunSetState").identity == RUN_SET_STATE_SCHEMA_ID
    assert (
        default_spec_registry.resolve("RunSetState").current_version == RUN_SET_STATE_SCHEMA_VERSION
    )
    stale_state = old.model_dump(mode="json")
    stale_state["schema_version"] = RUN_SET_STATE_SCHEMA_VERSION_V1
    store.path.write_text(json.dumps(stale_state), encoding="utf-8")
    with pytest.raises(ValidationError, match=RUN_SET_STATE_SCHEMA_VERSION_V1):
        store.load()
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate("RunSetState", stale_state)
    old_payload = _bundle(tmp_path).model_dump(mode="json")
    for old_version in (
        RUN_BUNDLE_SCHEMA_VERSION_V1,
        RUN_BUNDLE_SCHEMA_VERSION_V2,
        RUN_BUNDLE_SCHEMA_VERSION_V3,
        RUN_BUNDLE_SCHEMA_VERSION_V4,
        RUN_BUNDLE_SCHEMA_VERSION_V5,
        RUN_BUNDLE_SCHEMA_VERSION_V6,
    ):
        old_payload["schema_version"] = old_version
        with pytest.raises(UnsupportedSpecVersion, match="reassemble from a current"):
            default_spec_registry.migrate("RunBundle", old_payload)
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "RunBundle",
            {"schema_version": "feedbax.orchestration.run_bundle.v0"},
        )


def test_stage_engine_hands_typed_row_state_and_stop_authorization_to_conformance(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    row = bundle.rows[0]
    stopped = RowState(
        status="stopped",
        completed_at=stages.utc_now(),
        error="operator-stop-after-checkpoint",
    )
    runtime_inputs = RowConformanceRuntimeInputs(
        authorized_batch_stop=AuthorizedBatchStop(stop_after_batches=50)
    )
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={row.row_id: stopped},
    )
    state = _with_local_realized_proof(state).model_copy(update={"rows": {row.row_id: stopped}})

    artifacts = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        row_conformance_inputs={row.row_id: runtime_inputs},
    )._conformance_artifacts(row, state)

    assert artifacts.row_state == stopped
    assert artifacts.runtime_inputs == runtime_inputs


def test_failed_completed_certificate_can_be_reset_for_explicit_retry(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={bundle.rows[0].row_id: RowState(status="completed")},
        stages={
            STAGE_CERTIFY: stages.StageState(
                status="completed",
                attempts=1,
                started_at=stages.utc_now(),
                completed_at=stages.utc_now(),
                outputs={"overall": "fail", "certificate_sha256": "old"},
            )
        },
        certificate_ref="old-conformance.json",
    )

    reset = StageEngine(bundle=bundle, driver=FakeDriver())._reset_failed_certification(state)

    certify = reset.stage(STAGE_CERTIFY)
    assert certify.status == "pending"
    assert certify.attempts == 1
    assert certify.started_at is None
    assert certify.completed_at is None
    assert certify.outputs == {}
    assert reset.certificate_ref is None


def test_passing_completed_certificate_is_not_reset(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={bundle.rows[0].row_id: RowState(status="completed")},
        stages={
            STAGE_CERTIFY: stages.StageState(
                status="completed",
                outputs={"overall": "pass"},
            )
        },
        certificate_ref="passing-conformance.json",
    )

    assert (
        StageEngine(bundle=bundle, driver=FakeDriver())._reset_failed_certification(state) is state
    )


def test_request_engine_adopts_constructed_driver_poll_interval(tmp_path: Path) -> None:
    request, context, registry = _assembly_parts(tmp_path)
    driver = FakeDriver()
    driver.poll_interval_seconds = 7.0
    engine = StageEngine.from_request(
        request,
        context=context,
        registry=registry,
        driver_factory=lambda _bundle: driver,
    )

    engine.run(stop_after_stage="ASSEMBLE")

    assert engine.poll_interval_seconds == 7.0


@pytest.mark.parametrize("stop_after", STAGE_ORDER[:-1])
def test_stage_engine_resumes_from_every_stage_boundary(
    tmp_path: Path,
    stop_after: str,
) -> None:
    run_set_id = "2026-01-02-deadbeef"
    request, context, registry = _assembly_parts(tmp_path, run_set_id=run_set_id)
    store = RunSetStateStore(tmp_path / run_set_id / "state.json")
    first_driver = FakeDriver()
    StageEngine.from_request(
        request,
        context=context,
        registry=registry,
        driver_factory=lambda _bundle: first_driver,
        run_set_id=run_set_id,
        store=store,
        conformance_registry=_fixture_pass_registry(),
    ).run(stop_after_stage=stop_after)

    resumed_driver = FakeDriver()
    state = StageEngine.from_request(
        request,
        context=context,
        registry=registry,
        driver_factory=lambda _bundle: resumed_driver,
        run_set_id=run_set_id,
        store=store,
        conformance_registry=_fixture_pass_registry(),
    ).run()

    assert state.stage("REGISTER").status == "completed"
    if stop_after in (
        "PROVISION",
        "REALIZE_ENV",
        "STAGE_INPUTS",
        "LAUNCH",
        "MONITOR",
        "COLLECT",
        "CERTIFY",
        "TEARDOWN",
    ):
        assert "provision" not in resumed_driver.calls


def test_stage_retry_accounting_and_abort_teardown(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    retry_driver = FakeDriver(fail={"provision": 2})

    state = StageEngine(
        bundle=bundle,
        driver=retry_driver,
        store=store,
        conformance_registry=_fixture_pass_registry(),
    ).run()

    assert state.stage("PROVISION").attempts == 3
    assert retry_driver.calls.count("provision") == 3

    failing_bundle = _bundle(tmp_path / "abort", run_set_id="2026-01-02-feedface")
    failing_store = RunSetStateStore(failing_bundle.run_set_dir / "state.json")
    failing_driver = FakeDriver(fail={"realize_env": 3})

    with pytest.raises(RuntimeError, match="realize_env failed"):
        StageEngine(bundle=failing_bundle, driver=failing_driver, store=failing_store).run()

    failed_state = failing_store.load()
    assert failed_state.stage("REALIZE_ENV").attempts == 3
    assert failed_state.stage("TEARDOWN").status == "completed"
    assert "teardown" in failing_driver.calls


@pytest.mark.parametrize(
    ("collection_raises", "failure_logs_raise", "teardown_raises"),
    [(False, False, False), (True, True, True)],
)
def test_executor_failure_remains_primary_when_declared_collection_output_is_absent(
    tmp_path: Path,
    collection_raises: bool,
    failure_logs_raise: bool,
    teardown_raises: bool,
) -> None:
    class FailedExecutorDriver(FakeDriver):
        def probe(
            self,
            bundle: RunBundle,
            row: RunRowSpec,
            state: RunSetState,
        ) -> DriverRowProbe:
            self._call(f"probe:{row.row_id}")
            return DriverRowProbe(status="failed", detail="duplicate plugin registration")

        def collect(
            self,
            bundle: RunBundle,
            row: RunRowSpec,
            state: RunSetState,
        ) -> dict[str, str]:
            self._call(f"collect:{row.row_id}")
            if collection_raises:
                raise RuntimeError("manifest.json is absent")
            return {}

        def collect_failure_logs(
            self,
            bundle: RunBundle,
            state: RunSetState,
        ) -> dict[str, str]:
            self._call("collect_failure_logs")
            if failure_logs_raise:
                raise RuntimeError("failure log pull timed out")
            return {"failure_logs": str(bundle.run_set_dir / "failure-logs")}

    bundle = _bundle(
        tmp_path,
        rows=[_compiled_row("row-a", collect=["manifest.json"])],
    )
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = FailedExecutorDriver(fail={"teardown": 1} if teardown_raises else None)

    with pytest.raises(
        OrchestrationStageError,
        match="executor failed for row 'row-a': duplicate plugin registration",
    ):
        StageEngine(
            bundle=bundle,
            driver=driver,
            store=store,
            conformance_registry=_fixture_pass_registry(),
        ).run()

    failed_state = store.load()
    collect_stage = failed_state.stage("COLLECT")
    assert collect_stage.status == "failed"
    assert collect_stage.attempts == 1
    assert collect_stage.outputs["executor_failures"] == [
        {"row_id": "row-a", "error": "duplicate plugin registration"}
    ]
    evidence = collect_stage.outputs["secondary_evidence"]
    assert {
        "kind": "absent_collection_outputs",
        "row_id": "row-a",
        "missing_outputs": ["manifest.json"],
    } in evidence
    if collection_raises:
        assert {
            "kind": "collection_error_after_executor_failure",
            "row_id": "row-a",
            "detail": "manifest.json is absent",
        } in evidence
    assert failed_state.rows["row-a"].error == "duplicate plugin registration"
    assert driver.calls.count("collect:row-a") == 1
    assert driver.calls.count("collect_failure_logs") == 1
    assert driver.calls.count("teardown") == 1
    teardown_stage = failed_state.stage("TEARDOWN")
    assert teardown_stage.status == ("failed" if teardown_raises else "completed")
    diagnostic = teardown_stage.outputs["failure_log_collection"]
    assert diagnostic["status"] == ("failed" if failure_logs_raise else "completed")
    if failure_logs_raise:
        assert diagnostic["error"] == "failure log pull timed out"


def test_collection_error_for_completed_executor_remains_primary_and_retries(
    tmp_path: Path,
) -> None:
    class MissingOutputDriver(FakeDriver):
        def collect(
            self,
            bundle: RunBundle,
            row: RunRowSpec,
            state: RunSetState,
        ) -> dict[str, str]:
            self._call(f"collect:{row.row_id}")
            raise RuntimeError("manifest collection failed")

    bundle = _bundle(
        tmp_path,
        rows=[_compiled_row("row-a", collect=["manifest.json"])],
    )
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = MissingOutputDriver()

    with pytest.raises(RuntimeError, match="manifest collection failed"):
        StageEngine(
            bundle=bundle,
            driver=driver,
            store=store,
            conformance_registry=_fixture_pass_registry(),
        ).run()

    failed_state = store.load()
    assert failed_state.stage("COLLECT").attempts == 5
    assert "executor failed" not in (failed_state.stage("COLLECT").error or "")
    assert driver.calls.count("collect:row-a") == 5
    assert driver.calls.count("teardown") == 1


def test_absent_declared_output_for_completed_executor_fails_collection(
    tmp_path: Path,
) -> None:
    class MissingOutputDriver(FakeDriver):
        def collect(
            self,
            bundle: RunBundle,
            row: RunRowSpec,
            state: RunSetState,
        ) -> dict[str, str]:
            self._call(f"collect:{row.row_id}")
            return {}

    bundle = _bundle(
        tmp_path,
        rows=[_compiled_row("row-a", collect=["manifest.json"])],
    )
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = MissingOutputDriver()

    with pytest.raises(
        OrchestrationStageError,
        match="declared collection outputs are absent",
    ):
        StageEngine(
            bundle=bundle,
            driver=driver,
            store=store,
            conformance_registry=_fixture_pass_registry(),
        ).run()

    assert store.load().stage("COLLECT").attempts == 5
    assert driver.calls.count("collect:row-a") == 5
    assert driver.calls.count("teardown") == 1


def test_later_executor_failure_precedes_earlier_row_collection_error(tmp_path: Path) -> None:
    class MixedOutcomeDriver(FakeDriver):
        def probe(
            self,
            bundle: RunBundle,
            row: RunRowSpec,
            state: RunSetState,
        ) -> DriverRowProbe:
            self._call(f"probe:{row.row_id}")
            if row.row_id == "failed-row":
                return DriverRowProbe(status="failed", detail="executor payload mismatch")
            return DriverRowProbe(status="completed")

        def collect(
            self,
            bundle: RunBundle,
            row: RunRowSpec,
            state: RunSetState,
        ) -> dict[str, str]:
            self._call(f"collect:{row.row_id}")
            if row.row_id == "completed-row":
                raise RuntimeError("completed row output transport failed")
            return {}

    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row("completed-row", collect=["manifest.json"]),
            _compiled_row("failed-row", collect=["manifest.json"]),
        ],
    )
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = MixedOutcomeDriver()

    with pytest.raises(
        OrchestrationStageError,
        match="executor failed for row 'failed-row': executor payload mismatch",
    ):
        StageEngine(
            bundle=bundle,
            driver=driver,
            store=store,
            conformance_registry=_fixture_pass_registry(),
        ).run()

    failed_state = store.load()
    evidence = failed_state.stage("COLLECT").outputs["secondary_evidence"]
    assert {
        "kind": "collection_error_after_executor_failure",
        "row_id": "completed-row",
        "detail": "completed row output transport failed",
    } in evidence
    assert driver.calls.count("collect:completed-row") == 1
    assert driver.calls.count("collect:failed-row") == 1


def test_local_executor_failure_records_absent_declared_output_as_secondary(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                command=[sys.executable, "-c", "raise SystemExit(7)"],
                collect=["manifest.json"],
            )
        ],
    )
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",))

    with pytest.raises(
        OrchestrationStageError,
        match="executor failed for row 'row-a': exit=7",
    ):
        StageEngine(
            bundle=bundle,
            driver=driver,
            store=store,
            conformance_registry=_fixture_pass_registry(),
            poll_interval_seconds=0.01,
        ).run()

    failed_state = store.load()
    assert failed_state.stage("COLLECT").outputs["secondary_evidence"] == [
        {
            "kind": "absent_collection_outputs",
            "row_id": "row-a",
            "missing_outputs": ["manifest.json"],
        }
    ]
    assert failed_state.rows["row-a"].error == "exit=7"
    assert failed_state.stage("TEARDOWN").status == "completed"


def test_capped_remote_execution_requires_observed_billing_evidence(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        driver="runpod",
        max_spend_usd=1.0,
    )
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = FakeDriver()

    with pytest.raises(
        OrchestrationStageError,
        match="usable observed provider billing evidence",
    ):
        StageEngine(bundle=bundle, driver=driver, store=store, wall_time=lambda: 0.0).run()

    state = store.load()
    assert state.stage("REALIZE_ENV").status == "failed"
    assert state.stage("REALIZE_ENV").attempts == 1
    assert state.abort_reason == "budget-evidence-unavailable"
    assert state.budget_counters["budget_exceeded"] == "spend-evidence-unavailable"
    assert not any(call.startswith("launch:") for call in driver.calls)
    assert state.stage("TEARDOWN").status == "completed"
    assert "teardown" in driver.calls


def test_runpod_go_timestamp_allows_realize_env_under_spend_cap(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, driver="runpod", max_spend_usd=1.0)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    provision_record = {
        **_remote_billing_record(),
        **project_runpod_provision_facts(
            {
                "createdAt": "2026-07-19 18:05:00.898 +0000 UTC",
                "costPerHr": 0.99,
            }
        ),
    }
    driver = BillingFakeDriver(provision_record=provision_record)
    observed_at = datetime(2026, 7, 19, 18, 6, tzinfo=timezone.utc).timestamp()

    state = StageEngine(
        bundle=bundle,
        driver=driver,
        store=store,
        wall_time=lambda: observed_at,
    ).run(stop_after_stage=STAGE_REALIZE_ENV)

    assert state.stage(STAGE_PROVISION).status == "completed"
    assert state.stage(STAGE_REALIZE_ENV).status == "completed"
    assert state.provision_record["billing_started_at"] == "2026-07-19T18:05:00.898000+00:00"
    assert state.provision_record["billing_started_at_raw"] == ("2026-07-19 18:05:00.898 +0000 UTC")
    assert state.budget_counters["accrued_cost_usd"] == pytest.approx(0.99 * 59.102 / 3600)


@pytest.mark.parametrize(
    "raw_timestamp",
    [None, "not-a-timestamp", "2026-07-19T18:07:00Z"],
)
def test_runpod_unusable_timestamp_still_fails_closed_before_realize_env(
    tmp_path: Path,
    raw_timestamp: str | None,
) -> None:
    bundle = _bundle(tmp_path, driver="runpod", max_spend_usd=1.0)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    provision_record = {
        **_remote_billing_record(),
        **project_runpod_provision_facts({"createdAt": raw_timestamp, "costPerHr": 0.99}),
    }
    driver = BillingFakeDriver(provision_record=provision_record)
    observed_at = datetime(2026, 7, 19, 18, 6, tzinfo=timezone.utc).timestamp()

    with pytest.raises(OrchestrationStageError, match="billing_started_at"):
        StageEngine(
            bundle=bundle,
            driver=driver,
            store=store,
            wall_time=lambda: observed_at,
        ).run(stop_after_stage=STAGE_REALIZE_ENV)

    state = store.load()
    assert state.stage(STAGE_REALIZE_ENV).status == "failed"
    assert state.abort_reason == "budget-evidence-unavailable"
    assert "realize_env" not in driver.calls
    assert "teardown" in driver.calls


def test_capped_remote_execution_rejects_spend_at_cap(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        driver="runpod",
        max_spend_usd=1.0,
    )
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = BillingFakeDriver(provision_record=_remote_billing_record())

    with pytest.raises(stages.BudgetExceeded, match="before REALIZE_ENV"):
        StageEngine(bundle=bundle, driver=driver, store=store, wall_time=lambda: 3600.0).run()

    state = store.load()
    assert state.stage("REALIZE_ENV").attempts == 1
    assert state.budget_counters["accrued_cost_usd"] == pytest.approx(1.0)
    assert state.budget_counters["budget_exceeded"] == "spend"
    assert state.abort_reason == "budget-exceeded"
    assert not any(call.startswith("launch:") for call in driver.calls)
    assert "teardown" in driver.calls


def test_capped_remote_monitor_stops_at_observed_spend_cap(tmp_path: Path) -> None:
    @dataclass
    class Clock:
        now: float = 0.0

        def time(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += 1800.0

    clock = Clock()
    bundle = _bundle(
        tmp_path,
        driver="runpod",
        max_spend_usd=0.5,
    )
    driver = BillingFakeDriver(
        provision_record=_remote_billing_record(),
        probe_status="running",
    )

    state = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=_fixture_pass_registry(),
        wall_time=clock.time,
        sleep=clock.sleep,
        poll_interval_seconds=1.0,
    ).run()

    assert state.abort_reason == "budget-exceeded"
    assert state.budget_counters["accrued_cost_usd"] == pytest.approx(0.5)
    assert state.budget_counters["budget_exceeded"] == "spend"
    assert state.rows["row-a"].status == "stopped"
    assert "stop:row-a" in driver.calls
    assert "collect:row-a" in driver.calls
    assert "teardown" in driver.calls
    assert state.stage(STAGE_TEARDOWN).outputs["final_pod_inventory"]["verified"] is True
    assert (
        state.registration_payload["final_pod_inventory"]
        == state.stage(STAGE_TEARDOWN).outputs["final_pod_inventory"]
    )


def test_capped_remote_monitor_rejects_existing_spend_before_first_probe(
    tmp_path: Path,
) -> None:
    calls = 0

    def wall_time() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 2 else 1800.0

    bundle = _bundle(tmp_path, driver="runpod", max_spend_usd=0.5)
    driver = BillingFakeDriver(
        provision_record=_remote_billing_record(),
        probe_status="running",
    )

    state = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=_fixture_pass_registry(),
        wall_time=wall_time,
    ).run()

    assert state.abort_reason == "budget-exceeded"
    assert state.budget_counters["budget_exceeded"] == "spend"
    assert not any(call.startswith("probe:") for call in driver.calls)
    assert "stop:row-a" in driver.calls


def test_local_spend_cap_does_not_require_provider_billing_evidence(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, max_spend_usd=0.0)
    driver = FakeDriver()

    state = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=_fixture_pass_registry(),
    ).run()

    assert state.stage("REGISTER").status == "completed"
    assert "budget_exceeded" not in state.budget_counters


def test_request_assembly_certifies_all_core_checks_with_independent_identity(
    tmp_path: Path,
) -> None:
    """Prove executor identity independently agrees with ASSEMBLE, then tamper it."""
    authored = {
        "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
        "total_batches": 10,
        "training_config": {"fixture": "authored-intent-v1"},
    }
    executable_payload = _identity_training_payload()
    resolved_semantics = {
        "fixture": "resolved-semantics-v1",
        "training": executable_payload,
    }
    expected_intent_hash = "da602d442a5356281bf648ca49032739ba6255cdb427bb0da09cfa65bb4d332f"
    expected_root_hash = "4d61e465261b5115df71d07ea60d53b6121322a5527e0a033b62047fcb2310ad"
    expected_execution_hash = "1d6dd9eb9f07694aeec192fe6b2bb53785711ddad17854569ac8f63b16aec921"
    expected_artifact_hashes = {
        "authored": "e1aeb77d847c6b24011becca0db24da2f25e68f3cf542fdb4639615a266f8dc9",
        "payload": "7b32a1b9d28a5ccfd907a14904ed14c9af90074d996b0a27647d7ba1f3a3adbf",
        "snapshot": "a36d64645240d7f3742b5af8367c4ad1f145981e2d0378528945d26a2fbfec8d",
        "capsule": "a0657205c9b16c9247bf8177e588a6c48ce0a61f8940439b158f06ef83a5f14c",
    }
    assert (
        training_spec_sha256(StudioTrainingAssemblySpec.model_validate(authored).worker_payload())
        == expected_intent_hash
    )
    assert build_resolved_semantics_snapshot(resolved_semantics)["root_hash"] == expected_root_hash
    assert training_run_execution_hash(expected_root_hash, []) == expected_execution_hash
    normalized = preflight_training_run_manifest_payloads(executable_payload)

    diagnostics = {
        "completed_batches": 10,
        "checkpoint_coordinates": [5, 10],
        "lr_trace": {str(step): 0.001 for step in (0, 5, 10)},
        "optimizer_build_context": _schedule_context(
            schedule_origin_step=0,
            current_step=0,
            optimizer_count_at_current_step=0,
        ),
        "resume_context": _schedule_context(
            schedule_origin_step=0,
            current_step=0,
            optimizer_count_at_current_step=0,
        ),
        "seeds": {"controller": 17},
        "terminal_status": "completed",
    }

    def run_fixture(*, root: Path, run_set_id: str, manifest_intent_hash: str) -> RunSetState:
        authored_bytes = training_spec_canonical_bytes(authored)
        authored_path = root / "independent-authored.json"
        root.mkdir(parents=True, exist_ok=True)
        authored_path.write_bytes(authored_bytes)
        compiler_id = "feedbax.tests.identity-proof"
        compiler_version = "feedbax.tests.identity-proof.v1"
        request = RunAssemblyRequest(
            authored=SchemaArtifactRef(
                schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
                schema_version=STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
                artifact_id=f"fixture:{run_set_id}:independent-authored",
                sha256=hashlib.sha256(authored_bytes).hexdigest(),
                uri=str(authored_path),
            ),
            compiler=CompilerIdentity(
                compiler_id=compiler_id,
                compiler_version=compiler_version,
            ),
            deployment_policy=_deployment_policy(),
            environment=EnvironmentDeclaration(python_version="3.13"),
            budget=BudgetPolicy(max_wall_clock_seconds=10),
            orchestration_root=str(root),
        )
        registry = AssemblyCompilerRegistry()
        registry.register(
            schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
            compiler_id=compiler_id,
            compiler_version=compiler_version,
            compiler=_FixtureCompiler(
                (
                    CompiledExecutionRow(
                        row_id="identity-row",
                        payload=executable_payload,
                        resolved_semantics=resolved_semantics,
                        immutable_inputs=[],
                        launch=RowLaunchSpec(command=["identity-fake"]),
                    ),
                )
            ),
            identity_adapter=StudioTrainingIdentityAdapter(),
        )
        manifest = TrainingRunManifest(
            id=f"feedbax-training-run:{run_set_id}",
            metadata={
                "environment_fingerprint": "fake-fingerprint",
                "seeds": {"controller": 17},
            },
            training_spec=normalized.training_spec,
            task_spec=normalized.task_spec,
            graph_spec=normalized.graph_spec,
            summary_metrics={"completed_batches": 10},
            intent_hash=manifest_intent_hash,
            resolved_semantics_root_hash=expected_root_hash,
            execution_hash=expected_execution_hash,
            input_data_identities=[],
        ).model_dump(mode="json", exclude_none=True)
        driver = _IdentityFakeDriver(manifest=manifest, diagnostics=diagnostics)
        engine = StageEngine.from_request(
            request,
            context=AssemblyContext(custody_root=root / "assembly-custody"),
            registry=registry,
            driver_factory=lambda _bundle: driver,
            run_set_id=run_set_id,
            conformance_registry=build_default_check_registry(include_plugins=False),
        )
        return engine.run()

    passing_root = tmp_path / "passing"
    passing = run_fixture(
        root=passing_root,
        run_set_id="independent-identity-pass",
        manifest_intent_hash=expected_intent_hash,
    )
    certificate = json.loads(
        (passing_root / "independent-identity-pass" / "conformance.json").read_text(
            encoding="utf-8"
        )
    )
    checks = {check["check_id"]: check for check in certificate["rows"]["identity-row"]["checks"]}
    assembled_bundle = json.loads(
        (passing_root / "independent-identity-pass" / "bundle.json").read_text(encoding="utf-8")
    )
    execution = assembled_bundle["rows"][0]["execution"]
    assert {
        "authored": execution["authored_intent"]["sha256"],
        "payload": execution["payload"]["sha256"],
        "snapshot": execution["resolved_snapshot"]["sha256"],
        "capsule": execution["execution_capsule"]["sha256"],
    } == expected_artifact_hashes
    assert passing.stage("CERTIFY").status == "completed"
    assert passing.stage("REGISTER").status == "completed"
    assert certificate["overall"] == "pass"
    assert set(checks) == {
        "checkpoint_cadence",
        "completed_batches",
        "environment_fingerprint",
        "events_terminal",
        "execution_identity",
        "lr_trace",
        "manifest_valid",
        "realized_deployment",
        "seeds",
    }
    assert all(check["status"] == "pass" for check in checks.values())
    assert checks["execution_identity"]["expected"]
    assert checks["execution_identity"]["observed"]
    assert checks["execution_identity"]["expected"] == checks["execution_identity"]["observed"]
    assert checks["execution_identity"]["expected"] == {
        "intent_hash": expected_intent_hash,
        "resolved_semantics_root_hash": expected_root_hash,
        "execution_hash": expected_execution_hash,
        "input_data_identities": [],
    }
    realized = certificate["rows"]["identity-row"]["realized_deployment"]
    assert realized["provider"] == "local"
    assert realized["accrued_cost"] == 0.0
    assert realized["environment_fingerprint"] == "fake-fingerprint"

    tampered_root = tmp_path / "tampered"
    with pytest.raises(ValueError, match="phase=completed"):
        run_fixture(
            root=tampered_root,
            run_set_id="independent-identity-tampered",
            manifest_intent_hash="f" * 64,
        )
    tampered_state = RunSetStateStore(
        tampered_root / "independent-identity-tampered" / "state.json"
    ).load()
    tampered_certificate = json.loads(
        (tampered_root / "independent-identity-tampered" / "conformance.json").read_text(
            encoding="utf-8"
        )
    )
    tampered_checks = {
        check["check_id"]: check for check in tampered_certificate["rows"]["identity-row"]["checks"]
    }
    assert tampered_certificate["overall"] == "fail"
    assert tampered_checks["execution_identity"]["status"] == "fail"
    assert "intent_hash" in tampered_checks["execution_identity"]["detail"]
    assert tampered_state.stage("CERTIFY").status == "completed"
    assert tampered_state.stage("REGISTER").status == "failed"


def test_preflight_failures_record_named_checks_and_do_not_call_driver(tmp_path: Path) -> None:
    invalid = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                run_spec={"schema_version": "feedbax.spec.training_run.v0"},
            )
        ],
        python_version=None,
    )
    driver = FakeDriver()

    with pytest.raises(PreflightFailed):
        StageEngine(bundle=invalid, driver=driver).run()

    state = RunSetStateStore(invalid.run_set_dir / "state.json").load()
    checks = {check.name: check for check in state.stage(STAGE_PREFLIGHT).checks}
    assert checks["environment-declaration"].status == "fail"
    assert checks["manifest-payload-normalization"].status == "fail"
    assert driver.calls == []


def test_preflight_consumes_only_deployment_policy(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, driver="runpod")
    check = {entry.name: entry for entry in run_preflight_checks(bundle)}["deployment-policy"]

    assert check.status == "pass"
    assert check.observed == bundle.deployment_policy.model_dump(mode="json")


def test_preflight_rejects_registered_native_row_without_checkpoint_collection(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                command=["python", "-m", "feedbax", "execute-training-run-spec"],
                collect=["manifest.json", "training-diagnostics.json"],
            )
        ],
    )
    row = bundle.rows[0]
    bundle = bundle.model_copy(
        update={
            "rows": [
                row.model_copy(
                    update={
                        "launch": row.launch.model_copy(
                            update={"payload_routing": {"kind": "registered-execution-payload"}}
                        )
                    }
                )
            ]
        }
    )

    check = {entry.name: entry for entry in run_preflight_checks(bundle)}["native-output-custody"]

    assert check.status == "fail"
    assert check.detail == "row-a: missing ['checkpoints']"
    assert check.observed["row-a"]["required_for_registered_native_training"] == [
        "manifest.json",
        "training-diagnostics.json",
        "checkpoints",
    ]


@pytest.mark.parametrize(
    ("updates", "detail"),
    [
        ({"cloud_authorized": False}, "cloud authorization"),
        (
            {"review_required": True, "review_authorized": False},
            "review has not been explicitly authorized",
        ),
        ({"venue": "local"}, "requires venue='remote'"),
    ],
)
def test_stage_engine_rejects_invalid_deployment_policy_before_driver_calls(
    tmp_path: Path,
    updates: dict[str, Any],
    detail: str,
) -> None:
    bundle = _bundle(tmp_path, driver="runpod")
    invalid_policy = bundle.deployment_policy.__class__.model_construct(
        **{**bundle.deployment_policy.__dict__, **updates}
    )
    bundle = bundle.model_copy(update={"deployment_policy": invalid_policy})
    driver = FakeDriver()
    driver.preflight_checks = lambda _bundle: pytest.fail("driver preflight called")

    with pytest.raises(PreflightFailed, match="deployment-policy"):
        StageEngine(bundle=bundle, driver=driver).run()

    state = RunSetStateStore(bundle.run_set_dir / "state.json").load()
    check = {entry.name: entry for entry in state.stage(STAGE_PREFLIGHT).checks}[
        "deployment-policy"
    ]
    assert check.status == "fail"
    assert detail in (check.detail or "")
    assert driver.calls == []


def test_preflight_schedule_realization_uses_optimizer_builder(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                run_spec={
                    "optimizer": {
                        "type": "adamw",
                        "params": {"learning_rate": 0.001},
                    }
                },
            )
        ],
    )
    checks = {check.name: check for check in run_preflight_checks(bundle)}

    assert checks["schedule-realization"].status == "pass"
    assert checks["schedule-realization"].observed == {
        "row-a": [{"optimizer_index": 0, "scheduled": False, "points": 0}]
    }

    invalid = _bundle(
        tmp_path / "invalid",
        rows=[
            _compiled_row(
                "row-a",
                run_spec={"optimizer": {"type": "adamw", "params": {}}},
            )
        ],
        run_set_id="invalid-optimizer",
    )
    invalid_checks = {check.name: check for check in run_preflight_checks(invalid)}
    assert invalid_checks["schedule-realization"].status == "fail"
    assert "/params/learning_rate is required" in (
        invalid_checks["schedule-realization"].detail or ""
    )


def test_preflight_schedule_realization_discovers_controller_optimizer_metadata_contexts(
    tmp_path: Path,
) -> None:
    context = _schedule_context(
        schedule_origin_step=12_000,
        current_step=12_000,
        optimizer_count_at_current_step=12_000,
    )
    run_spec = _third_party_controller_training_payload(
        metadata={
            "resume_context": context,
            "optimizer_build_context": context,
        }
    )
    bundle = _bundle(
        tmp_path,
        rows=[_compiled_row("row-a", run_spec=run_spec)],
    )

    checks = {check.name: check for check in run_preflight_checks(bundle)}

    schedule_check = checks["schedule-realization"]
    assert schedule_check.status == "pass"
    row_observed = schedule_check.observed["row-a"][0]
    assert row_observed["scheduled"] is True
    assert row_observed["expected_context"] == context
    assert row_observed["observed_context"] == context
    assert len(row_observed["samples"]) >= 4


def test_preflight_discovers_nested_method_training_optimizer(tmp_path: Path) -> None:
    payload = (run_spec := _identity_training_payload())["method_payload"]["payload"]
    payload.pop("optimizer")
    payload["training"] = {
        "optimizer": {
            "type": "adamw",
            "params": {},
            "lr_schedule": LrScheduleSpec(kind="constant", learning_rate_0=0.1).model_dump(
                mode="json"
            ),
        }
    }
    bundle = _bundle(tmp_path, rows=[_compiled_row("row-a", run_spec=run_spec)])
    assert {check.name: check for check in run_preflight_checks(bundle)}[
        "schedule-realization"
    ].status == "pass"


def test_preflight_schedule_realization_requires_controller_optimizer_metadata_contexts(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                run_spec=_third_party_controller_training_payload(),
            )
        ],
    )

    checks = {check.name: check for check in run_preflight_checks(bundle)}

    schedule_check = checks["schedule-realization"]
    assert schedule_check.status == "fail"
    assert "resume_context missing" in (schedule_check.detail or "")
    assert schedule_check.observed == {"row-a": []}


def test_preflight_schedule_realization_fails_miswired_resume_before_driver(
    tmp_path: Path,
) -> None:
    declared_restart_context = _schedule_context(
        schedule_origin_step=12_000,
        current_step=12_000,
        optimizer_count_at_current_step=12_000,
    )
    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                run_spec={
                    "optimizer": _scheduled_optimizer_payload(),
                    "resume_context": declared_restart_context,
                    "optimizer_build_context": _schedule_context(
                        schedule_origin_step=0,
                        current_step=0,
                        optimizer_count_at_current_step=0,
                    ),
                },
            )
        ],
    )
    driver = FakeDriver()

    with pytest.raises(PreflightFailed):
        StageEngine(bundle=bundle, driver=driver).run()

    assert driver.calls == []
    state = RunSetStateStore(bundle.run_set_dir / "state.json").load()
    checks = {check.name: check for check in state.stage(STAGE_PREFLIGHT).checks}
    schedule_check = checks["schedule-realization"]
    assert schedule_check.status == "fail"
    assert "learning-rate mismatch" in (schedule_check.detail or "")
    row_observed = schedule_check.observed["row-a"][0]
    assert row_observed["expected_context"] == declared_restart_context
    assert row_observed["observed_context"] == {
        "schedule_origin_step": 0,
        "current_step": 0,
        "optimizer_count_at_current_step": 0,
    }
    assert len(row_observed["samples"]) >= 4
    assert row_observed["mismatches"][0]["expected"] != row_observed["mismatches"][0]["observed"]


def test_preflight_schedule_realization_passes_correct_resume_context(tmp_path: Path) -> None:
    resume_context = _schedule_context(
        schedule_origin_step=12_000,
        current_step=12_000,
        optimizer_count_at_current_step=12_000,
    )
    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                run_spec={
                    "optimizer": _scheduled_optimizer_payload(),
                    "resume_context": resume_context,
                    "optimizer_build_context": resume_context,
                },
            )
        ],
    )

    checks = {check.name: check for check in run_preflight_checks(bundle)}

    schedule_check = checks["schedule-realization"]
    assert schedule_check.status == "pass"
    row_observed = schedule_check.observed["row-a"][0]
    assert row_observed["scheduled"] is True
    assert row_observed["expected_context"] == resume_context
    assert row_observed["observed_context"] == resume_context
    assert len(row_observed["samples"]) >= 4


def test_preflight_schedule_realization_certifies_post_terminal_hold_through_run_end(
    tmp_path: Path,
) -> None:
    context = _schedule_context(
        schedule_origin_step=0,
        current_step=0,
        optimizer_count_at_current_step=0,
    )
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=0.03,
            total_steps=3500,
            constant_lr_iterations=1000,
            warmup_init_fraction=0.1,
            cosine_annealing_alpha=0.001,
        ),
    ).model_dump(mode="json")
    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                run_spec={
                    "optimizer": optimizer,
                    "n_batches": 4500,
                    "resume_context": context,
                    "optimizer_build_context": context,
                },
            )
        ],
    )

    checks = {check.name: check for check in run_preflight_checks(bundle)}

    schedule_check = checks["schedule-realization"]
    assert schedule_check.status == "pass"
    samples = schedule_check.observed["row-a"][0]["samples"]
    by_position = {sample["schedule_position"]: sample for sample in samples}
    assert set((3500, 3501, 4499, 4500)).issubset(by_position)
    for position in (3500, 3501, 4499, 4500):
        assert by_position[position]["expected"] == pytest.approx(3e-5)
        assert by_position[position]["observed"] == pytest.approx(3e-5)


def test_preflight_schedule_realization_fails_when_resume_context_is_dropped(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                run_spec={
                    "optimizer": _scheduled_optimizer_payload(),
                    "optimizer_build_context": _schedule_context(
                        schedule_origin_step=0,
                        current_step=0,
                        optimizer_count_at_current_step=0,
                    ),
                },
            )
        ],
    )

    checks = {check.name: check for check in run_preflight_checks(bundle)}

    assert checks["schedule-realization"].status == "fail"
    assert "resume_context missing" in (checks["schedule-realization"].detail or "")


def test_schedule_preflight_and_conformance_share_schedule_eval_helper() -> None:
    assert (
        conformance.learning_rate_from_build_optimizer
        is schedule_eval.learning_rate_from_build_optimizer
    )
    assert conformance.extract_resume_context is schedule_eval.extract_resume_context
    assert stages.schedule_eval is schedule_eval


def test_schedule_context_metadata_is_last_fallback_and_build_stays_independent() -> None:
    row_context = _schedule_context(
        schedule_origin_step=1,
        current_step=1,
        optimizer_count_at_current_step=1,
    )
    diagnostics_context = _schedule_context(
        schedule_origin_step=2,
        current_step=2,
        optimizer_count_at_current_step=2,
    )
    metadata_context = _schedule_context(
        schedule_origin_step=3,
        current_step=3,
        optimizer_count_at_current_step=3,
    )
    row = {
        "resume_context": row_context,
        "metadata": {
            "resume_context": metadata_context,
            "optimizer_build_context": metadata_context,
        },
    }
    diagnostics = {
        "resume_context": diagnostics_context,
        "optimizer_build_context": diagnostics_context,
    }

    assert schedule_eval.extract_resume_context(row, diagnostics) == row_context
    assert schedule_eval.extract_optimizer_build_context(row, diagnostics) == diagnostics_context
    with pytest.raises(schedule_eval.MissingScheduleContext, match="optimizer_build_context"):
        schedule_eval.require_schedule_context(
            schedule_eval.extract_optimizer_build_context(
                {"metadata": {"resume_context": metadata_context}}
            ),
            label="optimizer_build_context",
        )


def test_production_stage_engine_call_sites_supply_nonempty_registry() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for relative in (
        "feedbax/bin/orchestrate.py",
        "feedbax/web/services/training_service.py",
    ):
        tree = ast.parse((repo_root / relative).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "StageEngine")
                or (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "StageEngine"
                    and node.func.attr == "from_request"
                )
            )
        ]
        assert calls, relative
        for call in calls:
            registry = next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "conformance_registry"
                ),
                None,
            )
            assert isinstance(registry, ast.Call), f"{relative}:{call.lineno}"
            assert isinstance(registry.func, ast.Name), f"{relative}:{call.lineno}"
            assert registry.func.id == "build_default_check_registry", f"{relative}:{call.lineno}"


def test_conformance_discovery_prefers_typed_diagnostics_over_manifest_metrics(
    tmp_path: Path,
) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    manifest = TrainingRunManifest(
        id="feedbax-training-run:selection",
        completed_batches=10,
        summary_metrics={"completed_batches": 10},
    ).model_dump(mode="json", exclude_none=True)
    diagnostics = TrainingDiagnostics(
        manifest_id=manifest["id"],
        run_id="selection",
        terminal_status="completed",
        completed_batches=10,
        segment_completed_batches=10,
        cumulative_completed_batches=10,
        lr_trace=[{"step": 10, "learning_rate": 3e-4}],
        checkpoint_coordinates=[10],
        checkpoint_transactions=[
            {
                "transaction_id": "checkpoint-10",
                "completed_batches": 10,
                "cumulative_completed_batches": 10,
                "coordinate": {
                    "run_id": "selection",
                    "phase": "train",
                    "program_step": 10,
                },
            }
        ],
    ).model_dump(mode="json", exclude_none=True)
    (collected / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (collected / "training-diagnostics.json").write_text(
        json.dumps(diagnostics, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    discovered = stages._discover_conformance_artifacts({"collection": str(collected)})

    assert discovered["manifest_payload"] == manifest
    assert discovered["training_diagnostics"] == diagnostics
    assert discovered["training_diagnostics"]["lr_trace"] == [{"step": 10, "learning_rate": 3e-4}]
    assert discovered["training_diagnostics"]["checkpoint_coordinates"] == [10]
    assert discovered["training_diagnostics"]["checkpoint_transactions"] == [
        {
            "transaction_id": "checkpoint-10",
            "completed_batches": 10,
            "cumulative_completed_batches": 10,
            "coordinate": {
                "run_id": "selection",
                "phase": "train",
                "program_step": 10,
                "metrics": {},
            },
        }
    ]


def test_conformance_discovery_leaves_missing_diagnostics_absent(tmp_path: Path) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    manifest = TrainingRunManifest(
        id="feedbax-training-run:missing-diagnostics",
        completed_batches=10,
        summary_metrics={"completed_batches": 10},
    ).model_dump(mode="json", exclude_none=True)
    (collected / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    discovered = stages._discover_conformance_artifacts({"collection": str(collected)})

    assert discovered["manifest_payload"] == manifest
    assert "training_diagnostics" not in discovered


def test_conformance_discovery_leaves_ambiguous_typed_diagnostics_absent(
    tmp_path: Path,
) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    for index in (1, 2):
        diagnostics = TrainingDiagnostics(
            manifest_id=f"feedbax-training-run:ambiguous-{index}",
            run_id=f"ambiguous-{index}",
            terminal_status="completed",
            completed_batches=10,
            segment_completed_batches=10,
            cumulative_completed_batches=10,
            lr_trace=[{"step": 10, "learning_rate": 3e-4}],
        ).model_dump(mode="json", exclude_none=True)
        (collected / f"candidate-{index}.json").write_text(
            json.dumps(diagnostics, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    discovered = stages._discover_conformance_artifacts({"collection": str(collected)})

    assert "training_diagnostics" not in discovered


def test_conformance_discovery_ignores_partial_typed_identity_for_legacy_fallback(
    tmp_path: Path,
) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    malformed_typed = {
        "kind": "TrainingDiagnostics",
        "schema_id": TRAINING_DIAGNOSTICS_SCHEMA_ID,
        "completed_batches": 10,
        "lr_trace": [{"step": 10, "learning_rate": 1e-3}],
    }
    legacy = {
        "completed_batches": 10,
        "lr_trace": [{"step": 10, "learning_rate": 3e-4}],
        "checkpoint_coordinates": [10],
    }
    (collected / "a-partial-typed.json").write_text(
        json.dumps(malformed_typed, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (collected / "z-legacy.json").write_text(
        json.dumps(legacy, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    discovered = stages._discover_conformance_artifacts({"collection": str(collected)})

    assert discovered["training_diagnostics"] == legacy


def test_conformance_discovery_rejects_schema_less_run_spec_context(
    tmp_path: Path,
) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    context = _schedule_context(
        schedule_origin_step=0,
        current_step=10,
        optimizer_count_at_current_step=10,
    )
    run_spec = {
        "completed_batches": 10,
        "seeds": {"controller": 7},
        "resume_context": context,
        "optimizer_build_context": context,
    }
    (collected / "run-spec.json").write_text(
        json.dumps(run_spec, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    discovered = stages._discover_conformance_artifacts({"collection": str(collected)})

    assert "training_diagnostics" not in discovered


def test_production_default_certificate_rejects_declared_rewarm_with_flat_lr(
    tmp_path: Path,
) -> None:
    run_spec = {
        "optimizer": _scheduled_optimizer_payload(),
        "resume_context": _schedule_context(
            schedule_origin_step=0,
            current_step=0,
            optimizer_count_at_current_step=0,
        ),
        "optimizer_build_context": _schedule_context(
            schedule_origin_step=0,
            current_step=0,
            optimizer_count_at_current_step=0,
        ),
        "n_batches": 3500,
        "checkpoint_interval": 500,
        "seeds": {"controller": 7},
    }
    bundle = _bundle(
        tmp_path,
        rows=[_compiled_row("rewarm", run_spec=run_spec)],
        run_set_id="negative-canary",
    )
    row = bundle.row("rewarm")
    collected = bundle.run_set_dir / "collected" / row.row_id
    collected.mkdir(parents=True)
    manifest = collected / "training_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "TrainingRunManifest",
                "metadata": {
                    "environment_fingerprint": "fake-fingerprint",
                    "seeds": {"controller": 7},
                },
                "training_spec": {"inline": run_spec},
                "summary_metrics": {"completed_batches": 3500},
            }
        ),
        encoding="utf-8",
    )
    diagnostics = collected / "training_diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "completed_batches": 3500,
                "checkpoint_coordinates": list(range(500, 3501, 500)),
                "lr_trace": {"0": 3e-5, "500": 3e-5, "3500": 3e-5},
                "optimizer_build_context": run_spec["optimizer_build_context"],
                "resume_context": run_spec["resume_context"],
                "seeds": {"controller": 7},
            }
        ),
        encoding="utf-8",
    )
    events = bundle.run_set_dir / "events"
    events.mkdir(parents=True)
    (events / "rewarm.events.jsonl").write_text('{"type":"complete"}\n', encoding="utf-8")
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        environment_fingerprint="fake-fingerprint",
        rows={
            row.row_id: RowState(
                status="completed",
                collected_outputs={
                    manifest.name: str(manifest),
                    diagnostics.name: str(diagnostics),
                },
            )
        },
    )

    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        conformance_registry=build_default_check_registry(include_plugins=False),
    )
    state = _with_local_realized_proof(state)
    _state, outputs = engine._stage_certify(state)
    certificate = json.loads((bundle.run_set_dir / "conformance.json").read_text())
    checks = {entry["check_id"]: entry for entry in certificate["rows"]["rewarm"]["checks"]}

    assert outputs["overall"] == "fail"
    assert set(checks) == {
        check_id for check_id, _check in build_default_check_registry(include_plugins=False).items()
    }
    assert all(entry["status"] in {"pass", "fail"} for entry in checks.values())
    assert checks["lr_trace"]["status"] == "fail"


def test_local_driver_warm_first_max_parallel_budget_and_demo(tmp_path: Path) -> None:
    script = tmp_path / "row_script.py"
    script.write_text(
        """
from pathlib import Path
import os
import time
from feedbax.orchestration.events import RunEventEmitter

row = os.environ["FEEDBAX_ROW_ID"]
row_dir = Path(os.environ["FEEDBAX_ROW_DIR"])
with RunEventEmitter.from_env(heartbeat_seconds=None) as emitter:
    if row == "warm":
        emitter.emit("ready", {"row": row})
        time.sleep(0.15)
    else:
        time.sleep(0.02)
    (row_dir / "payload.json").write_text('{"row": "%s"}\\n' % row, encoding="utf-8")
    emitter.emit_terminal("complete", {"row": row})
""".strip(),
        encoding="utf-8",
    )
    rows = [
        _compiled_row("warm", command=[sys.executable, str(script)], collect=["payload.json"]),
        _compiled_row("second", command=[sys.executable, str(script)], collect=["payload.json"]),
    ]
    bundle = _bundle(
        tmp_path,
        rows=rows,
        launch_policy=LaunchPolicy(max_parallel_rows=1, warm_first=True),
        run_set_id="2026-01-02-cafebabe",
    )

    state = StageEngine(
        bundle=bundle,
        driver=LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",)),
        conformance_registry=_fixture_pass_registry(),
        poll_interval_seconds=0.01,
    ).run()

    assert state.stage("REGISTER").status == "completed"
    assert state.registration_payload and state.registration_payload["status"] == "completed"
    assert (
        state.registration_payload["certificate_sha256"]
        == hashlib.sha256((bundle.run_set_dir / "conformance.json").read_bytes()).hexdigest()
    )
    stage_inputs_sha256 = hashlib.sha256(
        json.dumps(
            state.stage(STAGE_STAGE_INPUTS).outputs,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert state.stage(STAGE_CERTIFY).outputs["stage_inputs_sha256"] == stage_inputs_sha256
    assert state.registration_payload["stage_inputs_sha256"] == stage_inputs_sha256
    assert {row_id: row.status for row_id, row in state.rows.items()} == {
        "warm": "completed",
        "second": "completed",
    }
    assert (bundle.run_set_dir / "events" / "warm.events.jsonl").exists()
    assert (bundle.run_set_dir / "collected" / "second" / "payload.json").exists()

    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(2)\n", encoding="utf-8")
    budget_bundle = _bundle(
        tmp_path / "budget",
        rows=[_compiled_row("slow", command=[sys.executable, str(slow)])],
        max_wall_clock_seconds=0.05,
        run_set_id="2026-01-02-badf00d",
    )
    budget_state = StageEngine(
        bundle=budget_bundle,
        driver=LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",)),
        conformance_registry=_fixture_pass_registry(),
        poll_interval_seconds=0.01,
    ).run()

    assert budget_state.abort_reason == "budget-exceeded"
    assert budget_state.rows["slow"].status == "stopped"
    assert budget_state.registration_payload
    assert budget_state.registration_payload["status"] == "aborted"


def test_local_monitor_requests_checkpoint_stop_and_records_provenance(tmp_path: Path) -> None:
    script = tmp_path / "interruptible_row.py"
    script.write_text(
        """
import signal
import time
from feedbax.orchestration.events import RunEventEmitter

emitter = RunEventEmitter.from_env(heartbeat_seconds=None)
assert emitter is not None

def stop_at_checkpoint(_signum, _frame):
    emitter.emit_terminal("complete", {"status": "cancelled"})
    emitter.close()
    raise SystemExit(0)

signal.signal(signal.SIGINT, stop_at_checkpoint)
emitter.emit("ready", {"phase": "train"})
emitter.emit_progress(
    {"phase": "train", "batch": 1, "total_batches": 10},
    batch=1,
    total_batches=10,
    force=True,
)
while True:
    time.sleep(0.01)
""".strip(),
        encoding="utf-8",
    )
    bundle = _bundle(
        tmp_path,
        rows=[_compiled_row("row", command=[sys.executable, str(script)])],
        run_set_id="checkpoint-stop",
    )
    event_path = bundle.run_set_dir / "events" / "row.events.jsonl"
    decision = CancellationDecision("stop", "test", 123.0)
    dispatched = False

    def interruption_probe() -> CancellationDecision | None:
        nonlocal dispatched
        if not dispatched and event_path.exists() and '"type":"ready"' in event_path.read_text():
            dispatched = True
            return decision
        return None

    state = StageEngine(
        bundle=bundle,
        driver=LocalOrchestrationDriver(cwd=Path.cwd(), freeze_lines=("feedbax==test",)),
        poll_interval_seconds=0.01,
        interruption_probe=interruption_probe,
    ).run(stop_after_stage="MONITOR")

    assert dispatched
    assert state.abort_reason == "operator-stop-after-checkpoint"
    assert state.rows["row"].status == "stopped"
    assert state.budget_counters["cancellation"] == decision.as_provenance()


@pytest.mark.parametrize(
    ("action", "expected_abort_reason", "expected_row_status"),
    [
        ("continue", None, "completed"),
        ("terminate", "operator-terminate", "stopped"),
    ],
)
def test_local_monitor_applies_continue_and_terminate_decisions(
    tmp_path: Path,
    action: CancellationAction,
    expected_abort_reason: str | None,
    expected_row_status: str,
) -> None:
    script = tmp_path / "row.py"
    script.write_text(
        """
import time
from feedbax.orchestration.events import RunEventEmitter

with RunEventEmitter.from_env(heartbeat_seconds=None) as emitter:
    assert emitter is not None
    emitter.emit("ready", {"phase": "train"})
    emitter.emit_progress(
        {"phase": "train", "batch": 1, "total_batches": 1},
        batch=1,
        total_batches=1,
        force=True,
    )
    time.sleep(0.1)
    emitter.emit_terminal("complete", {"status": "completed"})
""".strip(),
        encoding="utf-8",
    )
    bundle = _bundle(
        tmp_path,
        rows=[_compiled_row("row", command=[sys.executable, str(script)])],
        run_set_id=f"{action}-decision",
    )
    event_path = bundle.run_set_dir / "events" / "row.events.jsonl"
    decision = CancellationDecision(action, "test", 123.0)
    dispatched = False

    def interruption_probe() -> CancellationDecision | None:
        nonlocal dispatched
        if not dispatched and event_path.exists() and '"type":"ready"' in event_path.read_text():
            dispatched = True
            return decision
        return None

    state = StageEngine(
        bundle=bundle,
        driver=LocalOrchestrationDriver(cwd=Path.cwd(), freeze_lines=("feedbax==test",)),
        poll_interval_seconds=0.01,
        interruption_probe=interruption_probe,
    ).run(stop_after_stage="MONITOR")

    assert dispatched
    assert state.abort_reason == expected_abort_reason
    assert state.rows["row"].status == expected_row_status
    if action == "terminate":
        assert state.budget_counters["cancellation"] == decision.as_provenance()


def test_register_writes_failed_certificate_payload_and_reentry_is_idempotent(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    registry = CheckRegistry(
        {
            "fixture_fail": lambda row: CheckEntry(
                check_id="fixture_fail",
                status="fail",
                expected="pass",
                observed="fail",
            )
        }
    )

    with pytest.raises(ValueError, match="phase=completed"):
        StageEngine(
            bundle=bundle,
            driver=FakeDriver(),
            store=store,
            conformance_registry=registry,
        ).run()

    register_path = bundle.run_set_dir / "registration.json"
    certificate_path = bundle.run_set_dir / "conformance.json"
    payload = json.loads(register_path.read_text(encoding="utf-8"))
    certificate_digest = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
    failed_state = store.load()

    assert payload == {
        "abort_reason": None,
        "certificate_overall": "fail",
        "certificate_ref": str(certificate_path),
        "certificate_sha256": certificate_digest,
        "failure_reason": "conformance-failed",
        "run_set_id": bundle.run_set_id,
        "stage_inputs_sha256": failed_state.stage(STAGE_CERTIFY).outputs["stage_inputs_sha256"],
        "status": "failed",
    }
    assert failed_state.stage("REGISTER").status == "failed"
    assert failed_state.registration_payload == payload

    registration_mtime = register_path.stat().st_mtime_ns
    with pytest.raises(ValueError, match="phase=completed"):
        StageEngine(
            bundle=bundle,
            driver=FakeDriver(),
            store=store,
            conformance_registry=registry,
        ).run()
    assert register_path.stat().st_mtime_ns == registration_mtime

    tampered = dict(payload)
    tampered["status"] = "completed"
    register_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        OrchestrationStageError,
        match=r"registration payload mismatch at .*registration\.json.*conformance\.json",
    ):
        StageEngine(
            bundle=bundle,
            driver=FakeDriver(),
            store=store,
            conformance_registry=registry,
        ).run()


def test_explicit_recertification_preserves_failed_registration_history(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    failing_registry = CheckRegistry(
        {
            "fixture_fail": lambda row: CheckEntry(
                check_id="fixture_fail",
                status="fail",
                expected="pass",
                observed="fail",
            )
        }
    )
    with pytest.raises(ValueError, match="phase=completed"):
        StageEngine(
            bundle=bundle,
            driver=FakeDriver(),
            store=store,
            conformance_registry=failing_registry,
        ).run()

    register_path = bundle.run_set_dir / "registration.json"
    history_path = bundle.run_set_dir / "registration-history.json"
    failed_registration_bytes = register_path.read_bytes()
    failed_registration = json.loads(failed_registration_bytes)
    failed_certificate_sha256 = failed_registration["certificate_sha256"]

    passed = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        store=store,
        conformance_registry=_fixture_pass_registry(),
    ).run(retry_failed_certification=True)

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history == {
        "entries": [
            {
                "certificate_sha256": failed_certificate_sha256,
                "original_certificate_ref": failed_registration["certificate_ref"],
                "registration_payload": failed_registration,
                "registration_sha256": hashlib.sha256(failed_registration_bytes).hexdigest(),
            }
        ],
        "run_set_id": bundle.run_set_id,
        "schema_id": "feedbax.orchestration.registration_history",
        "schema_version": "feedbax.orchestration.registration_history.v1",
    }
    assert passed.registration_payload
    assert passed.registration_payload["status"] == "completed"
    assert passed.registration_payload["certificate_overall"] == "pass"
    assert passed.registration_payload["certificate_sha256"] != failed_certificate_sha256
    assert json.loads(register_path.read_text(encoding="utf-8")) == passed.registration_payload

    history_bytes = history_path.read_bytes()
    registration_bytes = register_path.read_bytes()
    reentered = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        store=store,
        conformance_registry=_fixture_pass_registry(),
    ).run()

    assert reentered.registration_payload == passed.registration_payload
    assert history_path.read_bytes() == history_bytes
    assert register_path.read_bytes() == registration_bytes


def test_explicit_recertification_rejects_tampered_registration_history(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    failing_registry = CheckRegistry(
        {
            "fixture_fail": lambda row: CheckEntry(
                check_id="fixture_fail",
                status="fail",
                expected="pass",
                observed="fail",
            )
        }
    )
    with pytest.raises(ValueError, match="phase=completed"):
        StageEngine(
            bundle=bundle,
            driver=FakeDriver(),
            store=store,
            conformance_registry=failing_registry,
        ).run()

    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        store=store,
        conformance_registry=_fixture_pass_registry(),
    )
    engine._preserve_failed_registration_history(store.load())
    history_path = bundle.run_set_dir / "registration-history.json"
    tampered = json.loads(history_path.read_text(encoding="utf-8"))
    tampered["entries"][0]["certificate_sha256"] = "0" * 64
    history_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(
        OrchestrationStageError,
        match="registration history mismatch",
    ):
        engine.run(retry_failed_certification=True)


@pytest.mark.parametrize(
    "inventory",
    [
        None,
        {"scope": "exact-owned-pod", "pod_ids": []},
        {
            "scope": "provider-account",
            "verified": False,
            "observed_at": "2026-07-18T20:00:00+00:00",
            "observation_basis": "runpodctl pod list --output json",
            "outcome": "unavailable",
            "pod_count": 0,
            "pod_ids": [],
        },
        {
            "scope": "provider-account",
            "verified": True,
            "observed_at": "2026-07-18T20:00:00+00:00",
            "observation_basis": "runpodctl pod list --output json",
            "outcome": "non-empty",
            "pod_count": 1,
            "pod_ids": ["pod-leftover"],
        },
        {
            "scope": "provider-account",
            "verified": True,
            "observation_basis": "runpodctl pod list --output json",
            "outcome": "empty",
            "pod_count": 0,
            "pod_ids": [],
        },
    ],
)
def test_register_refuses_runpod_without_verified_globally_empty_inventory(
    tmp_path: Path,
    inventory: dict[str, Any] | None,
) -> None:
    bundle = _bundle(tmp_path, driver="runpod")
    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        conformance_registry=_fixture_pass_registry(),
    )
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={"row-a": RowState(status="completed")},
    )
    teardown = state.stage(STAGE_TEARDOWN).model_copy(
        update={
            "status": "completed",
            "outputs": ({"final_pod_inventory": inventory} if inventory else {}),
        }
    )
    state = state.with_stage(STAGE_TEARDOWN, teardown)

    with pytest.raises(
        OrchestrationStageError,
        match="globally empty RunPod provider inventory",
    ):
        engine._stage_register(state)


def test_register_accepts_verified_globally_empty_runpod_inventory_gate(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, driver="runpod")
    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        conformance_registry=_fixture_pass_registry(),
    )
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={"row-a": RowState(status="completed")},
    )
    teardown = state.stage(STAGE_TEARDOWN).model_copy(
        update={
            "status": "completed",
            "outputs": {
                "final_pod_inventory": {
                    "scope": "provider-account",
                    "verified": True,
                    "observed_at": "2026-07-18T20:00:00+00:00",
                    "observation_basis": "runpodctl pod list --output json",
                    "outcome": "empty",
                    "pod_count": 0,
                    "pod_ids": [],
                }
            },
        }
    )
    state = state.with_stage(STAGE_TEARDOWN, teardown)

    with pytest.raises(OrchestrationStageError, match="certificate_ref"):
        engine._stage_register(state)


@pytest.mark.parametrize(
    ("rows", "abort_reason", "expected_status", "expected_outcomes"),
    [
        (
            {
                "row-b": RowState(status="completed"),
                "row-a": RowState(status="completed"),
            },
            None,
            "completed",
            None,
        ),
        (
            {
                "row-b": RowState(
                    status="stopped",
                    error="operator-stop-after-checkpoint",
                ),
                "row-a": RowState(
                    status="stopped",
                    error="operator-stop-after-checkpoint",
                ),
            },
            None,
            "stopped",
            {
                "row-a": {
                    "reason": "operator-stop-after-checkpoint",
                    "status": "stopped",
                },
                "row-b": {
                    "reason": "operator-stop-after-checkpoint",
                    "status": "stopped",
                },
            },
        ),
        (
            {
                "row-b": RowState(
                    status="stopped",
                    error="operator-stop-after-checkpoint",
                ),
                "row-a": RowState(status="completed"),
            },
            None,
            "mixed",
            {
                "row-a": {"status": "completed"},
                "row-b": {
                    "reason": "operator-stop-after-checkpoint",
                    "status": "stopped",
                },
            },
        ),
        (
            {"row-a": RowState(status="stopped", error="budget-exceeded")},
            "budget-exceeded",
            "aborted",
            None,
        ),
    ],
)
def test_register_derives_passing_status_from_durable_row_lifecycle(
    tmp_path: Path,
    rows: dict[str, RowState],
    abort_reason: str | None,
    expected_status: str,
    expected_outcomes: dict[str, dict[str, str]] | None,
) -> None:
    compiled_rows = [_compiled_row(row_id) for row_id in rows]
    bundle = _bundle(tmp_path, rows=compiled_rows)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        store=store,
        conformance_registry=_fixture_pass_registry(),
    )
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows=rows,
        abort_reason=abort_reason,
    )
    state = _with_local_realized_proof(state)
    state, certify_outputs = engine._stage_certify(state)
    state = state.with_stage(
        STAGE_CERTIFY,
        state.stage(STAGE_CERTIFY).model_copy(
            update={"status": "completed", "outputs": dict(certify_outputs)}
        ),
    )

    registered, payload = engine._stage_register(state)

    assert payload["status"] == expected_status
    assert payload["abort_reason"] == abort_reason
    assert payload.get("row_outcomes") == expected_outcomes
    assert registered.registration_payload == payload


def test_register_stopped_payload_reentry_is_idempotent(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    register_path = bundle.run_set_dir / "registration.json"
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        store=store,
        conformance_registry=_fixture_pass_registry(),
    )
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={
            "row-a": RowState(
                status="stopped",
                error="operator-stop-after-checkpoint",
            )
        },
    )
    state = _with_local_realized_proof(state)
    state, certify_outputs = engine._stage_certify(state)
    state = state.with_stage(
        STAGE_CERTIFY,
        state.stage(STAGE_CERTIFY).model_copy(
            update={"status": "completed", "outputs": dict(certify_outputs)}
        ),
    )
    state, first_payload = engine._stage_register(state)
    first_mtime = register_path.stat().st_mtime_ns

    _state, second_payload = engine._stage_register(state)

    assert first_payload["certificate_overall"] == "pass"
    assert first_payload["certificate_ref"] == state.certificate_ref
    assert json.loads(register_path.read_text(encoding="utf-8")) == first_payload
    assert second_payload == first_payload
    assert register_path.stat().st_mtime_ns == first_mtime


@pytest.mark.parametrize("tamper_certificate", [False, True])
def test_register_requires_the_digest_recorded_by_completed_certify(
    tmp_path: Path,
    tamper_certificate: bool,
) -> None:
    bundle = _bundle(tmp_path)
    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        conformance_registry=_fixture_pass_registry(),
    )
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={"row-a": RowState(status="completed")},
    )
    state = _with_local_realized_proof(state)
    state, outputs = engine._stage_certify(state)
    recorded_outputs = dict(outputs)
    if tamper_certificate:
        certificate_path = Path(state.certificate_ref or "")
        certificate_path.write_bytes(certificate_path.read_bytes() + b" ")
    else:
        recorded_outputs.pop("certificate_sha256")
    state = state.with_stage(
        STAGE_CERTIFY,
        state.stage(STAGE_CERTIFY).model_copy(
            update={"status": "completed", "outputs": recorded_outputs}
        ),
    )

    with pytest.raises(OrchestrationStageError, match="digest does not match"):
        engine._stage_register(state)


@pytest.mark.parametrize(
    ("tamper_surface", "message"),
    [
        ("status", "completed STAGE_INPUTS authority"),
        ("outputs", "STAGE_INPUTS digest does not match"),
        ("certify_digest", "STAGE_INPUTS digest does not match"),
    ],
)
def test_register_binds_resumed_stage_inputs_to_completed_certify(
    tmp_path: Path,
    tamper_surface: str,
    message: str,
) -> None:
    bundle = _bundle(tmp_path)
    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        conformance_registry=_fixture_pass_registry(),
    )
    state = _with_local_realized_proof(
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={"row-a": RowState(status="completed")},
        )
    )
    state, outputs = engine._stage_certify(state)
    state = state.with_stage(
        STAGE_CERTIFY,
        state.stage(STAGE_CERTIFY).model_copy(
            update={"status": "completed", "outputs": dict(outputs)}
        ),
    )
    if tamper_surface == "status":
        stage_inputs = state.stage(STAGE_STAGE_INPUTS).model_copy(update={"status": "pending"})
        state = state.with_stage(STAGE_STAGE_INPUTS, stage_inputs)
    elif tamper_surface == "outputs":
        stage_inputs = state.stage(STAGE_STAGE_INPUTS).model_copy(
            update={"outputs": {"substituted": True}}
        )
        state = state.with_stage(STAGE_STAGE_INPUTS, stage_inputs)
    else:
        certify_outputs = dict(state.stage(STAGE_CERTIFY).outputs)
        certify_outputs.pop("stage_inputs_sha256")
        state = state.with_stage(
            STAGE_CERTIFY,
            state.stage(STAGE_CERTIFY).model_copy(update={"outputs": certify_outputs}),
        )

    with pytest.raises(OrchestrationStageError, match=message):
        engine._stage_register(state)


@pytest.mark.parametrize(
    ("tamper_surface", "message"),
    [
        ("state_run", "state run_set_id"),
        ("certificate_run", "certificate run_set_id"),
        ("certificate_row_omission", "row-ID equality"),
        ("certificate_row_addition", "row-ID equality"),
        ("state_row_addition", "row-ID equality"),
        ("path_substitution", "certificate_ref"),
    ],
)
def test_register_binds_certificate_to_run_rows_and_certify_path(
    tmp_path: Path,
    tamper_surface: str,
    message: str,
) -> None:
    bundle = _bundle(tmp_path, rows=[_compiled_row("row-a"), _compiled_row("row-b")])
    engine = StageEngine(
        bundle=bundle,
        driver=FakeDriver(),
        conformance_registry=_fixture_pass_registry(),
    )
    state = _with_local_realized_proof(
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={
                "row-a": RowState(status="completed"),
                "row-b": RowState(status="completed"),
            },
        )
    )
    state, outputs = engine._stage_certify(state)
    state = state.with_stage(
        STAGE_CERTIFY,
        state.stage(STAGE_CERTIFY).model_copy(
            update={"status": "completed", "outputs": dict(outputs)}
        ),
    )
    certificate_path = Path(state.certificate_ref or "")

    if tamper_surface == "state_run":
        state = state.model_copy(update={"run_set_id": "substituted-run"})
    elif tamper_surface == "state_row_addition":
        state = state.model_copy(update={"rows": {**state.rows, "extra": RowState()}})
    elif tamper_surface == "path_substitution":
        substitute = bundle.run_set_dir / "substituted-conformance.json"
        substitute.write_bytes(certificate_path.read_bytes())
        state = state.model_copy(update={"certificate_ref": str(substitute)})
    else:
        payload = json.loads(certificate_path.read_text(encoding="utf-8"))
        if tamper_surface == "certificate_run":
            payload["run_set_id"] = "substituted-run"
        elif tamper_surface == "certificate_row_omission":
            del payload["rows"]["row-b"]
        else:
            payload["rows"]["extra"] = payload["rows"]["row-a"]
        certificate_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        certified_outputs = dict(state.stage(STAGE_CERTIFY).outputs)
        certified_outputs["certificate_sha256"] = hashlib.sha256(
            certificate_path.read_bytes()
        ).hexdigest()
        state = state.with_stage(
            STAGE_CERTIFY,
            state.stage(STAGE_CERTIFY).model_copy(update={"outputs": certified_outputs}),
        )

    with pytest.raises(OrchestrationStageError, match=message):
        engine._stage_register(state)


def test_local_driver_adopts_live_started_pid_without_spawning(tmp_path: Path) -> None:
    marker = tmp_path / "spawned.txt"
    compiled_row = _compiled_row(
        "row-a",
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('spawned')",
        ],
    )
    bundle = _bundle(tmp_path, rows=[compiled_row])
    row = bundle.row("row-a")
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",))
    driver.provision(bundle, RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()}))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        sentinels = bundle.run_set_dir / "sentinels"
        (sentinels / "row-a.started").write_text("1\n", encoding="utf-8")
        (sentinels / "row-a.pid").write_text(f"{process.pid}\n", encoding="utf-8")

        outputs = driver.launch_row(
            bundle,
            row,
            RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()}),
        )
    finally:
        process.terminate()
        process.wait(timeout=5)

    assert outputs["pid"] == process.pid
    assert outputs["adopted"] is True
    assert not marker.exists()


def test_local_driver_injects_native_execution_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    original = bundle.row("row-a")
    row = original.model_copy(
        update={
            "launch": RowLaunchSpec(
                command=[
                    sys.executable,
                    "-m",
                    "feedbax",
                    "execute-training-run-spec",
                    "specs/row-a.json",
                ]
            ),
            "execution": original.execution.model_copy(
                update={
                    "row_provenance": TrainingRowProvenance(
                        row_id="row-a",
                        row_index=4,
                        planned_run_id="feedbax-training-run:planned-local",
                        authored_payload_hash="e" * 64,
                        lowered_execution_payload_hash=original.execution.payload.sha256,
                        axis_coordinates={"seed": 9},
                        seed=9,
                        lowerer_identities=[
                            RowLowererIdentity(
                                lowerer_id="feedbax.tests.local",
                                lowerer_version="v2",
                            )
                        ],
                    )
                }
            ),
        }
    )
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 12345

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",))
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={"row-a": RowState()},
        environment_fingerprint="fingerprint-local",
    )
    driver.provision(bundle, state)

    outputs = driver.launch_row(bundle, row, state)

    assert outputs["pid"] == 12345
    command = captured["command"]
    assert command[-2] == "--execution-context-json"
    context = json.loads(command[-1])
    assert context["execution"]["row_provenance"]["planned_run_id"] == (
        "feedbax-training-run:planned-local"
    )
    assert context["execution"]["row_provenance"]["lowerer_identities"] == [
        {
            "lowerer_id": "feedbax.tests.local",
            "lowerer_version": "v2",
        }
    ]
    assert context["environment_fingerprint"] == "fingerprint-local"
    assert context["collection_root"].endswith("/rows/row-a")


def test_local_driver_marks_dead_started_pid_failed_without_spawning(tmp_path: Path) -> None:
    marker = tmp_path / "spawned.txt"
    compiled_row = _compiled_row(
        "row-a",
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('spawned')",
        ],
    )
    bundle = _bundle(tmp_path, rows=[compiled_row])
    row = bundle.row("row-a")
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",))
    driver.provision(bundle, RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()}))
    sentinels = bundle.run_set_dir / "sentinels"
    (sentinels / "row-a.started").write_text("1\n", encoding="utf-8")
    (sentinels / "row-a.pid").write_text("999999999\n", encoding="utf-8")

    outputs = driver.launch_row(
        bundle,
        row,
        RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()}),
    )

    assert outputs["status"] == "failed"
    assert outputs["event_discrepancies"][0]["code"] == "orphaned_launch"
    assert "orphaned launch" in (sentinels / "row-a.failed").read_text(encoding="utf-8")
    assert not marker.exists()


def test_stage_resume_records_orphaned_started_pid_as_failed(tmp_path: Path) -> None:
    marker = tmp_path / "spawned.txt"
    compiled_row = _compiled_row(
        "row-a",
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('spawned')",
        ],
    )
    bundle = _bundle(tmp_path, rows=[compiled_row])
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",))
    state = RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()})
    driver.provision(bundle, state)
    sentinels = bundle.run_set_dir / "sentinels"
    (sentinels / "row-a.started").write_text("1\n", encoding="utf-8")
    (sentinels / "row-a.pid").write_text("999999999\n", encoding="utf-8")

    with pytest.raises(OrchestrationStageError, match="executor failed for row 'row-a'"):
        StageEngine(
            bundle=bundle,
            driver=driver,
            store=store,
            conformance_registry=_fixture_pass_registry(),
            poll_interval_seconds=0.01,
        ).run()

    final_state = store.load()
    assert final_state.rows["row-a"].status == "failed"
    assert final_state.rows["row-a"].event_discrepancies[0]["code"] == "orphaned_launch"
    assert final_state.stage("TEARDOWN").status == "completed"
    assert not marker.exists()


@pytest.mark.parametrize(
    ("second_probe_status", "expected_status"),
    [
        ("completed", "completed"),
        ("failed", "failed"),
    ],
)
def test_monitor_reconciles_terminal_event_before_local_sentinel(
    tmp_path: Path,
    second_probe_status: str,
    expected_status: str,
) -> None:
    class TerminalEventFirstDriver(FakeDriver):
        def __init__(self) -> None:
            super().__init__()
            self.probe_count = 0

        def launch_row(
            self,
            bundle: RunBundle,
            row: RunRowSpec,
            state: RunSetState,
        ) -> dict[str, Any]:
            outputs = super().launch_row(bundle, row, state)
            events = bundle.run_set_dir / "events"
            events.mkdir(parents=True, exist_ok=True)
            status = "completed" if second_probe_status == "completed" else "failed"
            (events / f"{row.row_id}.events.jsonl").write_text(
                json.dumps(
                    {
                        "run_set_id": bundle.run_set_id,
                        "row_id": row.row_id,
                        "seq": 0,
                        "emitted_at_ms": 1,
                        "type": "complete" if status == "completed" else "failed",
                        "payload": {"status": status},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return outputs

        def probe(
            self,
            bundle: RunBundle,
            row: RunRowSpec,
            state: RunSetState,
        ) -> DriverRowProbe:
            self._call(f"probe:{row.row_id}")
            self.probe_count += 1
            if self.probe_count == 1:
                return DriverRowProbe(status="running")
            if second_probe_status == "completed":
                sentinels = bundle.run_set_dir / "sentinels"
                sentinels.mkdir(parents=True, exist_ok=True)
                (sentinels / f"{row.row_id}.done").write_text("0\n", encoding="utf-8")
            return DriverRowProbe(status=second_probe_status)

    bundle = _bundle(tmp_path)
    driver = TerminalEventFirstDriver()
    state = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=_fixture_pass_registry(),
        poll_interval_seconds=0,
    ).run(stop_after_stage="MONITOR")

    assert driver.probe_count == 2
    assert state.rows["row-a"].status == expected_status
    if expected_status == "completed":
        assert state.rows["row-a"].event_discrepancies == []
    else:
        assert state.rows["row-a"].event_discrepancies == [
            {"code": "terminal_event_without_sentinel", "event_status": "failed"}
        ]


@pytest.mark.parametrize(
    ("first_status", "second_status", "abort_reason", "expected_call"),
    [
        ("ready", "launched", None, "launch:second"),
        ("completed", "launched", None, "launch:second"),
        ("failed", "stopped", "warm-first-failed", "stop:second"),
    ],
)
def test_warm_first_gate_releases_ready_and_completed_first_rows(
    tmp_path: Path,
    first_status: str,
    second_status: str,
    abort_reason: str | None,
    expected_call: str,
) -> None:
    rows = [
        _compiled_row("warm"),
        _compiled_row("second"),
    ]
    bundle = _bundle(
        tmp_path,
        rows=rows,
        launch_policy=LaunchPolicy(max_parallel_rows=2, warm_first=True),
    )
    driver = FakeDriver()
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={
            "warm": RowState(status=first_status),
            "second": RowState(status="pending"),
        },
    )

    updated = StageEngine(bundle=bundle, driver=driver)._launch_pending_if_allowed(state)

    assert updated.rows["second"].status == second_status
    assert updated.abort_reason == abort_reason
    assert expected_call in driver.calls


def test_fingerprint_stability_package_changes_and_dirty_policy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    bundle = _bundle(
        tmp_path,
        rows=[_compiled_row("row")],
    ).model_copy(
        update={
            "environment": EnvironmentDeclaration(
                python_version="3.12",
                repo_revisions=[RepoRevision(path=".", revision="HEAD", dirty_allowed=True)],
                image_id="local",
            )
        }
    )

    first = compute_environment_fingerprint(bundle, cwd=repo, freeze_lines=("a==1", "b==2"))
    second = compute_environment_fingerprint(bundle, cwd=repo, freeze_lines=("b==2", "a==1"))
    changed = compute_environment_fingerprint(bundle, cwd=repo, freeze_lines=("a==1", "b==3"))

    assert first == second
    assert first != changed

    disallow_dirty = bundle.model_copy(
        update={
            "environment": EnvironmentDeclaration(
                python_version="3.12",
                repo_revisions=[RepoRevision(path=".", revision="HEAD", dirty_allowed=False)],
            )
        }
    )
    with pytest.raises(LocalDriverError, match="dirty repo not allowed"):
        compute_environment_fingerprint(disallow_dirty, cwd=repo, freeze_lines=("a==1",))


def test_fingerprint_inventory_does_not_require_pip(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    dist_info = site_packages / "example_pkg-1.2.3.dist-info"
    dist_info.mkdir(parents=True)
    metadata = dist_info / "METADATA"
    metadata.write_text(
        "Metadata-Version: 2.1\nName: example-pkg\nVersion: 1.2.3\n",
        encoding="utf-8",
    )
    (dist_info / "direct_url.json").write_text(
        '{"url":"file:///example-pkg","dir_info":{"editable":true}}\n',
        encoding="utf-8",
    )
    python = tmp_path / "pipless-python"
    python.write_text(
        (
            "#!/bin/sh\n"
            f"PYTHONPATH={shlex.quote(str(site_packages))} "
            f'exec {shlex.quote(sys.executable)} -S "$@"\n'
        ),
        encoding="utf-8",
    )
    python.chmod(0o755)
    pip_probe = subprocess.run(
        [str(python), "-c", "import importlib.util; print(importlib.util.find_spec('pip'))"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert pip_probe.stdout.strip() == "None"

    bundle = _bundle(tmp_path, rows=[_compiled_row("row")])
    driver = LocalOrchestrationDriver(
        cwd=tmp_path,
        python_executable=str(python),
    )
    state = RunSetState(run_set_id=bundle.run_set_id)
    first = driver.realize_env(bundle, state)
    second = driver.realize_env(bundle, state)
    metadata.write_text(
        "Metadata-Version: 2.1\nName: example-pkg\nVersion: 1.2.4\n",
        encoding="utf-8",
    )
    changed = driver.realize_env(bundle, state)

    assert first == second
    assert first != changed


def test_dependency_inventory_normalizes_names_and_order() -> None:
    first = [
        {"direct_url": None, "name": "Z_pkg", "version": "2"},
        {"direct_url": None, "name": "a.pkg", "version": "1"},
    ]
    reordered = [
        {"direct_url": None, "name": "A-PKG", "version": "1"},
        {"direct_url": None, "name": "z.pkg", "version": "2"},
    ]

    assert _canonicalize_dependency_inventory(first, executable="fixture") == (
        _canonicalize_dependency_inventory(reordered, executable="fixture")
    )


def test_dependency_inventory_rejects_conflicting_normalized_duplicates() -> None:
    inventory = [
        {"direct_url": None, "name": "example_pkg", "version": "1"},
        {"direct_url": None, "name": "example.pkg", "version": "2"},
    ]

    with pytest.raises(LocalDriverError, match="conflicting distributions"):
        _canonicalize_dependency_inventory(inventory, executable="fixture")


def test_fingerprint_uses_selected_interpreter_identity(tmp_path: Path) -> None:
    probe_payload = tmp_path / "probe.json"
    python = tmp_path / "identity-python"
    python.write_text(
        f"#!/bin/sh\ncat {shlex.quote(str(probe_payload))}\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    bundle = _bundle(tmp_path, rows=[_compiled_row("row")])

    def write_probe(version: str) -> None:
        probe_payload.write_text(
            json.dumps(
                {
                    "schema_version": "feedbax.local_dependency_inventory.v1",
                    "interpreter": {
                        "cache_tag": "cpython-313",
                        "executable": "/selected/python",
                        "implementation": "cpython",
                        "version": version,
                    },
                    "distributions": [],
                }
            ),
            encoding="utf-8",
        )

    write_probe("3.13.5")
    first = compute_environment_fingerprint(
        bundle,
        cwd=tmp_path,
        python_executable=str(python),
    )
    write_probe("3.13.6")
    changed = compute_environment_fingerprint(
        bundle,
        cwd=tmp_path,
        python_executable=str(python),
    )

    assert first != changed


def test_fingerprint_rejects_malformed_probe_payload(tmp_path: Path) -> None:
    python = tmp_path / "malformed-python"
    python.write_text("#!/bin/sh\necho '{\"distributions\":[]}'\n", encoding="utf-8")
    python.chmod(0o755)
    bundle = _bundle(tmp_path, rows=[_compiled_row("row")])

    with pytest.raises(LocalDriverError, match="invalid structure"):
        compute_environment_fingerprint(
            bundle,
            cwd=tmp_path,
            python_executable=str(python),
        )


def test_fingerprint_inventory_failure_is_not_replaced_with_empty_inventory(
    tmp_path: Path,
) -> None:
    broken_python = tmp_path / "broken-python"
    broken_python.write_text("#!/bin/sh\necho inventory-broken >&2\nexit 17\n", encoding="utf-8")
    broken_python.chmod(0o755)
    bundle = _bundle(tmp_path, rows=[_compiled_row("row")])

    with pytest.raises(LocalDriverError, match="inventory-broken"):
        compute_environment_fingerprint(
            bundle,
            cwd=tmp_path,
            python_executable=str(broken_python),
        )


def test_system_exit_runs_teardown_and_restores_signal_handlers(tmp_path: Path) -> None:
    class SystemExitDriver(FakeDriver):
        def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
            self._call("stage_inputs")
            raise SystemExit(7)

    bundle = _bundle(tmp_path)
    driver = SystemExitDriver()
    before = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    with pytest.raises(SystemExit) as raised:
        StageEngine(bundle=bundle, driver=driver).run()

    assert raised.value.code == 7
    assert driver.calls[-1] == "teardown"
    state = RunSetStateStore(bundle.run_set_dir / "state.json").load()
    assert state.stage(STAGE_TEARDOWN).status == "completed"
    assert {signum: signal.getsignal(signum) for signum in before} == before


def test_child_transport_timeout_runs_teardown_with_verified_absence(tmp_path: Path) -> None:
    class TimeoutDriver(FakeDriver):
        def provision(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
            self._call("provision")
            return {"driver": "fixture", "pod_id": "pod-timeout"}

        def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
            self._call("stage_inputs")
            result = _run_command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout_seconds=0.1,
            )
            result.check("fixture transport")
            raise AssertionError("unreachable")

        def teardown(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
            self._call("teardown")
            return {
                "teardown": "removed",
                "pod_absence": {"verified": True, "pod_id": "pod-timeout"},
            }

    bundle = _bundle(tmp_path)
    driver = TimeoutDriver()
    store = RunSetStateStore(bundle.run_set_dir / "state.json")

    with pytest.raises(RuntimeError, match="timed out after 0.1s"):
        StageEngine(bundle=bundle, driver=driver, store=store).run()

    assert driver.calls[-1] == "teardown"
    teardown = store.load().stage(STAGE_TEARDOWN)
    assert teardown.status == "completed"
    assert teardown.outputs["pod_absence"]["verified"] is True


def test_run_returns_post_teardown_state_instead_of_stale_precleanup_state(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    driver = FakeDriver()
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    teardown_index = STAGE_ORDER.index(STAGE_TEARDOWN)
    initial = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={row.row_id: RowState(status="completed") for row in bundle.rows},
        stages={
            stage_id: StageState(status="completed" if index < teardown_index else "pending")
            for index, stage_id in enumerate(STAGE_ORDER)
        },
    )
    store.save(initial)

    returned = StageEngine(bundle=bundle, driver=driver, store=store).run(
        stop_after_stage=STAGE_TEARDOWN
    )

    assert returned.stage(STAGE_TEARDOWN).status == "completed"
    assert returned.stage(STAGE_TEARDOWN).outputs["torn_down"] is True
    assert returned == store.load()


def test_scoped_signal_supervisor_is_noop_off_main_thread() -> None:
    before = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    entered: list[bool] = []

    def enter_supervisor() -> None:
        with _ScopedSignalSupervisor():
            entered.append(True)

    thread = threading.Thread(target=enter_supervisor)
    thread.start()
    thread.join(timeout=2)

    assert entered == [True]
    assert {signum: signal.getsignal(signum) for signum in before} == before


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_real_signal_during_stage_runs_teardown_then_prior_handler(
    tmp_path: Path,
    signum: signal.Signals,
) -> None:
    bundle = _bundle(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(bundle.model_dump_json(), encoding="utf-8")
    ready_path = tmp_path / "stage-inputs.ready"
    teardown_path = tmp_path / "teardown.json"
    reraised_path = tmp_path / "signal.reraised"
    script_path = tmp_path / "signal_stage.py"
    script_path.write_text(
        """
import json
import signal
import sys
import time
from pathlib import Path

from feedbax.orchestration.bundle import RunBundle
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.stages import StageEngine

bundle = RunBundle.model_validate_json(Path(sys.argv[1]).read_text())
ready = Path(sys.argv[2])
teardown = Path(sys.argv[3])
reraised = Path(sys.argv[4])
signum = int(sys.argv[5])

def prior_handler(received, _frame):
    reraised.write_text(str(received))
    raise SystemExit(128 + received)

signal.signal(signum, prior_handler)

class Driver:
    def provision(self, bundle, state):
        return {"driver": "fixture", "pod_id": "pod-signal"}
    def realize_env(self, bundle, state):
        return "fixture-fingerprint"
    def stage_inputs(self, bundle, state):
        ready.write_text("ready")
        time.sleep(30)
        return {}
    def teardown(self, bundle, state):
        outputs = {
            "teardown": "removed",
            "pod_absence": {"verified": True, "pod_id": "pod-signal"},
        }
        teardown.write_text(json.dumps(outputs))
        time.sleep(0.3)
        return outputs
    def launch_row(self, bundle, row, state):
        return {}
    def probe(self, bundle, row, state):
        return DriverRowProbe(status="completed")
    def stop_row(self, bundle, row, state):
        return {}
    def collect(self, bundle, row, state):
        return {}

StageEngine(bundle=bundle, driver=Driver()).run()
""".lstrip(),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(script_path),
            str(bundle_path),
            str(ready_path),
            str(teardown_path),
            str(reraised_path),
            str(int(signum)),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(Path.cwd()), os.environ.get("PYTHONPATH")))
            ),
        },
    )
    deadline = time.monotonic() + 8
    while not ready_path.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"signal fixture exited early with {process.returncode}")
        time.sleep(0.02)
    assert ready_path.exists()

    process.send_signal(signum)

    cleanup_deadline = time.monotonic() + 5
    while not teardown_path.exists() and time.monotonic() < cleanup_deadline:
        time.sleep(0.01)
    assert teardown_path.exists()
    process.send_signal(signum)

    assert process.wait(timeout=8) == 128 + signum
    assert json.loads(teardown_path.read_text())["pod_absence"]["verified"] is True
    assert reraised_path.read_text() == str(int(signum))
    persisted = RunSetStateStore(bundle.run_set_dir / "state.json").load()
    assert persisted.stage(STAGE_TEARDOWN).status == "completed"
    assert persisted.stage(STAGE_TEARDOWN).outputs["pod_absence"]["verified"] is True


def test_first_real_sigint_during_exception_teardown_is_deferred(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(bundle.model_dump_json(), encoding="utf-8")
    cleanup_started_path = tmp_path / "cleanup.started"
    teardown_path = tmp_path / "teardown.json"
    reraised_path = tmp_path / "signal.reraised"
    script_path = tmp_path / "signal_exception_teardown.py"
    script_path.write_text(
        """
import json
import signal
import sys
import time
from pathlib import Path

from feedbax.orchestration.bundle import RunBundle
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.stages import StageEngine

bundle = RunBundle.model_validate_json(Path(sys.argv[1]).read_text())
cleanup_started = Path(sys.argv[2])
teardown = Path(sys.argv[3])
reraised = Path(sys.argv[4])

def prior_handler(received, _frame):
    reraised.write_text(str(received))
    raise SystemExit(128 + received)

signal.signal(signal.SIGINT, prior_handler)

class Driver:
    def provision(self, bundle, state):
        return {"driver": "fixture", "pod_id": "pod-exception-signal"}
    def realize_env(self, bundle, state):
        return "fixture-fingerprint"
    def stage_inputs(self, bundle, state):
        raise RuntimeError("ordinary stage failure")
    def teardown(self, bundle, state):
        cleanup_started.write_text("started")
        time.sleep(0.3)
        outputs = {
            "teardown": "removed",
            "pod_absence": {"verified": True, "pod_id": "pod-exception-signal"},
        }
        teardown.write_text(json.dumps(outputs))
        return outputs
    def launch_row(self, bundle, row, state):
        return {}
    def probe(self, bundle, row, state):
        return DriverRowProbe(status="completed")
    def stop_row(self, bundle, row, state):
        return {}
    def collect(self, bundle, row, state):
        return {}

StageEngine(bundle=bundle, driver=Driver()).run()
""".lstrip(),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(script_path),
            str(bundle_path),
            str(cleanup_started_path),
            str(teardown_path),
            str(reraised_path),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(Path.cwd()), os.environ.get("PYTHONPATH")))
            ),
        },
    )
    deadline = time.monotonic() + 8
    while not cleanup_started_path.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"signal fixture exited early with {process.returncode}")
        time.sleep(0.02)
    assert cleanup_started_path.exists()

    process.send_signal(signal.SIGINT)

    assert process.wait(timeout=8) == 128 + signal.SIGINT
    assert json.loads(teardown_path.read_text())["pod_absence"]["verified"] is True
    assert reraised_path.read_text() == str(int(signal.SIGINT))
    persisted = RunSetStateStore(bundle.run_set_dir / "state.json").load()
    assert persisted.stage(STAGE_TEARDOWN).status == "completed"
    assert persisted.stage(STAGE_TEARDOWN).outputs["pod_absence"]["verified"] is True
