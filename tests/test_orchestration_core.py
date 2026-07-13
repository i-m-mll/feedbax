from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.manifest import TrainingRunManifest
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
from feedbax.orchestration import conformance, schedule_eval, stages
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    CompiledExecutionRow,
    CompiledRunSet,
    CompilerIdentity,
    RunAssemblyRequest,
    assemble_run_bundle,
)
from feedbax.orchestration.bundle import (
    RUN_BUNDLE_SCHEMA_ID,
    RUN_BUNDLE_SCHEMA_VERSION,
    RUN_BUNDLE_SCHEMA_VERSION_V1,
    RUN_BUNDLE_SCHEMA_VERSION_V2,
    BudgetPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    RepoRevision,
    RunBundle,
    RunRowSpec,
    RowLaunchSpec,
    SchemaArtifactRef,
)
from feedbax.orchestration.conformance import (
    CheckEntry,
    CheckRegistry,
    build_default_check_registry,
)
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.drivers.local import (
    LocalDriverError,
    LocalOrchestrationDriver,
    compute_environment_fingerprint,
)
from feedbax.orchestration.stages import (
    STAGE_ORDER,
    STAGE_PREFLIGHT,
    OrchestrationStageError,
    PreflightFailed,
    StageEngine,
    run_preflight_checks,
)
from feedbax.orchestration.state import (
    RUN_SET_STATE_SCHEMA_ID,
    RUN_SET_STATE_SCHEMA_VERSION,
    RowState,
    RunSetState,
    RunSetStateStore,
    StateLockError,
)
from feedbax.training.diagnostics import TRAINING_DIAGNOSTICS_SCHEMA_ID, TrainingDiagnostics
from feedbax.training.interruption import CancellationAction, CancellationDecision
from feedbax.training.manifest_preflight import preflight_training_run_manifest_payloads


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
        return {"provisioned": True}

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

    def compile(
        self,
        request: RunAssemblyRequest,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        del request, authored, run_set_id, context
        return CompiledRunSet(rows=list(self.rows))


def _compiled_row(
    row_id: str,
    *,
    command: list[str] | None = None,
    collect: list[str] | None = None,
    run_spec: dict[str, Any] | None = None,
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
    run_set_id: str = "2026-01-02-deadbeef",
    python_version: str | None = "3.12",
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
        environment=EnvironmentDeclaration(python_version=python_version),
        launch_policy=launch_policy or LaunchPolicy(max_parallel_rows=2),
        budget=BudgetPolicy(max_wall_clock_seconds=max_wall_clock_seconds),
        orchestration_root=str(tmp_path),
    )
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        compiler_id=compiler_id,
        compiler_version=compiler_version,
        compiler=_FixtureCompiler(tuple(rows or [_compiled_row("row-a")])),
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
    run_set_id: str = "2026-01-02-deadbeef",
    python_version: str | None = "3.12",
) -> RunBundle:
    request, context, registry = _assembly_parts(
        tmp_path,
        rows=rows,
        launch_policy=launch_policy,
        max_wall_clock_seconds=max_wall_clock_seconds,
        run_set_id=run_set_id,
        python_version=python_version,
    )
    return assemble_run_bundle(
        request,
        run_set_id=run_set_id,
        context=context,
        registry=registry,
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
    old_payload = _bundle(tmp_path).model_dump(mode="json")
    for old_version in (RUN_BUNDLE_SCHEMA_VERSION_V1, RUN_BUNDLE_SCHEMA_VERSION_V2):
        old_payload["schema_version"] = old_version
        with pytest.raises(UnsupportedSpecVersion, match="reassemble from the authored"):
            default_spec_registry.migrate("RunBundle", old_payload)
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "RunBundle",
            {"schema_version": "feedbax.orchestration.run_bundle.v0"},
        )


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


def test_request_assembly_certifies_all_eight_core_checks_with_independent_identity(
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
    expected_root_hash = "0d41d0f6fed921a7092f9a2a6d1ed349fc890f6c1c3f7074d32b4d8d806a9b96"
    expected_execution_hash = "fb332b40b7e18210f18e15172fdf86137a750e8f076da8f68122c77a90cd4f73"
    expected_artifact_hashes = {
        "authored": "e1aeb77d847c6b24011becca0db24da2f25e68f3cf542fdb4639615a266f8dc9",
        "payload": "c479f4c01118bcb1c5b2d72ae190cde4bd594fe1b8a80df592c126042e922e94",
        "snapshot": "aeb7acfcdde86b680403131d81367e38d0742369a92411096ba068f3f799c5d7",
        "capsule": "cc26ba45b954643009fb4a9498b68a6512993c588a543d3cd34192e072bb17bf",
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
    assert discovered["training_diagnostics"]["lr_trace"] == [
        {"step": 10, "learning_rate": 3e-4}
    ]
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

    assert payload == {
        "abort_reason": None,
        "certificate_overall": "fail",
        "certificate_ref": str(certificate_path),
        "certificate_sha256": certificate_digest,
        "failure_reason": "conformance-failed",
        "run_set_id": bundle.run_set_id,
        "status": "failed",
    }
    failed_state = store.load()
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

    final_state = StageEngine(
        bundle=bundle,
        driver=driver,
        store=store,
        conformance_registry=_fixture_pass_registry(),
        poll_interval_seconds=0.01,
    ).run()

    assert final_state.rows["row-a"].status == "failed"
    assert final_state.rows["row-a"].event_discrepancies[0]["code"] == "orphaned_launch"
    assert not marker.exists()


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
