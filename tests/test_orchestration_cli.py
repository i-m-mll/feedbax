from __future__ import annotations

import json
import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from feedbax.bin import orchestrate
import feedbax.contracts.training as training_contracts
import feedbax.plugins.discovery as plugin_discovery
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.studio_training import (
    StudioTrainingAssemblySpec,
    StudioTrainingIdentityAdapter,
)
from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_descriptor,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.orchestration import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    BudgetPolicy,
    CompiledExecutionRow,
    CompiledRunSet,
    CompilerIdentity,
    DeploymentPolicy,
    DeploymentResourceRequest,
    EnvironmentDeclaration,
    LaunchPolicy,
    RowLaunchSpec,
    RunAssemblyRequest,
    RunBundle,
    RunEventEmitter,
    RunSetState,
    RunSetStateStore,
    SchemaArtifactRef,
    StageState,
    StateLockError,
    assemble_run_bundle,
)
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.orchestration.drivers import runpod as runpod_driver_module
from feedbax.orchestration.conformance import CheckRegistry, pass_check
from feedbax.orchestration.drivers.runpod import RunPodOrchestrationDriver
from feedbax.orchestration.stages import PreflightFailed
from feedbax.orchestration.state import RowState
from feedbax.plugins.discovery import load_training_method_plugins
from feedbax.training.preparation import ExecutionPreparationProviderRegistry
from feedbax.training.spec_storage import (
    TRAINING_RUN_MATRIX_COMPILER_ID,
    TRAINING_RUN_MATRIX_COMPILER_VERSION,
    register_training_run_matrix_compiler,
)


_PLUGIN_METHOD_REF = "tests/orchestration_plugin/v1"
_PLUGIN_SCHEMA_ID = "tests.spec.orchestration_plugin"
_PLUGIN_SCHEMA_VERSION = "tests.spec.orchestration_plugin.v1"


class _FixtureCompiler:
    def __init__(self, launches: list[tuple[str, RowLaunchSpec]]) -> None:
        self.launches = launches

    def compile(
        self,
        *,
        authored: dict[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        del run_set_id, context
        return CompiledRunSet(
            rows=[
                CompiledExecutionRow(
                    row_id=row_id,
                    payload=authored,
                    resolved_semantics={**authored, "fixture_row_id": row_id},
                    immutable_inputs=[],
                    launch=launch,
                )
                for row_id, launch in self.launches
            ]
        )


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


def _assembly_request(
    tmp_path: Path,
    *,
    launches: list[tuple[str, RowLaunchSpec]] | None = None,
    driver: str = "local",
    environment: EnvironmentDeclaration | None = None,
    max_wall_clock_seconds: float = 10.0,
) -> tuple[RunAssemblyRequest, AssemblyCompilerRegistry]:
    spec = StudioTrainingAssemblySpec(total_batches=1)
    authored_path = tmp_path / "studio-training.json"
    authored_bytes = spec.model_dump_json(exclude_none=True).encode("utf-8")
    authored_path.parent.mkdir(parents=True, exist_ok=True)
    authored_path.write_bytes(authored_bytes)
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=spec.schema_id,
            schema_version=spec.schema_version,
            artifact_id=f"fixture:{hashlib.sha256(authored_bytes).hexdigest()}",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id="feedbax.test.cli",
            compiler_version="feedbax.test.cli.v1",
        ),
        deployment_policy=_deployment_policy(driver),
        environment=environment or EnvironmentDeclaration(python_version="3.12"),
        launch_policy=LaunchPolicy(max_parallel_rows=2),
        budget=BudgetPolicy(max_wall_clock_seconds=max_wall_clock_seconds),
        orchestration_root=str(tmp_path),
    )
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=spec.schema_id,
        compiler_id=request.compiler.compiler_id,
        compiler_version=request.compiler.compiler_version,
        compiler=_FixtureCompiler(
            launches
            or [("row-a", RowLaunchSpec(command=[sys.executable, "-c", "pass"]))]
        ),
        identity_adapter=StudioTrainingIdentityAdapter(),
    )
    return request, registry


def _bundle(
    tmp_path: Path,
    *,
    run_set_id: str = "2026-01-02-cli",
    launches: list[tuple[str, RowLaunchSpec]] | None = None,
    driver: str = "local",
    environment: EnvironmentDeclaration | None = None,
    max_wall_clock_seconds: float = 10.0,
) -> RunBundle:
    request, registry = _assembly_request(
        tmp_path,
        launches=launches,
        driver=driver,
        environment=environment,
        max_wall_clock_seconds=max_wall_clock_seconds,
    )
    return assemble_run_bundle(
        request,
        run_set_id=run_set_id,
        context=AssemblyContext(custody_root=tmp_path / "custody"),
        registry=registry,
    )


def _write_request(request: RunAssemblyRequest, path: Path) -> Path:
    path.write_text(request.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _write_bundle(bundle: RunBundle, path: Path) -> Path:
    path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _register_orchestration_plugin_method(registry: Any) -> None:
    contract = standard_supervised_method_contract().model_copy(
        update={
            "method_ref": _PLUGIN_METHOD_REF,
            "method_payload_schema_version": _PLUGIN_SCHEMA_VERSION,
        }
    )
    registry.register_descriptor(
        replace(
            standard_supervised_method_descriptor(),
            method_ref=_PLUGIN_METHOD_REF,
            payload_schema_id=_PLUGIN_SCHEMA_ID,
            payload_schema_version=_PLUGIN_SCHEMA_VERSION,
            contract_compiler=lambda _payload: contract,
            rejected_payload_versions=(),
            owner="tests.test_orchestration_cli",
            package="tests",
        )
    )


def _standard_training_run_payload() -> dict[str, Any]:
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
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=1, batch_size=1, learning_rate=0.01),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state",
                label="target",
                selector="port:gain.output",
                target_value=[0.0],
            )
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    ).model_dump(mode="json", exclude_none=True)


def _plugin_training_run_payload() -> dict[str, Any]:
    payload = _standard_training_run_payload()
    payload["method_ref"] = {
        "package": "tests",
        "name": "orchestration_plugin",
        "version": "v1",
    }
    method_payload = standard_supervised_method_payload().model_dump(
        mode="json", exclude_none=True
    )
    method_payload["schema_id"] = _PLUGIN_SCHEMA_ID
    method_payload["schema_version"] = _PLUGIN_SCHEMA_VERSION
    method_payload["payload"]["optimizer"]["params"]["learning_rate"] = 0.01
    payload["method_payload"] = method_payload
    worker_execution = payload["worker_execution"]
    worker_execution["method_contract"]["method_ref"] = _PLUGIN_METHOD_REF
    worker_execution["method_contract"][
        "method_payload_schema_version"
    ] = _PLUGIN_SCHEMA_VERSION
    worker_execution["effective_phase"]["method_ref"] = _PLUGIN_METHOD_REF
    return payload


def _matrix_request(
    tmp_path: Path,
    *,
    training_run_payload: dict[str, Any],
) -> tuple[RunAssemblyRequest, AssemblyCompilerRegistry]:
    matrix = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "orchestration plugin discovery",
        "base": {"kind": "inline", "inline": training_run_payload},
        "rows": [{"row_id": "plugin-row", "seed": 7}],
    }
    authored_bytes = json.dumps(matrix, sort_keys=True).encode("utf-8")
    authored_path = tmp_path / "training-matrix.json"
    authored_path.write_bytes(authored_bytes)
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            schema_version=TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            artifact_id=f"fixture:{hashlib.sha256(authored_bytes).hexdigest()}",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
            compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
        ),
        deployment_policy=_deployment_policy(),
        environment=EnvironmentDeclaration(python_version="3.12"),
        launch_policy=LaunchPolicy(max_parallel_rows=1),
        budget=BudgetPolicy(max_wall_clock_seconds=10.0),
        orchestration_root=str(tmp_path / "orchestration"),
    )
    registry = AssemblyCompilerRegistry()
    register_training_run_matrix_compiler(registry, allow_inline_base=True)
    return request, registry


def _plugin_matrix_request(tmp_path: Path) -> tuple[RunAssemblyRequest, AssemblyCompilerRegistry]:
    return _matrix_request(tmp_path, training_run_payload=_plugin_training_run_payload())


def _save_state(bundle: RunBundle, state: RunSetState) -> None:
    bundle.run_set_dir.mkdir(parents=True, exist_ok=True)
    _write_bundle(bundle, bundle.run_set_dir / "bundle.json")
    RunSetStateStore(bundle.run_set_dir / "state.json").save(state)


def test_preflight_loads_non_builtin_training_method_entry_point_before_matrix_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_registry = training_contracts.DEFAULT_TRAINING_METHOD_REGISTRY
    monkeypatch.setattr(
        method_registry,
        "_registrations",
        method_registry._registrations.copy(),
    )
    monkeypatch.setattr(
        method_registry,
        "_descriptors",
        method_registry._descriptors.copy(),
    )
    preparation_registry = ExecutionPreparationProviderRegistry()
    plugin = SimpleNamespace(
        register_feedbax_training_methods=_register_orchestration_plugin_method
    )
    monkeypatch.setattr(
        orchestrate,
        "load_training_method_plugins",
        lambda **kwargs: load_training_method_plugins(
            preparation_registry=preparation_registry,
            entry_points=[SimpleNamespace(name="orchestration-method", load=lambda: plugin)],
            **kwargs,
        ),
    )
    request, assembly_registry = _plugin_matrix_request(tmp_path)
    request_path = _write_request(request, tmp_path / "assembly-request.json")
    monkeypatch.setattr(
        orchestrate,
        "build_default_assembly_registry",
        lambda: assembly_registry,
    )
    monkeypatch.setattr(
        orchestrate,
        "build_default_check_registry",
        lambda: CheckRegistry({"fixture_pass": lambda _row: pass_check("fixture_pass")}),
    )

    assert orchestrate.main(["preflight", "--assembly-request", str(request_path)]) == 0
    assert _PLUGIN_METHOD_REF in method_registry.available_keys()


@pytest.mark.parametrize("command", ["preflight", "launch"])
def test_matrix_commands_load_training_plugins_before_request_validation(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _assembly_request(tmp_path)
    request_path = _write_request(request, tmp_path / "assembly-request.json")
    events: list[tuple[str, bool] | str] = []

    monkeypatch.setattr(
        orchestrate,
        "load_training_method_plugins",
        lambda *, fail_on_load_error: events.append(("plugins", fail_on_load_error)),
    )
    monkeypatch.setattr(
        orchestrate,
        "_load_assembly_request",
        lambda _path: events.append("request") or request,
    )

    class FakeEngine:
        def run(self, **_kwargs: Any) -> RunSetState:
            return RunSetState(
                run_set_id="plugin-order",
                rows={"row": RowState(status="completed")},
                stages={
                    "PREFLIGHT": StageState(
                        status="completed",
                        checks=[{"name": "fixture_pass", "status": "pass"}],
                    )
                },
            )

    monkeypatch.setattr(orchestrate, "_request_engine", lambda *_args, **_kwargs: FakeEngine())

    assert orchestrate.main([command, "--assembly-request", str(request_path)]) == 0
    assert events == [("plugins", True), "request"]


def test_broken_installed_plugin_fails_before_builtin_matrix_engine_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setattr(
        plugin_discovery,
        "feedbax_plugin_entry_points",
        lambda _group: [
            SimpleNamespace(
                name="broken-orchestration-method",
                load=lambda: (_ for _ in ()).throw(RuntimeError("broken plugin")),
            )
        ],
    )
    request, _ = _matrix_request(
        tmp_path,
        training_run_payload=_standard_training_run_payload(),
    )
    request_path = _write_request(request, tmp_path / "assembly-request.json")
    monkeypatch.setattr(
        orchestrate,
        "_request_engine",
        lambda *_args, **_kwargs: pytest.fail("assembly engine must not be constructed"),
    )
    monkeypatch.setattr(
        orchestrate,
        "LocalOrchestrationDriver",
        lambda *_args, **_kwargs: pytest.fail("provider driver must not be constructed"),
    )

    assert orchestrate.main(["preflight", "--assembly-request", str(request_path)]) == 1
    assert capsys.readouterr().err.strip().endswith(
        "Failed to load Feedbax training-method plugin "
        "entry-point:broken-orchestration-method: broken plugin"
    )


def test_status_line_format_is_stable(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    run_set_dir = bundle.run_set_dir
    event_path = run_set_dir / "events" / "row-a.events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    RunEventEmitter(
        run_set_id=bundle.run_set_id,
        row_id="row-a",
        path=event_path,
        heartbeat_seconds=None,
    ).emit_progress(
        {"batch": 3, "total_batches": 8, "loss": 0.125, "phase": "train"},
        batch=3,
        total_batches=8,
        force=True,
    )
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={"row-a": RowState(status="running", event_seq_high_water_mark=0)},
        stages={
            "ASSEMBLE": StageState(status="completed"),
            "PREFLIGHT": StageState(status="completed"),
            "LAUNCH": StageState(status="completed"),
        },
    )

    line = orchestrate.format_status_line(
        state,
        "row-a",
        run_set_dir=run_set_dir,
        now_ms=event_path.stat().st_mtime_ns // 1_000_000 + 10_000,
    )

    assert line.startswith("row=row-a status=running batch=3/8 last_loss=0.125 last_event_age_s=")
    assert " seq=0 stages=ASSEMBLE:completed,PREFLIGHT:completed,PROVISION:pending" in line


def test_json_status_validates_state_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    bundle = _bundle(tmp_path, run_set_id="json-status")
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={"row-a": RowState(status="completed")},
    )
    _save_state(bundle, state)

    assert orchestrate.main(["status", "--run-set", bundle.run_set_id, "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert RunSetState.model_validate(payload).run_set_id == "json-status"


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (PreflightFailed("bad bundle"), orchestrate.EXIT_PREFLIGHT),
        (StateLockError("locked"), orchestrate.EXIT_LOCK),
        (RuntimeError("other"), orchestrate.EXIT_OTHER),
        (
            RunSetState(
                run_set_id="set",
                rows={"row": RowState(status="failed")},
            ),
            orchestrate.EXIT_ROW_FAILURE,
        ),
        (
            RunSetState(
                run_set_id="set",
                rows={"row": RowState(status="stopped")},
                abort_reason="budget-exceeded",
            ),
            orchestrate.EXIT_BUDGET,
        ),
    ],
)
def test_exit_code_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: BaseException | RunSetState,
    expected: int,
) -> None:
    request, _ = _assembly_request(tmp_path)
    request_path = _write_request(request, tmp_path / "assembly-request.json")

    class FakeEngine:
        def run(self) -> RunSetState:
            if isinstance(result, BaseException):
                raise result
            return result

    monkeypatch.setattr(orchestrate, "_request_engine", lambda *_args, **_kwargs: FakeEngine())

    assert orchestrate.main(["launch", "--assembly-request", str(request_path)]) == expected


def test_watch_exits_after_all_rows_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    bundle = _bundle(
        tmp_path,
        run_set_id="watch-terminal",
        launches=[
            ("a", RowLaunchSpec(command=[sys.executable, "-c", "pass"])),
            ("b", RowLaunchSpec(command=[sys.executable, "-c", "pass"])),
        ],
    )
    events_dir = bundle.run_set_dir / "events"
    for row_id in ("a", "b"):
        emitter = RunEventEmitter(
            run_set_id=bundle.run_set_id,
            row_id=row_id,
            path=events_dir / f"{row_id}.events.jsonl",
            heartbeat_seconds=None,
        )
        emitter.emit_terminal("complete", {"row": row_id})
        emitter.close()
    _save_state(
        bundle,
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={"a": RowState(status="completed"), "b": RowState(status="completed")},
        ),
    )

    assert orchestrate.main(["watch", "--run-set", bundle.run_set_id, "--poll-interval", "0"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        'row=a seq=0 type=complete payload={"row":"a"}',
        'row=b seq=0 type=complete payload={"row":"b"}',
    ]


def test_collect_and_teardown_are_idempotent_after_completed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    bundle = _bundle(tmp_path, run_set_id="idempotent")
    stages = {stage: StageState(status="completed") for stage in orchestrate.STAGE_ORDER}
    _save_state(
        bundle,
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={"row-a": RowState(status="completed")},
            stages=stages,
        ),
    )

    assert orchestrate.main(["collect", "--run-set", bundle.run_set_id]) == 0
    assert orchestrate.main(["collect", "--run-set", bundle.run_set_id]) == 0
    assert orchestrate.main(["teardown", "--run-set", bundle.run_set_id]) == 0
    assert orchestrate.main(["teardown", "--run-set", bundle.run_set_id, "--force"]) == 0


def test_certify_explicitly_retries_a_completed_failed_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    def run_existing(_run_set_id: str, **kwargs: Any) -> RunSetState:
        calls.append(("run", kwargs))
        return RunSetState(
            run_set_id="failed-certificate",
            rows={"row-a": RowState(status="completed")},
            stages={
                "CERTIFY": StageState(
                    status="completed",
                    outputs={"overall": "pass"},
                )
            },
        )

    monkeypatch.setattr(
        orchestrate,
        "load_training_method_plugins",
        lambda **kwargs: calls.append(("plugins", kwargs)),
    )
    monkeypatch.setattr(orchestrate, "_run_existing", run_existing)

    assert orchestrate.main(["certify", "--run-set", "failed-certificate"]) == 0
    assert calls == [
        ("plugins", {"fail_on_load_error": True}),
        (
            "run",
            {
                "stop_after_stage": "CERTIFY",
                "retry_failed_certification": True,
            },
        ),
    ]


def test_two_row_local_driver_demo_through_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orch"))
    script = tmp_path / "row.py"
    script.write_text(
        """
from pathlib import Path
import os
from feedbax.orchestration.events import RunEventEmitter

row = os.environ["FEEDBAX_ROW_ID"]
row_dir = Path(os.environ["FEEDBAX_ROW_DIR"])
with RunEventEmitter.from_env(heartbeat_seconds=None) as emitter:
    emitter.emit("ready", {"row": row})
    emitter.emit_progress(
        {"phase": "train", "batch": 1, "total_batches": 1, "loss": 0.5},
        batch=1,
        total_batches=1,
        force=True,
    )
    (row_dir / "payload.json").write_text('{"row": "%s"}\\n' % row, encoding="utf-8")
    emitter.emit_terminal("complete", {"row": row})
""".strip(),
        encoding="utf-8",
    )
    request, registry = _assembly_request(
        tmp_path / "orch",
        launches=[
            (
                "row-a",
                RowLaunchSpec(
                    command=[sys.executable, str(script)],
                    collect=["payload.json"],
                ),
            ),
            (
                "row-b",
                RowLaunchSpec(
                    command=[sys.executable, str(script)],
                    collect=["payload.json"],
                ),
            ),
        ],
    )
    request_path = _write_request(request, tmp_path / "assembly-request.json")

    class FastLocalDriver(LocalOrchestrationDriver):
        seen_bindings = ()
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(cwd=tmp_path, freeze_lines=("feedbax==test",), **kwargs)
            type(self).seen_bindings = self.input_provider_bindings

    monkeypatch.setattr(orchestrate, "LocalOrchestrationDriver", FastLocalDriver)
    monkeypatch.setattr(orchestrate, "build_default_assembly_registry", lambda: registry)
    monkeypatch.setattr(
        orchestrate,
        "build_default_check_registry",
        lambda: CheckRegistry({"fixture_pass": lambda _row: pass_check("fixture_pass")}),
    )

    assert orchestrate.main(["preflight", "--assembly-request", str(request_path)]) == 0
    assert orchestrate.main(
        [
            "launch",
            "--assembly-request",
            str(request_path),
            "--driver",
            "local",
            "--resume-run-set",
            "local-demo",
        ]
    ) == 0
    assert orchestrate.main(["status", "--run-set", "local-demo"]) == 0
    assert orchestrate.main(["certify", "--run-set", "local-demo"]) == 0
    assert orchestrate.main(["teardown", "--run-set", "local-demo"]) == 0
    assert orchestrate.main(["resume", "--run-set", "local-demo", "--input-provider", f"checkpoint.inputs={tmp_path}"]) == 0
    assert FastLocalDriver.seen_bindings[0].name == "checkpoint.inputs"

    status_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("row=")
    ]
    assert status_lines == [
        "row=row-a status=completed batch=1/1 last_loss=0.5 "
        "last_event_age_s=0 seq=2 stages=ASSEMBLE:completed,PREFLIGHT:completed,"
        "PROVISION:completed,REALIZE_ENV:completed,STAGE_INPUTS:completed,"
        "LAUNCH:completed,MONITOR:completed,COLLECT:completed,CERTIFY:completed,"
        "TEARDOWN:completed,REGISTER:completed",
        "row=row-b status=completed batch=1/1 last_loss=0.5 "
        "last_event_age_s=0 seq=2 stages=ASSEMBLE:completed,PREFLIGHT:completed,"
        "PROVISION:completed,REALIZE_ENV:completed,STAGE_INPUTS:completed,"
        "LAUNCH:completed,MONITOR:completed,COLLECT:completed,CERTIFY:completed,"
        "TEARDOWN:completed,REGISTER:completed",
    ]


def test_runpod_driver_is_constructed_from_typed_deployment_policy(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        driver="runpod",
        environment=EnvironmentDeclaration(
            python_version="3.12",
            image_id="runpod/pytorch:1.0.3",
            metadata={
                "runpod_pod_id": "pod-123",
                "runpod_ssh_host": "198.51.100.10",
                "runpod_ssh_port": 2222,
                "runpod_path_patches": [
                    {
                        "remote_file": "/workspace/feedbax/pyproject.toml",
                        "from": "/local/feedbax",
                        "to": "/workspace/feedbax",
                    }
                ],
            },
        ),
    )

    bindings = orchestrate._input_provider_bindings([f"checkpoint.inputs={tmp_path}"])
    driver = orchestrate._driver_for_bundle(bundle, bindings)

    assert isinstance(driver, RunPodOrchestrationDriver)
    assert driver.config.pod_id == "pod-123"
    assert driver.config.gpu_id == "NVIDIA GeForce RTX 4090"
    assert driver.config.datacenters == ("CA-MTL-1", "US-OR-1")
    assert driver.config.path_patches[0][0] == "/workspace/feedbax/pyproject.toml"
    assert driver.input_provider_bindings == bindings


def test_collection_recovery_binding_requires_row_and_absolute_root(tmp_path: Path) -> None:
    bindings = orchestrate._collection_recovery_bindings([f"r5={tmp_path}"])

    assert bindings[0].row_id == "r5"
    assert Path(bindings[0].root) == tmp_path
    with pytest.raises(ValueError, match="ROW=ABSOLUTE_PATH"):
        orchestrate._collection_recovery_bindings(["r5=relative"])


def test_certify_cli_accepts_input_provider_binding(tmp_path: Path) -> None:
    args = orchestrate.build_parser().parse_args(
        [
            "certify",
            "--run-set",
            "run-set",
            "--input-provider",
            f"checkpoint.source={tmp_path}",
        ]
    )

    assert orchestrate._input_provider_bindings(args.input_provider)[0].name == (
        "checkpoint.source"
    )


def test_resume_loads_training_method_plugins_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        orchestrate,
        "load_training_method_plugins",
        lambda **kwargs: calls.append(("plugins", kwargs)),
    )
    monkeypatch.setattr(
        orchestrate,
        "_run_existing",
        lambda *_args, **kwargs: (
            calls.append(("run", kwargs))
            or RunSetState(
                run_set_id="resumed",
                rows={"row-a": RowState(status="completed")},
            )
        ),
    )

    assert orchestrate.main(["resume", "--run-set", "resumed"]) == 0
    assert calls[0] == ("plugins", {"fail_on_load_error": True})
    assert calls[1][0] == "run"


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5", "v6"])
def test_load_bundle_rejects_legacy_versions_for_launch(tmp_path: Path, version: str) -> None:
    path = tmp_path / "bundle-v1.json"
    payload = _bundle(tmp_path).model_dump(mode="json")
    payload["schema_version"] = f"feedbax.orchestration.run_bundle.{version}"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="reassemble from a current RunAssemblyRequest"):
        orchestrate._load_bundle(path)


def test_load_assembly_request_rejects_v1_without_review_authorization(tmp_path: Path) -> None:
    request, _ = _assembly_request(tmp_path)
    payload = request.model_dump(mode="json")
    payload["schema_version"] = "feedbax.spec.run_assembly_request.v1"
    path = tmp_path / "request-v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="re-author a current request"):
        orchestrate._load_assembly_request(path)


def test_launch_driver_override_conflict_fails_before_engine_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = _assembly_request(tmp_path)
    path = _write_request(request, tmp_path / "assembly-request.json")
    monkeypatch.setattr(
        orchestrate,
        "_request_engine",
        lambda *_args, **_kwargs: pytest.fail("engine must not be constructed"),
    )

    assert orchestrate.main(
        ["launch", "--assembly-request", str(path), "--driver", "runpod"]
    ) == orchestrate.EXIT_OTHER


def test_launch_dry_run_binds_rows_without_credentials_or_stage_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request, _ = _assembly_request(tmp_path, driver="runpod")
    path = _write_request(request, tmp_path / "assembly-request.json")
    bundle = _bundle(tmp_path / "bundle", driver="runpod")
    monkeypatch.setattr(orchestrate, "assemble_run_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(orchestrate, "load_training_method_plugins", lambda **_kwargs: None)
    monkeypatch.setattr(
        orchestrate,
        "load_runpod_api_key",
        lambda: pytest.fail("dry-run must not load RunPod credentials"),
    )
    monkeypatch.setattr(
        orchestrate,
        "_request_engine",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not create a stage engine"),
    )
    monkeypatch.setattr(
        runpod_driver_module,
        "SubprocessRunPodTransport",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not construct a transport"),
    )

    assert orchestrate.main(["launch", "--assembly-request", str(path), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "row=row-a dry-run=accepted" in output


def test_launch_dry_run_rejects_local_deployment_policy(tmp_path: Path) -> None:
    request, _ = _assembly_request(tmp_path, driver="local")
    path = _write_request(request, tmp_path / "assembly-request.json")

    assert orchestrate.main(["launch", "--assembly-request", str(path), "--dry-run"]) == (
        orchestrate.EXIT_OTHER
    )


def test_launch_cli_exposes_deadman_request_overrides() -> None:
    args = orchestrate.build_parser().parse_args(
        [
            "launch",
            "--assembly-request",
            "assembly-request.json",
            "--deadman", "--deadman-silence-seconds", "900",
            "--input-provider", "/invalid-relative",
        ]
    )

    assert args.deadman is True
    assert args.deadman_silence_seconds == 900
    with pytest.raises(ValueError, match="NAME=ABSOLUTE_PATH"):
        orchestrate._input_provider_bindings(args.input_provider)
