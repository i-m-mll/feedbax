from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

from feedbax.bin import orchestrate
from feedbax.contracts.studio_training import (
    StudioTrainingAssemblySpec,
    StudioTrainingIdentityAdapter,
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
from feedbax.orchestration.conformance import CheckRegistry, pass_check
from feedbax.orchestration.drivers.runpod import RunPodOrchestrationDriver
from feedbax.orchestration.stages import PreflightFailed
from feedbax.orchestration.state import RowState


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


def _save_state(bundle: RunBundle, state: RunSetState) -> None:
    bundle.run_set_dir.mkdir(parents=True, exist_ok=True)
    _write_bundle(bundle, bundle.run_set_dir / "bundle.json")
    RunSetStateStore(bundle.run_set_dir / "state.json").save(state)


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
        def __init__(self) -> None:
            super().__init__(cwd=tmp_path, freeze_lines=("feedbax==test",))

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
    assert orchestrate.main(["resume", "--run-set", "local-demo"]) == 0

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

    driver = orchestrate._driver_for_bundle(bundle)

    assert isinstance(driver, RunPodOrchestrationDriver)
    assert driver.config.pod_id == "pod-123"
    assert driver.config.gpu_id == "NVIDIA GeForce RTX 4090"
    assert driver.config.datacenters == ("CA-MTL-1", "US-OR-1")
    assert driver.config.path_patches[0][0] == "/workspace/feedbax/pyproject.toml"


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5"])
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


def test_launch_cli_exposes_deadman_request_overrides() -> None:
    args = orchestrate.build_parser().parse_args(
        [
            "launch",
            "--assembly-request",
            "assembly-request.json",
            "--deadman",
            "--deadman-silence-seconds",
            "900",
        ]
    )

    assert args.deadman is True
    assert args.deadman_silence_seconds == 900
