from __future__ import annotations

import ast
import copy
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from feedbax.execution.backends import render_modal_app
from feedbax.execution.local import run_local_execution
from feedbax.execution.models import (
    EXECUTION_PLAN_SCHEMA_VERSION,
    EXECUTION_SPEC_SCHEMA_VERSION,
    ExecutionCell,
    ExecutionSpec,
    LOCAL_EXECUTION_RESULT_SCHEMA_VERSION,
    RepoSource,
    validate_materialized_execution_artifact,
)
from feedbax.execution.planning import (
    default_feedbax_sources,
    prepare_execution_plan,
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
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.integrations.provider import provider_manifest
from feedbax.web.app import create_app


class _StubModalImage:
    @staticmethod
    def debian_slim(*args: object, **kwargs: object) -> "_StubModalImage":
        return _StubModalImage()

    def apt_install(self, *args: object, **kwargs: object) -> "_StubModalImage":
        return self

    def pip_install(self, *args: object, **kwargs: object) -> "_StubModalImage":
        return self

    def env(self, *args: object, **kwargs: object) -> "_StubModalImage":
        return self

    def add_local_dir(self, *args: object, **kwargs: object) -> "_StubModalImage":
        return self

    def workdir(self, *args: object, **kwargs: object) -> "_StubModalImage":
        return self

    def run_commands(self, *args: object, **kwargs: object) -> "_StubModalImage":
        return self


class _StubModalApp:
    def __init__(self, name: str) -> None:
        self.name = name

    def function(self, **kwargs: object) -> object:
        def _decorator(function: object) -> object:
            return function

        return _decorator

    def local_entrypoint(self) -> object:
        def _decorator(function: object) -> object:
            return function

        return _decorator


class _StubModalVolume:
    @staticmethod
    def from_name(name: str, *, create_if_missing: bool = False) -> "_StubModalVolume":
        return _StubModalVolume()

    def commit(self) -> None:
        return None


class _StubModalSecret:
    @staticmethod
    def from_name(name: str) -> object:
        return {"name": name}


def _stub_modal_module() -> types.ModuleType:
    modal = types.ModuleType("modal")
    modal.Image = _StubModalImage
    modal.App = _StubModalApp
    modal.Volume = _StubModalVolume
    modal.Secret = _StubModalSecret
    return modal


def _minimal_graph() -> dict[str, object]:
    return {
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


def _training_run_payload() -> dict[str, object]:
    worker = WorkerExecutionSpec(
        method_contract=standard_supervised_method_contract(),
        effective_phase=standard_supervised_effective_phase_spec(),
    )
    spec = TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=2, batch_size=3),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=worker,
    )
    return spec.model_dump(mode="json")


def _training_source() -> dict[str, object]:
    return {
        "kind": "inline",
        "identity": "studio:training-run-spec:toy",
        "inline": _training_run_payload(),
    }


def _training_source_with_json_literals() -> dict[str, object]:
    source = copy.deepcopy(_training_source())
    inline = source["inline"]
    assert isinstance(inline, dict)
    inline["metadata"] = {
        "enabled": True,
        "disabled": False,
        "nullable": None,
        "nested": [{"present": True, "missing": None}],
    }
    task = inline["task"]
    assert isinstance(task, dict)
    params = task["params"]
    assert isinstance(params, dict)
    params["json_literals"] = {
        "enabled": True,
        "disabled": False,
        "nullable": None,
    }
    return source


def _json_literal_cell_params() -> dict[str, object]:
    return {
        "enabled": True,
        "disabled": False,
        "nullable": None,
        "nested": [{"present": True, "missing": None}],
    }


def _exec_rendered_modal_app(rendered: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setitem(sys.modules, "modal", _stub_modal_module())
    namespace: dict[str, object] = {}
    exec(compile(rendered, "<feedbax_modal_execution.py>", "exec"), namespace)
    return namespace


def _forbidden_json_literal_names(rendered: str) -> set[str]:
    parsed = ast.parse(rendered)
    return {
        node.id
        for node in ast.walk(parsed)
        if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}
    }


def test_execution_spec_declares_and_rejects_schema_versions() -> None:
    spec = ExecutionSpec(command="python train.py")

    assert spec.schema_version == EXECUTION_SPEC_SCHEMA_VERSION
    assert spec.model_dump(mode="json")["schema_version"] == EXECUTION_SPEC_SCHEMA_VERSION

    with pytest.raises(ValidationError):
        ExecutionSpec.model_validate(
            {
                "schema_version": "feedbax.spec.execution.v1",
                "command": "python train.py",
            }
        )


def test_training_execution_plan_derives_local_command_from_training_run_spec() -> None:
    spec = ExecutionSpec(
        backend="local",
        job_id="training-local",
        command="python rlrmp_train.py --duplicated-flag 123",
        training_run_spec=_training_source(),
    )

    plan = prepare_execution_plan(spec)

    assert "python -m feedbax execute-training-run-spec" in plan.command
    assert "rlrmp_train.py" not in plan.command
    assert "--duplicated-flag" not in plan.command
    assert "--run-id training-local" in plan.command
    training_record = plan.reproducibility["training_run_spec"]
    assert plan.schema_version == EXECUTION_PLAN_SCHEMA_VERSION
    assert training_record["identity"] == "studio:training-run-spec:toy"
    assert len(training_record["content_sha256"]) == 64
    assert training_record["path"].endswith("/feedbax_runs/training-local/training-run-spec.json")
    spec_route = next(route for route in plan.artifact_routes if route.role == "training_run_spec")
    assert spec_route.logical_name == "training-run-spec.json"
    assert spec_route.uri is not None
    assert spec_route.uri.endswith("/feedbax_runs/training-local/training-run-spec.json")
    assert spec_route.sha256 is None
    assert spec_route.size_bytes is None
    assert spec_route.metadata["hash_status"] == "deferred_until_materialization"
    assert spec_route.metadata["size_status"] == "deferred_until_materialization"
    manifest_route = next(
        route for route in plan.artifact_routes if route.role == "training_run_manifest"
    )
    assert manifest_route.uri is not None
    assert manifest_route.uri.endswith(
        "/feedbax_runs/training-local/manifests/training_runs/"
        "feedbax-training-run_training-local.json"
    )
    assert any("ignore command" in warning for warning in plan.warnings)


def test_training_execution_plan_accepts_referenced_training_run_spec() -> None:
    spec = ExecutionSpec(
        backend="local",
        job_id="training-ref",
        training_run_spec={
            "kind": "ref",
            "ref": "repo://specs/training-run.json",
            "content_sha256": "a" * 64,
        },
    )

    plan = prepare_execution_plan(spec)

    record = plan.reproducibility["training_run_spec"]
    assert "python -m feedbax execute-training-run-spec" in plan.command
    assert record["source_kind"] == "ref"
    assert record["identity"] == "repo://specs/training-run.json"
    assert record["ref"] == "repo://specs/training-run.json"
    assert record["content_sha256"] == "a" * 64


def test_runpod_plan_uses_ssh_worker_contract() -> None:
    spec = ExecutionSpec(
        backend="runpod",
        job_id="runpod-smoke",
        command="uv run --no-sync python scripts/train.py",
        repos=[
            *default_feedbax_sources(feedbax_ref="abc123"),
            RepoSource(
                name="rlrmp",
                role="tooling",
                install_mode="github-ref",
                package="rlrmp",
                git_url="https://github.com/i-m-mll/rlrmp.git",
                git_ref="def456",
            ),
        ],
        primary_repo="feedbax",
        issues=["c6b1b73"],
    )

    plan = prepare_execution_plan(spec)

    assert plan.backend == "runpod"
    assert plan.run_directory == "/workspace/feedbax_runs/runpod-smoke"
    assert plan.cloud_payload["provider"] == "runpod"
    assert plan.cloud_payload["pod_request"]["imageName"].startswith("runpod/pytorch")
    assert plan.cloud_payload["pod_request"]["volumeMountPath"] == "/workspace"
    assert "22/tcp" in plan.cloud_payload["pod_request"]["ports"]
    assert "--gpu-id 'NVIDIA GeForce RTX 5090'" in plan.cloud_payload["runpodctl_create"]
    assert "ssh.ip" in plan.cloud_payload["readiness"][0]
    assert plan.health_checks[0].id == "ssh"
    assert any(check.id == "gpu" for check in plan.health_checks)
    assert "runtime.ports" in plan.cloud_payload["readiness"][1]
    assert plan.launch.command is not None
    assert "nohup bash -lc" in plan.launch.command
    assert "/workspace/feedbax_runs/runpod-smoke/done" in plan.launch.command
    assert plan.reproducibility["repo_refs"]["feedbax"]["git_ref"] == "abc123"


def test_runpod_training_plan_records_training_source_without_provider_contact() -> None:
    spec = ExecutionSpec(
        backend="runpod",
        job_id="runpod-training-spec",
        training_run_spec=_training_source(),
        repos=default_feedbax_sources(feedbax_ref="abc123"),
        primary_repo="feedbax",
    )

    plan = prepare_execution_plan(spec)

    assert "python -m feedbax execute-training-run-spec" in plan.command
    assert plan.command.count("--") == 3
    assert "scripts/train" not in plan.command
    assert plan.cloud_payload["provider"] == "runpod"
    assert plan.cloud_payload["training_run_spec"] == plan.reproducibility["training_run_spec"]
    assert plan.cloud_payload["training_run_spec"]["path"] == (
        "/workspace/feedbax_runs/runpod-training-spec/training-run-spec.json"
    )
    assert any(check.id == "ssh" for check in plan.health_checks)
    assert any(check.id == "gpu" for check in plan.health_checks)


def test_runpod_plan_marks_local_rsync_as_dev_override() -> None:
    spec = ExecutionSpec(
        backend="runpod",
        command="python -m feedbax.bin.train",
        repos=[
            RepoSource(
                name="feedbax",
                role="project",
                install_mode="local-rsync",
                local_path="/tmp/feedbax",
            )
        ],
    )

    plan = prepare_execution_plan(spec)

    assert any("local-rsync is a development override" in warning for warning in plan.warnings)
    sync = next(step for step in plan.bootstrap if step.id == "sync-feedbax")
    assert sync.command is not None
    assert "--no-owner --no-group" in sync.command
    assert "--stats" in sync.command


def test_modal_plan_represents_parallel_cells_without_ssh() -> None:
    spec = ExecutionSpec(
        backend="modal",
        job_id="modal-cells",
        command="python scripts/run_cell.py",
        cells=[
            ExecutionCell(id="cell-a", env={"CELL": "a"}, params={"gain": 1}),
            ExecutionCell(id="cell-b", env={"CELL": "b"}, params={"gain": 2}),
        ],
        repos=[
            RepoSource(
                name="feedbax",
                role="project",
                install_mode="github-ref",
                package="feedbax",
                git_url="https://github.com/i-m-mll/feedbax.git",
                git_ref="abc123",
            )
        ],
        modal={"max_containers": 8, "secrets": ["feedbax-github"]},
        issues=["c6b1b73"],
    )

    plan = prepare_execution_plan(spec)

    assert plan.backend == "modal"
    assert plan.cloud_payload["provider"] == "modal"
    assert plan.cloud_payload["parallel_submission"] == "spawn_map"
    assert plan.cloud_payload["generated_app"]["entrypoint"] == "feedbax_modal_execution.py"
    assert plan.cloud_payload["max_containers"] == 8
    assert plan.cloud_payload["secrets"] == ["feedbax-github"]
    assert plan.cloud_payload["volume"]["commit_required"] is True
    assert (
        "feedbax @ git+https://github.com/i-m-mll/feedbax.git@abc123"
        in plan.cloud_payload["computed_image_packages"]
    )
    assert "feedbax" not in plan.cloud_payload["computed_image_packages"]
    assert [cell["id"] for cell in plan.cloud_payload["cells"]] == ["cell-a", "cell-b"]
    assert all(check.id != "ssh" for check in plan.health_checks)
    assert next(step for step in plan.bootstrap if step.id == "modal-source-feedbax")
    assert plan.launch.command == "modal run --detach feedbax_modal_execution.py  # generated app uses spawn_map/map for independent cells"
    assert any("disjoint output paths" in warning for warning in plan.warnings)


def test_modal_training_plan_and_app_use_training_run_spec_runner() -> None:
    spec = ExecutionSpec(
        backend="modal",
        job_id="modal-training-spec",
        training_run_spec=_training_source(),
        modal={"use_spawn_map": False},
    )

    plan = prepare_execution_plan(spec)
    rendered = render_modal_app(spec)

    assert "python -m feedbax execute-training-run-spec" in plan.command
    assert "python -m feedbax execute-training-run-spec" in rendered
    assert "rlrmp" not in plan.command
    assert "rlrmp" not in rendered
    assert plan.cloud_payload["provider"] == "modal"
    assert plan.cloud_payload["training_run_spec"] == plan.reproducibility["training_run_spec"]
    assert plan.cloud_payload["training_run_spec"]["path"] == (
        "/vol/feedbax_runs/modal-training-spec/training-run-spec.json"
    )
    assert plan.cloud_payload["cells"][0]["command"].startswith(
        "python -m feedbax execute-training-run-spec"
    )


def test_modal_app_renderer_contains_volume_health_and_map_semantics() -> None:
    spec = ExecutionSpec(
        backend="modal",
        job_id="modal-render",
        command="python scripts/run_cell.py",
        cells=[ExecutionCell(id="cell-a", params={"seed": 1})],
        modal={"use_spawn_map": False, "secrets": ["feedbax-github"]},
    )

    rendered = render_modal_app(spec)

    assert "modal.App(APP_NAME)" in rendered
    assert "modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)" in rendered
    assert "modal.Secret.from_name(name)" in rendered
    assert "def health() -> dict:" in rendered
    assert "volume.commit()" in rendered
    assert "run_cell.map(CELLS)" in rendered
    assert '\\"cell-a\\"' in rendered
    assert '\\"feedbax-github\\"' in rendered


def test_modal_app_json_payload_globals_exec_without_json_name_shim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell_params = _json_literal_cell_params()
    common_kwargs = {
        "backend": "modal",
        "training_run_spec": _training_source_with_json_literals(),
        "cells": [ExecutionCell(id="cell-a", params=cell_params)],
        "env": {"STRING_FALSE": "false"},
        "modal": {"gpu": ["L40S", "A100"], "volume_name": None},
    }
    pip_spec = ExecutionSpec(job_id="modal-json-pip", **common_kwargs)
    local_root = tmp_path / "feedbax"
    local_root.mkdir()
    local_spec = ExecutionSpec(
        job_id="modal-json-local",
        repos=[
            RepoSource(
                name="feedbax",
                role="project",
                install_mode="local-embed",
                local_path=str(local_root),
            )
        ],
        primary_repo="feedbax",
        **common_kwargs,
    )

    for spec in (pip_spec, local_spec):
        namespace = _exec_rendered_modal_app(render_modal_app(spec), monkeypatch)

        assert namespace["DEFAULT_ENV"] == spec.env
        assert namespace["CELLS"][0]["params"] == cell_params
        assert namespace["GPU"] == ["L40S", "A100"]
        assert namespace["TRAINING_RUN_SPEC_PATH"].endswith("/training-run-spec.json")
        assert namespace["TRAINING_RUN_SPEC_PAYLOAD"] == spec.training_run_spec.inline_payload()
        assert namespace["VOLUME_NAME"] is None


def test_modal_app_renderer_does_not_emit_bare_json_literal_names(tmp_path: Path) -> None:
    cell_params = _json_literal_cell_params()
    common_kwargs = {
        "backend": "modal",
        "training_run_spec": _training_source_with_json_literals(),
        "cells": [ExecutionCell(id="cell-a", params=cell_params)],
        "env": {"STRING_FALSE": "false"},
        "modal": {"gpu": ["L40S", "A100"], "volume_name": None},
    }
    local_root = tmp_path / "feedbax"
    local_root.mkdir()
    specs = [
        ExecutionSpec(job_id="modal-json-pip", **common_kwargs),
        ExecutionSpec(
            job_id="modal-json-local",
            repos=[
                RepoSource(
                    name="feedbax",
                    role="project",
                    install_mode="local-embed",
                    local_path=str(local_root),
                )
            ],
            primary_repo="feedbax",
            **common_kwargs,
        ),
    ]

    for spec in specs:
        rendered = render_modal_app(spec)

        assert _forbidden_json_literal_names(rendered) == set()


def test_local_execution_emits_manifest_and_logs(tmp_path: Path) -> None:
    spec = ExecutionSpec(
        backend="local",
        job_id="local-smoke",
        command="python -c 'print(\"feedbax local smoke\")'",
        issues=["c6b1b73"],
        metadata={"purpose": "unit-test"},
    )

    result = run_local_execution(spec, root=tmp_path, timeout=10)

    assert result.schema_version == LOCAL_EXECUTION_RESULT_SCHEMA_VERSION
    assert result.status == "completed"
    assert result.return_code == 0
    assert Path(result.stdout_path).read_text(encoding="utf-8") == "feedbax local smoke\n"
    assert Path(result.stderr_path).read_text(encoding="utf-8") == ""
    assert Path(result.manifest_path).exists()
    assert result.stdout.role == "execution_stdout"
    assert result.stdout.sha256 is not None
    assert result.stdout.size_bytes == len("feedbax local smoke\n")
    assert result.stderr.role == "execution_stderr"
    assert result.stderr.sha256 is not None
    assert result.manifest.role == "training_run_manifest"
    assert result.execution_plan.role == "execution_plan"
    assert {artifact.role for artifact in result.produced_artifacts} == {
        "execution_plan",
        "execution_stdout",
        "execution_stderr",
        "training_run_manifest",
    }
    validate_materialized_execution_artifact(result.stdout, expected_role="execution_stdout")
    with pytest.raises(ValueError, match="role mismatch"):
        validate_materialized_execution_artifact(result.stdout, expected_role="execution_stderr")
    Path(result.stdout_path).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_materialized_execution_artifact(result.stdout, expected_role="execution_stdout")
    assert result.manifest_payload["kind"] == "TrainingRunManifest"
    assert result.manifest_payload["job_id"] == "local-smoke"
    assert result.manifest_payload["provenance"]["entrypoint"]["kind"] == "feedbax-execution"
    assert result.manifest_payload["provenance"]["issues"] == ["c6b1b73"]
    assert result.plan.job_id == "local-smoke"
    assert (tmp_path / "executions" / "local-smoke" / "execution-plan.json").exists()


def test_execution_plan_cli_round_trip(tmp_path: Path) -> None:
    spec_path = tmp_path / "execution-spec.json"
    plan_path = tmp_path / "execution-plan.json"
    spec_path.write_text(
        json.dumps(
            {
                "backend": "modal",
                "job_id": "cli-plan",
                "command": "python train.py",
                "modal": {"use_spawn_map": False},
            }
        ),
        encoding="utf-8",
    )

    from feedbax.bin.provider import main

    assert main(["execution-plan", str(spec_path), "--output", str(plan_path)]) == 0
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["job_id"] == "cli-plan"
    assert payload["cloud_payload"]["parallel_submission"] == "map"

    modal_app_path = tmp_path / "feedbax_modal_execution.py"
    assert main(["modal-app", str(spec_path), "--output", str(modal_app_path)]) == 0
    assert "modal.App(APP_NAME)" in modal_app_path.read_text(encoding="utf-8")


def test_provider_manifest_and_http_expose_execution_contract() -> None:
    manifest = provider_manifest()

    assert manifest.capabilities["prepare_execution_plan"].input_schema == "ExecutionSpec"
    assert manifest.capabilities["run_local_execution"].output_schema == "LocalExecutionResult"
    assert "ExecutionPlan" in manifest.schemas
    assert "execution_plan" in manifest.artifact_roles

    client = TestClient(create_app())
    response = client.post(
        "/api/provider/execution/plan",
        json={
            "backend": "runpod",
            "job_id": "http-plan",
            "command": "python train.py",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "runpod"
    assert payload["job_id"] == "http-plan"
    assert payload["cloud_payload"]["provider"] == "runpod"
