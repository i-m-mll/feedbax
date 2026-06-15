from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from feedbax.cloud_backends import render_modal_app
from feedbax.execution_models import (
    ExecutionCell,
    ExecutionSpec,
    RepoSource,
)
from feedbax.execution_plan import (
    default_feedbax_sources,
    prepare_execution_plan,
)
from feedbax.local_execution import run_local_execution
from feedbax.integrations.provider import provider_manifest
from feedbax.web.app import create_app


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
    assert '"cell-a"' in rendered
    assert '"feedbax-github"' in rendered


def test_local_execution_emits_manifest_and_logs(tmp_path: Path) -> None:
    spec = ExecutionSpec(
        backend="local",
        job_id="local-smoke",
        command="python -c 'print(\"feedbax local smoke\")'",
        issues=["c6b1b73"],
        metadata={"purpose": "unit-test"},
    )

    result = run_local_execution(spec, root=tmp_path, timeout=10)

    assert result.status == "completed"
    assert result.return_code == 0
    assert Path(result.stdout_path).read_text(encoding="utf-8") == "feedbax local smoke\n"
    assert Path(result.stderr_path).read_text(encoding="utf-8") == ""
    assert Path(result.manifest_path).exists()
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
