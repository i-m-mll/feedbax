from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNPOD_DEPLOY = REPO_ROOT / "scripts" / "deploy" / "runpod_deploy.sh"
POLL_RUN = REPO_ROOT / "scripts" / "deploy" / "poll_run.sh"


def run_script(
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUNPOD_DEPLOY), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def write_config(tmp_path: Path) -> Path:
    rlrmp = tmp_path / "rlrmp"
    feedbax = tmp_path / "feedbax"
    cookbook = tmp_path / "jax-cookbook"
    for root in (rlrmp, feedbax, cookbook):
        root.mkdir()
        (root / "pyproject.toml").write_text("[project]\nname = 'placeholder'\n")

    config = tmp_path / "deploy.conf"
    config.write_text(
        "\n".join(
            [
                f"RLRMP_ROOT={str(rlrmp)!r}",
                f"FEEDBAX_ROOT={str(feedbax)!r}",
                f"JAX_COOKBOOK_ROOT={str(cookbook)!r}",
                "RUNPOD_NAME='test-runpod-deploy'",
                "REMOTE_RUN_DIR='/workspace/feedbax_runs/test-runpod-deploy'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_training_launch_requires_confirmed_spec(tmp_path: Path) -> None:
    config = write_config(tmp_path)

    result = run_script(
        "--dry-run",
        "--config",
        str(config),
        "--launch-command",
        "uv run --no-sync python train.py",
    )

    assert result.returncode == 2
    assert "Training spec confirmation required" in result.stderr
    assert "train_spec" in result.stderr
    assert "runpodctl pod create" not in result.stdout


def test_dry_run_prints_deterministic_deploy_commands(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    spec = tmp_path / "train-spec.json"
    spec.write_text(
        json.dumps(
            {
                "user_confirmed": True,
                "experiment": "dry-run",
                "command": "uv run --no-sync python train.py",
            }
        ),
        encoding="utf-8",
    )

    result = run_script(
        "--dry-run",
        "--config",
        str(config),
        "--train-spec",
        str(spec),
        "--launch-command",
        "uv run --no-sync python train.py",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "curl" in output
    assert "hub.docker.com/v2/repositories/runpod/pytorch/tags/" in output
    assert "runpodctl pod create" in output
    assert "--ports 22/tcp\\,8080/http" in output or "--ports 22/tcp,8080/http" in output
    assert "acquire_status=endpoint_assigned pod=dry-run-pod" in output
    assert "nvidia-smi" in output
    assert "rsync -az --delete --no-owner --no-group --stats" in output
    assert "--exclude /_artifacts" in output
    assert "feedbax/" in output
    assert "web/node_modules" in output
    assert "perl" in output
    assert "-0pi" in output
    assert "-e" in output
    assert "\\\\Q\\$ENV\\{PATCH_FROM\\}\\\\E" in output
    normalized = (
        output.replace("\\ ", " ")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\(", "(")
        .replace("\\)", ")")
    )
    assert "uv sync" in normalized
    assert "uv pip install -U" in normalized
    assert "jax\\[cuda12\\]" in output
    assert "uv run --no-sync python" in normalized


def test_reused_pod_dry_run_probes_before_install_and_skips_on_success(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    spec = tmp_path / "train-spec.json"
    spec.write_text(json.dumps({"user_confirmed": True}), encoding="utf-8")

    result = run_script(
        "--dry-run",
        "--config",
        str(config),
        "--ssh-host",
        "198.51.100.10",
        "--ssh-port",
        "2222",
        "--train-spec",
        str(spec),
        "--launch-command",
        "uv run --no-sync python train.py",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    normalized = (
        output.replace("\\ ", " ")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\(", "(")
        .replace("\\)", ")")
    )
    assert "starting venv consistency probe" in output
    assert "uv run --no-sync python -c" in normalized
    assert "import jax, jax.numpy; print(jax.devices())" in normalized
    assert "venv_probe_branch=probe_ok" in output
    assert "uv venv --clear" not in normalized
    assert "uv pip install -U" not in normalized


def test_acquire_only_dry_run_stops_before_deploy(tmp_path: Path) -> None:
    config = write_config(tmp_path)

    result = run_script(
        "--dry-run",
        "--config",
        str(config),
        "--acquire-only",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "runpodctl pod create" in output
    assert "acquire_status=endpoint_assigned pod=dry-run-pod" in output
    assert "nvidia-smi" in output
    assert "endpoint_source=ssh_object" in output
    assert "endpoint_classification=direct_endpoint_ready" in output
    assert "rsync -az" not in output
    assert "uv sync" not in output


def test_known_endpoint_acquire_only_skips_runpod_discovery(tmp_path: Path) -> None:
    config = write_config(tmp_path)

    result = run_script(
        "--dry-run",
        "--config",
        str(config),
        "--acquire-only",
        "--ssh-host",
        "198.51.100.10",
        "--ssh-port",
        "2222",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "runpodctl pod create" not in output
    assert "runpodctl pod get" not in output
    assert "root@198.51.100.10" in output
    assert "-p 2222" in output
    assert "endpoint_source=provided" in output
    assert "endpoint_classification=direct_endpoint_ready" in output


def test_acquire_only_classifies_missing_direct_endpoint_quickly(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_runpodctl = bin_dir / "runpodctl"
    fake_runpodctl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' "
        "'{\"desiredStatus\":\"RUNNING\",\"ssh\":{\"error\":\"no direct endpoint\"}}'\n",
        encoding="utf-8",
    )
    fake_runpodctl.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "ENDPOINT_CLASSIFIER_TIMEOUT_SECONDS": "0",
    }

    result = run_script(
        "--config",
        str(config),
        "--acquire-only",
        "--pod-id",
        "pod-missing-endpoint",
        "--skip-image-check",
        env=env,
    )

    assert result.returncode == 1
    assert "no direct ssh endpoint after 0s" in result.stderr
    assert "endpoint_classification=missing_direct_endpoint" in result.stderr
    assert "ssh_error=no_direct_endpoint" in result.stderr
    assert "rsync -az" not in result.stdout + result.stderr


def test_poll_run_dry_run_prints_status_before_sleep() -> None:
    result = subprocess.run(
        [
            str(POLL_RUN),
            "--dry-run",
            "--pod-id",
            "pod-123",
            "--cadence-seconds",
            "0",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "+ sleep 0" in result.stdout
    status_lines = [line for line in result.stdout.splitlines() if " pod=pod-123 " in line]
    assert len(status_lines) == 1
    assert "pod_status=DRY_RUN" in status_lines[0]
    assert "endpoint_source=ssh_object" in status_lines[0]
    assert "endpoint_classification=direct_endpoint_discovered" in status_lines[0]
    assert "ssh_error=none" in status_lines[0]
    assert "gpu=dry-run" in status_lines[0]
    assert result.stdout.index(" pod=pod-123 ") < result.stdout.index("+ sleep 0")
