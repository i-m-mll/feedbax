from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNPOD_DEPLOY = REPO_ROOT / "scripts" / "deploy" / "runpod_deploy.sh"
LIB_RUN_PREP = REPO_ROOT / "scripts" / "deploy" / "lib_run_prep.sh"
LIB_ACQUIRE = REPO_ROOT / "scripts" / "deploy" / "lib_acquire.sh"
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


def write_training_run_spec(path: Path) -> None:
    from tests.test_training_run_executor import _run_spec

    path.write_text(
        json.dumps(_run_spec().model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def init_repo(path: Path, branch: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", branch], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_deploy_starts_log_with_checkout_provenance_and_warns_on_pin_skew(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
    feedbax = tmp_path / "feedbax"
    rlrmp = tmp_path / "rlrmp"
    feedbax_sha = init_repo(feedbax, "feedbax-test")
    consumer_sha = init_repo(rlrmp, "consumer-test")
    (feedbax / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    (rlrmp / "ci").mkdir()
    (rlrmp / "ci" / "feedbax-ref.toml").write_text(
        f'rev = "{"1" * 40}"\n', encoding="utf-8"
    )

    result = run_script(
        "--config",
        str(config),
        "--help",
    )

    lines = result.stderr.splitlines()
    assert result.returncode == 0
    assert lines[0].startswith("provenance ")
    assert f"feedbax_sha={feedbax_sha}" in lines[0]
    assert "feedbax_branch=feedbax-test" in lines[0]
    assert "feedbax_dirty=true" in lines[0]
    assert f"consumer_sha={consumer_sha}" in lines[0]
    assert "consumer_branch=consumer-test" in lines[0]
    assert f"script={RUNPOD_DEPLOY}" in lines[0]
    assert lines[1].startswith("WARNING: feedbax checkout/pin skew")


def test_poll_starts_log_with_checkout_provenance(tmp_path: Path) -> None:
    feedbax = tmp_path / "feedbax"
    rlrmp = tmp_path / "rlrmp"
    feedbax_sha = init_repo(feedbax, "feedbax-poll")
    consumer_sha = init_repo(rlrmp, "consumer-poll")
    (rlrmp / "ci").mkdir()
    (rlrmp / "ci" / "feedbax-ref.toml").write_text(
        f'rev = "{feedbax_sha}"\n', encoding="utf-8"
    )

    result = subprocess.run(
        [str(POLL_RUN), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "FEEDBAX_ROOT": str(feedbax),
            "RLRMP_ROOT": str(rlrmp),
        },
    )

    lines = result.stderr.splitlines()
    assert result.returncode == 0
    assert len(lines) == 1
    assert lines[0].startswith("provenance ")
    assert f"feedbax_sha={feedbax_sha}" in lines[0]
    assert "feedbax_branch=feedbax-poll" in lines[0]
    assert "feedbax_dirty=false" in lines[0]
    assert f"consumer_sha={consumer_sha}" in lines[0]
    assert "consumer_branch=consumer-poll" in lines[0]
    assert f"script={POLL_RUN}" in lines[0]


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
    assert "rsync -az --delete --no-owner --no-group --progress --stats" in output
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
    assert "JAX_COMPILATION_CACHE_DIR" in output
    assert "/workspace/jax_cache" in output


def test_rows_dry_run_uses_cache_env_and_warm_first_order(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    spec = tmp_path / "train-spec.json"
    spec.write_text(json.dumps({"user_confirmed": True}), encoding="utf-8")
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {"id": "row_a", "command": "uv run --no-sync python train.py --row a"},
                    {"id": "row_b", "command": "uv run --no-sync python train.py --row b"},
                    {"id": "row_c", "command": "uv run --no-sync python train.py --row c"},
                ],
            }
        ),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "JAX_COMPILATION_CACHE_DIR": "/workspace/custom_jax_cache",
        "ROW_LAUNCH_STAGGER_SECONDS": "0",
        "MAX_PARALLEL_ROWS": "2",
        "WARM_COMPILE_READY_REGEX": "TRAINING_READY",
    }
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
        "--rows-manifest",
        str(rows),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "warm_compile_first=1" in output
    assert "JAX_COMPILATION_CACHE_DIR" in output
    assert "/workspace/custom_jax_cache" in output
    first_launch = output.index("launching row row_a")
    warm_wait = output.index(
        "dry-run: warm compile first would poll row row_a structured readiness or log"
    )
    second_launch = output.index("launching row row_b")
    assert first_launch < warm_wait < second_launch
    warm_gate_output = output[warm_wait:second_launch]
    assert "regex 'TRAINING_READY'" in warm_gate_output
    assert "row_a.pid" not in warm_gate_output
    assert "ssh -n" in output
    assert "</dev/null" in output
    assert "PYTHONUNBUFFERED=1" in output


def run_warm_gate_probe(
    tmp_path: Path,
    *,
    sentinel: str | None = None,
    log_text: str = "",
    event_text: str = "",
    timeout_seconds: int = 1,
    ready_regex: str = "TRAINING_READY",
) -> subprocess.CompletedProcess[str]:
    run_dir = tmp_path / "run"
    sentinel_dir = run_dir / "sentinels"
    log_dir = run_dir / "logs"
    sentinel_dir.mkdir(parents=True)
    log_dir.mkdir()
    if sentinel is not None:
        for suffix in sentinel.split("+"):
            (sentinel_dir / f"row_a.{suffix}").touch()
    (log_dir / "row_a.log").write_text(log_text, encoding="utf-8")
    event_dir = run_dir / "events"
    event_dir.mkdir()
    (event_dir / "row_a.events.jsonl").write_text(event_text, encoding="utf-8")
    script = f"""
set -euo pipefail
log() {{ printf '==> %s\\n' "$*" >&2; }}
die() {{ printf 'error: %s\\n' "$*" >&2; exit 1; }}
print_cmd() {{ :; }}
run_cmd() {{ "$@"; }}
capture_cmd() {{ "$@"; }}
sq() {{ local v=${{1-}}; v=${{v//\\'/\\'\\\\\\'\\'}}; printf "'%s'" "$v"; }}
source {str(LIB_RUN_PREP)!r}
RUN_PREP_PROVIDER=local
DRY_RUN=0
REMOTE_RUN_DIR={str(run_dir)!r}
REMOTE_SENTINEL_DIR={str(sentinel_dir)!r}
SENTINEL_TIMEOUT_SECONDS={timeout_seconds}
SENTINEL_POLL_SECONDS=1
wait_for_row_active_training row_a {ready_regex!r}
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_warm_gate_completed_row_releases_without_readiness_marker(tmp_path: Path) -> None:
    result = run_warm_gate_probe(tmp_path, sentinel="done")

    assert result.returncode == 0, result.stderr
    assert "completed before readiness marker matched" in result.stderr
    assert "failed" not in result.stderr.lower()


def test_warm_gate_readiness_marker_releases_fanout(tmp_path: Path) -> None:
    result = run_warm_gate_probe(tmp_path, log_text="TRAINING_READY\n")

    assert result.returncode == 0, result.stderr
    assert "reached active training" in result.stderr


def test_warm_gate_structured_ready_event_releases_without_log_output(tmp_path: Path) -> None:
    result = run_warm_gate_probe(
        tmp_path,
        event_text='{"run_set_id":"set","row_id":"row_a","type":"ready"}\n',
    )

    assert result.returncode == 0, result.stderr
    assert "emitted a ready event" in result.stderr
    assert (tmp_path / "run" / "sentinels" / "row_a.ready").exists()


def test_buffered_child_trips_structured_readiness_before_stdout_flush(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    sentinel_dir = run_dir / "sentinels"
    child = tmp_path / "buffered_child.py"
    child.write_text(
        """import json
import os
import pathlib
import time

events = pathlib.Path(os.environ["FEEDBAX_RUN_EVENTS_DIR"])
events.mkdir(parents=True, exist_ok=True)
event_file = events / f"{os.environ['FEEDBAX_ROW_ID']}.events.jsonl"
event_file.write_text(json.dumps({"type": "ready"}) + "\\n", encoding="utf-8")
print("TRAINING_READY", end="")
time.sleep(3)
""",
        encoding="utf-8",
    )
    command = f"env -u PYTHONUNBUFFERED {shlex.quote(sys.executable)} {shlex.quote(str(child))}"
    script = f"""
set -euo pipefail
log() {{ printf '==> %s\\n' "$*" >&2; }}
die() {{ printf 'error: %s\\n' "$*" >&2; exit 1; }}
run_cmd() {{ "$@"; }}
capture_cmd() {{ "$@"; }}
validate_row_id() {{ [[ $1 =~ ^[A-Za-z0-9_.-]+$ ]]; }}
sq() {{ local v=${{1-}}; v=${{v//\\'/\\'\\\\\\'\\'}}; printf "'%s'" "$v"; }}
source {str(LIB_RUN_PREP)!r}
RUN_PREP_PROVIDER=local
DRY_RUN=0
REMOTE_RUN_DIR={str(run_dir)!r}
REMOTE_SENTINEL_DIR={str(sentinel_dir)!r}
JAX_COMPILATION_CACHE_DIR={str(tmp_path / 'cache')!r}
SENTINEL_TIMEOUT_SECONDS=5
SENTINEL_POLL_SECONDS=0.05
launch_row row_a {str(tmp_path)!r} {shlex.quote(command)}
wait_for_row_active_training row_a TRAINING_READY
if grep -F TRAINING_READY {str(run_dir / 'logs' / 'row_a.log')!r}; then
    exit 3
fi
touch {str(tmp_path / 'gate-before-flush')!r}
"""

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert "emitted a ready event" in result.stderr
    assert (tmp_path / "gate-before-flush").exists()
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not (sentinel_dir / "row_a.done").exists():
        time.sleep(0.05)
    assert (sentinel_dir / "row_a.done").exists()


def test_provider_mocked_multi_row_launch_completes_without_manual_release_or_kill(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    sentinel_dir = run_dir / "sentinels"
    first = tmp_path / "first.py"
    first.write_text(
        """import json
import os
import pathlib
import time

events = pathlib.Path(os.environ["FEEDBAX_RUN_EVENTS_DIR"])
events.mkdir(parents=True, exist_ok=True)
(events / f"{os.environ['FEEDBAX_ROW_ID']}.events.jsonl").write_text(
    json.dumps({"type": "ready"}) + "\\n", encoding="utf-8"
)
print("buffered", end="")
time.sleep(0.3)
""",
        encoding="utf-8",
    )
    second_marker = tmp_path / "second-ran"
    second = tmp_path / "second.py"
    second.write_text(
        f"from pathlib import Path\nPath({str(second_marker)!r}).touch()\n",
        encoding="utf-8",
    )
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "id": "row_a",
                        "command": (
                            f"env -u PYTHONUNBUFFERED {shlex.quote(sys.executable)} "
                            f"{shlex.quote(str(first))}"
                        ),
                    },
                    {
                        "id": "row_b",
                        "command": f"{shlex.quote(sys.executable)} {shlex.quote(str(second))}",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    script = f"""
set -euo pipefail
log() {{ printf '==> %s\\n' "$*" >&2; }}
die() {{ printf 'error: %s\\n' "$*" >&2; exit 1; }}
run_cmd() {{ "$@"; }}
capture_cmd() {{ "$@"; }}
sq() {{ printf '%q' "${{1-}}"; }}
source {str(LIB_ACQUIRE)!r}
source {str(LIB_RUN_PREP)!r}
RUN_PREP_PROVIDER=local
DRY_RUN=0
REMOTE_RLRMP_ROOT={str(tmp_path)!r}
REMOTE_RUN_DIR={str(run_dir)!r}
REMOTE_SENTINEL_DIR={str(sentinel_dir)!r}
JAX_COMPILATION_CACHE_DIR={str(tmp_path / 'cache')!r}
ROWS_MANIFEST={str(rows)!r}
ROW_LAUNCH_STAGGER_SECONDS=0
MAX_PARALLEL_ROWS=2
WARM_COMPILE_FIRST=1
WARM_COMPILE_READY_REGEX=TRAINING_READY
SENTINEL_TIMEOUT_SECONDS=5
SENTINEL_POLL_SECONDS=0.1
launch_training
"""

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert "emitted a ready event" in result.stderr
    assert "launched 2 row(s)" in result.stderr
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not all(
        (sentinel_dir / f"{row}.done").exists() for row in ("row_a", "row_b")
    ):
        time.sleep(0.05)
    assert second_marker.exists()
    assert (sentinel_dir / "row_a.done").exists()
    assert (sentinel_dir / "row_b.done").exists()
    assert (sentinel_dir / "row_a.ready").exists()


def test_warm_gate_completed_row_wins_over_stale_failed_sentinel(tmp_path: Path) -> None:
    result = run_warm_gate_probe(tmp_path, sentinel="done+failed")

    assert result.returncode == 0, result.stderr
    assert "completed before readiness marker matched" in result.stderr
    assert "error:" not in result.stderr


def test_warm_gate_failed_row_aborts_fanout(tmp_path: Path) -> None:
    result = run_warm_gate_probe(tmp_path, sentinel="failed")

    assert result.returncode == 1
    assert "warm compile row row_a failed" in result.stderr


def test_warm_gate_readiness_timeout_warns_and_releases(tmp_path: Path) -> None:
    result = run_warm_gate_probe(tmp_path, timeout_seconds=0)

    assert result.returncode == 0, result.stderr
    assert "WARNING: timed out waiting for warm compile row row_a" in result.stderr
    assert "marking the warm row failed" in result.stderr


def test_rows_manifest_configures_warm_gate_readiness_marker(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    spec = tmp_path / "train-spec.json"
    spec.write_text(json.dumps({"user_confirmed": True}), encoding="utf-8")
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "warm_compile_ready_string": "ROW_READY",
                "rows": [
                    {"id": "row_a", "command": "uv run --no-sync python train.py --row a"},
                    {"id": "row_b", "command": "uv run --no-sync python train.py --row b"},
                ],
            }
        ),
        encoding="utf-8",
    )

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
        "--rows-manifest",
        str(rows),
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "string 'ROW_READY'" in output


def test_rows_manifest_payload_preflight_fails_before_acquisition(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    run_spec = tmp_path / "training-run-spec.json"
    write_training_run_spec(run_spec)
    spec = tmp_path / "train-spec.json"
    spec.write_text(json.dumps({"user_confirmed": True}), encoding="utf-8")
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "id": "flat_3e-5",
                        "command": (
                            f"{sys.executable} -m feedbax execute-training-run-spec {run_spec}"
                        ),
                        "training_run_spec": str(run_spec),
                        "training_payload": {
                            "schema_version": "rlrmp.cs_stochastic_gru.v1",
                            "experiment": "flat_3e-5",
                        },
                        "training_payload_kind": "RLRMPRunSpec",
                        "training_payload_schema_id": "rlrmp.run_spec",
                        "training_payload_schema_version": "rlrmp.run_spec.v2",
                    }
                ],
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
        "--rows-manifest",
        str(rows),
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "preflighting TrainingRunManifest payload for row flat_3e-5" in output
    assert "Embedded SpecPayload schema version disagrees with inline payload" in output
    assert "row_id='flat_3e-5'" in output
    assert str(run_spec) in output
    assert "runpodctl pod create" not in output


def test_rows_manifest_payload_preflight_temp_files_are_parallel_safe(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
    run_spec = tmp_path / "training-run-spec.json"
    write_training_run_spec(run_spec)
    spec = tmp_path / "train-spec.json"
    spec.write_text(json.dumps({"user_confirmed": True}), encoding="utf-8")
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "id": "parallel_preflight",
                        "command": (
                            f"{sys.executable} -m feedbax execute-training-run-spec {run_spec}"
                        ),
                        "training_run_spec": str(run_spec),
                        "training_payload": {
                            "schema_version": "rlrmp.cs_stochastic_gru.v1",
                            "experiment": "parallel_preflight",
                        },
                        "training_payload_kind": "RLRMPRunSpec",
                        "training_payload_schema_id": "rlrmp.run_spec",
                        "training_payload_schema_version": "rlrmp.run_spec.v2",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    temp_dir = tmp_path / "preflight-tmp"
    temp_dir.mkdir()
    env = {**os.environ, "TMPDIR": str(temp_dir)}
    command = [
        str(RUNPOD_DEPLOY),
        "--dry-run",
        "--config",
        str(config),
        "--train-spec",
        str(spec),
        "--rows-manifest",
        str(rows),
    ]

    processes = [
        subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        for _ in range(2)
    ]
    results = []
    for process in processes:
        stdout, stderr = process.communicate()
        results.append((process.returncode, stdout, stderr))

    for returncode, stdout, stderr in results:
        output = stdout + stderr
        assert returncode == 1
        assert "Embedded SpecPayload schema version disagrees with inline payload" in output
        assert "mktemp:" not in output
        assert "File exists" not in output
    assert list(temp_dir.glob("feedbax-manifest-preflight.*")) == []


def test_rows_manifest_preflight_resolves_relative_spec_against_workdir(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
    row_workdir = tmp_path / "row-workdir"
    row_workdir.mkdir()
    run_spec = row_workdir / "training-run-spec.json"
    write_training_run_spec(run_spec)
    spec = tmp_path / "train-spec.json"
    spec.write_text(json.dumps({"user_confirmed": True}), encoding="utf-8")
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "id": "row_rel",
                        "workdir": str(row_workdir),
                        "command": (
                            f"{sys.executable} -m feedbax execute-training-run-spec "
                            "training-run-spec.json"
                        ),
                        "training_run_spec": "training-run-spec.json",
                        "training_payload": {
                            "schema_version": "rlrmp.cs_stochastic_gru.v1",
                            "experiment": "relative-path",
                        },
                        "training_payload_kind": "RLRMPRunSpec",
                        "training_payload_schema_id": "rlrmp.run_spec",
                        "training_payload_schema_version": "rlrmp.run_spec.v2",
                    }
                ],
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
        "--rows-manifest",
        str(rows),
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    resolved = str(run_spec)
    assert f"preflighting TrainingRunManifest payload for row row_rel ({resolved})" in output
    assert "Embedded SpecPayload schema version disagrees with inline payload" in output
    assert "runpodctl pod create" not in output


def test_resume_baseline_missing_source_fails_preflight(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    missing = tmp_path / "rlrmp" / "_artifacts" / "missing-run" / "checkpoint_100"
    spec = tmp_path / "train-spec.json"
    spec.write_text(
        json.dumps(
            {
                "user_confirmed": True,
                "resume": {
                    "baseline_checkpoint_path": str(missing),
                    "baseline_completed_batch": 100,
                },
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
        "uv run --no-sync python train.py --resume",
    )

    assert result.returncode == 1
    assert "baseline preflight failed: source checkpoint not found" in result.stderr
    assert str(missing) in result.stderr
    assert "runpodctl pod create" not in result.stdout + result.stderr


def test_resume_baseline_missing_custody_pointer_fails_preflight(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    baseline = tmp_path / "rlrmp" / "_artifacts" / "run-a" / "checkpoint_100"
    baseline.mkdir(parents=True)
    spec = tmp_path / "train-spec.json"
    spec.write_text(
        json.dumps(
            {
                "user_confirmed": True,
                "resume": {
                    "baseline_checkpoint_path": str(baseline),
                    "baseline_completed_batch": 100,
                },
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
        "uv run --no-sync python train.py --resume",
    )

    assert result.returncode == 1
    assert "baseline preflight failed: custody latest.json not found" in result.stderr
    assert "runpodctl pod create" not in result.stdout + result.stderr


def test_resume_baseline_completed_batch_mismatch_fails_preflight(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    baseline = tmp_path / "rlrmp" / "_artifacts" / "run-a" / "checkpoint_100"
    baseline.mkdir(parents=True)
    (baseline / "latest.json").write_text(json.dumps({"completed_batch": 99}), encoding="utf-8")
    spec = tmp_path / "train-spec.json"
    spec.write_text(
        json.dumps(
            {
                "user_confirmed": True,
                "resume": {
                    "baseline_checkpoint_path": str(baseline),
                    "baseline_completed_batch": 100,
                },
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
        "uv run --no-sync python train.py --resume",
    )

    assert result.returncode == 1
    assert "baseline preflight failed: completed_batch mismatch" in result.stderr
    assert "declared 100 but latest.json has 99" in result.stderr
    assert "runpodctl pod create" not in result.stdout + result.stderr


def test_latest_pointer_completed_batches_prefers_explicit_total_over_coordinate(
    tmp_path: Path,
) -> None:
    latest = tmp_path / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "completed_coordinate": {"program_step": 12009},
                "completed_training_batches": 16500,
            }
        ),
        encoding="utf-8",
    )
    script = (
        f"source {str(REPO_ROOT / 'scripts' / 'deploy' / 'lib_acquire.sh')!r}\n"
        f"latest_pointer_completed_batches {str(latest)!r}\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "16500"


def test_latest_pointer_completed_batches_rejects_coordinate_as_batch_fallback(
    tmp_path: Path,
) -> None:
    latest = tmp_path / "latest.json"
    latest.write_text(
        json.dumps({"completed_coordinate": {"program_step": 24}}),
        encoding="utf-8",
    )
    script = (
        f"source {str(REPO_ROOT / 'scripts' / 'deploy' / 'lib_acquire.sh')!r}\n"
        f"latest_pointer_completed_batches {str(latest)!r}\n"
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_resume_baseline_is_preflighted_and_staged_despite_artifact_exclude(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
    baseline = tmp_path / "rlrmp" / "_artifacts" / "run-a" / "checkpoint_100"
    baseline.mkdir(parents=True)
    (baseline / "latest.json").write_text(json.dumps({"completed_batch": 100}), encoding="utf-8")
    spec = tmp_path / "train-spec.json"
    spec.write_text(
        json.dumps(
            {
                "user_confirmed": True,
                "resume": {
                    "baseline_checkpoint_path": str(baseline),
                    "baseline_completed_batch": 100,
                },
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
        "uv run --no-sync python train.py --resume",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "--exclude /_artifacts" in output
    assert "staging declared baseline" in output
    assert str(baseline) in output
    assert "/workspace/rlrmp/_artifacts/run-a/checkpoint_100/" in output
    assert "mkdir -p" in output
    assert "--no-owner --no-group --progress --stats" in output
    run_config = tmp_path / "feedbax" / ".runpod" / "run-config.json"
    payload = json.loads(run_config.read_text(encoding="utf-8"))
    assert payload["remote_run_dir"] == "/workspace/feedbax_runs/test-runpod-deploy"
    assert payload["baselines"][0]["completed_batch"] == "100"


def test_string_train_spec_rows_and_resume_only_preflight_are_fail_closed(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
    baseline = tmp_path / "rlrmp" / "_artifacts" / "run-a" / "checkpoint_100"
    baseline.mkdir(parents=True)
    (baseline / "latest.json").write_text(
        json.dumps({"completed_batch": 100}), encoding="utf-8"
    )
    train_spec = tmp_path / "train-spec.json"
    train_spec.write_text(
        json.dumps({"user_confirmed": True, "rows": ["resume_row"]}),
        encoding="utf-8",
    )
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "id": "resume_row",
                        "command": (
                            "uv run --no-sync python scripts/train_cs_nominal_gru.py "
                            "--run-spec results/run.json --full-train --resume"
                        ),
                        "workdir": "/workspace/rlrmp",
                        "resume": {
                            "baseline_checkpoint_path": str(baseline),
                            "baseline_completed_batch": 100,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_script(
        "--dry-run",
        "--config",
        str(config),
        "--train-spec",
        str(train_spec),
        "--rows-manifest",
        str(rows),
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "Cannot index string with string" not in output
    assert "verifying staged resume checkpoint for row resume_row" in output
    assert "--verify-resume-only" in output
    assert output.index("--verify-resume-only") < output.index("launched 1 row(s)")


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
        '\'{"desiredStatus":"RUNNING","ssh":{"error":"no direct endpoint"}}\'\n',
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
    run_config = REPO_ROOT / ".pytest-runpod-run-config.json"
    run_config.write_text(
        json.dumps(
            {
                "remote_run_dir": "/workspace/feedbax_runs/non-default",
                "remote_sentinel_dir": "/workspace/feedbax_runs/non-default/sentinels",
                "remote_checkpoint_dir": "/workspace/feedbax_runs/non-default",
                "remote_log_dir": "/workspace/feedbax_runs/non-default/logs",
            }
        ),
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [
                str(POLL_RUN),
                "--dry-run",
                "--pod-id",
                "pod-123",
                "--run-config",
                str(run_config),
                "--cadence-seconds",
                "0",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        run_config.unlink(missing_ok=True)

    assert result.returncode == 0, result.stderr
    assert "/workspace/feedbax_runs/non-default/sentinels" in result.stderr
    assert "+ sleep 0" in result.stdout
    status_lines = [line for line in result.stdout.splitlines() if " pod=pod-123 " in line]
    assert len(status_lines) == 1
    assert "pod_status=DRY_RUN" in status_lines[0]
    assert "endpoint_source=ssh_object" in status_lines[0]
    assert "endpoint_classification=direct_endpoint_discovered" in status_lines[0]
    assert "ssh_error=none" in status_lines[0]
    assert "gpu=dry-run" in status_lines[0]
    assert "row_telemetry=none" in status_lines[0]
    assert result.stdout.index(" pod=pod-123 ") < result.stdout.index("+ sleep 0")


def test_poll_run_requires_run_dir_or_config() -> None:
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

    assert result.returncode == 1
    assert "remote run dir is required" in result.stderr


def test_poll_run_reports_started_without_terminal_sentinel_as_stale(
    tmp_path: Path,
) -> None:
    sentinel_dir = tmp_path / "sentinels"
    checkpoint_dir = tmp_path / "run"
    log_dir = checkpoint_dir / "logs"
    events_dir = checkpoint_dir / "events"
    sentinel_dir.mkdir()
    log_dir.mkdir(parents=True)
    events_dir.mkdir()
    (sentinel_dir / "row_a.started").touch()
    (sentinel_dir / "row_b.started").touch()
    (sentinel_dir / "row_b.done").touch()
    (events_dir / "row_b.events.jsonl").write_text(
        json.dumps(
            {
                "type": "ready",
                "payload": {
                    "runtime_telemetry": {
                        "start_to_first_progress_seconds": 12.5,
                        "persistent_cache_effectiveness_proxy": (
                            "preexisting_entries_no_new_entries_observed"
                        ),
                    }
                },
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    function_source = subprocess.run(
        [
            "sed",
            "-n",
            "/^build_remote_status_command() {/,/^}/p",
            str(POLL_RUN),
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    script = (
        f"{function_source}\n"
        "cmd=$(build_remote_status_command "
        f"{str(sentinel_dir)!r} {str(checkpoint_dir)!r} {str(log_dir)!r} "
        f"{str(events_dir)!r})\n"
        'bash -c "$cmd"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "rows_done=1 rows_failed=0 rows_running=0 rows_stale=1" in result.stdout
    assert result.stdout.index("last_batch=") < result.stdout.index("last_checkpoint=")
    assert "row_a:stale_started" in result.stdout
    assert "row_b:done" in result.stdout
    assert "train_process=" in result.stdout
    assert "row_b:first_progress_s=12.5:cache_proxy=" in result.stdout
    assert "preexisting_entries_no_new_entries_observed" in result.stdout


def test_deploy_launch_wrapper_marks_failure_on_signals(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    spec = tmp_path / "train-spec.json"
    spec.write_text(json.dumps({"user_confirmed": True}), encoding="utf-8")

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
    script = LIB_RUN_PREP.read_text(encoding="utf-8")
    assert "mark_failed()" in script
    assert "trap mark_failed EXIT" in script
    assert "trap 'signal_failed 143' TERM" in script
    assert "training.failed" in output


def test_poll_run_pgrep_pattern_cannot_match_the_inspection_command() -> None:
    script = POLL_RUN.read_text(encoding="utf-8")

    assert "pgrep -af '[t]rain_|[p]ython .*train|[u]v run .*train'" in script
    assert 'pgrep -af "train_"' not in script
