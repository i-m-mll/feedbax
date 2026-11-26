from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from feedbax.contracts import strict_json
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.orchestration import runpod_bootstrap
from feedbax.orchestration.project_packages import pin_project_package
from feedbax.orchestration.repo_snapshot import RepoSnapshotRecord, RepoSnapshotSourceFault, SealedRepoSnapshot
from feedbax.orchestration.runpod import SubprocessRunPodTransport
from feedbax.orchestration.runpod_session import RunPodEnvironment, RunPodSessionProvider
from feedbax.orchestration.runpod_session_provider import RunPodAcquisition
from feedbax.orchestration.session import SessionError
from feedbax.plugins.declarations import DistributionSource
from tests.support.hang import HANG_SECONDS


def _environment(root: Path, **uv_source) -> RunPodEnvironment:
    return RunPodEnvironment(
        workspace=str(root), bootstrap_python=sys.executable, uv_version="0.13.8",
        snapshot=RepoSnapshotRecord(
            commit="a" * 40, dirty=False, content_sha256="b" * 64,
            source_state_sha256="c" * 64, file_count=1,
        ),
        backend="cpu", **uv_source,
    )


def test_image_uv_retains_environment_identity(tmp_path):
    environment = _environment(tmp_path, uv="/usr/bin/uv")
    original = environment.model_dump(mode="json", exclude={"uv_sha256"})
    assert environment.digest == hashlib.sha256(canonical_json_v2_bytes(original)).hexdigest()
    with pytest.raises(ValueError, match="Declare an image uv path or a staged uv SHA-256"):
        _environment(tmp_path, uv="/usr/bin/uv", uv_sha256="d" * 64)


def test_provider_rejects_wrong_local_uv_before_opening(tmp_path):
    local_uv = tmp_path / "uv"
    local_uv.write_bytes(b"not the approved binary")
    environment = _environment(tmp_path, uv_sha256="d" * 64)
    snapshot = SealedRepoSnapshot("test", tmp_path, tmp_path, environment.snapshot)
    acquisition = RunPodAcquisition(
        tmp_path / "acquisition", SubprocessRunPodTransport(),
        api_key="fixture-api-key", ssh_key_path=tmp_path / "unused-key",
    )
    with pytest.raises(SessionError, match="Local uv binary differs"):
        RunPodSessionProvider(acquisition, environment, snapshot, uv_binary=local_uv)


def test_venue_stages_and_runs_exact_uv_offline(tmp_path):
    executable = shutil.which("uv")
    if executable is None:
        pytest.skip("uv is needed to exercise local RunPod staging")
    binary = Path(executable).read_bytes()
    digest = hashlib.sha256(binary).hexdigest()
    root = tmp_path / "session"
    env = {**os.environ, "UV_OFFLINE": "1", "UV_CACHE_DIR": str(tmp_path / "uv-cache")}
    uv_version = subprocess.check_output([executable, "--version"], timeout=HANG_SECONDS).decode().split()[1]
    # The CUDA probe is independent of staging and requires JAX in the venue venv.
    script = "\n".join((
        Path(strict_json.__file__).read_text(),
        Path(runpod_bootstrap.__file__).read_text(),
        "_PROBE = 'print(\"{}\")'",
        'bootstrap(lambda _: ("repo-digest", 1), strict_json_loads)',
    ))

    def call(action: str, payload: bytes, *arguments: str, check: bool = True):
        return subprocess.run(
            [sys.executable, "-I", "-c", script, action, str(root), *arguments],
            input=payload, check=check, capture_output=True, env=env, timeout=HANG_SECONDS,
        )

    prepare = json.dumps({"uv_sha256": digest}).encode()
    assert json.loads(call("prepare", prepare).stdout)["send_uv"]
    truncated = call("stage-uv", binary[:32], digest, check=False)
    assert b"Transferred uv differs from the authorized SHA-256" in truncated.stderr
    staged = root / "tools" / "uv"
    assert not staged.exists()
    call("stage-uv", binary, digest)
    assert staged.stat().st_mode & 0o777 == 0o500
    assert not json.loads(call("prepare", prepare).stdout)["send_uv"]

    incoming = root / "incoming"
    (incoming / "pyproject.toml").write_text(
        '[project]\nname = "venue-fixture"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
    )
    subprocess.run(
        [executable, "lock", "--offline", "--python", sys.executable],
        cwd=incoming, env=env, timeout=HANG_SECONDS, capture_output=True, check=True,
    )
    install = json.dumps({
        "uv_sha256": digest, "uv_version": uv_version, "content_sha256": "repo-digest", "file_count": 1,
        "timeout": 5, "groups": [], "extras": [], "backend": "cpu", "worker_env": {},
    }).encode()

    staged.chmod(0o700)
    staged.write_bytes(b"wrong approved bytes")
    staged.chmod(0o500)
    result = call("install", install, check=False)
    assert b"Staged uv differs from the authorized SHA-256" in result.stderr
    assert not (root / "venv").exists()

    staged.unlink()
    call("stage-uv", binary, digest)
    assert call("install", install).stdout.strip() == b"{}"
    assert (root / "venv" / "bin" / "python").exists()

    (root / "probe.json").write_bytes(b'{"project_packages":[{}],"project_packages":[]}')
    duplicate_probe = call("install", install, check=False)
    assert duplicate_probe.returncode != 0
    assert b"duplicate JSON object key 'project_packages'" in duplicate_probe.stderr


def test_project_package_rejects_ambiguous_editable_metadata(tmp_path, monkeypatch):
    class Distribution:
        version = "1.0"

        def read_text(self, name):
            assert name == "direct_url.json"
            return '{"url":"file:///chosen","dir_info":{"editable":false,"editable":true}}'

    monkeypatch.setattr(metadata, "distribution", lambda _name: Distribution())
    source = DistributionSource("fixture", "fixture", "1.0")
    with pytest.raises(RepoSnapshotSourceFault) as refused:
        pin_project_package(source, tmp_path)
    assert isinstance(refused.value.__cause__, strict_json.DuplicateJsonKeyError)
