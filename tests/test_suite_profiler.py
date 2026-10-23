"""Exercise diagnostic aggregation against real, tiny serial and xdist sessions."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("workers", ["0", "2"])
def test_profiler_reports_phases_fixtures_and_group_occupancy(tmp_path, workers):
    root = Path(__file__).resolve().parents[1]
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text("""import pytest
pytestmark = pytest.mark.xdist_group("shared_fixture")
@pytest.fixture(scope="module")
def shared():
    return object()
def test_pass(shared):
    assert shared is not None
def test_fail(shared):
    assert False
def test_skip():
    pytest.skip("intentional")
""")
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\nmarkers = xdist_group(name)\n")
    output = tmp_path / "profile.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tests),
            "-c",
            str(config),
            "-n",
            workers,
            "--dist=loadgroup",
            "-p",
            "scripts.pytest_suite_profile",
            "-q",
        ],
        cwd=root,
        env={**os.environ, "FEEDBAX_SUITE_PROFILE_RAW": str(output)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    text = output.read_text()
    assert str(tmp_path) not in text
    data = json.loads(text)
    assert data["schema_version"] == 1
    assert data["exitstatus"] == 1
    assert data["collected"] == 3
    assert data["worker_count"] == int(workers)
    assert data["active_worker_count"] == 1
    assert data["fixtures"]["module:shared"]["count"] == 1
    (row,) = data["files"].values()
    assert row["outcomes"] == {"passed": 1, "failed": 1, "skipped": 1}
    assert set(row["phases"]) == {"setup", "call", "teardown"}
    assert len(row["workers"]) == 1


def test_profiler_retains_worker_crash_without_worker_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_crash.py").write_text("def test_crash():\n    import os\n    os._exit(3)\n")
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    output = tmp_path / "profile.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tests),
            "-c",
            str(config),
            "-n",
            "1",
            "--max-worker-restart=0",
            "-p",
            "scripts.pytest_suite_profile",
            "-q",
        ],
        cwd=root,
        env={**os.environ, "FEEDBAX_SUITE_PROFILE_RAW": str(output)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    data = json.loads(output.read_text())
    assert data["exitstatus"] == 1
    assert data["worker_errors"] == 1
    assert data["worker_count"] == 1
    assert "AttributeError" not in result.stdout + result.stderr
