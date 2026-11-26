"""A project's code comes from the environment of the uv project around its folder, and its absence is one Problem."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from fastapi.testclient import TestClient

from feedbax.project_folder import ProjectFolderLocation
from feedbax.web.app import create_app
from tests.support.environment_project import OPERATION, PACKAGE_SOURCE, environment_project, rollout_project, write_project
from tests.support.hang import HANG_SECONDS

REPO = Path(__file__).resolve().parents[1]


def _window_map(client: TestClient, project_id: str) -> dict:
    path = f"/api/projects/{project_id}"
    revision = client.get(path).json()["revision"]
    response = client.get(f"{path}/train-runs", params={"revision": revision, "window_map": True})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_package_the_project_cannot_load_is_one_problem_and_its_runs_cannot_run(host, tmp_path, monkeypatch):
    """Caught: one "Unknown operation" per step that needs the package, or the steps planning as if it were there."""
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    projects = tmp_path / "projects"
    write_project(projects / "absent", rollout_project(OPERATION, OPERATION, "feedbax.no_such_operation"))
    with TestClient(create_app(projects_dir=projects, automatic_run="off")) as client:
        plan = _window_map(client, "absent")
    problems = plan["window_map"]["problems"]
    unavailable = [problem for problem in problems if problem["code"] == "package_unavailable"]
    assert [problem["package"] for problem in unavailable] == ["environment_fixture"]
    assert unavailable[0]["step"] is None
    assert {problem["step"] for problem in problems if problem["code"] == "missing_package"} == {"analysis-0", "analysis-1"}
    assert {problem["step"] for problem in problems if problem.get("package") is None} == {"analysis-2"}, (
        "a misspelled key of a loaded package stays a Problem of its own step"
    )
    status = {row["key"]["step"]: row["status"] for row in plan["window_map"]["rows"]}
    assert status["analysis-0"] == status["analysis-1"] == "cannot_run"


def test_a_project_without_an_environment_is_told_so(host, tmp_path, monkeypatch):
    """Caught: a project outside any uv project is told to install a package, which nothing would ever load."""
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    folder = write_project(tmp_path / "loose", rollout_project(OPERATION))
    assert not any((directory / "pyproject.toml").exists() for directory in (folder, *folder.parents))
    with TestClient(create_app(projects_dir=tmp_path / "projects", automatic_run="off", open_folder=folder)) as client:
        problems = _window_map(client, ProjectFolderLocation.selected(folder).project_id)["window_map"]["problems"]
    (unavailable,) = [problem for problem in problems if problem["code"] == "package_unavailable"]
    assert "pyproject.toml" in unavailable["detail"] and str(folder) in unavailable["detail"]
    assert not (tmp_path / "projects").exists(), "opening a folder in place copies nothing under the projects root"


def test_a_stale_lock_stops_the_launch_with_uvs_reason(tmp_path):
    """Caught: Studio rewriting a project's lock, or serving the project from an environment its lock does not describe."""
    folder = environment_project(tmp_path / "repository")
    pyproject = tmp_path / "repository" / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('dependencies = ["feedbax[web]"]', 'dependencies = ["feedbax[web]", "idna"]'))
    lock = (tmp_path / "repository" / "uv.lock").read_bytes()
    launched = subprocess.run(
        [sys.executable, "-B", "-m", "scripts.studio_dev", "--home", str(tmp_path / "home"), "--project", str(folder)],
        cwd=REPO, capture_output=True, text=True, timeout=HANG_SECONDS,
    )
    assert launched.returncode != 0
    assert "--check" in launched.stderr and "uv lock" in launched.stderr, launched.stderr
    assert (tmp_path / "repository" / "uv.lock").read_bytes() == lock
    assert not (tmp_path / "repository" / ".venv").exists()
    assert not (tmp_path / "home").exists()


def test_a_package_module_that_fails_to_import_is_reported_not_fatal(tmp_path):
    """Caught: a typo in one module of a project package takes the Studio backend down."""
    repository = tmp_path / "repository"
    environment_project(repository)
    source = repository / PACKAGE_SOURCE / "__init__.py"
    source.write_text(source.read_text() + "\ndef broken(:\n")
    site = tmp_path / "site"
    (site / "environment_fixture-1.0.0.dist-info").mkdir(parents=True)
    (site / "environment_fixture-1.0.0.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: environment-fixture\nVersion: 1.0.0\n",
    )
    (site / "environment_fixture-1.0.0.dist-info" / "entry_points.txt").write_text(
        "[feedbax.plugins]\nenvironment_fixture = environment_fixture\n",
    )
    shutil.copytree(repository / PACKAGE_SOURCE, site / "environment_fixture")
    probe = (
        "from feedbax.application.composition import compose_application, load_application; import asyncio; "
        "state = asyncio.run(compose_application()); (failure,) = state.failures; "
        "print(failure.source.distribution, failure.module, failure.error)"
    )
    served = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=HANG_SECONDS,
        env=os.environ | {"PYTHONPATH": str(site)},
    )
    assert served.returncode == 0, served.stderr
    assert served.stdout.startswith("environment-fixture environment_fixture ") and "line" in served.stdout
    strict = subprocess.run(
        [sys.executable, "-c", "from feedbax.application.composition import load_application; load_application()"],
        capture_output=True, text=True, timeout=HANG_SECONDS, env=os.environ | {"PYTHONPATH": str(site)},
    )
    assert strict.returncode != 0 and "environment_fixture" in strict.stderr
