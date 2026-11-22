"""A warm worker serves deliveries one after another until a retirement rule ends it."""

from pathlib import Path

import pytest

from feedbax.host import HostClient, Job
from tests.support import minimal_worker
from tests.support.host import FRESH_WORKERS_ENV, admit, job_row, wait_until


def _run(host, control: Path, name: str, **modes) -> dict:
    job = Job(name, name, f"request-{name}", minimal_worker.launch(control, **modes))
    minimal_worker.release(control, job.id)
    admit(host, name, job)
    return wait_until(lambda: (row := job_row(host, job.id)) and row["state"] not in ("queued", "assigned", "running") and row)


def test_worker_serves_other_projects_until_it_loads_project_code(reusing_host, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for project in (first, second):
        project.mkdir()
    (second / "plugin.py").write_text("")
    a = _run(reusing_host, tmp_path, "a", project_root=first)
    b = _run(reusing_host, tmp_path, "b", project_root=second)
    loads = _run(reusing_host, tmp_path, "loads", project_root=second, load=second / "plugin.py")
    same = _run(reusing_host, tmp_path, "same", project_root=second)
    other = _run(reusing_host, tmp_path, "other", project_root=first)
    assert [row["state"] for row in (a, b, loads, same, other)] == ["succeeded"] * 5
    assert a["worker_pid"] == b["worker_pid"] == loads["worker_pid"] == same["worker_pid"]
    assert b["delivery"] == same["delivery"] == 1
    assert other["worker_pid"] != a["worker_pid"]


@pytest.mark.parametrize("rule", ["failed", "thread", "source"])
def test_worker_retires_after_a_failure_a_left_thread_or_a_changed_source(reusing_host, tmp_path, rule):
    module = tmp_path / "loaded.py"
    module.write_text("")
    first = _run(
        reusing_host, tmp_path, "first", fail=rule == "failed", leave_thread=rule == "thread",
        load=module if rule == "source" else None,
    )
    assert first["state"] == ("failed" if rule == "failed" else "succeeded")
    if rule == "source":
        module.write_text("CHANGED = True\n")
    after = _run(reusing_host, tmp_path, "after")
    assert after["state"] == "succeeded"
    assert after["worker_pid"] != first["worker_pid"]


def test_reused_worker_trains_the_same_checkpoints_as_a_fresh_one(reusing_host, tmp_path, monkeypatch, private_hosts):
    from feedbax.jobs import JobService
    from feedbax.application.bundle import new_application_registry_bundle
    from feedbax.web.services.project_folder import ProjectFolderService
    from tests.test_desktop_background_activity import _project

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    projects = {name: tmp_path / name / "projects" for name in ("earlier", "reused", "fresh")}
    _project(projects["earlier"] / "reach", batches=3)
    for name in ("reused", "fresh"):
        _project(projects[name] / "reach", batches=2)
    registries = new_application_registry_bundle()

    def train(host: HostClient, name: str) -> dict:
        folders = ProjectFolderService(projects[name])
        plan = folders.read_plan("reach", revision=None)
        (run,) = plan.plan.runs
        (item,) = JobService(host, registries).admit_plan(folders, plan, run_keys=(run.key,), token=name).items
        return wait_until(lambda: job_row(host, item.job_id, "succeeded"))

    def blobs(name: str) -> set[str]:
        return {path.name for path in (projects[name] / "reach" / "results").glob("**/transactions/*/blobs/*.safetensors")}

    earlier = train(reusing_host, "earlier")
    reused = train(reusing_host, "reused")
    monkeypatch.setenv(FRESH_WORKERS_ENV, "1")
    fresh = train(HostClient(private_hosts / "fresh"), "fresh")
    assert reused["worker_pid"] == earlier["worker_pid"] and reused["delivery"] == 1
    assert fresh["worker_pid"] != reused["worker_pid"]
    assert blobs("reused") and blobs("reused") == blobs("fresh")
