"""The Unix-socket client exercises the detached host and real command workers."""

import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from feedbax.host import AdmissionItem, Edge, HostClient, HostError, Job, Launch


@pytest.fixture
def host(tmp_path):
    client = HostClient(tmp_path / "host")
    client.ping()
    try:
        yield client
    finally:
        live = [job["id"] for job in client.jobs() if job["outcome"] is None]
        if live:
            client.stop(live, "force")
            wait_until(lambda: all(job["outcome"] is not None for job in client.jobs()))
        os.kill(client.ping()["pid"], signal.SIGTERM)
        wait_until(lambda: not client.address.exists())


def wait_until(read, *, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = read()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("The host did not reach the expected state.")


def run_job(name, source, *, project="project", request=None, edges=(), slot="compute"):
    return Job(
        name,
        project,
        request or name,
        Launch((sys.executable, "-c", source)),
        edges=edges,
        slot=slot,
    )


def admit(host, token, *jobs):
    return host.admit(token, [AdmissionItem(job.id, job) for job in jobs], record={})


def job_row(host, name, state=None):
    rows = host.jobs(job_id=name)
    return rows[0] if rows and (state is None or rows[0]["state"] == state) else None


def restart(host):
    before = host.ping()["pid"]
    os.kill(before, signal.SIGKILL)

    def new_pid():
        try:
            pid = host.ping()["pid"]
        except HostError as error:
            if error.code != "disconnected":
                raise
            return None
        return pid if pid != before else None

    wait_until(new_pid)


def test_host_keeps_attempt_and_cursor_across_daemon_restart(host, tmp_path):
    executions = tmp_path / "executions"
    release = tmp_path / "release"
    source = f"""
from pathlib import Path
import time
with open({str(executions)!r}, "a") as stream:
    print("started", file=stream)
print("live", flush=True)
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
print("completed", flush=True)
"""
    main = run_job("main", source)
    alias = run_job("alias", source, request=main.request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(admit, host, "same-token", main)
        second = pool.submit(admit, host, "same-token", main)
        assert first.result() == second.result()
    attached = admit(host, "attach", alias)
    assert attached["items"][0]["job_id"] == main.id
    assert attached["items"][0]["disposition"] == "attached"
    wait_until(lambda: executions.exists() and job_row(host, main.id, "running"))
    before = host.jobs_snapshot()
    assert before["jobs"][0]["state"] == "running"
    restart(host)
    release.touch()
    finished = wait_until(lambda: job_row(host, main.id, "succeeded"))
    assert finished["worker_pid"] == before["jobs"][0]["worker_pid"]
    assert finished["delivery"] == 1
    assert executions.read_text().splitlines() == ["started"]
    assert admit(host, "same-token", main) == first.result()
    assert host.submissions(token="same-token") == [first.result()]
    assert admit(host, "repeat-job-id", main)["items"][0]["job_id"] == main.id
    assert len(host.jobs()) == 1
    tail = host.events(before["cursor"], projects=["project"])
    assert sum(event["kind"] == "terminal" for event in tail["events"]) == 1
    assert any(event["data"].get("line") == "completed" for event in tail["events"])
    other = host.events(before["cursor"], projects=["other"])
    assert other["events"] == []
    assert other["cursor"] == tail["head"]


def test_publication_releases_one_dependent_without_client(host, tmp_path):
    release = tmp_path / "release"
    side_effect = tmp_path / "dependent"
    parent = run_job(
        "parent",
        f"""
import os
import time
from pathlib import Path
with open(os.environ["FEEDBAX_HOST_EVENT_LOG"], "a") as stream:
    stream.write('{{"kind":"publication","data":{{"key":"checkpoint:7","checkpoint":{{"step":7}}}}}}\\n')
    stream.flush()
    os.fsync(stream.fileno())
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
""",
    )
    dependent = run_job(
        "dependent",
        f"from pathlib import Path; Path({str(side_effect)!r}).touch()",
        edges=(Edge(parent.id, "checkpoint:7"),),
        slot="light",
    )
    blocked = run_job("blocked", "", edges=(Edge(parent.id, "checkpoint:8"),))
    admit(host, "dag", blocked, dependent, parent)
    wait_until(side_effect.exists)
    assert job_row(host, parent.id)["state"] == "running"
    assert job_row(host, parent.id)["checkpoint"] == {"step": 7}
    release.touch()
    wait_until(lambda: job_row(host, dependent.id, "succeeded"))
    wait_until(lambda: job_row(host, blocked.id, "blocked"))


def test_warm_entrypoint_reports_checkpoint_stop_and_typed_error(host, tmp_path):
    module = tmp_path / "worker_fixture.py"
    module.write_text("""
import json
import os
import time
from pathlib import Path

def main(args):
    payload = json.loads(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_text())
    log = Path(os.environ["FEEDBAX_HOST_EVENT_LOG"])
    if payload["fail"]:
        result = {"outcome": "failed", "error": {"code": "update_failed", "message": "Update failed."}}
    else:
        while not Path(os.environ["FEEDBAX_HOST_STOP_FILE"]).exists():
            time.sleep(0.01)
        result = {"outcome": "stopped", "checkpoint": {"step": 4}}
    with log.open("a") as stream:
        stream.write(json.dumps({"kind": "result", "data": result}) + "\\n")
    return int(payload["fail"])
""")

    def make_launch(*, fail):
        return Launch(
            (sys.executable,),
            cwd=str(tmp_path),
            warm_entrypoint="worker_fixture:main",
            payload={"fail": fail},
        )

    host.warm(make_launch(fail=False))
    first = Job("first", "project", "request", make_launch(fail=False))
    admit(host, "stop", first)
    wait_until(lambda: job_row(host, first.id, "running"))
    assert host.stop([first.id], "checkpoint")[0]["state"] == "stopping"
    stopped = wait_until(lambda: job_row(host, first.id, "stopped"))
    assert stopped["checkpoint"] == {"step": 4}
    continued = Job("continued", "project", "request", make_launch(fail=True), continues=first.id)
    admit(host, "resume", continued)
    failed = wait_until(lambda: job_row(host, continued.id, "failed"))
    assert failed["continues"] == first.id
    assert failed["outcome"]["error"]["code"] == "update_failed"


def test_force_escalates_and_signal_without_intent_fails(host, tmp_path):
    child_pid = tmp_path / "child-pid"
    ignores_term = run_job(
        "ignores-term",
        f"""
import os
import signal
import time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(child_pid)!r}).write_text(str(os.getpid()))
time.sleep(120)
""",
    )
    admit(host, "force", ignores_term)
    wait_until(lambda: child_pid.exists() and child_pid.read_text())
    start = time.monotonic()
    host.stop([ignores_term.id], "force")
    wait_until(lambda: job_row(host, ignores_term.id, "stopped"), timeout=5)
    assert time.monotonic() - start < 5

    crash = run_job("crash", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
    admit(host, "crash", crash)
    outcome = wait_until(lambda: job_row(host, crash.id, "failed"))["outcome"]
    assert outcome["signal"] == signal.SIGKILL
    assert outcome["error"]["code"] == "worker_signal"


def test_admission_of_cyclic_dependencies_leaves_no_partial_jobs(host):
    a = run_job("a", "", edges=(Edge("b"),))
    b = run_job("b", "", edges=(Edge("a"),))
    with pytest.raises(HostError, match="cyclic dependency"):
        admit(host, "cycle", a, b)
    assert host.jobs() == []
    assert host.submissions(token="cycle") == []
    assert host.events()["events"] == []


def test_admit_rechecks_success_after_service_snapshot(host):
    first = run_job("first", "", request="request")
    followup = run_job("followup", "", request=first.request)
    initial_cursor = host.jobs_snapshot()["cursor"]
    admit(host, "first", first)
    wait_until(lambda: job_row(host, first.id, "succeeded"))
    unrelated = run_job("unrelated", "")
    items = [
        AdmissionItem("unrelated", unrelated),
        AdmissionItem("followup", followup, checked_through=initial_cursor),
    ]
    with pytest.raises(HostError) as changed:
        host.admit("followup", items, record={})
    assert changed.value.code == "admission_changed"
    assert job_row(host, "unrelated") is None
    assert host.submissions(token="followup") == []
    # Explicit rerun is independent of the service's previously observed success.
    admit(host, "followup", followup)
    wait_until(lambda: job_row(host, followup.id, "succeeded"))
