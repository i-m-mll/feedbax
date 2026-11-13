"""The Unix-socket client exercises the detached host and real command workers."""

import asyncio
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

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


@pytest.mark.parametrize("stop_first", [False, True])
def test_completion_edges_are_bound_to_one_attempt(host, tmp_path, stop_first):
    """Exercise real daemon scheduling with controlled worker events."""
    release = tmp_path / "release"
    deliveries = tmp_path / "deliveries"
    (tmp_path / "phase_fixture.py").write_text("""
import json
import os
from pathlib import Path
import time

def main(args):
    payload = json.loads(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_text())
    stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    release = Path(payload["release"])
    with Path(os.environ["FEEDBAX_HOST_EVENT_LOG"]).open("a", buffering=1) as stream:
        checkpoint = {"completed_updates": 1}
        print(json.dumps({
            "kind": "publication", "data": {"key": "checkpoint:1", "checkpoint": checkpoint},
        }), file=stream)
        while not release.exists() and not stop.exists():
            time.sleep(0.01)
        print(json.dumps({
            "kind": "result",
            "data": {"outcome": "stopped", "checkpoint": checkpoint}
                if stop.exists() else {"outcome": "succeeded"},
        }), file=stream)
    return 0
""")
    launch = Launch(
        (sys.executable,),
        cwd=str(tmp_path),
        warm_entrypoint="phase_fixture:main",
        payload={"release": str(release)},
    )
    prefix = Job("prefix", "project", "prefix-request", launch)
    successor = run_job(
        "successor",
        f"from pathlib import Path; Path({str(deliveries)!r}).open('a').write('started\\n')",
        edges=(Edge(prefix.id),),
        slot="light",
    )
    submission = admit(host, "phases", successor, prefix)
    wait_until(lambda: job_row(host, prefix.id)["checkpoint"] == {"completed_updates": 1})
    assert job_row(host, successor.id)["state"] == "queued"
    assert not deliveries.exists()
    if stop_first:
        assert host.stop([prefix.id], "checkpoint")[0]["state"] == "stopping"
        assert wait_until(lambda: job_row(host, prefix.id, "stopped"))["checkpoint"]
        blocked = wait_until(lambda: job_row(host, successor.id, "blocked"))
        assert blocked["outcome"]["error"]["code"] == "dependency_ended"
        continued = Job("continued", prefix.project, prefix.request, launch, continues=prefix.id)
        admit(host, "continuation", continued)
        release.touch()
        wait_until(lambda: job_row(host, continued.id, "succeeded"))
        assert job_row(host, successor.id)["state"] == "blocked"
        assert not deliveries.exists()
    else:
        release.touch()
        finished = wait_until(lambda: job_row(host, successor.id, "succeeded"))
        assert finished["delivery"] == 1
        assert admit(host, "phases", successor, prefix) == submission
        assert deliveries.read_text().splitlines() == ["started"]


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
    independent = Job("independent", "project", "request", make_launch(fail=False))
    admit(host, "rerun", independent)
    wait_until(lambda: job_row(host, independent.id, "running"))
    with pytest.raises(HostError) as refusal:
        admit(host, "resume", continued)
    assert refusal.value.code == "invalid_resume"
    host.stop([independent.id], "checkpoint")
    wait_until(lambda: job_row(host, independent.id, "stopped"))
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


@pytest.fixture
def completion_worker(tmp_path):
    (tmp_path / "completion_fixture.py").write_text("""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

def main(args):
    mode, = args
    payload = json.loads(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_text())
    stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    root = Path(payload["manifest_root"])
    job_id = os.environ["FEEDBAX_HOST_JOB_ID"]
    with Path(os.environ["FEEDBAX_HOST_EVENT_LOG"]).open("a", buffering=1) as events:
        def emit(kind, data):
            print(json.dumps({"kind": kind, "data": data}), file=events)
        def member(label, value):
            relative = f"{job_id}-{label}.json"
            raw = json.dumps(value).encode()
            (root / relative).write_bytes(raw)
            return {"path": relative, "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
        input_member = member("binding", payload)
        digest = input_member["sha256"]
        emit("binding", {"binding_sha256": digest, "resume": payload.get("resume")})
        if mode == "failed":
            emit("result", {"outcome": "failed"})
            return 1
        checkpoint = {"completed_updates": 1, "attempt": job_id}
        emit("publication", {"key": "checkpoint:1", "checkpoint": checkpoint})
        while not (root / f"release-{job_id}").exists() and not stop.exists():
            time.sleep(0.01)
        if stop.exists():
            emit("result", {"outcome": "stopped", "checkpoint": checkpoint})
            return 0
        if mode == "missing":
            emit("result", {"outcome": "succeeded"})
            return 0
        checkpoint = {"completed_updates": 2, "attempt": job_id}
        receipt_file = member("receipt", checkpoint)
        checkpoint_file = member("checkpoint", checkpoint)
        receipt = {"manifest_id": job_id, "manifest_sha256": receipt_file["sha256"]}
        completion = {
            "producer_job_id": job_id, "project_id": payload["project_id"],
            "request_id": payload["request_id"] if mode != "wrong_request" else "different",
            "manifest_root": str(root), "binding_sha256": digest,
            "receipt": receipt, "checkpoint": checkpoint,
            "files": [input_member, receipt_file, checkpoint_file],
        }
        emit("result", {
            "outcome": "succeeded", "receipt": receipt, "checkpoint": checkpoint,
            "completion": completion,
        })
        return 0
""")

    def producer(job_id, *, request="same-request", continues=None, resume=None, mode="complete"):
        launch = Launch(
            (sys.executable, mode),
            cwd=str(tmp_path),
            warm_entrypoint="completion_fixture:main",
            payload={
                "manifest_root": str(tmp_path),
                "project_id": "project",
                "request_id": request,
                "resume": resume,
            },
        )
        return Job(job_id, "project", request, launch, continues=continues)

    return producer


def _completion_consumer(tmp_path, anchor, *, gated=True):
    output = tmp_path / "consumed.jsonl"
    source = f"""
import os
from pathlib import Path
bindings = Path(os.environ["FEEDBAX_HOST_COMPLETION_BINDINGS"]).read_text().strip()
with open({str(output)!r}, "a") as stream:
    print(bindings, file=stream)
"""
    edges = (Edge(anchor, authenticated_completion=True),)
    if gated:
        edges = (*edges, Edge("gate"))
    return run_job("consumer", source, edges=edges, slot="light")


def _completion_gate(tmp_path, *, project="project"):
    release = tmp_path / "open-gate"
    return run_job(
        "gate",
        f"""
import time
from pathlib import Path
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
""",
        project=project,
    )


def _bound_edge(host, job_id="consumer"):
    row = job_row(host, job_id)
    return next(edge["binding"] for edge in row["edges"] if edge.get("authenticated_completion"))


def test_authenticated_completion_pins_the_explicit_attempt_across_restart(
    host,
    tmp_path,
    completion_worker,
):
    original = completion_worker("original")
    consumer = _completion_consumer(tmp_path, original.id)
    gate = _completion_gate(tmp_path)
    submission = admit(host, "authenticated-phases", consumer, original, gate)
    with pytest.raises(HostError, match="has no such dependency"):
        host.mark_satisfied(consumer.id, Edge(original.id))
    wait_until(lambda: job_row(host, original.id)["checkpoint"])
    assert job_row(host, consumer.id)["state"] == "queued"

    host.stop([original.id], "checkpoint")
    stopped = wait_until(lambda: job_row(host, original.id, "stopped"))
    restart(host)
    assert job_row(host, consumer.id)["state"] == "queued"
    assert _bound_edge(host) is None

    rerun = completion_worker("unrelated-rerun")
    admit(host, "unrelated", rerun)
    (tmp_path / "release-unrelated-rerun").touch()
    wait_until(lambda: job_row(host, rerun.id, "succeeded"))
    assert _bound_edge(host) is None

    continued = completion_worker("explicit", continues=original.id, resume=stopped["checkpoint"])
    admit(host, "explicit", continued)
    (tmp_path / "release-explicit").touch()
    pinned = wait_until(lambda: _bound_edge(host))
    assert pinned["dependency_job_id"] == original.id
    assert pinned["completion"]["producer_job_id"] == continued.id
    assert pinned["completion"]["checkpoint"]["completed_updates"] == 2
    restart(host)
    assert _bound_edge(host) == pinned
    assert admit(host, "authenticated-phases", consumer, original, gate) == submission
    (tmp_path / "open-gate").touch()
    assert wait_until(lambda: job_row(host, consumer.id, "succeeded"))["delivery"] == 1
    assert json.loads((tmp_path / "consumed.jsonl").read_text()) == {"bindings": [pinned]}
    events = [event for event in host.events()["events"] if event["job_id"] == consumer.id]
    assert [event["data"] for event in events if event["kind"] == "dependency_bound"] == [pinned]


def test_authenticated_completion_rejects_tamper_after_pinning(
    host,
    tmp_path,
    completion_worker,
):
    producer = completion_worker("producer")
    consumer = _completion_consumer(tmp_path, producer.id)
    admit(host, "tamper", producer, consumer, _completion_gate(tmp_path))
    (tmp_path / "release-producer").touch()
    pinned = wait_until(lambda: _bound_edge(host))
    (tmp_path / "producer-checkpoint.json").write_text('{"changed":true}')
    (tmp_path / "open-gate").touch()
    blocked = wait_until(lambda: job_row(host, consumer.id, "blocked"))
    assert blocked["outcome"]["error"]["code"] == "dependency_corrupt"
    assert blocked["delivery"] == 0
    assert _bound_edge(host) == pinned
    assert not (tmp_path / "consumed.jsonl").exists()


@pytest.mark.parametrize(
    ("first_result", "success_source"),
    [
        ("stopped", "original"),
        ("failed", "original"),
        ("missing-product", "original"),
        ("wrong-input", "original"),
        ("corrupt-success", "original"),
        ("stopped", "first"),
    ],
)
def test_authenticated_completion_resolves_exact_resume_branches(
    host, tmp_path, completion_worker, first_result, success_source
):
    original = completion_worker("original")
    consumer = _completion_consumer(tmp_path, original.id)
    gate = _completion_gate(tmp_path)
    submission = admit(host, "resume-branches", original, consumer, gate)

    def stop(job):
        wait_until(lambda: job_row(host, job.id)["checkpoint"])
        host.stop([job.id], "checkpoint")
        return wait_until(lambda: job_row(host, job.id, "stopped"))

    original_checkpoint = stop(original)["checkpoint"]
    first_input = original_checkpoint
    if first_result == "wrong-input":
        first_input = original_checkpoint | {"completed_updates": True}
    first = completion_worker(
        "first",
        continues=original.id,
        resume=first_input,
        mode={
            "failed": "failed",
            "missing-product": "missing",
            "corrupt-success": "wrong_request",
        }.get(first_result, "complete"),
    )
    admit(host, "first-resume", first)
    if first_result == "stopped":
        first_row = stop(first)
    else:
        (tmp_path / "release-first").touch()
        first_state = "failed" if first_result == "failed" else "succeeded"
        first_row = wait_until(lambda: job_row(host, first.id, first_state))
    restart(host)
    assert job_row(host, consumer.id)["state"] == "queued"
    assert _bound_edge(host) is None
    rejection_stage = {"wrong-input": "continuation", "corrupt-success": "completion"}.get(
        first_result
    )

    def check_rejections(dependents):
        rejections = [
            event for event in host.events()["events"] if event["kind"] == "dependency_rejected"
        ]
        assert [event["job_id"] for event in rejections] == (
            [dependent.id for dependent in dependents] if rejection_stage is not None else []
        )
        for event in rejections:
            assert event["data"]["dependency_job_id"] == original.id
            assert event["data"]["candidate_job_id"] == first.id
            assert event["data"]["stage"] == rejection_stage
            assert event["data"]["error"]["code"] == "dependency_corrupt"

    check_rejections((consumer,))

    source_checkpoint = (
        original_checkpoint if success_source == original.id else first_row["checkpoint"]
    )
    selected = completion_worker("selected", continues=success_source, resume=source_checkpoint)
    admit(host, "successful-resume", selected)
    (tmp_path / "release-selected").touch()
    settled = wait_until(lambda: job_row(host, selected.id, "succeeded"))
    pinned = wait_until(lambda: _bound_edge(host))
    assert pinned == {
        "dependency_job_id": original.id,
        "completion": settled["outcome"]["completion"],
    }

    # A descendant on a different branch can succeed after the first pin.
    later_source = (
        first.id if first_result == "stopped" and success_source == original.id else original.id
    )
    later_checkpoint = first_row["checkpoint"] if later_source == first.id else original_checkpoint
    later = completion_worker("later", continues=later_source, resume=later_checkpoint)
    admit(host, "later-resume", later)
    (tmp_path / "release-later").touch()
    newer = wait_until(lambda: job_row(host, later.id, "succeeded"))
    assert newer["outcome"]["receipt"] != pinned["completion"]["receipt"]
    fresh = replace(consumer, id="fresh-consumer", request="fresh-consumer")
    fresh_submission = admit(host, "fresh-consumer", fresh)
    assert wait_until(lambda: _bound_edge(host, fresh.id)) == pinned
    restart(host)
    assert _bound_edge(host) == _bound_edge(host, fresh.id) == pinned
    assert admit(host, "resume-branches", original, consumer, gate) == submission
    assert admit(host, "fresh-consumer", fresh) == fresh_submission

    (tmp_path / "open-gate").touch()
    for dependent in (consumer, fresh):
        assert wait_until(lambda: job_row(host, dependent.id, "succeeded"))["delivery"] == 1
    restart(host)
    deliveries = (tmp_path / "consumed.jsonl").read_text().splitlines()
    assert [json.loads(row) for row in deliveries] == [{"bindings": [pinned]}] * 2
    check_rejections((consumer, fresh))
    assert [
        (event["job_id"], event["data"])
        for event in host.events()["events"]
        if event["kind"] == "dependency_bound"
    ] == [(consumer.id, pinned), (fresh.id, pinned)]


def test_authenticated_completion_keeps_a_corrupt_pin_when_a_sibling_succeeds(
    host, tmp_path, completion_worker
):
    original = completion_worker("original")
    consumer = _completion_consumer(tmp_path, original.id)
    admit(host, "corrupt-pin", original, consumer, _completion_gate(tmp_path))
    wait_until(lambda: job_row(host, original.id)["checkpoint"])
    host.stop([original.id], "checkpoint")
    checkpoint = wait_until(lambda: job_row(host, original.id, "stopped"))["checkpoint"]

    first = completion_worker("first", continues=original.id, resume=checkpoint)
    admit(host, "first-resume", first)
    (tmp_path / "release-first").touch()
    pinned = wait_until(lambda: _bound_edge(host))
    (tmp_path / "first-checkpoint.json").write_text('{"changed":true}')

    later = completion_worker("later", continues=original.id, resume=checkpoint)
    admit(host, "later-resume", later)
    (tmp_path / "release-later").touch()
    succeeded = wait_until(lambda: job_row(host, later.id, "succeeded"))
    fresh = replace(consumer, id="fresh-consumer", request="fresh-consumer")
    admit(host, "fresh-consumer", fresh)
    healthy = wait_until(lambda: _bound_edge(host, fresh.id))
    assert healthy["completion"] == succeeded["outcome"]["completion"]

    restart(host)
    assert _bound_edge(host) == pinned
    assert _bound_edge(host, fresh.id) == healthy
    (tmp_path / "open-gate").touch()
    blocked = wait_until(lambda: job_row(host, consumer.id, "blocked"))
    assert blocked["outcome"]["error"]["code"] == "dependency_corrupt"
    assert blocked["delivery"] == 0
    assert wait_until(lambda: job_row(host, fresh.id, "succeeded"))["delivery"] == 1
    assert json.loads((tmp_path / "consumed.jsonl").read_text()) == {"bindings": [healthy]}


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("failed", "input_unavailable"),
        ("missing", "input_unavailable"),
        ("wrong_request", "dependency_corrupt"),
    ],
)
def test_authenticated_completion_reports_definitive_dependency_errors(
    host,
    tmp_path,
    completion_worker,
    mode,
    code,
):
    producer = completion_worker("producer", mode=mode)
    consumer = _completion_consumer(tmp_path, producer.id, gated=False)
    admit(host, "definitive", producer, consumer)
    (tmp_path / "release-producer").touch()
    blocked = wait_until(lambda: job_row(host, consumer.id, "blocked"))
    assert blocked["outcome"]["error"]["code"] == code
    assert blocked["delivery"] == 0
    assert not (tmp_path / "consumed.jsonl").exists()


def test_real_studio_worker_resume_binds_completion_after_daemon_restart(
    host, tmp_path, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from feedbax.job_completion import verify_bound_training_completion
    from feedbax.jobs import JobService
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.training.checkpoint_custody import LATEST_POINTER_NAME
    from feedbax.training_host_worker import main as training_worker
    from feedbax.web.api import jobs, projects
    from feedbax.web.services.project_folder import ProjectFolderService
    from feedbax.contracts.project import Project
    from tests.test_canonical_evaluate_admission import _save_project
    from tests.test_document_train_pair import authored_documents, document_project

    # Pause only the original attempt at its first committed update so the
    # actual host can deliver the checkpoint stop before the next update.
    stop_ready = tmp_path / "stop-ready"
    (tmp_path / "gated_training.py").write_text(f"""
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from feedbax.training import executor
from {training_worker.__module__} import {training_worker.__name__} as run_training

def main(args):
    launch = json.loads(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_text())
    if launch.get("resume") is not None:
        return run_training(args)
    execute = executor.execute_managed_training_run

    def pause_first_update(*args, **kwargs):
        callback = kwargs["committed_update_callback"]

        def wait_for_stop(coordinate, completed, slots):
            callback(coordinate, completed, slots)
            if completed != 1:
                return
            Path({str(stop_ready)!r}).touch()
            stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
            deadline = time.monotonic() + 15
            while not stop.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Host did not deliver checkpoint stop.")
                time.sleep(0.01)

        kwargs["committed_update_callback"] = wait_for_stop
        return execute(*args, **kwargs)

    with patch.object(executor, "execute_managed_training_run", pause_first_update):
        return run_training(args)
""")

    service = JobService(host, new_application_registry_bundle())
    monkeypatch.setattr(
        service,
        "_training_worker",
        replace(
            service._training_worker,
            env={
                "JAX_PLATFORMS": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).resolve().parents[1]))),
            },
            warm_entrypoint="gated_training:main",
        ),
    )
    app = FastAPI()
    app.include_router(jobs.router)
    app.include_router(projects.router)
    app.state.job_service = service
    app.state.project_folders = ProjectFolderService(tmp_path / "projects")
    client = TestClient(app)
    project = "document-resume"
    _save_project(
        app.state.project_folders, project, Project.model_validate(document_project(authored_documents()))
    )
    revision = app.state.project_folders.read_plan(project, revision=None).revision
    submission = {
        "token": "ordinary-m1",
        "revision": revision,
        "run_keys": [{"variant": "default", "step": "train", "row": "default"}],
    }
    run_path = f"/api/projects/{project}/train-runs"
    submitted = client.post(run_path, json=submission)
    assert submitted.status_code == 202, submitted.text
    producer = submitted.json()["items"][0]["job_id"]
    consumer = run_job(
        "consumer",
        f"""
import json
import os
from pathlib import Path
from {verify_bound_training_completion.__module__} import {verify_bound_training_completion.__name__}
bindings = json.loads(Path(os.environ["FEEDBAX_HOST_COMPLETION_BINDINGS"]).read_text())["bindings"]
assert len(bindings) == 1
verify_bound_training_completion(bindings[0]["completion"])
Path({str(tmp_path / "verified.json")!r}).write_text(json.dumps(bindings[0]))
""",
        project=project,
        edges=(Edge(producer, authenticated_completion=True), Edge("gate")),
        slot="light",
    )
    # Give this probe Studio job metadata so its events reach the public
    # projection. The withheld publication keeps it out of worker delivery.
    visible_consumer = replace(
        run_job(
            "visible-consumer",
            "raise AssertionError('Withheld publication must prevent delivery')",
            project=project,
            edges=(
                Edge(producer, authenticated_completion=True),
                Edge("gate", "withheld"),
            ),
            slot="light",
        ),
        data=job_row(host, producer)["data"],
    )
    internal = admit(
        host,
        "real-completion",
        consumer,
        visible_consumer,
        _completion_gate(tmp_path, project=project),
    )
    listed = client.get("/api/submissions", params={"project": project})
    assert listed.status_code == 200, listed.text
    assert listed.json() == [submitted.json()]
    assert client.get(f"/api/submissions/{internal['id']}").status_code == 404
    collision = client.post(run_path, json=submission | {"token": "real-completion"})
    assert collision.status_code == 409, collision.text
    hidden_stop = client.post("/api/jobs/stop", json={"job_ids": [consumer.id], "mode": "force"})
    assert hidden_stop.status_code == 202, hidden_stop.text
    assert hidden_stop.json()[0]["error"]["code"] == "unknown_job"
    assert client.post(f"/api/jobs/{consumer.id}/seen").status_code == 404
    hidden_resume = client.post(
        f"/api/jobs/{consumer.id}/resume", json={"token": "internal-resume"}
    )
    assert hidden_resume.status_code == 404
    wait_until(lambda: stop_ready.exists() or job_row(host, producer)["outcome"], timeout=20)
    assert stop_ready.exists(), job_row(host, producer)
    stopping = client.post("/api/jobs/stop", json={"job_ids": [producer], "mode": "checkpoint"})
    assert stopping.status_code == 202, stopping.text
    wait_until(lambda: job_row(host, producer)["outcome"], timeout=20)
    stopped = job_row(host, producer)
    assert stopped["state"] == "stopped", stopped
    checkpoint = stopped["checkpoint"]
    assert checkpoint["completed_updates"] == 1
    assert job_row(host, consumer.id)["state"] == "queued"
    assert _bound_edge(host) is None
    cursor = host.jobs_snapshot()["cursor"]
    restart(host)
    assert job_row(host, consumer.id)["state"] == "queued"

    mismatched = replace(
        run_job("mismatched-resume", "pass", project=project, request=stopped["request"]),
        continues=producer,
    )
    admit(host, "mismatched-continuation", mismatched)
    wait_until(lambda: job_row(host, mismatched.id, "succeeded"))
    assert job_row(host, consumer.id)["state"] == "queued"
    assert _bound_edge(host) is None

    resumed = client.post(f"/api/jobs/{producer}/resume", json={"token": "ordinary-m1-resumed"})
    assert resumed.status_code == 202, resumed.text
    continuation = resumed.json()
    continued = continuation["job_id"]
    assert continuation["continues"] == producer
    assert continuation["request_id"] == stopped["request"]
    resumed_launch = host.jobs(job_id=continued, include_launch=True)[0]["launch"]
    assert resumed_launch["payload"]["resume"] == checkpoint
    wait_until(lambda: job_row(host, continued)["outcome"], timeout=20)
    finished = job_row(host, continued)
    assert finished["state"] == "succeeded", finished
    pinned = wait_until(lambda: _bound_edge(host) or job_row(host, consumer.id, "blocked"))
    assert "completion" in pinned, (pinned, finished["outcome"])
    completion = pinned["completion"]
    assert pinned["dependency_job_id"] == producer
    assert completion["producer_job_id"] == continued
    assert completion["receipt"] == finished["outcome"]["receipt"]
    assert completion["checkpoint"] == finished["outcome"]["checkpoint"]
    assert completion["checkpoint"]["completed_updates"] == 2
    restart(host)
    assert _bound_edge(host) == pinned
    latest = Path(completion["checkpoint"]["checkpoint_root"]) / LATEST_POINTER_NAME
    latest.write_text("This mutable pointer is outside the pinned completion closure.")
    (tmp_path / "open-gate").touch()
    terminal = wait_until(lambda: job_row(host, consumer.id)["outcome"], timeout=20)
    assert terminal["outcome"] == "succeeded", host.events()["events"][-10:]
    assert job_row(host, consumer.id)["delivery"] == 1
    assert json.loads((tmp_path / "verified.json").read_text()) == pinned
    raw = host.events(after=cursor, projects=(project,))
    assert [
        event["data"]
        for event in raw["events"]
        if event["job_id"] == consumer.id and event["kind"] == "dependency_bound"
    ] == [pinned]
    (refusal,) = [
        event
        for event in raw["events"]
        if event["job_id"] == visible_consumer.id and event["kind"] == "dependency_rejected"
    ]
    assert refusal["data"]["candidate_job_id"] == mismatched.id
    assert refusal["data"]["stage"] == "continuation"
    hidden_history = client.get(
        "/api/events/history",
        params={"project": project, "after": refusal["cursor"] - 1, "through": refusal["cursor"]},
    )
    assert hidden_history.status_code == 200, hidden_history.text
    assert hidden_history.json() == {"cursor": refusal["cursor"], "oldest_cursor": 0, "events": []}
    history = client.get(
        "/api/events/history", params={"project": project, "after": cursor, "through": raw["head"]}
    )
    assert history.status_code == 200, history.text
    assert history.json()["cursor"] == raw["head"]
    public = history.json()["events"]
    assert any(event["job_id"] == visible_consumer.id for event in public)
    assert {event["job_id"] for event in public} == {continued, visible_consumer.id}
    (published,) = [
        event["data"]
        for event in public
        if event["job_id"] == continued and event["kind"] == "terminal"
    ]
    assert published["receipt"] == completion["receipt"]
    assert published["checkpoint"] == completion["checkpoint"]
    assert not any(
        event["kind"] in {"dependency_bound", "dependency_rejected"}
        or {"completion", "completion_error"} & event["data"].keys()
        for event in public
    ), "Detailed history exposed host-only completion evidence."

    async def observe_folder():
        from httpx import ASGITransport, AsyncClient

        notices = []
        replay = []
        changed = asyncio.Event()

        async def receive():
            await changed.wait()
            return {"type": "http.disconnect"}

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http:

            async def send(message):
                if message["type"] != "http.response.body":
                    return
                body = message.get("body", b"").decode()
                if body.startswith("event: project_revision\n"):
                    assert "id:" not in body
                    notice = json.loads(body.split("data: ", 1)[1])
                    notices.append(notice["revision"])
                    if len(notices) == 1:
                        edit = await http.post(
                            f"/api/projects/{project}/files",
                            json={
                                "revision": revision,
                                "file": {
                                    "kind": "step",
                                    "id": "observe",
                                    "step": {"kind": "train", "base": {"inline": {}}},
                                },
                            },
                        )
                        assert edit.status_code == 201, edit.text
                        assert edit.json()["snapshot"]["revision"] != revision
                    else:
                        changed.set()
                        await asyncio.Future()
                elif body.startswith("id: "):
                    replay.append(body)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "method": "GET",
                "path": "/api/events",
                "query_string": f"machine=true&project={project}&folder={project}".encode(),
                "headers": [(b"last-event-id", str(cursor).encode())],
                "http_version": "1.1",
                "scheme": "http",
                "server": ("testserver", 80),
            }
            await asyncio.wait_for(app(scope, receive, send), timeout=5)
        return notices, replay

    notices, replay = asyncio.run(observe_folder())
    current = client.get(f"/api/projects/{project}").json()
    assert notices == [revision, current["revision"]]
    assert any(f"id: {raw['head']}\nevent: replay_complete\n" in frame for frame in replay)
    assert not any(
        '"kind":"dependency_bound"' in frame
        or '"kind":"dependency_rejected"' in frame
        or '"completion":' in frame
        or '"completion_error":' in frame
        for frame in replay
    )
    assert any('"kind":"terminal"' in frame and continued in frame for frame in replay)


def test_resumed_consumer_inherits_its_pinned_completion(host, tmp_path, completion_worker):
    producer = completion_worker("producer")
    consumer = _completion_consumer(tmp_path, producer.id)
    admit(host, "dependencies", producer, consumer, _completion_gate(tmp_path))
    (tmp_path / "release-producer").touch()
    pinned = wait_until(lambda: _bound_edge(host))
    assert host.stop([consumer.id], "checkpoint")[0]["state"] == "stopped"
    assert job_row(host, consumer.id)["delivery"] == 0

    newer = completion_worker("new-producer")
    admit(host, "newer", newer)
    (tmp_path / "release-new-producer").touch()
    wait_until(lambda: job_row(host, newer.id, "succeeded"))
    resumed = Job(
        "resumed-consumer",
        consumer.project,
        consumer.request,
        consumer.launch,
        slot=consumer.slot,
        continues=consumer.id,
    )
    admit(host, "resume-consumer", resumed)
    wait_until(lambda: job_row(host, resumed.id, "succeeded"))
    assert json.loads((tmp_path / "consumed.jsonl").read_text()) == {"bindings": [pinned]}
