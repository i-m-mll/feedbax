"""A venue uses the detached Host's normal RPCs with durable controller ownership."""

import asyncio
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.host import AdmissionItem, HostClient, HostError, Job, Launch
from feedbax.host.client import StdioAttachment
from feedbax.host.daemon import Host
from feedbax.host.protocol import VERSION, SessionAccess
from feedbax.host.session import SessionFence, initialize_session

SESSION = "venue-session"
OPERATOR = "operator"
ADOPTION = "adoption-" * 5
INITIAL = "initial-" * 5
NEXT = "next-" * 9
OTHER = "other-" * 8


def initialize(root: Path) -> float:
    expiry = time.time() + 120
    initialize_session(
        root, session_id=SESSION, operator=OPERATOR, expires_at=expiry,
        adoption_token=ADOPTION, controller_token=INITIAL,
    )
    return expiry


def attached(root: Path, access: SessionAccess | None = None) -> HostClient:
    def command(start_host: bool) -> list[str]:
        argv = [sys.executable, "-m", "feedbax.host", "--root", str(root), "attach"]
        return argv if start_host else [*argv, "--no-start"]

    return HostClient(root, attachment=StdioAttachment(command), session=access)


def expect_code(code: str, call, /, *args, **kwargs):
    with pytest.raises(HostError) as raised:
        call(*args, **kwargs)
    assert raised.value.code == code


def until(read):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if value := read():
            return value
        time.sleep(0.02)
    pytest.fail("The Host did not reach the expected state.")


def stop_daemon(client: HostClient, pid: int) -> None:
    os.kill(pid, signal.SIGTERM)
    until(lambda: not client.address.exists())


def held_job(release: Path) -> Job:
    source = f"""
from pathlib import Path
import time
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
"""
    return Job("held", "project", "request", Launch((sys.executable, "-c", source)))


def test_real_attachments_share_host_and_adoption_fences_old_controller(tmp_path):
    root = tmp_path / "venue"
    old_access = SessionAccess(SESSION, 0, INITIAL)
    old = attached(root, old_access)
    unscoped = attached(root)
    expect_code("session_uninitialized", old.ping)
    assert not root.exists()
    initialize(root)
    expect_code("disconnected", old.jobs_snapshot, start_host=False)
    old_ping = old.ping()
    pid = old_ping["pid"]
    assert type(pid) is int
    release = tmp_path / "release"
    current = old
    try:
        assert attached(root, old_access).ping() == old_ping
        expect_code("session_fenced", HostClient(root).record_put, "direct", {})
        expect_code("session_fenced", unscoped.record_put, "unscoped", {})
        expect_code("session_fenced", unscoped.session_epoch, SESSION, OPERATOR, INITIAL)
        expect_code("session_not_closing", unscoped.recover_session_close, SESSION, OPERATOR, ADOPTION)
        job = held_job(release)
        admission = old.admit("token", [AdmissionItem("key", job)], record={})
        assert admission["items"][0]["job_id"] == job.id
        running = until(lambda: next(
            (row for row in old.jobs(job_id=job.id) if row["state"] == "running"), None,
        ))
        expect_code("session_busy", old.claim_session_close)

        def adopt(token):
            try:
                return unscoped.adopt_session(SESSION, OPERATOR, 0, ADOPTION, token), token
            except HostError as error:
                assert error.code == "session_fenced"
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            winners = [result for result in pool.map(adopt, [NEXT, OTHER]) if result is not None]
        assert len(winners) == 1
        epoch, token = winners[0]
        assert epoch == unscoped.session_epoch(SESSION, OPERATOR, ADOPTION) == 1
        expect_code("session_fenced", old.stop, [job.id], "force")
        expect_code("session_fenced", old.acknowledge_collection, job.id, "a" * 64)
        expect_code("session_fenced", old.claim_session_close)
        current = attached(root, SessionAccess(SESSION, epoch, token))
        assert current.ping() == old_ping
        assert current.jobs(job_id=job.id)[0]["worker_pid"] == running["worker_pid"]
        release.touch()
        until(lambda: not current.session_status()["live_jobs"])
        claim = current.claim_session_close()
        assert claim == {"session_id": SESSION, "operator": OPERATOR, "epoch": 1}
        expect_code("session_closing", current.policy, compute=1)
        expect_code("session_closing", unscoped.adopt_session, SESSION, OPERATOR, epoch, ADOPTION, INITIAL)
        expect_code("session_fenced", unscoped.recover_session_close, SESSION, OPERATOR, token)
        assert unscoped.recover_session_close(SESSION, OPERATOR, ADOPTION) == claim
        stop_daemon(current, pid)
        reopened = current.ping()
        pid = reopened["pid"]
        assert type(pid) is int and reopened != old_ping
        assert reopened["stream_id"] == old_ping["stream_id"]
        assert current.claim_session_close() == claim
        assert unscoped.recover_session_close(SESSION, OPERATOR, ADOPTION) == claim
    finally:
        release.touch()
        if current.address.exists():
            stop_daemon(current, pid)


def test_close_requires_exact_acknowledgements_and_retains_evidence(tmp_path, monkeypatch):
    root = tmp_path / "venue"
    expires_at = initialize(root)
    access = asdict(SessionAccess(SESSION, 0, INITIAL))

    async def exchange():
        host = Host(root)

        async def call(op, **params):
            return await host._call({"op": op, "version": VERSION, "session": access, **params})

        try:
            digest = "a" * 64
            ref = ContentObjectRef(
                object_id=f"content-object://sha256/{digest}", sha256=digest, size_bytes=1,
            ).model_dump(mode="json")
            job = held_job(tmp_path / "release")
            await call("admit", token="seed", items=[asdict(AdmissionItem("key", job))], record={})
            row = host.store.get(job.id)
            assert row is not None
            # Supply the approved publication at the Store/Host boundary. The
            # managed worker's artifact verification is tested by its own suite.
            with host.store.db:
                host.store.finished(row, {"outcome": "succeeded", "store_success": ref})
            with pytest.raises(HostError, match="unacknowledged") as busy:
                await call("claim_session_close")
            assert busy.value.code == "session_busy"
            assert (await call("session_status"))["pending"] == {job.id: digest}
            with pytest.raises(HostError) as mismatch:
                await call("acknowledge_collection", job_id=job.id, sha256="b" * 64)
            assert mismatch.value.code == "collection_conflict"
            await call("acknowledge_collection", job_id=job.id, sha256=digest)
            await call("acknowledge_collection", job_id=job.id, sha256=digest)
            assert (await call("session_status"))["collections"] == {job.id: digest}
            persisted = SessionFence.load(root)
            assert persisted is not None and persisted.state.collections == {job.id: digest}

            queued = Job("queued", "project", "later", job.launch)
            await call("admit", token="later", items=[asdict(AdmissionItem("key", queued))], record={})
            monkeypatch.setattr("feedbax.host.session.time", SimpleNamespace(time=lambda: expires_at))
            host._tick()
            assert host.store.get(queued.id)["state"] == "queued"
            assert host.workers == {}
            with pytest.raises(HostError) as expired:
                await call("stop", job_ids=[queued.id], mode="force")
            assert expired.value.code == "session_expired"
        finally:
            host.file_changes.close()
            host.store.db.close()
            host.lock.close()

    asyncio.run(exchange())

    persisted = SessionFence.load(root)
    assert persisted is not None
    persisted.path.unlink()

    async def reopen_without_fence():
        expect_code("session_unavailable", Host, root)

    asyncio.run(reopen_without_fence())


def test_worker_that_finishes_cold_import_after_expiry_never_invokes_job(tmp_path):
    root = tmp_path / "venue"
    release = tmp_path / "release"
    invoked = tmp_path / "invoked"
    (tmp_path / "delayed_worker.py").write_text(f"""
import time
from pathlib import Path
while not Path({str(release)!r}).exists():
    time.sleep(0.01)

def main(args):
    Path({str(invoked)!r}).touch()
""")
    expires_at = time.time() + 4
    initialize_session(
        root, session_id=SESSION, operator=OPERATOR, expires_at=expires_at,
        adoption_token=ADOPTION, controller_token=INITIAL,
    )
    client = attached(root, SessionAccess(SESSION, 0, INITIAL))
    pid = client.ping()["pid"]
    assert type(pid) is int
    try:
        launch = Launch((sys.executable,), cwd=str(tmp_path), warm_entrypoint="delayed_worker:main")
        job = Job("delayed", "project", "request", launch)
        client.admit("token", [AdmissionItem("key", job)], record={})
        until(lambda: client.jobs(job_id=job.id)[0]["state"] == "assigned")
        until(lambda: time.time() >= expires_at)
        release.touch()
        outcome = until(lambda: client.jobs(job_id=job.id)[0]["outcome"])
        assert outcome["error"]["code"] == "session_expired"
        assert not invoked.exists()
    finally:
        release.touch()
        if client.address.exists():
            stop_daemon(client, pid)
