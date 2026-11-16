"""Freeze a loopback SSH link while the venue and client clocks keep running."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import getpass
import secrets
import select
import shutil
import socket
import subprocess
import sys
from threading import Event
from time import monotonic

import pytest

from feedbax.host import HostClient, HostError
from feedbax.host.client import StdioAttachment
from feedbax.orchestration.runpod import SubprocessRunPodTransport
from feedbax.orchestration.session_jobs import SessionJobs
from feedbax.persistence.ssh_store import SSHStoreConnection
from tests.support.session_venue import LocalVenueProvider
from tests.test_local_job_host import admit, job_row, run_job, wait_until
from tests.test_session_rehearsal import _authorization, _driver, _jobs, _request


@contextmanager
def _loopback_sshd(root):
    sshd = shutil.which("sshd")
    if not sshd or not shutil.which("ssh") or not shutil.which("ssh-keygen"):
        pytest.skip("requires an OpenSSH client and server")
    for key in ("host", "client"):
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(root / key)],
            check=True, timeout=5,
        )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = root / "sshd.conf"
    config.write_text(f"""ListenAddress 127.0.0.1
Port {port}
HostKey "{root / 'host'}"
PidFile "{root / 'sshd.pid'}"
AuthorizedKeysFile "{root / 'client.pub'}"
StrictModes no
UsePAM no
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitUserRC no
AllowTcpForwarding no
""")
    log = root / "sshd.log"
    with log.open("wb") as stderr:
        server = subprocess.Popen([sshd, "-D", "-e", "-f", str(config)], stdin=subprocess.DEVNULL, stderr=stderr)
        try:
            def listening():
                contents = log.read_text()
                assert server.poll() is None, contents
                return f"Server listening on 127.0.0.1 port {port}" in contents

            wait_until(listening, timeout=4)
            yield port
        finally:
            server.terminate()
            server.wait(timeout=5)


def _relay(listener, server_port, stalled, finished) -> bool:
    """Discard traffic after the freeze, leaving both TCP connections open."""
    client, _ = listener.accept()
    with client, socket.create_connection(("127.0.0.1", server_port), timeout=5) as server:
        sockets = {client: server, server: client}
        while not finished.is_set():
            readable, _, _ = select.select(sockets, (), (), 0.2)
            for source in readable:
                data = source.recv(65536)
                if not data:
                    return source is client
                if not stalled.is_set():
                    sockets[source].sendall(data)
    return False


@pytest.mark.skipif(sys.platform != "darwin", reason="uses an unprivileged macOS OpenSSH server")
def test_silent_runpod_ssh_stall_expires_and_replays_the_same_stream(tmp_path, monkeypatch, record_property):
    provider = LocalVenueProvider(tmp_path / "venue")
    driver = _driver(tmp_path / "controller", provider)
    request = _request(provider)
    opened = driver.open(request, _authorization(request), adoption_token=secrets.token_urlsafe())
    assert opened.venue is not None
    host_root = provider.directory(opened.venue) / "host"
    local_attach = provider.client
    ready, release = tmp_path / "relay-ready", tmp_path / "release-job"
    stalled, finished = Event(), Event()

    def ssh_client(port, *, mark_ready=False):
        boot_key = " ".join((tmp_path / "host.pub").read_text().split()[:2])
        transport = SubprocessRunPodTransport(
            ssh_host="127.0.0.1", ssh_port=port, ssh_key_path=tmp_path / "client",
            strict_ssh_host_key=True, ssh_known_hosts_path=str(tmp_path / f"known-hosts-{port}"),
            ssh_boot_public_key=boot_key,
        )
        connection = SSHStoreConnection(f"{getpass.getuser()}@127.0.0.1", transport.pinned_ssh_argv())
        attach = [sys.executable, "-I", "-m", "feedbax.host", "--root", str(host_root), "attach", "--no-start"]
        if mark_ready:
            attach = [
                sys.executable, "-I", "-c",
                f"import os; from pathlib import Path; Path({str(ready)!r}).touch(); os.execv({attach[0]!r}, {attach!r})",
            ]
        return HostClient(attachment=StdioAttachment(lambda _: connection.command(attach)),
                          session=driver.session_access(request.id))

    try:
        with _loopback_sshd(tmp_path) as port, socket.socket() as listener, ThreadPoolExecutor(2) as workers:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(5)
            monkeypatch.setattr(provider, "client", lambda _resource, _access: ssh_client(port))
            jobs = SessionJobs(driver, request.id, _jobs)
            worker = run_job("held", f"""
from pathlib import Path
from time import sleep
while not Path({str(release)!r}).exists():
    sleep(0.02)
print("finished", flush=True)
""")
            admit(jobs.host, "one-job", worker)
            wait_until(lambda: job_row(jobs.host, worker.id, "running"))
            jobs.poll()
            before = jobs.history.observation(request.id)
            assert before is not None and before[1] > 0

            relay = workers.submit(_relay, listener, port, stalled, finished)
            frozen = ssh_client(listener.getsockname()[1], mark_ready=True)
            call = workers.submit(frozen.events, before[1], wait=25)
            try:
                wait_until(ready.is_file, timeout=4)
                assert not call.done()
                started = monotonic()
                stalled.set()
                # OpenSSH sends two 5-second probes, then exits on the next timer tick.
                with pytest.raises(HostError, match="exited with status 255") as error:
                    call.result(timeout=18)
                elapsed = monotonic() - started
                record_property("ssh_stall_seconds", round(elapsed, 3))
                assert error.value.code == "disconnected" and 8 < elapsed <= 18
                assert relay.result(timeout=3), "The peer closed before the SSH client detected the stall."
                assert jobs.history.observation(request.id) == before
            finally:
                finished.set()

            release.touch()
            wait_until(lambda: job_row(jobs.host, worker.id, "succeeded"))
            reattached = SessionJobs(driver, request.id, _jobs)
            page = reattached.poll()
            assert page["events"] and page["cursor"] > before[1]
            assert all(event["cursor"] > before[1] and event["job_id"] == worker.id for event in page["events"])
            assert reattached.history.cursors()[request.id] == (before[0], page["cursor"])
            _, replayed = reattached.history.events(
                request.id, before[0], after=before[1], through=page["cursor"], projects=[worker.project],
            )
            assert list(replayed) == page["events"]
            assert reattached.poll()["events"] == []
            record_property("cursor_before_stall", before[1])
            record_property("cursor_after_replay", page["cursor"])
            record_property("replayed_events", len(replayed))
    finally:
        release.touch(exist_ok=True)
        monkeypatch.setattr(provider, "client", local_attach)
        assert driver.close(request.id).remaining == ()
