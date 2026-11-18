"""Host fixtures, job helpers and the event-driven wait of the Host tests.

`host` is one Host per pytest process, shared by the tests that use it one after another. Its inputs are
fixed: it is started with the environment captured at session start, each test's keys are namespaced
(`run_job`, `token`), every job still live when a test ends is stopped, and the policy read at session start
is restored. A test that finds a different Host answering, or none, fails and the Host is replaced.
`FEEDBAX_TEST_PRIVATE_HOSTS=1` gives every test its own Host through the same fixture.

`private_host` is a Host of the test's own, started from the environment of the moment, for tests whose
subject is the Host's life, its pool, or Host-wide state.

Loaded as a plugin by `tests/conftest.py`, which also brings the follower-thread check, the session-end
sweep of leaked Hosts and the `round` marker's rule.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import cache
import itertools
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import TypeVar

import pytest

from feedbax.host import AdmissionItem, HostClient, HostError, Job, Launch
from feedbax.host.io import private_directory
from feedbax.host.python import same_python_argv
from tests.support.hang import HANG_SECONDS

T = TypeVar("T")

PRIVATE_HOSTS_ENV = "FEEDBAX_TEST_PRIVATE_HOSTS"
FOLLOWER_THREAD = "studio-job-activity"
_STEP_SECONDS = 0.02
_LISTEN_SECONDS = 0.2


@dataclass
class _Started:
    """A Host this module started, with the environment every start of it uses."""

    root: Path
    env: Mapping[str, str]
    pid: int


_started: dict[Path, _Started] = {}
_listening: HostClient | None = None
_namespace = ""
_tests = itertools.count()


def _client(root: Path) -> HostClient:
    client = HostClient(root)
    client.prohibit_start()
    return client


def _answering(root: Path) -> int | None:
    try:
        return _client(root).ping()["pid"]
    except HostError as error:
        if error.code != "disconnected":
            raise
        return None


def _launch(root: Path, env: Mapping[str, str]) -> _Started:
    private_directory(root)
    log_path = root / "host.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            same_python_argv("-m", "feedbax.host", "--root", str(root)),
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env=dict(env),
        )
    threading.Thread(target=process.wait, daemon=True).start()
    deadline = time.monotonic() + HANG_SECONDS
    while (pid := _answering(root)) is None:
        if process.poll() is not None:
            raise AssertionError(f"The Host exited at start with status {process.returncode}; see {log_path}.")
        if time.monotonic() >= deadline:
            raise AssertionError(f"The Host did not answer within {HANG_SECONDS} s; see {log_path}.")
        time.sleep(_STEP_SECONDS)
    started = _Started(root.resolve(), env, pid)
    _started[started.root] = started
    return started


def _exited(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _end(pid: int) -> None:
    """End one Host this process started and wait until its process has exited."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    wait_until(lambda: _exited(pid))


def _stop_live(client: HostClient) -> None:
    live = [job["id"] for job in client.jobs() if job["outcome"] is None]
    if live:
        client.stop(live, "force")
        wait_until(lambda: all(job_row(client, job_id)["outcome"] is not None for job_id in live))


@cache
def _registries():
    from feedbax.plugins.application import new_application_registry_bundle

    return new_application_registry_bundle()


def warm_training(host: HostClient) -> None:
    """Have the Host keep the Train worker that `JobService` launches from this process imported."""
    from feedbax.jobs import JobService

    JobService(host, _registries()).warm()


def worker_environment(service, env: Mapping[str, str]) -> None:
    """Add `env` to every worker launch of a `JobService`, so it reaches workers without the Host's environment."""
    for name in ("_training_worker", "_evaluation_worker", "_light_worker"):
        launch = getattr(service, name)
        setattr(service, name, replace(launch, env=launch.env | dict(env)))


def restart(host: HostClient) -> None:
    """Kill the Host at `host.root` and wait until a new Host answers there.

    A Host started by `host` comes back with its fixed environment and its Train worker warmed again; a
    `private_host` comes back through the client, with the environment of the moment.
    """
    before = host.ping()["pid"]
    started = _started.get(host.root)
    os.kill(before, signal.SIGKILL)
    if started is None:
        def new_pid():
            try:
                pid = host.ping()["pid"]
            except HostError as error:
                if error.code != "disconnected":
                    raise
                return None
            return pid if pid != before else None

        wait_until(new_pid)
        return
    wait_until(lambda: _answering(host.root) is None)
    _launch(host.root, started.env)
    warm_training(host)


def wait_until(read: Callable[[], T]) -> T:
    """Return the first truthy value of `read`, evaluating it again whenever the current test's Host changes.

    Between evaluations it long-polls the Host's events, so it waits at most `_LISTEN_SECONDS` for a change
    the Host does not record (a file a worker writes); without a Host it evaluates every `_STEP_SECONDS`.
    It fails after `HANG_SECONDS` with the states of the Host's live jobs.
    """
    listening = _listening
    deadline = time.monotonic() + HANG_SECONDS
    cursor = None
    while True:
        if listening is not None and cursor is None:
            cursor = _head(listening)
        value = read()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"The expected state did not arrive within {HANG_SECONDS} s. {_live_jobs(listening)}"
            )
        began = time.monotonic()
        if listening is not None and cursor is not None:
            try:
                cursor = listening.events(cursor, limit=1, wait=_LISTEN_SECONDS, start_host=False)["head"]
            except HostError as error:
                if error.code != "disconnected":
                    raise
                cursor = None
        time.sleep(max(0.0, _STEP_SECONDS - (time.monotonic() - began)))


def _head(client: HostClient) -> int | None:
    try:
        return client.events(0, limit=1, start_host=False)["head"]
    except HostError as error:
        if error.code != "disconnected":
            raise
        return None


def _live_jobs(client: HostClient | None) -> str:
    if client is None:
        return "No Host was being followed."
    try:
        live = [f"{job['id']}: {job['state']}" for job in client.jobs() if job["outcome"] is None]
    except HostError as error:
        return f"The Host did not answer: {error}."
    return f"Live jobs: {', '.join(live) or 'none'}."


def token(name: str) -> str:
    """A submission token of the current test, distinct from every other test's on a shared Host."""
    return f"{_namespace}{name}"


def run_job(name, source, *, project=None, request=None, edges=(), slot="compute"):
    return Job(
        f"{_namespace}{name}",
        project or f"{_namespace}project",
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


@contextmanager
def _following(root: Path, namespace: str) -> Iterator[None]:
    global _listening, _namespace
    _listening, _namespace = _client(root), namespace
    try:
        yield
    finally:
        _listening, _namespace = None, ""


@dataclass
class _Served:
    """A Host that `host` serves, replaced by a clean one under a new root when it is found changed."""

    base: Path
    env: Mapping[str, str]
    roots: Iterator[int]
    started: _Started | None = None
    policy: dict[str, int] | None = None

    def start(self) -> _Started:
        self.started = _launch(self.base / f"host-{next(self.roots)}", self.env)
        if self.policy is None:
            self.policy = _client(self.started.root).policy()
        return self.started

    def answering(self) -> str | None:
        """Why the Host answering at the root is not the one this process started there, if it is not."""
        assert self.started is not None
        expected = _started[self.started.root].pid
        pid = _answering(self.started.root)
        if pid == expected:
            return None
        return f"Host {expected} was expected at {self.started.root}, and {pid or 'no Host'} answered"

    def replace(self) -> None:
        assert self.started is not None
        root = self.started.root
        for pid in {_started[root].pid, _answering(root)} - {None}:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        wait_until(lambda: _answering(root) is None)
        self.start()

    def stop(self) -> None:
        if self.started is not None:
            _end(_started[self.started.root].pid)


@pytest.fixture(scope="session")
def _host_environment() -> dict[str, str]:
    return dict(os.environ)


@pytest.fixture(scope="session")
def _shared_host(tmp_path_factory, _host_environment) -> Iterator[_Served]:
    shared = _Served(tmp_path_factory.mktemp("shared-host"), _host_environment, itertools.count())
    shared.start()
    try:
        yield shared
    finally:
        shared.stop()


@pytest.fixture
def host(request, tmp_path, _host_environment) -> Iterator[HostClient]:
    """The pytest process's shared Host, or with `FEEDBAX_TEST_PRIVATE_HOSTS=1` one Host per test."""
    namespace = f"t{next(_tests)}-"
    if os.environ.get(PRIVATE_HOSTS_ENV) != "1":
        yield from _serve(request.getfixturevalue("_shared_host"), namespace)
        return
    private = _Served(tmp_path, _host_environment, itertools.count())
    private.start()
    try:
        yield from _serve(private, namespace)
    finally:
        private.stop()


def _serve(served: _Served, namespace: str) -> Iterator[HostClient]:
    if (changed := served.answering()) is not None:
        served.replace()
        pytest.fail(f"The Host changed before this test: {changed}. It was replaced.")
    root = served.started.root
    with _following(root, namespace):
        yield _client(root)
        if (changed := served.answering()) is not None:
            served.replace()
            pytest.fail(f"This test changed the Host: {changed}. It was replaced.")
        admin = _client(root)
        _stop_live(admin)
        admin.policy(**served.policy)


@pytest.fixture
def private_host(tmp_path) -> Iterator[HostClient]:
    """A Host of the test's own, started on first use with the environment of the moment."""
    client = HostClient(tmp_path / "host")
    client.ping()
    with _following(client.root, ""):
        try:
            yield client
        finally:
            _stop_live(client)
            _end(client.ping()["pid"])


@pytest.fixture(autouse=True)
def _no_follower_left() -> Iterator[None]:
    """Fail a test that leaves a Studio job-activity follower running."""
    before = set(threading.enumerate())
    yield
    left = [thread for thread in threading.enumerate() if thread.name == FOLLOWER_THREAD and thread not in before]
    if left:
        pytest.fail(f"The test left {len(left)} {FOLLOWER_THREAD} thread(s) running; close each JobActivity.")


def _hosts_under(root: Path) -> list[tuple[int, Path]]:
    listing = subprocess.run(
        ["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True, check=True,
        timeout=HANG_SECONDS,
    ).stdout
    hosts = []
    for line in listing.splitlines():
        pid, _, command = line.strip().partition(" ")
        argv = command.split()
        if "attach" in argv or "--root" not in argv[:-1]:
            continue
        if not any(argv[i:i + 2] == ["-m", "feedbax.host"] for i in range(len(argv))):
            continue
        host_root = Path(argv[argv.index("--root") + 1]).resolve()
        if host_root.is_relative_to(root):
            hosts.append((int(pid), host_root))
    return hosts


def pytest_sessionfinish(session: pytest.Session) -> None:
    """In the controller, end every Host still running under the invocation's temporary root and fail the run."""
    if hasattr(session.config, "workerinput"):
        return
    root = session.config._tmp_path_factory.getbasetemp().resolve()
    leaked = _hosts_under(root)
    if not leaked:
        return
    for pid, _ in leaked:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    listing = "\n".join(f"  {pid}: {host_root}" for pid, host_root in leaked)
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    message = f"Hosts were still running at the end of the session and were ended:\n{listing}"
    if reporter is not None:
        reporter.write_sep("=", "leaked Hosts", red=True)
        reporter.write_line(message)
    else:
        print(message, file=sys.stderr)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Refuse a `round` mark that does not say, in one sentence, what must really happen."""
    for item in items:
        for mark in item.iter_markers("round"):
            if mark.kwargs or len(mark.args) != 1 or not isinstance(mark.args[0], str) or not mark.args[0].strip():
                raise pytest.UsageError(
                    f"{item.nodeid}: `round` takes one sentence saying what must really happen."
                )
