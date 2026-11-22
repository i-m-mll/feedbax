"""Host fixtures, job helpers and the event-driven wait of the Host tests.

`host` is one Host per pytest process, shared by the tests that use it one after another. Its inputs are
fixed: it is started with the environment captured at session start, each test's keys are namespaced
(`run_job`, `token`), every job still live when a test ends is stopped, and the policy read at session start
is restored. A test that finds a different Host answering, or none, fails and the Host is replaced.
`FEEDBAX_TEST_PRIVATE_HOSTS=1` gives every test its own Host through the same fixture. Every Host a test starts,
directly or through a child process, has its pytest process as owner and exits when that process exits.

A test starts a Host only where it has said it will. Every start by a `HostClient` in this process is
recorded, and a test that is not marked `round` fails for each start at a root outside its `private_hosts`
directory. `private_hosts` is where a test whose subject is a Host's start, life, pool, Host-wide state or
environment starts its own Hosts, and it ends them afterwards; `private_host` is a started Host there. The
support's own starts (the shared Host, its restart and replacement, and every Host under
`FEEDBAX_TEST_PRIVATE_HOSTS=1`) are not client starts. A Host started by a child process of the test is the
product's own and is not recorded; the session-end sweep fails the run if one is left.

A job that holds its slot until released or stopped is `held_until_released` or `HELD_UNTIL_STOPPED`.

Loaded as a plugin by `tests/conftest.py`, which also brings the follower-thread check, the session-end
sweep of leaked processes and the `round` marker's rule.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import cache
import asyncio
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
from feedbax.host import client as _client_module
from feedbax.host.io import OWNER_ENV
from tests.support.hang import HANG_SECONDS

T = TypeVar("T")

PRIVATE_HOSTS_ENV = "FEEDBAX_TEST_PRIVATE_HOSTS"
FRESH_WORKERS_ENV = "FEEDBAX_HOST_FRESH_WORKERS"
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
_launch_host = _client_module.launch_host
_client_starts: list[Path] = []
_declared: Path | None = None
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


def _recorded_launch(root: Path, env: Mapping[str, str] | None = None) -> subprocess.Popen:
    _client_starts.append(root.resolve())
    return _launch_host(root, env)


def pytest_configure(config: pytest.Config) -> None:
    os.environ[OWNER_ENV] = str(os.getpid())
    _client_module.launch_host = _recorded_launch


def _launch(root: Path, env: Mapping[str, str]) -> _Started:
    process = _launch_host(root, env)
    log_path = root / "host.log"
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


def _stop_live(client: HostClient) -> None:
    live = [job["id"] for job in client.jobs() if job["outcome"] is None]
    if live:
        client.stop(live, "force")
        wait_until(lambda: all(job_row(client, job_id)["outcome"] is not None for job_id in live))


@cache
def _registries():
    from feedbax.application.bundle import new_application_registry_bundle

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


async def wait_until_async(read: Callable[[], T]) -> T:
    """Return the first truthy value of `read` within a running event loop, evaluating it every `_STEP_SECONDS`.

    It fails after `HANG_SECONDS`.
    """
    deadline = time.monotonic() + HANG_SECONDS
    while not (value := read()):
        if time.monotonic() >= deadline:
            raise AssertionError(f"The expected state did not arrive within {HANG_SECONDS} s.")
        await asyncio.sleep(_STEP_SECONDS)
    return value


def until_exists(*paths: str, indent: str = "") -> str:
    """Source of a statement that waits until a file exists at one of `paths`, each a Python expression.

    It goes into the source of a job or a worker module a test writes; `indent` prefixes each of its lines.
    """
    source = (
        "import pathlib as _pathlib, time as _time\n"
        f"while not any(_pathlib.Path(_path).exists() for _path in ({', '.join(paths)},)):\n"
        f"    _time.sleep({_STEP_SECONDS / 2})\n"
    )
    return "".join(f"{indent}{line}\n" for line in source.splitlines())


def held_until_released(release: Path) -> str:
    """Source of a job that holds its slot until a file exists at `release`."""
    return until_exists(repr(str(release)))


HELD_UNTIL_STOPPED = "import threading\nthreading.Event().wait()\n"
"""Source of a job that holds its slot until it is stopped."""


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
            _client(self.started.root).shutdown()


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
def private_hosts(tmp_path) -> Iterator[Path]:
    """The directory under which this test starts Hosts of its own; each is ended when the test ends."""
    global _declared
    directory = (tmp_path / "hosts").resolve()
    directory.mkdir()
    _declared = directory
    yield directory
    for root in dict.fromkeys(_client_starts):
        if root.is_relative_to(directory):
            _client(root).shutdown()


@pytest.fixture
def private_host(private_hosts) -> Iterator[HostClient]:
    """A Host of the test's own under `private_hosts`, started with the environment of the moment."""
    client = HostClient(private_hosts / "host")
    client.ping()
    with _following(client.root, ""):
        yield client


@pytest.fixture
def reusing_host(monkeypatch, request) -> HostClient:
    """A `private_host` whose workers serve deliveries one after another in every run of `make test-round`."""
    monkeypatch.delenv(FRESH_WORKERS_ENV, raising=False)
    return request.getfixturevalue("private_host")


_UNDECLARED = (
    "This test started a Host at {root} that it did not declare. Take the `host` fixture to run on the shared"
    " Host. If a Host of its own is part of what the test is about (its start, life, pool, Host-wide state or"
    " environment), take `private_host`, or take `private_hosts` and start it under that directory. If it"
    " needs cold real workers or a real clock, mark it `round`."
)


@pytest.fixture(autouse=True)
def _hosts_declared(request) -> Iterator[None]:
    """Fail a fast-tier test for each Host its clients started outside its `private_hosts` directory.

    Starts made between tests (by a fixture of a wider scope) count for the next test.
    """
    global _declared
    yield
    starts, declared = list(dict.fromkeys(_client_starts)), _declared
    _client_starts.clear()
    _declared = None
    if request.node.get_closest_marker("round") is not None:
        return
    undeclared = [root for root in starts if declared is None or not root.is_relative_to(declared)]
    if undeclared:
        pytest.fail("\n".join(_UNDECLARED.format(root=root) for root in undeclared))


@pytest.fixture(autouse=True)
def _no_follower_left() -> Iterator[None]:
    """Fail a test that leaves a Studio job-activity follower running."""
    before = set(threading.enumerate())
    yield
    left = [thread for thread in threading.enumerate() if thread.name == FOLLOWER_THREAD and thread not in before]
    if left:
        pytest.fail(f"The test left {len(left)} {FOLLOWER_THREAD} thread(s) running; close each JobActivity.")


_LEAK_GRACE_SECONDS = 1.0
_interrupted: pytest.StashKey[list[str]] = pytest.StashKey()


def _processes() -> dict[int, tuple[int, str]]:
    listing = subprocess.run(
        ["ps", "-axww", "-o", "pid=,ppid=,command="], capture_output=True, text=True, check=True,
        timeout=HANG_SECONDS,
    ).stdout
    processes = {}
    for line in listing.splitlines():
        pid, ppid, *command = line.split(None, 2)
        processes[int(pid)] = (int(ppid), command[0] if command else "")
    return processes


def _running_under(roots: list[Path]) -> dict[int, tuple[Path, str]]:
    """Python processes other than this one and its ancestors whose command line names a path under `roots`.

    That is every Host, relay, venue watchdog, launcher and worker a test started there, whatever its role.
    """
    processes = _processes()
    ancestors, pid = set(), os.getpid()
    while pid in processes and pid not in ancestors:
        ancestors.add(pid)
        pid = processes[pid][0]
    names = {root.name for root in roots}
    found = {}
    for pid, (_, command) in processes.items():
        if pid in ancestors or not Path(command.partition(" ")[0]).name.lower().startswith("python"):
            continue
        for token in command.replace("=", " ").split():
            if not token.startswith("/") or not any(name in token for name in names):
                continue
            path = Path(token).resolve()
            if root := next((root for root in roots if path.is_relative_to(root)), None):
                found[pid] = (root, command)
                break
    return found


def _end_all_under(roots: list[Path]) -> dict[int, tuple[Path, str]]:
    """End every process under `roots` until none is left, and return those that were found running.

    A process still running after a short grace is ended; a watchdog may restart a Host in the meantime,
    so the scan repeats until it finds nothing.
    """
    deadline = time.monotonic() + _LEAK_GRACE_SECONDS
    while (running := _running_under(roots)) and time.monotonic() < deadline:
        time.sleep(_STEP_SECONDS)
    ended = dict(running)
    deadline = time.monotonic() + HANG_SECONDS
    while running:
        if time.monotonic() >= deadline:
            raise AssertionError(f"Processes under {roots} could not be ended: {sorted(running)}")
        for pid in running:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(_STEP_SECONDS)
        running = _running_under(roots)
        ended.update(running)
    return ended


def _owned(root: Path) -> bool:
    """Whether the pytest process that made the numbered temporary root `root` is still running."""
    try:
        pid = int((root / ".lock").read_text())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _listing(ended: dict[int, tuple[Path, str]]) -> list[str]:
    return [f"  {pid}: {command[:300]}" for pid, (_, command) in sorted(ended.items())]


def _report(session: pytest.Session, title: str, lines: list[str]) -> None:
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", title, red=True)
        for line in lines:
            reporter.write_line(line)
    else:
        print("\n".join([title, *lines]), file=sys.stderr)


def pytest_sessionstart(session: pytest.Session) -> None:
    """In the controller, end what an earlier run that is no longer running left under its temporary root.

    A run that is killed never reaches its own sweep; its numbered root keeps a `.lock` naming a dead pid.
    """
    if hasattr(session.config, "workerinput") or session.config.option.basetemp:
        return
    parent = session.config._tmp_path_factory.getbasetemp().resolve().parent
    orphans = [root.resolve() for root in parent.glob("pytest-*")
               if root.is_dir() and not root.is_symlink() and not _owned(root)]
    ended = _end_all_under(orphans) if orphans else {}
    if ended:
        roots = sorted({str(root) for root, _ in ended.values()})
        session.config.stash[_interrupted] = [
            f"Processes left by an earlier test run that is no longer running, under {', '.join(roots)},"
            " were still running and were ended:",
            *_listing(ended),
        ]


def pytest_sessionfinish(session: pytest.Session) -> None:
    """In the controller, end every process still running under the invocation's temporary root and fail the run."""
    if hasattr(session.config, "workerinput"):
        return
    root = session.config._tmp_path_factory.getbasetemp().resolve()
    lines = session.config.stash.get(_interrupted, [])
    if ended := _end_all_under([root]):
        lines = [*lines, "Processes were still running at the end of the session and were ended:", *_listing(ended)]
    if lines:
        _report(session, "leaked processes", lines)
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
