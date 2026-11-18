"""Pytest plugin: label child processes with the running test, record its phase timings, and end leaked Hosts.

Every Host root in these tests belongs to one test, so a Host that is still running after its test's teardown
is a leak. It is recorded as `host_leaked` and terminated, so it does not load the tests that follow.
"""

import os
import signal
import time

import pytest

from sitecustomize import STARTED_HOSTS, record

GRACE_SECONDS = 2.0


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    os.environ["FEEDBAX_HOST_PROFILE_TEST"] = item.nodeid
    record("test_start")


def pytest_runtest_logreport(report):
    if os.environ.get("PYTEST_XDIST_WORKER") is None and hasattr(report, "node"):
        return
    record("test_phase", nodeid=report.nodeid, when=report.when, start=report.start, stop=report.stop,
           outcome=report.outcome)


def _alive(process) -> bool:
    if process.returncode is not None:
        return False
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        return False
    # The client reaps its Host on a thread; a zombie has exited.
    try:
        return process.poll() is None
    except ChildProcessError:
        return False


def _end(process) -> None:
    for signum, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.kill(process.pid, signum)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if not _alive(process):
                return
            time.sleep(0.05)


def _reap(test) -> None:
    live = [(process, owner) for process, owner in STARTED_HOSTS
            if (test is None or owner == test) and _alive(process)]
    deadline = time.monotonic() + GRACE_SECONDS
    while live and time.monotonic() < deadline:
        time.sleep(0.05)
        live = [(process, owner) for process, owner in live if _alive(process)]
    for process, owner in live:
        record("host_leaked", host=process.pid, owner=owner)
        _end(process)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    _reap(item.nodeid)


def pytest_unconfigure(config):
    _reap(None)
