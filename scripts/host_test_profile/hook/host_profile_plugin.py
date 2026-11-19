"""Pytest plugin: label child processes with the running test, record its phase timings, and name leaked Hosts.

A Host started by a session-scoped fixture, and any later Host on its root, belongs to the session. Every other
Host belongs to the test that started it, so one still running after that test's teardown is a leak, recorded
as `host_leaked` with the test that owns it. It is left running: the suite's own sweep ends it and fails the run.
"""

import os
import time

import pytest

from sitecustomize import SESSION_SETUP, STARTED_HOSTS, record

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


def _record_leaks(test) -> None:
    live = [(process, owner) for process, owner in STARTED_HOSTS
            if (test is None or owner == test) and _alive(process)]
    deadline = time.monotonic() + GRACE_SECONDS
    while live and time.monotonic() < deadline:
        time.sleep(0.05)
        live = [(process, owner) for process, owner in live if _alive(process)]
    for process, owner in live:
        record("host_leaked", host=process.pid, owner=owner)
        STARTED_HOSTS.remove((process, owner))


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    session = fixturedef.scope == "session"
    SESSION_SETUP[0] += session
    try:
        yield
    finally:
        SESSION_SETUP[0] -= session


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    _record_leaks(item.nodeid)


def pytest_unconfigure(config):
    _record_leaks(None)
