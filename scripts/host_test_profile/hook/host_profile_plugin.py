"""Pytest plugin: label child processes with the running test and record its phase timings."""

import os

import pytest

from sitecustomize import record


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    os.environ["FEEDBAX_HOST_PROFILE_TEST"] = item.nodeid
    record("test_start")


def pytest_runtest_logreport(report):
    if os.environ.get("PYTEST_XDIST_WORKER") is None and hasattr(report, "node"):
        return
    record("test_phase", nodeid=report.nodeid, when=report.when, start=report.start, stop=report.stop,
           outcome=report.outcome)
