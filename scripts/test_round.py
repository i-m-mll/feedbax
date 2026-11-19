"""Run the round tier and check that removing what the Host tests share changes only time.

Runs `pytest -m round`, then the fast tier's Host tests four times: as usual, with
`FEEDBAX_TEST_PRIVATE_HOSTS=1`, with `FEEDBAX_DISABLE_JAX_COMPILATION_CACHE=1` and with
`FEEDBAX_HOST_FRESH_WORKERS=1`. It fails
when the round tier fails or when any test id's outcome in the last three runs differs from the
first. Arguments (paths and selection options, `tests` by default) choose the tests of both.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree

import pytest

HOST_FIXTURES = {"host", "private_host", "reusing_host"}
REMOVALS = {
    "private Hosts": {"FEEDBAX_TEST_PRIVATE_HOSTS": "1"},
    "no compile cache": {"FEEDBAX_DISABLE_JAX_COMPILATION_CACHE": "1"},
    "fresh workers": {"FEEDBAX_HOST_FRESH_WORKERS": "1"},
}


class _HostTests:
    """Collects the ids of the selected tests that can reach a Host fixture of `tests.support.host`.

    A test can reach one when the fixture is in its fixture closure, or in its module's namespace, from which
    a test can take it through `request.getfixturevalue` without the closure showing it.
    """

    def __init__(self) -> None:
        self.ids: list[str] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        support = sys.modules["tests.support.host"]
        fixtures = [getattr(support, name) for name in HOST_FIXTURES]
        for item in session.items:
            closure = getattr(item, "_fixtureinfo").name2fixturedefs.values()
            if any(
                definition.func.__module__ == support.__name__ and definition.func.__name__ in HOST_FIXTURES
                for chain in closure for definition in chain
            ) or any(value is fixture for value in vars(item.module).values() for fixture in fixtures):
                self.ids.append(item.nodeid)


def host_test_ids(arguments: list[str]) -> list[str]:
    collector = _HostTests()
    status = pytest.main(["--collect-only", "-qq", "-n0", *arguments], plugins=[collector])
    if status != pytest.ExitCode.OK:
        sys.exit(f"Collection of the Host tests failed with status {status}.")
    return collector.ids


def outcomes(report: Path) -> dict[str, str]:
    """Each test case's outcome in a JUnit report, keyed by its class name and name."""
    result = {}
    for case in ElementTree.parse(report).iter("testcase"):
        kinds = {child.tag for child in case}
        outcome = "failed" if kinds & {"failure", "error"} else "skipped" if "skipped" in kinds else "passed"
        key = f"{case.get('classname')}::{case.get('name')}"
        result[key] = "failed" if result.get(key) == "failed" else outcome
    return result


def run(label: str, arguments: list[str], env: dict[str, str], report: Path) -> int:
    print(f"\n== {label}", flush=True)
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}", *arguments]
    return subprocess.run(command, env=os.environ | env).returncode


def main(arguments: list[str]) -> int:
    arguments = arguments or ["tests"]
    ids = host_test_ids(arguments)
    if not ids:
        sys.exit("No Host tests were selected.")
    with tempfile.TemporaryDirectory(prefix="test-round-") as directory:
        reports = Path(directory)
        round_status = run("round tier", ["-m", "round", "-n", "4", *arguments], {}, reports / "round.xml")
        run("Host tests, shared", ids, {}, reports / "shared.xml")
        shared = outcomes(reports / "shared.xml")
        differences = []
        for label, env in REMOVALS.items():
            report = reports / f"{label}.xml"
            run(f"Host tests, {label}", ids, env, report)
            removed = outcomes(report)
            differences += [
                f"  {key}: {shared.get(key, 'absent')} shared, {removed.get(key, 'absent')} with {label}"
                for key in sorted(shared.keys() | removed.keys()) if shared.get(key) != removed.get(key)
            ]
    print(f"\n{len(ids)} Host tests compared.")
    if differences:
        print("Outcomes that changed when what is shared was removed:", *differences, sep="\n")
    if round_status != 0:
        print(f"The round tier failed with status {round_status}.")
    return 1 if differences or round_status != 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
