"""Opt-in, local pytest timing diagnostics; no public artifact contract."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

import pytest

from scripts.full_suite import PROFILE_SCHEMA_ID, PROFILE_SCHEMA_VERSION


class SuiteProfiler:
    def __init__(self, config):
        self.config = config
        self.started = time.perf_counter()
        self.epoch = time.time()
        self.load_started = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        self.worker_errors = 0
        self.phases = []
        self.fixtures = defaultdict(lambda: {"count": 0, "seconds": 0.0})
        self.workers = set()
        self.worker_ids = set()
        self.collected = 0
        self.collection_errors = 0

    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef, request):
        start = time.perf_counter()
        yield
        key = f"{fixturedef.scope}:{fixturedef.argname}"
        self.fixtures[key]["count"] += 1
        self.fixtures[key]["seconds"] += time.perf_counter() - start

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_collectreport(self, report):
        self.collection_errors += int(report.failed)

    def pytest_runtest_logreport(self, report):
        if hasattr(self.config, "workerinput"):
            return
        worker = getattr(report, "worker_id", "main")
        self.workers.add(worker)
        # Parameter IDs can contain private paths or values. Occupancy needs only the file.
        file = report.nodeid.split("::", 1)[0]
        if not file.startswith("tests/") or Path(file).is_absolute():
            file = "external"
        self.phases.append(
            {
                "file": file,
                "phase": report.when,
                "seconds": report.duration,
                "outcome": report.outcome,
                "worker": worker,
                "start": report.start - self.epoch,
                "stop": report.stop - self.epoch,
            }
        )

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error):
        self.worker_ids.add(node.gateway.id)
        output = getattr(node, "workeroutput", {}).get("feedbax_profile", {})
        self.collected = max(self.collected, output.get("collected", 0))
        self.collection_errors = max(self.collection_errors, output.get("collection_errors", 0))
        self.worker_errors += int(error is not None)
        for key, value in output.get("fixtures", {}).items():
            self.fixtures[key]["count"] += value["count"]
            self.fixtures[key]["seconds"] += value["seconds"]

    def pytest_sessionfinish(self, session, exitstatus):
        common = {
            "fixtures": dict(self.fixtures),
            "collected": self.collected,
            "collection_errors": self.collection_errors,
        }
        if hasattr(self.config, "workerinput"):
            self.config.workeroutput["feedbax_profile"] = common
            return
        files = defaultdict(lambda: {"phases": Counter(), "outcomes": Counter(), "workers": set()})
        workers = defaultdict(Counter)
        spans = defaultdict(list)
        for phase in self.phases:
            row = files[phase["file"]]
            row["phases"][phase["phase"]] += phase["seconds"]
            row["workers"].add(phase["worker"])
            workers[phase["worker"]][phase["phase"]] += phase["seconds"]
            spans[phase["worker"]].extend((phase["start"], phase["stop"]))
            if phase["phase"] == "call" or phase["outcome"] != "passed":
                row["outcomes"][phase["outcome"]] += 1
        payload = {
            "schema_id": PROFILE_SCHEMA_ID,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "complete": True,
            "host_load_started": self.load_started,
            "host_load_finished": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "pytest_elapsed_seconds": time.perf_counter() - self.started,
            "exitstatus": int(exitstatus),
            **common,
            "worker_count": len(self.worker_ids),
            "active_worker_count": len(self.workers),
            "worker_errors": self.worker_errors,
            "workers": {
                name: {
                    "phases": dict(value),
                    "first_phase_seconds": min(spans[name]),
                    "last_phase_seconds": max(spans[name]),
                    "span_seconds": max(spans[name]) - min(spans[name]),
                }
                for name, value in workers.items()
            },
            "scheduler": self.config.getoption("dist", default="no") if self.worker_ids else "no",
            "files": {
                name: {**row, "workers": sorted(row["workers"])}
                for name, row in sorted(files.items())
            },
        }
        Path(os.environ["FEEDBAX_SUITE_PROFILE_RAW"]).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def pytest_configure(config):
    if os.environ.get("FEEDBAX_SUITE_PROFILE_RAW"):
        config.pluginmanager.register(SuiteProfiler(config), "feedbax-suite-profiler")
