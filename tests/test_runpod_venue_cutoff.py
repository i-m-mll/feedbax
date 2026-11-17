"""The pod credential authorizes only its exact owned RunPod REST v2 target."""

import json
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from feedbax.host import watchdog
from feedbax.orchestration import runpod_venue_cutoff as guard


@pytest.fixture
def runpod(tmp_path, monkeypatch):
    venue = guard._Venue(tmp_path, "feedbax-session-" + "f" * 64, time.time_ns() // 1_000_000 - 1, "owned")
    monkeypatch.setenv("RUNPOD_POD_ID", venue.pod_id)
    monkeypatch.setenv("RUNPOD_API_KEY", "feedbax-local-test-pod-key")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    calls: list[tuple[str, str, str | None]] = []
    responses: list[tuple[int, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def reply(self):
            calls.append((self.command, self.path, self.headers.get("Authorization")))
            status, body = responses.pop(0)
            self.send_response(status)
            if status == 307:
                self.send_header("Location", "/unexpected-redirect")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        do_GET = do_DELETE = reply

        def log_message(self, format: str, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        guard._private_root(venue.root)
        guard._publish(venue.paths[-1], venue.watchdog_policy(origin))
        try:
            yield venue, origin, calls, responses
        finally:
            server.shutdown()
            thread.join()


def test_termination_requires_observed_absence(runpod):
    venue, origin, calls, responses = runpod
    pod = {"id": venue.pod_id, "name": venue.intent}
    responses.extend([(200, pod), (204, {}), (200, pod), (200, pod), (204, {}), (404, {})])
    with pytest.raises(guard._GuardError, match="still reports"):
        guard._terminate(venue, origin)
    guard._terminate(venue, origin)
    assert [method for method, _, _ in calls] == ["GET", "DELETE", "GET", "GET", "DELETE", "GET"]
    assert {(path, header) for _, path, header in calls} == {
        ("/v2/pods/owned", "Bearer feedbax-local-test-pod-key"),
    }


@pytest.mark.parametrize("outcome", ["wrong_id", "wrong_intent", "redirect"])
def test_termination_refuses_mismatched_or_redirected_custody(runpod, outcome):
    venue, origin, calls, responses = runpod
    responses.append({
        "wrong_id": (200, {"id": "foreign", "name": venue.intent}),
        "wrong_intent": (200, {"id": venue.pod_id, "name": "foreign"}),
        "redirect": (307, {}),
    }[outcome])
    with pytest.raises(guard._GuardError):
        guard._terminate(venue, origin)
    assert [(method, path) for method, path, _ in calls] == [("GET", "/v2/pods/owned")]


def test_loopback_adapter_never_receives_a_nonsynthetic_key(runpod, monkeypatch):
    venue, origin, calls, _ = runpod
    monkeypatch.setenv("RUNPOD_API_KEY", "opaque-key")
    with pytest.raises(guard._GuardError, match="synthetic"):
        guard._terminate(venue, origin)
    assert calls == []


def test_termination_refuses_a_call_before_the_armed_cutoff(runpod):
    venue, origin, calls, _ = runpod
    venue = replace(venue, workspace=venue.workspace / "later", ends_at_ms=venue.ends_at_ms + 60_000)
    guard._private_root(venue.root)
    guard._publish(venue.paths[-1], venue.watchdog_policy(origin))
    with pytest.raises(guard._GuardError, match="has not been reached"):
        guard._terminate(venue, origin)
    assert calls == []


def test_boot_watchdog_does_not_pass_the_pod_key_to_host(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-key")
    monkeypatch.setenv("RUNPOD_POD_ID", "owned")
    root = tmp_path / "host"
    root.mkdir()
    (root / "host.lock").touch()
    ready = tmp_path / "ready"
    ready.touch(mode=0o600)
    child = object()

    def launch(_argv, *, env, **_kwargs):
        assert "RUNPOD_API_KEY" not in env
        assert env["RUNPOD_POD_ID"] == "owned"
        return child

    monkeypatch.setattr(watchdog.subprocess, "Popen", launch)
    config = watchdog._Config(1, root, Path("/python"), ready, ("/terminate",))
    assert watchdog._start_host(config) is child
