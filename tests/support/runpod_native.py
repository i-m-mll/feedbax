"""Execute the actual RunPod entrypoint against an injected, pod-scoped REST service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from tests.support.runpod_session import FakeRunPod


ACCOUNT_KEY = "feedbax-local-test-account-key"
POD_KEY = "feedbax-local-test-pod-key"


class NativeLoopbackRunPod(FakeRunPod):
    def __init__(self, resources, *, workspace: Path, scratch: Path, on_delete):
        super().__init__(resources, on_delete=on_delete)
        self.workspace = workspace
        self.scratch = scratch
        self.secret_value = ""
        self.controller_attached = True
        self.scoped_requests: list[tuple[str, str, int, int]] = []
        self.boot_armed = scratch / "startup-ran"
        self.refuse_delete_once = False

    @property
    def venue(self) -> Path:
        return self.workspace / "feedbax-sessions" / self.pod_id

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.runpod.io":
            assert self.controller_attached, "The detached controller made a RunPod request."
            assert request.headers["Authorization"] == f"Bearer {ACCOUNT_KEY}"
        result = super().handle(request)
        if request.method == "POST" and request.url.path == "/v2/account/secrets":
            self.secret_value = json.loads(request.content)["value"]
        if request.method == "POST" and request.url.path == "/v2/pods":
            self._run_entrypoint()
        return result

    def _run_entrypoint(self) -> None:
        assert self.created_pod["env"].keys() == {"FEEDBAX_BOOT_SSH_KEY"}
        argv = self.created_pod["entrypoint"].copy()
        assert ACCOUNT_KEY not in json.dumps(self.created_pod)
        assert POD_KEY not in json.dumps(self.created_pod)
        ssh_directory = self.scratch / "image-sshd"
        ssh_directory.mkdir()
        start = self.scratch / "start.sh"
        start.write_text(f"""#!{sys.executable}
import os
from pathlib import Path
assert "RUNPOD_API_KEY" not in os.environ
assert Path({str(self.venue / 'cutoff-proof.json')!r}).is_file()
Path({str(self.boot_armed)!r}).touch()
""")
        start.chmod(0o700)
        assert argv[5:7] == ["/etc/ssh", "/start.sh"]
        argv[5:7] = [str(ssh_directory), str(start)]
        result = subprocess.run(
            argv, timeout=12, capture_output=True, check=False,
            env=os.environ | {"RUNPOD_API_KEY": POD_KEY, "RUNPOD_POD_ID": self.pod_id,
                              "FEEDBAX_BOOT_SSH_KEY": self.secret_value},
        )
        assert result.returncode == 0, result.stderr.decode()
        assert self.boot_armed.is_file()

    @contextmanager
    def serve_scoped(self) -> Iterator[str]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def respond(self):
                received_ms = time.time_ns() // 1_000_000
                path = f"/v2/pods/{fake.pod_id}"
                if self.path != path or self.headers.get("Authorization") != f"Bearer {POD_KEY}":
                    status, content = 403, b""
                elif self.command == "DELETE" and fake.refuse_delete_once:
                    fake.refuse_delete_once = False
                    status, content = 503, b""
                else:
                    reply = FakeRunPod.handle(fake, httpx.Request(
                        self.command, "https://api.runpod.io" + self.path,
                    ))
                    status, content = reply.status_code, reply.content
                fake.scoped_requests.append((self.command, self.path, status, received_ms))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            do_GET = do_DELETE = respond

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            running = threading.Thread(target=server.serve_forever, daemon=True)
            running.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}"
            finally:
                server.shutdown()
                running.join(timeout=2)
