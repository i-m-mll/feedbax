"""Arm the provider-native RunPod watchdog at the pod's entrypoint."""

from __future__ import annotations

import base64
import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from feedbax.host import watchdog
from feedbax.host.protocol import JsonObject
from feedbax.orchestration import runpod_venue_cutoff as guard
from feedbax.orchestration.runpod_session_provider import RunPodCutoffArm
from feedbax.orchestration.session import ProviderResource, SessionError, SessionLaunch


_LOADER = """
import base64
import sys
import zlib
guard, watchdog = (zlib.decompress(base64.b64decode(x)) for x in sys.argv[1:3])
sys.argv = ["feedbax-runpod-cutoff", *sys.argv[3:]]
globals().update(BUNDLED_GUARD=guard, BUNDLED_WATCHDOG=watchdog)
exec(compile(guard, "<feedbax-runpod-cutoff>", "exec"), globals())
"""


@dataclass(frozen=True)
class NativeRunPodCutoff:
    """Use the pod credential supplied by RunPod to the pinned image at startup."""

    workspace: str
    bootstrap_python: str
    _test_origin: str | None = None

    def __post_init__(self) -> None:
        for location in (self.workspace, self.bootstrap_python):
            path = PurePosixPath(location)
            if not path.is_absolute() or ".." in path.parts or "\0" in location:
                raise ValueError("RunPod cutoff requires absolute venue paths.")
        if self._test_origin is not None and not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{1,5}", self._test_origin):
            raise ValueError("The injected RunPod cutoff endpoint must be on loopback.")

    def _args(self, launch: SessionLaunch) -> tuple[str, ...]:
        return (
            "--workspace", self.workspace, "--intent", launch.intent,
            "--ends-at-ms", str(launch.ends_at_ms),
        )

    def _origin(self) -> tuple[str, ...]:
        return ("--test-origin", self._test_origin) if self._test_origin is not None else ()

    def arm(self, launch: SessionLaunch) -> RunPodCutoffArm:
        files = guard.__file__, watchdog.__file__
        if not all(files):
            raise SessionError("RunPod cutoff requires the installed guard and watchdog sources.")
        sources = tuple(
            base64.b64encode(zlib.compress(Path(file).read_bytes())).decode("ascii")
            for file in files if file is not None
        )
        return RunPodCutoffArm(
            create_terms={},
            boot_argv=(self.bootstrap_python, "-I", "-c", _LOADER, *sources, "arm",
                       *self._args(launch), *self._origin()),
        )

    def _argv(self, action: str, resource: ProviderResource, launch: SessionLaunch) -> tuple[str, ...]:
        if (
            resource.kind != "venue" or resource.intent != launch.intent
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resource.id)
        ):
            raise SessionError("Native RunPod cutoff requires the exact owned pod.")
        script = PurePosixPath(self.workspace) / "feedbax-sessions" / resource.id / "cutoff_guard.py"
        return (
            self.bootstrap_python, "-I", str(script), action, *self._args(launch), "--pod-id", resource.id,
        )

    def terminate_argv(self, resource: ProviderResource, launch: SessionLaunch) -> tuple[str, ...]:
        return (*self._argv("terminate", resource, launch), *self._origin())

    def proof_argv(self, resource: ProviderResource, launch: SessionLaunch) -> list[str]:
        return list(self._argv("check", resource, launch))

    @staticmethod
    def read_proof(raw: bytes, resource: ProviderResource, launch: SessionLaunch) -> JsonObject:
        expected: JsonObject = {
            "version": 1, "pod_id": resource.id, "intent": launch.intent, "ends_at_ms": launch.ends_at_ms,
        }
        if json.loads(raw) != expected:
            raise SessionError("RunPod venue did not prove the exact pre-SSH cutoff.")
        return expected
