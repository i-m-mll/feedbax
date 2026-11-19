"""Filesystem-backed provider for detached production Host session tests."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.host.client import HostClient, StdioAttachment
from feedbax.host.io import TOKEN_ENV, socket_path, write_json
from feedbax.host.process import verify
from feedbax.host.protocol import SessionAccess
from feedbax.host.session import initialize_session
from feedbax.orchestration.session import (
    ProviderMachineObservation, ProviderResource, SessionError, SessionLaunch, SessionRealization, SessionRequest,
)
from tests.support.hang import HANG_SECONDS


ENVIRONMENT = hashlib.sha256(f"{sys.executable}:{sys.version}".encode()).hexdigest()


def evidence(fields, key):
    return hmac.new(key, canonical_json_v2_bytes(fields), "sha256").hexdigest()


def verify_authorization(authorization, key):
    fields = authorization.model_dump(mode="json", exclude={"evidence"})
    if not hmac.compare_digest(authorization.evidence, evidence(fields, key)):
        raise SessionError("Invalid test issuer authentication.")


class LocalVenueProvider:
    target = "local-rehearsal"
    enforces_cutoff = True
    supports_idle_close = False

    def __init__(self, root: Path, *, rate: int = 3_600_000):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.rate = rate

    def hourly_microusd(self, request: SessionRequest) -> int:
        return self.rate

    def directory(self, resource: ProviderResource) -> Path:
        return self.root / resource.id

    def _write_resource(self, resource: ProviderResource) -> Path:
        directory = self.directory(resource)
        directory.mkdir()
        assert write_json(directory / "resource.json", resource.model_dump(mode="json"), exclusive=True)
        return directory

    def inventory(self) -> list[ProviderResource]:
        result = []
        for record in sorted(self.root.glob("*/resource.json")):
            try:
                result.append(ProviderResource.model_validate_json(record.read_bytes()))
            except FileNotFoundError:
                continue
        return result

    def machine_observation(self, resource: ProviderResource) -> ProviderMachineObservation:
        if resource not in self.inventory():
            state = "off"
        else:
            state = "on" if (self.directory(resource) / "realization.json").is_file() else "starting"
        return ProviderMachineObservation(state=state, observed_at_ms=time.time_ns() // 1_000_000)

    def check_opening_deadline(self, launch: SessionLaunch) -> None:
        pass

    def acquire(self, launch: SessionLaunch) -> ProviderResource:
        resource = ProviderResource(id=launch.request.id.hex, intent=launch.intent, kind="venue")
        assert not (self.root / f".used-{resource.id}").exists()
        directory = self._write_resource(resource)
        (self.root / f".used-{resource.id}").touch(exist_ok=False)
        process = subprocess.Popen(
            [sys.executable, "-m", __name__, str(directory), str(launch.ends_at_ms)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        threading.Thread(target=process.wait, daemon=True).start()
        return resource

    def realize(
        self, resource: ProviderResource, launch: SessionLaunch, *,
        adoption_token: str, controller_token: str, opening,
    ) -> SessionRealization:
        directory = self.directory(resource)
        bootstrap = directory / "realization.json"
        if bootstrap.exists():
            return self.realization(resource)
        opening("starting_host")
        initialize_session(
            directory / "host", session_id=str(launch.request.id), operator=launch.authorization.operator,
            expires_at=launch.ends_at_ms / 1000, adoption_token=adoption_token, controller_token=controller_token,
        )
        host_token = secrets.token_hex(24)
        with (directory / "host.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "feedbax.host", "--root", str(directory / "host")],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                env=os.environ | {TOKEN_ENV: host_token},
            )
        threading.Thread(target=process.wait, daemon=True).start()
        write_json(directory / "host-process.json", {"pid": process.pid, "token": host_token})
        deadline = time.monotonic() + HANG_SECONDS
        while not socket_path((directory / "host").resolve()).exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise SessionError(f"Local venue Host did not start. See {directory / 'host.log'}.")
            time.sleep(.02)
        realized = SessionRealization(
            launch=launch, environment=ENVIRONMENT, probe={"python": sys.executable, "placement": {"device": "cpu"}},
        )
        assert write_json(bootstrap, realized.model_dump(mode="json"), exclusive=True)
        return realized

    def realization(self, resource: ProviderResource) -> SessionRealization:
        return SessionRealization.model_validate_json((self.directory(resource) / "realization.json").read_bytes())

    def client(self, resource: ProviderResource, access: SessionAccess | None) -> HostClient:
        root = self.directory(resource) / "host"

        def command(_start_host):
            return [sys.executable, "-m", "feedbax.host", "--root", str(root), "attach", "--no-start"]

        return HostClient(root, attachment=StdioAttachment(command), session=access)

    def claim_opening_close(
        self, resource: ProviderResource | None, launch: SessionLaunch,
    ) -> tuple[ProviderResource, ...] | None:
        if resource is None:
            return None
        assert resource.intent == launch.intent
        if (self.directory(resource) / "realization.json").exists():
            return None
        return (resource,)

    def remove(self, resource: ProviderResource) -> None:
        directory = self.directory(resource)
        try:
            current = ProviderResource.model_validate_json((directory / "resource.json").read_bytes())
        except FileNotFoundError:
            return
        if current != resource:
            raise SessionError("Resource custody changed before deletion.")
        remove_directory(directory)


def remove_directory(directory: Path) -> None:
    identity = directory / "host-process.json"
    if identity.is_file():
        process = json.loads(identity.read_bytes())
        pid, token = process["pid"], process["token"]
        state, error = verify(pid, token)
        if state == "unverifiable":
            raise SessionError(f"Local venue process identity could not be verified: {error}")
        if state == "verified":
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + HANG_SECONDS
            while verify(pid, token)[0] != "gone":
                if time.monotonic() >= deadline:
                    raise TimeoutError("Venue Host did not exit.")
                time.sleep(.02)
        socket_path((directory / "host").resolve()).unlink(missing_ok=True)
    shutil.rmtree(directory, ignore_errors=False)


if __name__ == "__main__":
    venue = Path(sys.argv[1])
    deadline_ms = int(sys.argv[2])
    while venue.exists():
        if time.time_ns() // 1_000_000 >= deadline_ms:
            remove_directory(venue)
            break
        time.sleep(.1)
