"""Challenge the installed native authenticator and issue a durable Jobs authorization."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.jobs import StudioMachineQuote
from feedbax.desktop.control import _read_private_json, write_private_json
from feedbax.orchestration.session import SessionAuthorization, SessionError, SessionRequest

_METHOD = "macos-touch-id"
_EVIDENCE_VERSION = "feedbax.studio.machine.v1"


def _native_proof(launcher: Path, challenge: str, reason: str) -> str:
    try:
        result = subprocess.run(
            [launcher, "service", "attest-machine"],
            input=json.dumps({"challenge": challenge, "reason": reason}),
            text=True, capture_output=True, check=True, timeout=50,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        evidence = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise SessionError("Touch ID was cancelled or is unavailable.") from error
    if not isinstance(evidence, dict) or set(evidence) != {"challenge"} or not isinstance(
        proof := evidence["challenge"], str
    ):
        raise SessionError("Studio received invalid Touch ID evidence.")
    return proof


def _usd(microusd: int) -> str:
    dollars, fraction = divmod(microusd, 1_000_000)
    cents = f"{fraction:06d}".rstrip("0").ljust(2, "0")
    return f"US${dollars}.{cents}"


class DesktopApproval:
    def __init__(
        self, key_path: Path, launcher: Path, *,
        native_proof: Callable[[Path, str, str], str] = _native_proof,
    ) -> None:
        try:
            record = _read_private_json(key_path)
        except FileNotFoundError:
            record = {"key": secrets.token_hex(32)}
            write_private_json(key_path, record)
        encoded = record.get("key")
        if not isinstance(encoded, str) or len(encoded) != 64:
            raise ValueError("Invalid Studio machine authorization key.")
        self._key = bytes.fromhex(encoded)
        if len(self._key) != 32:
            raise ValueError("Invalid Studio machine authorization key.")
        self._operator = f"macos-user:{os.getuid()}"
        self._launcher = launcher
        self._native_proof = native_proof
        self._prompt = Lock()

    def _signature(self, fields: dict[str, object], challenge: str) -> str:
        message = canonical_json_v2_bytes({
            "schema": _EVIDENCE_VERSION, "challenge": challenge, "authorization": fields,
        })
        return hmac.digest(self._key, message, "sha256").hex()

    def __call__(self, quote: StudioMachineQuote, request: SessionRequest) -> SessionAuthorization:
        fields = {
            "request_digest": request.digest, "operator": self._operator, "method": _METHOD,
            "expires_at_ms": quote.expires_at_ms, "ceiling_microusd": quote.ceiling_microusd,
        }
        challenge = hashlib.sha256(canonical_json_v2_bytes({
            "nonce": secrets.token_hex(32), "authorization": fields, "quote": quote.model_dump(mode="json"),
        })).hexdigest()
        count = len(quote.rows)
        reason = f"Approve {count} {'run' if count == 1 else 'runs'} up to {_usd(quote.ceiling_microusd)}"
        if not self._prompt.acquire(blocking=False):
            raise SessionError("Another machine approval is waiting for Touch ID.")
        try:
            proof = self._native_proof(self._launcher, challenge, reason)
        finally:
            self._prompt.release()
        if not proof.isascii() or not hmac.compare_digest(proof, challenge):
            raise SessionError("Touch ID evidence belongs to a different machine quote.")
        return SessionAuthorization.model_validate(fields | {
            "evidence": f"{_EVIDENCE_VERSION}:{challenge}:{self._signature(fields, challenge)}",
        })

    def verify(self, authorization: SessionAuthorization) -> None:
        version, _, evidence = authorization.evidence.partition(":")
        challenge, _, signature = evidence.partition(":")
        if (
            version != _EVIDENCE_VERSION
            or authorization.operator != self._operator
            or authorization.method != _METHOD
            or len(challenge) != 64 or not challenge.isascii()
            or len(signature) != 64 or not signature.isascii()
            or not hmac.compare_digest(signature, self._signature(
                authorization.model_dump(exclude={"evidence"}), challenge,
            ))
        ):
            raise SessionError("Machine authorization has no matching Studio Touch ID evidence.")
