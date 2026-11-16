"""Durable controller ownership and collection evidence for a venue Host."""

import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import stat
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from feedbax.host.io import private_directory, sync_directory, write_json
from feedbax.host.protocol import HostError, SessionCloseClaim


_FENCE_FILE = "controller-fence.json"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _token_hash(token: str) -> str:
    if not isinstance(token, str) or not 32 <= len(token) <= 256 or not token.isascii():
        raise ValueError("Session capabilities must be ASCII strings of 32 to 256 characters.")
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _matches(token: object, digest: str) -> bool:
    return isinstance(token, str) and 32 <= len(token) <= 256 and token.isascii() and hmac.compare_digest(
        hashlib.sha256(token.encode("ascii")).hexdigest(), digest,
    )


@dataclass(frozen=True, slots=True)
class _FenceState:
    session_id: str
    operator: str
    expires_at: float
    adoption_sha256: str
    controller_sha256: str
    epoch: int
    closing: bool
    collections: dict[str, str]

    def __post_init__(self) -> None:
        if (
            not _IDENTIFIER.fullmatch(self.session_id) or not self.operator.strip()
            or type(self.expires_at) not in {int, float} or not math.isfinite(self.expires_at)
            or self.expires_at <= 0
            or not _DIGEST.fullmatch(self.adoption_sha256)
            or not _DIGEST.fullmatch(self.controller_sha256)
            or self.adoption_sha256 == self.controller_sha256
            or type(self.epoch) is not int or self.epoch < 0
            or type(self.closing) is not bool or not isinstance(self.collections, dict)
            or any(not _IDENTIFIER.fullmatch(job) or not _DIGEST.fullmatch(digest)
                   for job, digest in self.collections.items())
        ):
            raise ValueError("Invalid saved session controller fence.")


def initialize_session(
    root: Path, *, session_id: str, operator: str, expires_at: float,
    adoption_token: str, controller_token: str,
) -> None:
    """Seed epoch zero before a trusted venue starts its Host."""
    state = _FenceState(
        session_id=session_id, operator=operator, expires_at=expires_at,
        adoption_sha256=_token_hash(adoption_token), controller_sha256=_token_hash(controller_token),
        epoch=0, closing=False, collections={},
    )
    if state.expires_at <= time.time():
        raise HostError("session_expired", "Cannot initialize an expired session.")
    root = private_directory(root.resolve())
    with (root / "host.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = SessionFence.load(root)
        if existing is not None:
            if existing.state != state:
                raise HostError("session_initialized", "This venue Host already has a controller fence.")
            return
        if (root / "host.sqlite").exists():
            raise HostError("session_initialized", "A session requires a fresh Host root.")
        if not write_json(root / _FENCE_FILE, asdict(state), exclusive=True):
            raise HostError("session_initialized", "This venue Host already has a controller fence.")


class SessionFence:
    """Mutated by the single Host process while it holds the venue's Host lock."""

    def __init__(self, path: Path, state: _FenceState):
        self.path = path
        self._state: _FenceState | None = state

    @classmethod
    def load(cls, root: Path) -> "SessionFence | None":
        path = root / _FENCE_FILE
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise PermissionError("The venue controller fence must be owned by the Host user.")
            state = _FenceState(**json.load(stream))
        return cls(path, state)

    @property
    def state(self) -> _FenceState:
        if self._state is None:
            raise HostError("session_unavailable", "The Host cannot verify its persisted controller fence.")
        return self._state

    @property
    def expired(self) -> bool:
        return self.state.expires_at <= time.time()

    @property
    def accepts_work(self) -> bool:
        state = self._state
        return state is not None and not state.closing and state.expires_at > time.time()

    def authenticate(self, access: object) -> None:
        state = self.state
        if (
            not isinstance(access, dict) or access.keys() != {"session_id", "epoch", "token"}
            or access["session_id"] != state.session_id
            or type(access["epoch"]) is not int or access["epoch"] != state.epoch
            or not _matches(access["token"], state.controller_sha256)
        ):
            raise HostError("session_fenced", "This request does not hold the current session controller fence.")

    def require_open(self) -> None:
        if self.expired:
            raise HostError("session_expired", "The venue session has expired.")
        if self.state.closing:
            raise HostError("session_closing", "The venue session has a durable close claim.")

    def epoch(self, session_id: str, operator: str, adoption_token: str) -> int:
        state = self.state
        if (session_id, operator) != (state.session_id, state.operator) or not _matches(
            adoption_token, state.adoption_sha256,
        ):
            raise HostError("session_fenced", "Session adoption requires the verified driver capability.")
        return state.epoch

    def adopt(
        self, session_id: str, operator: str, expected_epoch: int, adoption_token: str, controller_token: str,
    ) -> int:
        epoch = self.epoch(session_id, operator, adoption_token)
        self.require_open()
        if type(expected_epoch) is not int or expected_epoch != epoch:
            raise HostError("session_fenced", "The venue controller epoch changed before adoption.")
        digest = _token_hash(controller_token)
        if digest in {self.state.adoption_sha256, self.state.controller_sha256}:
            raise ValueError("Session adoption requires a fresh controller capability.")
        self._persist(replace(self.state, epoch=epoch + 1, controller_sha256=digest))
        return epoch + 1

    def acknowledge_collection(self, job_id: str, digest: str) -> None:
        recorded = self.state.collections.get(job_id)
        if recorded == digest:
            return
        if recorded is not None:
            raise HostError("collection_conflict", "This job has different acknowledged collection evidence.")
        self._persist(replace(self.state, collections=self.state.collections | {job_id: digest}))

    def claim_close(self) -> SessionCloseClaim:
        if not self.state.closing:
            self._persist(replace(self.state, closing=True))
        return self._close_claim()

    def recover_close(self, session_id: str, operator: str, adoption_token: str) -> SessionCloseClaim:
        self.epoch(session_id, operator, adoption_token)
        if not self.state.closing:
            raise HostError("session_not_closing", "The venue has no durable close claim.")
        return self._close_claim()

    def _close_claim(self) -> SessionCloseClaim:
        return {"session_id": self.state.session_id, "operator": self.state.operator, "epoch": self.state.epoch}

    def _persist(self, state: _FenceState) -> None:
        self._state = None
        try:
            write_json(self.path, asdict(state))
        except OSError:
            # A failed directory sync can follow rename. Never keep granting access
            # until the state on disk is durable again.
            sync_directory(self.path.parent)
            current = self.load(self.path.parent)
            if current is None:
                raise HostError("session_unavailable", "The venue controller fence disappeared.")
            self._state = current.state
            raise
        self._state = state
