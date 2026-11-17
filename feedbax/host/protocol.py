"""Local host requests and wire records."""

from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type JsonObject = dict[str, Json]
type Slot = Literal["compute", "light"]
type StopMode = Literal["checkpoint", "force"]
type State = Literal[
    "queued",
    "blocked",
    "assigned",
    "running",
    "stopping",
    "succeeded",
    "reused",
    "stopped",
    "failed",
    "interrupted",
]
type Disposition = Literal["admitted", "attached", "reused", "rejected"]
type ReusePolicy = Literal["selected", "automatic", "prerequisite"]

LIGHT_ENTRYPOINT = "feedbax.light_host_worker:main"
VALIDITY_CHECK_ENTRYPOINT = "feedbax.validity.check_worker:main"
MANAGED_ENTRYPOINTS = frozenset({
    LIGHT_ENTRYPOINT, "feedbax.training_host_worker:main", "feedbax.training.evaluate_worker:main",
})

VERSION = 1
MAX_RPC_BYTES = 32 * 1024 * 1024
LIVE_STATES = ("queued", "assigned", "running", "stopping")
TERMINAL_STATES = ("succeeded", "reused", "stopped", "failed", "interrupted", "blocked")


def can_resume_attempt(state: str, checkpoint: object, delivery: int) -> bool:
    return state in {"stopped", "failed", "interrupted"} and (
        checkpoint is not None or (state == "stopped" and delivery == 0)
    )


class HostError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SessionAccess:
    session_id: str
    epoch: int
    token: str = field(repr=False)


class SessionStatus(TypedDict):
    session_id: str
    stream_id: str
    operator: str
    epoch: int
    expires_at: float
    closing: bool
    expired: bool
    live_jobs: list[str]
    pending: dict[str, str]
    collections: dict[str, str]
    idle_since_ms: NotRequired[int | None]


class SessionCloseClaim(TypedDict):
    session_id: str
    operator: str
    epoch: int


@dataclass(frozen=True, slots=True)
class Launch:
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    payload: JsonObject = field(default_factory=dict)
    warm_entrypoint: str | None = None
    preload: tuple[str, ...] = ()
    project_root: str | None = None


@dataclass(frozen=True, slots=True)
class Edge:
    job_id: str
    publication: str | None = None
    authenticated_completion: bool = False


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    project: str
    request: str
    launch: Launch
    slot: Slot = "compute"
    continues: str | None = None
    edges: tuple[Edge, ...] = ()
    data: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TerminalReuse:
    receipt: JsonObject
    anchor: str | None


@dataclass(frozen=True, slots=True)
class DeferredReuse:
    attempt: str
    receipt: JsonObject
    policy: ReusePolicy
    quick_seconds: float


@dataclass(frozen=True, slots=True)
class AdmissionItem:
    run_key: str
    job: Job | None = None
    disposition: Literal["reused", "rejected"] | None = None
    reason: JsonObject | str | None = None
    checked_through: int | None = None
    terminal_reuse: TerminalReuse | None = None
    deferred_reuse: DeferredReuse | None = None
    force_execution: bool = False
    light_definition: JsonObject | None = None


class AdmissionResult(TypedDict):
    run_key: str
    disposition: Disposition
    job_id: str | None
    reason: JsonObject | str | None


class Admission(TypedDict):
    id: str
    token: str
    record: JsonObject
    items: list[AdmissionResult]


class Event(TypedDict):
    cursor: int
    job_id: str
    project: str
    request: str
    kind: str
    data: JsonObject
    time: float


class EventPage(TypedDict):
    cursor: int
    events: list[Event]
    head: int
    oldest_cursor: int


class JobsSnapshot(TypedDict):
    cursor: int
    oldest_cursor: int
    jobs: list[JsonObject]
    records: NotRequired[dict[str, JsonObject]]
    terminal_train_receipts: NotRequired[dict[str, JsonObject]]
    checkpoint_differences: NotRequired[dict[str, JsonObject]]
    checkpoint_producers: NotRequired[dict[str, str]]
