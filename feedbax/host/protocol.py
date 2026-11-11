"""Local host requests and wire records."""

from dataclasses import dataclass, field
from typing import Literal, TypedDict

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
    "stopped",
    "failed",
    "interrupted",
]
type Disposition = Literal["admitted", "attached", "reused", "rejected"]

VERSION = 1
MAX_RPC_BYTES = 32 * 1024 * 1024
LIVE_STATES = ("queued", "assigned", "running", "stopping")
TERMINAL_STATES = ("succeeded", "stopped", "failed", "interrupted", "blocked")


class HostError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class Launch:
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    payload: JsonObject = field(default_factory=dict)
    warm_entrypoint: str | None = None
    preload: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Edge:
    job_id: str
    publication: str | None = None


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
class AdmissionItem:
    run_key: str
    job: Job | None = None
    disposition: Literal["reused", "rejected"] | None = None
    reason: JsonObject | str | None = None
    checked_through: int | None = None


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
