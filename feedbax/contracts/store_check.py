"""Result of an explicit verification of immutable Store content."""

from typing import Literal

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.base import StrictModel


class StoreCheckFinding(StrictModel):
    code: Literal["damaged_object", "check_incomplete"]
    address: str
    path: str
    detail: str


class StoreCheckResult(StrictModel):
    store: str
    attempt: str | None
    receipt: ContentObjectRef | None
    verified_objects: int
    verified_bytes: int
    findings: tuple[StoreCheckFinding, ...]
