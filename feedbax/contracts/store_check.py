"""Verification of immutable Store content and advice on authored project members."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.base import StrictModel


class StoreCheckFinding(StrictModel):
    code: Literal["damaged_object", "check_incomplete"]
    address: str
    path: str
    detail: str


class UnrolledFamilyAdvice(StrictModel):
    code: Literal["unrolled_member_family"]
    address: str
    path: str
    members: tuple[str, ...] = Field(min_length=2)
    count: int = Field(ge=2)
    detail: str


class RestatedDefaultAdvice(StrictModel):
    code: Literal["restated_default"]
    address: str
    path: str
    detail: str


type CheckAdvice = Annotated[UnrolledFamilyAdvice | RestatedDefaultAdvice, Field(discriminator="code")]


class StoreCheckResult(StrictModel):
    store: str
    attempt: str | None
    receipt: ContentObjectRef | None
    verified_objects: int
    verified_bytes: int
    findings: tuple[StoreCheckFinding, ...]
    advice: tuple[CheckAdvice, ...] = Field(default=(), exclude_if=lambda value: not value)
