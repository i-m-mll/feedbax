"""Named execution profiles and the placement of a particular attempt."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.studio_refinements import validate_cross_field_refinements


LOCAL_TARGET_ID = "local"
TargetId = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
TargetRunKind = Literal["train", "evaluate", "analyze", "render", "report"]
SESSION_RUN_KINDS: frozenset[TargetRunKind] = frozenset({"train", "evaluate"})
LOCAL_RUN_KINDS: tuple[TargetRunKind, ...] = ("train", "evaluate", "analyze", "render", "report")


class TargetModel(StrictModel):
    model_config = ConfigDict(frozen=True)


class KeepOpen(TargetModel):
    kind: Literal["keep-open"] = "keep-open"


class CloseWhenIdle(TargetModel):
    kind: Literal["close-when-idle"] = "close-when-idle"
    seconds: int = Field(gt=0, strict=True)


SessionLifecycle = Annotated[KeepOpen | CloseWhenIdle, Field(discriminator="kind")]


class LocalTargetProfile(TargetModel):
    kind: Literal["local"]
    id: Literal["local"]
    label: Literal["This Mac"]


class SessionTargetProfile(TargetModel):
    kind: Literal["session"]
    id: TargetId
    label: str = Field(min_length=1)
    driver: TargetId
    resources: dict[str, JsonValue]
    lifecycle: SessionLifecycle
    maximum_microusd: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _remote_id(self) -> "SessionTargetProfile":
        validate_cross_field_refinements("SessionTargetProfile", self)
        return self


TargetProfile = Annotated[LocalTargetProfile | SessionTargetProfile, Field(discriminator="kind")]
LOCAL_PROFILE = LocalTargetProfile(kind="local", id=LOCAL_TARGET_ID, label="This Mac")


class LocalTarget(TargetModel):
    kind: Literal["local"]
    profile: Literal["local"]


class TargetRealization(TargetModel):
    environment: str = Field(pattern=r"^[0-9a-f]{64}$")
    placement: dict[str, JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)


class SessionTarget(TargetModel):
    kind: Literal["session"]
    profile: TargetId
    driver: TargetId
    session_id: UUID
    realization: TargetRealization | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _remote_id(self) -> "SessionTarget":
        validate_cross_field_refinements("SessionTarget", self)
        return self


AttemptTarget = Annotated[LocalTarget | SessionTarget, Field(discriminator="kind")]
LOCAL_TARGET = LocalTarget(kind="local", profile=LOCAL_TARGET_ID)
_ATTEMPT_TARGET = TypeAdapter(AttemptTarget)


def attempt_target(value: object) -> AttemptTarget:
    return LOCAL_TARGET if value is None else _ATTEMPT_TARGET.validate_python(value)


class ReadyTarget(TargetModel):
    state: Literal["ready"]


class UnavailableTarget(TargetModel):
    state: Literal["unavailable"]
    reason: str


TargetAvailability = Annotated[ReadyTarget | UnavailableTarget, Field(discriminator="state")]


class TargetOption(TargetModel):
    profile: TargetProfile
    run_kinds: tuple[TargetRunKind, ...]
    availability: TargetAvailability


class ExecutionTargets(TargetModel):
    default: Literal["local"]
    targets: tuple[TargetOption, ...]
