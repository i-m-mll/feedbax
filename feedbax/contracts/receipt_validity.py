"""Receipt-bound observations, assessments and judgements."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.code_validity import RegistryBinding, SourceDifference, SourcePath
from feedbax.contracts.request_refs import ContentRef, Name, ReferenceValue, RequestId, Sha256


class RegistryDifference(ReferenceValue):
    kind: Literal["registry"] = "registry"
    before: RegistryBinding
    after: RegistryBinding | None


class DriverDifference(ReferenceValue):
    kind: Literal["driver"] = "driver"
    name: Name
    before: Name
    after: Name


type CodeDifference = Annotated[
    SourceDifference | RegistryDifference | DriverDifference, Field(discriminator="kind")
]


class EnvironmentDifference(ReferenceValue):
    distribution: Name
    before: Name
    after: Name | None


class FileDifference(ReferenceValue):
    kind: Literal["file"] = "file"
    path: SourcePath
    before: Sha256
    after: Sha256 | None


class InputDifference(ReferenceValue):
    kind: Literal["input"] = "input"
    role: Name
    member: Name
    before: ContentRef
    after: ContentRef | None
    detail: Name | None = None


type CheckDomain = Literal["code", "environment"]


class CheckConditions(ReferenceValue):
    """All observed differences in both domains at the time of a replay."""

    code: tuple[CodeDifference, ...]
    outside_computation: tuple[CodeDifference, ...] = ()
    environment: tuple[EnvironmentDifference, ...]


class CheckOutcome(ReferenceValue):
    outcome: Literal["equal", "different", "unavailable"]
    detail: Name | None = None


class ValidityCheckEvidence(ReferenceValue):
    code: CheckOutcome | None = None
    environment: CheckOutcome | None = None


class ValidityCheckScope(ReferenceValue):
    request_id: RequestId
    receipt: ContentObjectRef
    checks: Annotated[tuple[CheckDomain, ...], Field(min_length=1)]
    conditions: CheckConditions


class ValidityCheckDelivery(ReferenceValue):
    project_root: Path
    store_root: Path
    request_id: RequestId
    receipt: ContentObjectRef
    checks: Annotated[tuple[CheckDomain, ...], Field(min_length=1)]
    producer_launch: dict[str, JsonValue] | None = None
    completion_bindings: tuple[dict[str, JsonValue], ...] = ()


class ValidityAssessment(ReferenceValue):
    basis: Literal["computation"] = "computation"
    receipt: ContentObjectRef
    conditions: CheckConditions
    evidence: ValidityCheckEvidence
    assessed_at: AwareDatetime

    @model_validator(mode="after")
    def _has_checked_difference(self) -> "ValidityAssessment":
        code, environment = self.evidence.code, self.evidence.environment
        if (
            (code is None and environment is None)
            or (code is not None and not self.conditions.code)
            or (environment is not None and not self.conditions.environment)
        ):
            raise ValueError("An assessment must name the differences it checked")
        return self


class ValidityJudgement(ReferenceValue):
    basis: Literal["judgement"] = "judgement"
    receipt: ContentObjectRef
    differences: Annotated[tuple[CodeDifference, ...], Field(min_length=1)]
    who: Name
    when: AwareDatetime
    why: Name


type ValidityRecord = Annotated[
    ValidityAssessment | ValidityJudgement, Field(discriminator="basis")
]


class CodeReason(ReferenceValue):
    kind: Literal["code"] = "code"
    basis: Literal["source", "computation", "unverifiable"]
    differences: tuple[CodeDifference, ...]
    detail: Name | None = None


class EnvironmentReason(ReferenceValue):
    kind: Literal["environment"] = "environment"
    differences: tuple[EnvironmentDifference, ...]
    detail: Name | None = None


type ValidityReason = Annotated[
    CodeReason | EnvironmentReason | FileDifference | InputDifference, Field(discriminator="kind")
]


class EnvironmentNotice(ReferenceValue):
    differences: tuple[EnvironmentDifference, ...]
    check: Literal["pending", "confirmed", "different", "not_checked"]
    detail: Name | None = None


class ReceiptValidity(ReferenceValue):
    receipt: ContentObjectRef
    state: Literal["current", "unchecked", "stale"]
    reasons: tuple[ValidityReason, ...] = ()
    checks: tuple[CheckDomain, ...] = ()
    environment: EnvironmentNotice | None = None


class ValidityAction(ReferenceValue):
    revision: Sha256
    request_id: RequestId
    receipt: ContentObjectRef


class ValidityAcceptance(ValidityAction):
    differences: Annotated[tuple[CodeDifference, ...], Field(min_length=1)]
    who: Name
    why: Name
