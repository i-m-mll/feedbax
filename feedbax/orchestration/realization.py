"""Inert backend plans and observed attempts for scientific invocations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from feedbax.contracts.base import (
    StrictModel,
    canonical_json_bytes,
)
from feedbax.execution.records import Invocation

if TYPE_CHECKING:
    from feedbax.orchestration.drivers.capabilities import (
        DriverCapabilityEnvelope,
        RealizedDriverCapabilities,
    )


BACKEND_PLAN_SCHEMA_ID = "feedbax.orchestration.backend_plan"
BACKEND_PLAN_SCHEMA_VERSION = "feedbax.orchestration.backend_plan.v1"
ATTEMPT_SCHEMA_ID = "feedbax.manifest.attempt"
ATTEMPT_SCHEMA_VERSION = "feedbax.manifest.attempt.v1"

_OBSERVED_REALIZATION_KEYS = frozenset(
    {
        "attempt_id",
        "instance_id",
        "pod_id",
        "provider_resource_handle",
        "reservation_id",
        "resource_handle",
        "worker_identity",
    }
)
_SECRET_VALUE_KEYS = frozenset(
    {"api_key", "credential", "credentials", "password", "secret", "token"}
)


class FrozenRecord(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MachineShape(FrozenRecord):
    accelerator_type: str | None = None
    accelerator_count: int = Field(default=0, ge=0)
    cpu_count: int | None = Field(default=None, ge=1)
    memory_gib: float | None = Field(default=None, gt=0)
    regions: tuple[str, ...] = ()


class ExpectedCost(FrozenRecord):
    currency: Literal["USD"] = "USD"
    maximum: float = Field(ge=0)
    basis: str = Field(min_length=1)


class BackendRealizationRequest(FrozenRecord):
    """Exact inert facts supplied to a backend adapter."""

    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    capability_variant: str = Field(min_length=1)
    code_bundle_id: str = Field(min_length=1)
    environment_bundle_id: str = Field(min_length=1)
    command: tuple[str, ...] = Field(min_length=1)
    machine: MachineShape = Field(default_factory=MachineShape)
    network_requirements: tuple[str, ...] = ()
    secret_names: tuple[str, ...] = ()
    timeout_seconds: float = Field(gt=0)
    retry_classification: Literal["never", "same-plan"] = "never"
    expected_cost: ExpectedCost | None = None
    billable_confirmation_class: str | None = None
    external_effect_key: str = Field(min_length=1)
    configuration: dict[str, Any] = Field(default_factory=dict)

    @field_validator("secret_names")
    @classmethod
    def _reject_secret_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("secret names must be unique and canonically ordered")
        return value

    @field_validator("configuration")
    @classmethod
    def _reject_observed_state_and_secret_material(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        _validate_inert_configuration(value, field_ref="backend realization configuration")
        return value


class BackendPlan(FrozenRecord):
    """Exact provider-specific realization of one immutable invocation."""

    schema_id: Literal["feedbax.orchestration.backend_plan"] = BACKEND_PLAN_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.backend_plan.v1"] = BACKEND_PLAN_SCHEMA_VERSION
    backend_plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_id: str = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    driver_capability_schema_id: str = Field(min_length=1)
    driver_capability_schema_version: str = Field(min_length=1)
    driver_capability_variant: str = Field(min_length=1)
    code_bundle_id: str = Field(min_length=1)
    environment_bundle_id: str = Field(min_length=1)
    command: tuple[str, ...] = Field(min_length=1)
    machine: MachineShape
    network_requirements: tuple[str, ...] = ()
    secret_names: tuple[str, ...] = ()
    timeout_seconds: float = Field(gt=0)
    retry_classification: Literal["never", "same-plan"]
    expected_cost: ExpectedCost | None = None
    billable_confirmation_class: str | None = None
    external_effect_key: str = Field(min_length=1)
    configuration: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_identity(self) -> "BackendPlan":
        _validate_inert_configuration(
            self.configuration,
            field_ref="backend plan configuration",
        )
        if self.backend_plan_id != _backend_plan_identity(self):
            raise ValueError("backend_plan_id does not match canonical realization content")
        return self


class Attempt(FrozenRecord):
    """One observed realization of a backend plan; never semantic workflow state."""

    schema_id: Literal["feedbax.manifest.attempt"] = ATTEMPT_SCHEMA_ID
    schema_version: Literal["feedbax.manifest.attempt.v1"] = ATTEMPT_SCHEMA_VERSION
    attempt_id: str = Field(min_length=1)
    invocation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    reservation_id: str | None = None
    provider_resource_handle: str | None = None
    worker_identity: str | None = None
    status: Literal["pending", "running", "succeeded", "failed", "cancelled", "unknown"]
    started_at: datetime | None = None
    terminal_at: datetime | None = None
    exit_classification: str | None = None
    observations: tuple[dict[str, Any], ...] = ()
    event_refs: tuple[str, ...] = ()
    publication_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_terminal_shape(self) -> "Attempt":
        terminal = self.status in {"succeeded", "failed", "cancelled"}
        if terminal != (self.terminal_at is not None):
            raise ValueError("terminal attempts require terminal_at and nonterminal attempts omit it")
        if terminal != (self.exit_classification is not None):
            raise ValueError(
                "terminal attempts require exit_classification and nonterminal attempts omit it"
            )
        if self.reservation_id is not None and not self.reservation_id.strip():
            raise ValueError("reservation_id must be non-empty when supplied")
        return self


class UnsupportedBackendRecordVersionError(ValueError):
    pass


class OrchestrationBackend:
    """BackendProtocol adapter over one existing orchestration capability envelope."""

    def __init__(
        self,
        *,
        backend_id: str,
        supported_scientific_capabilities: frozenset[str],
        driver_capabilities: "DriverCapabilityEnvelope",
    ) -> None:
        self.backend_id = backend_id
        self.supported_scientific_capabilities = supported_scientific_capabilities
        self.driver_capabilities = driver_capabilities

    def realize(self, capability: str, request: object) -> object:
        """Realize one exact invocation without constructing a driver or causing an effect."""

        if capability not in self.supported_scientific_capabilities:
            supported = sorted(self.supported_scientific_capabilities)
            raise ValueError(
                f"backend {self.backend_id!r} does not support capability {capability!r}; "
                f"supported capabilities are {supported!r}"
            )
        if not isinstance(request, tuple) or len(request) != 2:
            raise TypeError("backend realization request must be (Invocation, BackendRealizationRequest)")
        invocation, realization = request
        if not isinstance(invocation, Invocation) or not isinstance(
            realization, BackendRealizationRequest
        ):
            raise TypeError("backend realization requires typed invocation and realization records")
        missing = set(invocation.capabilities) - self.supported_scientific_capabilities
        if missing:
            raise ValueError(
                f"backend {self.backend_id!r} cannot realize invocation capabilities "
                f"{sorted(missing)!r}"
            )
        realized = self.driver_capabilities.realize(realization.capability_variant)
        return _build_backend_plan(invocation, self.backend_id, realized, realization)


def backend_plan_from_document(document: Any) -> BackendPlan:
    return _load_record(
        document,
        model=BackendPlan,
        schema_id=BACKEND_PLAN_SCHEMA_ID,
        schema_version=BACKEND_PLAN_SCHEMA_VERSION,
        label="backend plan",
    )


def attempt_from_document(document: Any) -> Attempt:
    return _load_record(
        document,
        model=Attempt,
        schema_id=ATTEMPT_SCHEMA_ID,
        schema_version=ATTEMPT_SCHEMA_VERSION,
        label="attempt",
    )


def _build_backend_plan(
    invocation: Invocation,
    backend_id: str,
    realized: "RealizedDriverCapabilities",
    request: BackendRealizationRequest,
) -> BackendPlan:
    billable = realized.facts.spend.value != "none"
    if billable and (
        request.expected_cost is None or request.billable_confirmation_class is None
    ):
        raise ValueError(
            "paid-resource-capable backend plans require expected cost and a billable "
            "confirmation class"
        )
    if not billable and (
        request.expected_cost is not None or request.billable_confirmation_class is not None
    ):
        raise ValueError("non-billable backend plans cannot carry billable confirmation fields")
    if realized.facts.authorization.value == "cloud-and-spend-required" and (
        request.billable_confirmation_class != "authenticated-effect-reservation"
    ):
        raise ValueError(
            "cloud-and-spend backend plans require authenticated-effect-reservation confirmation"
        )
    content = {
        "invocation_id": invocation.invocation_id,
        "backend_id": backend_id,
        "adapter_id": request.adapter_id,
        "adapter_version": request.adapter_version,
        "driver_capability_schema_id": realized.schema_id,
        "driver_capability_schema_version": realized.schema_version,
        "driver_capability_variant": realized.variant_id,
        "code_bundle_id": request.code_bundle_id,
        "environment_bundle_id": request.environment_bundle_id,
        "command": request.command,
        "machine": request.machine,
        "network_requirements": request.network_requirements,
        "secret_names": request.secret_names,
        "timeout_seconds": request.timeout_seconds,
        "retry_classification": request.retry_classification,
        "expected_cost": request.expected_cost,
        "billable_confirmation_class": request.billable_confirmation_class,
        "external_effect_key": request.external_effect_key,
        "configuration": request.configuration,
    }
    serializable = {
        key: value.model_dump(mode="json") if isinstance(value, StrictModel) else value
        for key, value in content.items()
    }
    return BackendPlan(backend_plan_id=_sha256(serializable), **content)


def _backend_plan_identity(plan: BackendPlan) -> str:
    return _sha256(
        plan.model_dump(
            mode="json",
            exclude={"schema_id", "schema_version", "backend_plan_id"},
        )
    )


def _load_record(
    document: Any,
    *,
    model: type[BackendPlan] | type[Attempt],
    schema_id: str,
    schema_version: str,
    label: str,
) -> BackendPlan | Attempt:
    if not isinstance(document, Mapping):
        raise UnsupportedBackendRecordVersionError(f"{label} document must be a mapping")
    if document.get("schema_id") != schema_id:
        raise UnsupportedBackendRecordVersionError(
            f"unsupported {label} schema_id {document.get('schema_id')!r}; expected {schema_id!r}"
        )
    if document.get("schema_version") != schema_version:
        raise UnsupportedBackendRecordVersionError(
            f"unsupported {label} schema_version {document.get('schema_version')!r}; "
            f"expected {schema_version!r}; no migration is defined"
        )
    return model.model_validate(document)


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_inert_configuration(value: Any, *, field_ref: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower().replace("-", "_")
            if key in _OBSERVED_REALIZATION_KEYS:
                raise ValueError(
                    f"{field_ref} contains observed realization field {raw_key!r}; "
                    "resource handles belong only in Attempt"
                )
            if key in _SECRET_VALUE_KEYS or any(
                key.endswith(f"_{suffix}") for suffix in _SECRET_VALUE_KEYS
            ):
                raise ValueError(
                    f"{field_ref} contains secret material field {raw_key!r}; "
                    "BackendPlan carries secret names only"
                )
            _validate_inert_configuration(item, field_ref=f"{field_ref}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_inert_configuration(item, field_ref=f"{field_ref}[{index}]")


__all__ = [
    "ATTEMPT_SCHEMA_ID",
    "ATTEMPT_SCHEMA_VERSION",
    "BACKEND_PLAN_SCHEMA_ID",
    "BACKEND_PLAN_SCHEMA_VERSION",
    "Attempt",
    "BackendPlan",
    "BackendRealizationRequest",
    "ExpectedCost",
    "MachineShape",
    "OrchestrationBackend",
    "UnsupportedBackendRecordVersionError",
    "attempt_from_document",
    "backend_plan_from_document",
]
