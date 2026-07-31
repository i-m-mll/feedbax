"""Versioned capability and construction contracts for orchestration drivers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from feedbax.orchestration.drivers.base import OrchestrationDriver


DRIVER_CAPABILITIES_SCHEMA_ID = "feedbax.orchestration.driver-capabilities"
DRIVER_CAPABILITIES_SCHEMA_VERSION_V1 = "1"
DRIVER_CAPABILITIES_SCHEMA_VERSION = DRIVER_CAPABILITIES_SCHEMA_VERSION_V1

_DRIVER_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)*")


class DriverStage(StrEnum):
    """Stage operations implemented by every orchestration driver."""

    PROVISION = "provision"
    REALIZE_ENV = "realize_env"
    STAGE_INPUTS = "stage_inputs"
    LAUNCH_ROW = "launch_row"
    PROBE = "probe"
    STOP_ROW = "stop_row"
    COLLECT = "collect"
    TEARDOWN = "teardown"


CORE_DRIVER_STAGES = frozenset(DriverStage)


class DriverHook(StrEnum):
    """Optional hook surfaces consumed by the stage engine."""

    HAS_PENDING_OWNED_RESOURCE = "has_pending_owned_resource"
    RESTORE_FROM_PROVISION_RECORD = "restore_from_provision_record"
    GOVERN_PROVISIONING_RETRIES = "govern_provisioning_retries"
    RESTORE_COMPLETED_PREFLIGHT = "restore_completed_preflight"
    STATIC_PREFLIGHT_CHECKS = "static_preflight_checks"
    PREFLIGHT_CHECKS = "preflight_checks"
    REPO_REALIZATION_PLAN = "repo_realization_plan"
    PREFLIGHT_EVIDENCE = "preflight_evidence"
    ENGINE_ACQUISITION = "engine_acquisition"
    PROVISION_RETRY_DELAY = "provision_retry_delay"
    REMOTE_SMOKE = "remote_smoke"
    SMOKE_FAILURE_EVIDENCE = "smoke_failure_evidence"
    COLLECTION_RECOVERY_EVIDENCE = "collection_recovery_evidence"
    COLLECT_FAILURE_LOGS = "collect_failure_logs"
    TEARDOWN_OWNERSHIP = "teardown_ownership"
    BATCH_PROBE = "batch_probe"
    CHECKPOINT_STOP = "checkpoint_stop"


class DriverVenue(StrEnum):
    """Location in which row execution occurs."""

    LOCAL_PROCESS = "local-process"
    REMOTE_SERVICE = "remote-service"
    CLOUD_RESOURCE = "cloud-resource"


class ResourceSemantics(StrEnum):
    """Ownership of execution resources visible to the driver."""

    LOCAL_PROCESS = "local-process"
    EXTERNALLY_MANAGED = "externally-managed"
    DRIVER_OWNED = "driver-owned"


class SpendSemantics(StrEnum):
    """How spend is created and observed by this driver."""

    NONE = "none"
    EXTERNALLY_MANAGED = "externally-managed"
    DRIVER_OBSERVED = "driver-observed"


class AuthorizationSemantics(StrEnum):
    """Authority required before the driver may use its execution venue."""

    NONE = "none"
    OPTIONAL_CALLER_CREDENTIAL = "optional-caller-credential"
    CLOUD_AND_SPEND_REQUIRED = "cloud-and-spend-required"


class EnvironmentSemantics(StrEnum):
    """Authority behind the environment fingerprint."""

    LOCAL_INVENTORY = "local-inventory"
    OPAQUE_DRIVER_IDENTITY = "opaque-driver-identity"
    REMOTE_REALIZATION = "remote-realization"


class MonitoringSemantics(StrEnum):
    """Strongest monitoring mechanism implemented by the driver."""

    ROW_POLL = "row-poll"
    EVENT_STREAM_AND_ROW_POLL = "event-stream-and-row-poll"
    PROVIDER_INVENTORY = "provider-inventory"


class RecoverySemantics(StrEnum):
    """Recovery evidence exposed beyond persisted core stage outputs."""

    NONE = "none"
    PROCESS_LOCAL = "process-local"
    DURABLE_REMOTE = "durable-remote"


class RetrySemantics(StrEnum):
    """Retry policy implemented beyond core idempotent stage re-entry."""

    NONE = "none"
    DRIVER_GOVERNED = "driver-governed"


class AcquisitionSemantics(StrEnum):
    """How execution resources are acquired."""

    NONE = "none"
    EXTERNALLY_PROVIDED = "externally-provided"
    ENGINE_GOVERNED = "engine-governed"


class TeardownSemantics(StrEnum):
    """Guarantee made by a successful driver teardown call."""

    LOCAL_PROCESS_STOP = "local-process-stop"
    EXTERNAL_RESOURCES_PRESERVED = "external-resources-preserved"
    VERIFIED_RESOURCE_ABSENCE = "verified-resource-absence"


class CustodySemantics(StrEnum):
    """Custody location for outputs before orchestration collection."""

    LOCAL_RUN_SET = "local-run-set"
    EXTERNAL_SERVICE = "external-service"
    EPHEMERAL_REMOTE_RESOURCE = "ephemeral-remote-resource"


@dataclass(frozen=True)
class DriverCapabilities:
    """Versioned, immutable facts used to reason about a driver without its name."""

    driver_name: str
    venue: DriverVenue
    resources: ResourceSemantics
    spend: SpendSemantics
    authorization: AuthorizationSemantics
    environment: EnvironmentSemantics
    monitoring: MonitoringSemantics
    recovery: RecoverySemantics
    retry: RetrySemantics
    acquisition: AcquisitionSemantics
    teardown: TeardownSemantics
    custody: CustodySemantics
    stages: frozenset[DriverStage] = CORE_DRIVER_STAGES
    optional_hooks: frozenset[DriverHook] = frozenset()
    schema_id: str = DRIVER_CAPABILITIES_SCHEMA_ID
    schema_version: str = DRIVER_CAPABILITIES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject malformed or internally contradictory declarations."""
        if self.schema_id != DRIVER_CAPABILITIES_SCHEMA_ID:
            raise ValueError(f"unsupported driver capability schema id: {self.schema_id!r}")
        if self.schema_version != DRIVER_CAPABILITIES_SCHEMA_VERSION:
            raise ValueError(
                "unsupported driver capability schema version: "
                f"{self.schema_version!r}; expected {DRIVER_CAPABILITIES_SCHEMA_VERSION!r}"
            )
        if _DRIVER_NAME_PATTERN.fullmatch(self.driver_name) is None:
            raise ValueError(f"invalid orchestration driver name: {self.driver_name!r}")
        enum_fields = (
            ("venue", self.venue, DriverVenue),
            ("resources", self.resources, ResourceSemantics),
            ("spend", self.spend, SpendSemantics),
            ("authorization", self.authorization, AuthorizationSemantics),
            ("environment", self.environment, EnvironmentSemantics),
            ("monitoring", self.monitoring, MonitoringSemantics),
            ("recovery", self.recovery, RecoverySemantics),
            ("retry", self.retry, RetrySemantics),
            ("acquisition", self.acquisition, AcquisitionSemantics),
            ("teardown", self.teardown, TeardownSemantics),
            ("custody", self.custody, CustodySemantics),
        )
        for field_name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise TypeError(f"{field_name} must be a {enum_type.__name__} value")
        if not isinstance(self.stages, frozenset) or not all(
            isinstance(stage, DriverStage) for stage in self.stages
        ):
            raise TypeError("driver capability stages must be a frozenset of DriverStage values")
        missing = CORE_DRIVER_STAGES - self.stages
        if missing:
            names = ", ".join(stage.value for stage in sorted(missing, key=str))
            raise ValueError(f"driver capability declaration omits core stages: {names}")
        if not isinstance(self.optional_hooks, frozenset) or not all(
            isinstance(hook, DriverHook) for hook in self.optional_hooks
        ):
            raise TypeError("optional_hooks must be a frozenset of DriverHook values")
        if self.acquisition is AcquisitionSemantics.ENGINE_GOVERNED and (
            DriverHook.ENGINE_ACQUISITION not in self.optional_hooks
        ):
            raise ValueError("engine-governed acquisition requires the engine_acquisition hook")
        if self.resources is ResourceSemantics.DRIVER_OWNED and (
            self.acquisition is AcquisitionSemantics.NONE
        ):
            raise ValueError("driver-owned resources require a non-none acquisition declaration")
        if self.spend is SpendSemantics.DRIVER_OBSERVED and (
            self.resources is not ResourceSemantics.DRIVER_OWNED
        ):
            raise ValueError("driver-observed spend requires driver-owned resources")
        if self.authorization is AuthorizationSemantics.CLOUD_AND_SPEND_REQUIRED and (
            self.venue is not DriverVenue.CLOUD_RESOURCE
            or self.spend is not SpendSemantics.DRIVER_OBSERVED
        ):
            raise ValueError(
                "cloud-and-spend authorization requires cloud venue and observed spend"
            )
        if self.retry is RetrySemantics.DRIVER_GOVERNED and not {
            DriverHook.GOVERN_PROVISIONING_RETRIES,
            DriverHook.PROVISION_RETRY_DELAY,
        }.issubset(self.optional_hooks):
            raise ValueError("driver-governed retry requires retry governance and delay hooks")
        if self.teardown is TeardownSemantics.VERIFIED_RESOURCE_ABSENCE and (
            self.resources is not ResourceSemantics.DRIVER_OWNED
        ):
            raise ValueError("verified resource absence requires driver-owned resources")

    def supports(self, hook: DriverHook) -> bool:
        """Return whether this declaration includes one optional hook."""
        return hook in self.optional_hooks


class DriverCapabilityProvider(Protocol):
    """Object that exposes an explicit versioned capability declaration."""

    capabilities: DriverCapabilities


@dataclass(frozen=True)
class DriverAuthority:
    """Caller-granted authority available during driver construction."""

    cloud_authorized: bool = False
    spend_authorized: bool = False
    credential_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.cloud_authorized, bool) or not isinstance(
            self.spend_authorized, bool
        ):
            raise TypeError("driver authority flags must be bool values")
        if not isinstance(self.credential_names, frozenset) or not all(
            isinstance(name, str) and name and name == name.strip()
            for name in self.credential_names
        ):
            raise TypeError("credential_names must be a frozenset of non-empty names")


@dataclass(frozen=True)
class DriverConstructionContext:
    """Explicit inputs available to a registered driver factory."""

    configuration: Mapping[str, object] = field(default_factory=dict)
    runtime_bindings: Mapping[str, object] = field(default_factory=dict)
    credentials: Mapping[str, str] = field(default_factory=dict, repr=False)
    authority: DriverAuthority = field(default_factory=DriverAuthority)
    recovery_inputs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Snapshot mappings so later caller mutation cannot change factory inputs."""
        object.__setattr__(self, "configuration", _frozen_mapping(self.configuration))
        object.__setattr__(self, "runtime_bindings", _frozen_mapping(self.runtime_bindings))
        object.__setattr__(self, "credentials", _frozen_string_mapping(self.credentials))
        object.__setattr__(self, "recovery_inputs", _frozen_mapping(self.recovery_inputs))


class DriverFactory(Protocol):
    """Construct one driver from explicit configuration and runtime authority."""

    def __call__(self, context: DriverConstructionContext) -> OrchestrationDriver:
        """Construct one driver instance."""
        ...


@dataclass(frozen=True)
class DriverRegistration:
    """One exact driver name, capability declaration, and factory."""

    name: str
    capabilities: DriverCapabilities
    factory: DriverFactory

    def __post_init__(self) -> None:
        if self.name != self.capabilities.driver_name:
            raise ValueError(
                "driver registration name must match its capability declaration: "
                f"{self.name!r} != {self.capabilities.driver_name!r}"
            )
        if not callable(self.factory):
            raise TypeError(f"driver factory for {self.name!r} must be callable")


class DriverRegistry:
    """Injected, deterministic registry for capability-aware driver construction."""

    def __init__(self, registrations: tuple[DriverRegistration, ...] = ()) -> None:
        self._registrations: dict[str, DriverRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: DriverRegistration) -> None:
        """Register one exact driver identity, rejecting duplicates."""
        if not isinstance(registration, DriverRegistration):
            raise TypeError("driver registration must be a DriverRegistration")
        if registration.name in self._registrations:
            raise ValueError(f"orchestration driver already registered: {registration.name!r}")
        self._registrations[registration.name] = registration

    def registered_names(self) -> tuple[str, ...]:
        """Return registered driver names in deterministic order."""
        return tuple(sorted(self._registrations))

    def resolve(self, name: str) -> DriverRegistration:
        """Resolve one exact driver name or fail closed with available names."""
        try:
            return self._registrations[name]
        except KeyError as exc:
            available = ", ".join(repr(item) for item in self.registered_names()) or "<none>"
            raise ValueError(
                f"unknown orchestration driver {name!r}; registered drivers: {available}"
            ) from exc

    def construct(
        self,
        name: str,
        context: DriverConstructionContext,
    ) -> OrchestrationDriver:
        """Construct a driver and verify its declaration matches the registry."""
        if not isinstance(context, DriverConstructionContext):
            raise TypeError("driver construction context must be a DriverConstructionContext")
        registration = self.resolve(name)
        driver = registration.factory(context)
        declared = driver.capabilities
        if declared != registration.capabilities:
            raise ValueError(
                f"constructed driver {name!r} capability declaration does not match its registry"
            )
        return driver


def _frozen_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError("driver construction inputs must be mappings")
    copied = dict(values)
    if not all(isinstance(key, str) for key in copied):
        raise TypeError("driver construction input keys must be strings")
    return MappingProxyType(copied)


def _frozen_string_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("driver credentials must be a mapping")
    copied = dict(values)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in copied.items()):
        raise TypeError("driver credential names and values must be strings")
    return MappingProxyType(copied)
