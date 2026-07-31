"""Versioned support, realization, and construction contracts for drivers."""

from __future__ import annotations

import copy
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
DRIVER_CAPABILITIES_SCHEMA_VERSION_V2 = "2"
DRIVER_CAPABILITIES_SCHEMA_VERSION = DRIVER_CAPABILITIES_SCHEMA_VERSION_V2

_IDENTITY_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)*")


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
    GLOBAL_RESOURCE_INVENTORY = "global_resource_inventory"
    DRY_RUN_LAUNCH = "dry_run_launch"


class DriverVenue(StrEnum):
    """Location in which row execution occurs."""

    LOCAL_PROCESS = "local-process"
    REMOTE_SERVICE = "remote-service"
    CLOUD_RESOURCE = "cloud-resource"


class ResourceSemantics(StrEnum):
    """Ownership of execution resources in one realized variant."""

    LOCAL_PROCESS = "local-process"
    EXTERNALLY_MANAGED = "externally-managed"
    DRIVER_OWNED = "driver-owned"


class SpendSemantics(StrEnum):
    """How spend is created and observed in one realized variant."""

    NONE = "none"
    EXTERNALLY_MANAGED = "externally-managed"
    DRIVER_OBSERVED = "driver-observed"


class AuthorizationSemantics(StrEnum):
    """Authority required before using one realized venue."""

    NONE = "none"
    OPTIONAL_CALLER_CREDENTIAL = "optional-caller-credential"
    CLOUD_AND_SPEND_REQUIRED = "cloud-and-spend-required"


class EnvironmentSemantics(StrEnum):
    """Authority behind the environment fingerprint."""

    LOCAL_INVENTORY = "local-inventory"
    OPAQUE_DRIVER_IDENTITY = "opaque-driver-identity"
    REMOTE_REALIZATION = "remote-realization"


class MonitoringSemantics(StrEnum):
    """Strongest monitoring mechanism in one realized variant."""

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
    """How execution resources are acquired in one realized variant."""

    NONE = "none"
    EXTERNALLY_PROVIDED = "externally-provided"
    ENGINE_GOVERNED = "engine-governed"


class TeardownSemantics(StrEnum):
    """Guarantee made by successful teardown in one realized variant."""

    LOCAL_PROCESS_STOP = "local-process-stop"
    EXTERNAL_RESOURCES_PRESERVED = "external-resources-preserved"
    VERIFIED_RESOURCE_ABSENCE = "verified-resource-absence"


class CustodySemantics(StrEnum):
    """Custody location before orchestration collection."""

    LOCAL_RUN_SET = "local-run-set"
    EXTERNAL_SERVICE = "external-service"
    EPHEMERAL_REMOTE_RESOURCE = "ephemeral-remote-resource"


@dataclass(frozen=True)
class DriverCapabilityFacts:
    """Immutable semantic facts for one supported or realized driver variant."""

    variant_id: str
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

    def __post_init__(self) -> None:
        """Reject malformed or internally contradictory facts."""
        _validate_identity(self.variant_id, field_name="driver capability variant id")
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
            raise ValueError(f"driver capability facts omit core stages: {names}")
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
        """Return whether this variant includes one optional hook."""
        return hook in self.optional_hooks


@dataclass(frozen=True)
class RealizedDriverCapabilities:
    """Per-instance selection of one exact supported capability variant."""

    driver_name: str
    facts: DriverCapabilityFacts
    schema_id: str = DRIVER_CAPABILITIES_SCHEMA_ID
    schema_version: str = DRIVER_CAPABILITIES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_id, self.schema_version)
        _validate_identity(self.driver_name, field_name="orchestration driver name")
        if not isinstance(self.facts, DriverCapabilityFacts):
            raise TypeError("realized driver facts must be DriverCapabilityFacts")

    @property
    def variant_id(self) -> str:
        """Return the selected supported variant identity."""
        return self.facts.variant_id


@dataclass(frozen=True)
class DriverCapabilityEnvelope:
    """Versioned set of variants a registered driver knows how to realize."""

    driver_name: str
    variants: Mapping[str, DriverCapabilityFacts]
    schema_id: str = DRIVER_CAPABILITIES_SCHEMA_ID
    schema_version: str = DRIVER_CAPABILITIES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_id, self.schema_version)
        _validate_identity(self.driver_name, field_name="orchestration driver name")
        if not isinstance(self.variants, Mapping) or not self.variants:
            raise ValueError("driver capability envelope requires at least one variant")
        copied = dict(self.variants)
        for variant_id, facts in copied.items():
            if not isinstance(facts, DriverCapabilityFacts):
                raise TypeError("driver capability variants must be DriverCapabilityFacts")
            if variant_id != facts.variant_id:
                raise ValueError(
                    "driver capability variant key must match its facts: "
                    f"{variant_id!r} != {facts.variant_id!r}"
                )
        object.__setattr__(self, "variants", MappingProxyType(copied))

    @classmethod
    def single(
        cls,
        driver_name: str,
        facts: DriverCapabilityFacts,
    ) -> DriverCapabilityEnvelope:
        """Build an envelope for a driver with one context-invariant variant."""
        return cls(driver_name=driver_name, variants={facts.variant_id: facts})

    def realize(self, variant_id: str) -> RealizedDriverCapabilities:
        """Select one supported variant or fail closed with available variants."""
        try:
            facts = self.variants[variant_id]
        except KeyError as exc:
            available = ", ".join(repr(item) for item in sorted(self.variants))
            raise ValueError(
                f"unsupported capability variant {variant_id!r} for driver "
                f"{self.driver_name!r}; supported variants: {available}"
            ) from exc
        return RealizedDriverCapabilities(driver_name=self.driver_name, facts=facts)

    def supports(self, realized: RealizedDriverCapabilities) -> bool:
        """Return whether ``realized`` is an exact variant in this envelope."""
        return (
            realized.driver_name == self.driver_name
            and self.variants.get(realized.variant_id) == realized.facts
            and realized.schema_id == self.schema_id
            and realized.schema_version == self.schema_version
        )


class DriverCapabilityProvider(Protocol):
    """Constructed driver exposing its exact per-instance capability state."""

    realized_capabilities: RealizedDriverCapabilities


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


@dataclass(frozen=True, init=False)
class DriverConstructionContext:
    """Deep-detached inputs available to capability resolvers and factories."""

    _configuration: Mapping[str, object] = field(repr=False)
    _runtime_bindings: Mapping[str, object] = field(repr=False)
    _credentials: Mapping[str, str] = field(repr=False)
    _authority: DriverAuthority = field(repr=False)
    _recovery_inputs: Mapping[str, object] = field(repr=False)

    def __init__(
        self,
        *,
        configuration: Mapping[str, object] | None = None,
        runtime_bindings: Mapping[str, object] | None = None,
        credentials: Mapping[str, str] | None = None,
        authority: DriverAuthority | None = None,
        recovery_inputs: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(self, "_configuration", _snapshot_mapping(configuration or {}))
        object.__setattr__(self, "_runtime_bindings", _snapshot_mapping(runtime_bindings or {}))
        object.__setattr__(self, "_credentials", _snapshot_credentials(credentials or {}))
        resolved_authority = authority or DriverAuthority()
        if not isinstance(resolved_authority, DriverAuthority):
            raise TypeError("driver authority must be a DriverAuthority")
        object.__setattr__(self, "_authority", copy.deepcopy(resolved_authority))
        object.__setattr__(self, "_recovery_inputs", _snapshot_mapping(recovery_inputs or {}))

    @property
    def configuration(self) -> Mapping[str, object]:
        """Return a detached immutable view of driver configuration."""
        return _detached_mapping(self._configuration)

    @property
    def runtime_bindings(self) -> Mapping[str, object]:
        """Return a detached immutable view of runtime bindings."""
        return _detached_mapping(self._runtime_bindings)

    @property
    def credentials(self) -> Mapping[str, str]:
        """Return a detached immutable view of credential values."""
        return MappingProxyType(dict(self._credentials))

    @property
    def authority(self) -> DriverAuthority:
        """Return detached construction authority."""
        return copy.deepcopy(self._authority)

    @property
    def recovery_inputs(self) -> Mapping[str, object]:
        """Return a detached immutable view of recovery inputs."""
        return _detached_mapping(self._recovery_inputs)


class DriverCapabilityResolver(Protocol):
    """Select truthful per-instance facts from explicit construction inputs."""

    def __call__(self, context: DriverConstructionContext) -> RealizedDriverCapabilities:
        """Resolve one supported variant for ``context``."""
        ...


class DriverFactory(Protocol):
    """Construct one driver with registry-validated realized capabilities."""

    def __call__(
        self,
        context: DriverConstructionContext,
        realized: RealizedDriverCapabilities,
    ) -> OrchestrationDriver:
        """Construct one driver instance."""
        ...


@dataclass(frozen=True)
class DriverRegistration:
    """One driver support envelope, realization resolver, and factory."""

    name: str
    supported_capabilities: DriverCapabilityEnvelope
    resolve_capabilities: DriverCapabilityResolver
    factory: DriverFactory

    def __post_init__(self) -> None:
        if self.name != self.supported_capabilities.driver_name:
            raise ValueError(
                "driver registration name must match its capability envelope: "
                f"{self.name!r} != {self.supported_capabilities.driver_name!r}"
            )
        if not callable(self.resolve_capabilities):
            raise TypeError(f"driver capability resolver for {self.name!r} must be callable")
        if not callable(self.factory):
            raise TypeError(f"driver factory for {self.name!r} must be callable")


class DriverRegistry:
    """Injected registry for context-aware capability realization and construction."""

    def __init__(self, registrations: tuple[DriverRegistration, ...] = ()) -> None:
        self._sealed = False
        self._registrations: dict[str, DriverRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: DriverRegistration) -> None:
        """Register one exact driver identity, rejecting duplicates."""
        if self._sealed:
            raise RuntimeError("orchestration driver registry is sealed")
        if not isinstance(registration, DriverRegistration):
            raise TypeError("driver registration must be a DriverRegistration")
        if registration.name in self._registrations:
            raise ValueError(f"orchestration driver already registered: {registration.name!r}")
        self._registrations[registration.name] = registration

    def seal(self) -> None:
        """Prevent registration after application bootstrap publication."""
        self._sealed = True

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
        """Resolve supported facts, construct, and verify instance realization."""
        if not isinstance(context, DriverConstructionContext):
            raise TypeError("driver construction context must be a DriverConstructionContext")
        registration = self.resolve(name)
        realized = registration.resolve_capabilities(context)
        if not isinstance(realized, RealizedDriverCapabilities):
            raise TypeError(
                f"capability resolver for driver {name!r} must return RealizedDriverCapabilities"
            )
        if not registration.supported_capabilities.supports(realized):
            raise ValueError(f"capability resolver for driver {name!r} selected unsupported facts")
        driver = registration.factory(context, realized)
        if driver.realized_capabilities != realized:
            raise ValueError(
                f"constructed driver {name!r} realized capabilities do not match "
                "the context selection"
            )
        return driver


def _validate_schema(schema_id: str, schema_version: str) -> None:
    if schema_id != DRIVER_CAPABILITIES_SCHEMA_ID:
        raise ValueError(f"unsupported driver capability schema id: {schema_id!r}")
    if schema_version != DRIVER_CAPABILITIES_SCHEMA_VERSION:
        raise ValueError(
            "unsupported driver capability schema version: "
            f"{schema_version!r}; expected {DRIVER_CAPABILITIES_SCHEMA_VERSION!r}"
        )


def _validate_identity(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _IDENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"invalid {field_name}: {value!r}")


def _snapshot_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError("driver construction inputs must be mappings")
    if not all(isinstance(key, str) for key in values):
        raise TypeError("driver construction input keys must be strings")
    return MappingProxyType({key: _freeze_nested(value) for key, value in values.items()})


def _snapshot_credentials(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("driver credentials must be a mapping")
    copied = dict(values)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in copied.items()):
        raise TypeError("driver credential names and values must be strings")
    return MappingProxyType(copied)


def _freeze_nested(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {copy.deepcopy(key): _freeze_nested(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_nested(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_nested(item) for item in value)
    return copy.deepcopy(value)


def _detached_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _detach_nested(value) for key, value in values.items()})


def _detach_nested(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {copy.deepcopy(key): _detach_nested(item) for key, item in value.items()}
        )
    if isinstance(value, tuple):
        return tuple(_detach_nested(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_detach_nested(item) for item in value)
    return copy.deepcopy(value)
