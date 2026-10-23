"""Inert GCP backend realization for durable-controller execution."""

from __future__ import annotations

from feedbax.orchestration.drivers.capabilities import (
    AcquisitionSemantics,
    AuthorizationSemantics,
    CustodySemantics,
    DriverCapabilityEnvelope,
    DriverCapabilityFacts,
    DriverHook,
    DriverVenue,
    EnvironmentSemantics,
    MonitoringSemantics,
    RecoverySemantics,
    ResourceSemantics,
    RetrySemantics,
    SpendSemantics,
    TeardownSemantics,
)


GCP_CONTROLLER_CAPABILITIES = DriverCapabilityEnvelope.single(
    "gcp",
    DriverCapabilityFacts(
        variant_id="controller-acquired",
        venue=DriverVenue.CLOUD_RESOURCE,
        resources=ResourceSemantics.DRIVER_OWNED,
        spend=SpendSemantics.DRIVER_OBSERVED,
        authorization=AuthorizationSemantics.CLOUD_AND_SPEND_REQUIRED,
        environment=EnvironmentSemantics.REMOTE_REALIZATION,
        monitoring=MonitoringSemantics.PROVIDER_INVENTORY,
        recovery=RecoverySemantics.DURABLE_REMOTE,
        retry=RetrySemantics.NONE,
        acquisition=AcquisitionSemantics.ENGINE_GOVERNED,
        teardown=TeardownSemantics.VERIFIED_RESOURCE_ABSENCE,
        custody=CustodySemantics.EPHEMERAL_REMOTE_RESOURCE,
        optional_hooks=frozenset({DriverHook.ENGINE_ACQUISITION}),
    ),
)


__all__ = ["GCP_CONTROLLER_CAPABILITIES"]
