from __future__ import annotations

from dataclasses import replace

import pytest

from feedbax.orchestration.drivers.capabilities import (
    DRIVER_CAPABILITIES_SCHEMA_ID,
    DRIVER_CAPABILITIES_SCHEMA_VERSION,
    AcquisitionSemantics,
    AuthorizationSemantics,
    CustodySemantics,
    DriverCapabilities,
    DriverConstructionContext,
    DriverHook,
    DriverRegistration,
    DriverRegistry,
    DriverStage,
    DriverVenue,
    EnvironmentSemantics,
    MonitoringSemantics,
    RecoverySemantics,
    ResourceSemantics,
    RetrySemantics,
    SpendSemantics,
    TeardownSemantics,
)
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.web.services.worker_driver import WorkerHttpDriver


def _fixture_capabilities(name: str = "fixture:driver") -> DriverCapabilities:
    return DriverCapabilities(
        driver_name=name,
        venue=DriverVenue.LOCAL_PROCESS,
        resources=ResourceSemantics.LOCAL_PROCESS,
        spend=SpendSemantics.NONE,
        authorization=AuthorizationSemantics.NONE,
        environment=EnvironmentSemantics.LOCAL_INVENTORY,
        monitoring=MonitoringSemantics.ROW_POLL,
        recovery=RecoverySemantics.NONE,
        retry=RetrySemantics.NONE,
        acquisition=AcquisitionSemantics.NONE,
        teardown=TeardownSemantics.LOCAL_PROCESS_STOP,
        custody=CustodySemantics.LOCAL_RUN_SET,
    )


def test_capability_contract_has_stable_schema_identity_and_core_stages() -> None:
    capabilities = _fixture_capabilities()

    assert capabilities.schema_id == DRIVER_CAPABILITIES_SCHEMA_ID
    assert capabilities.schema_version == DRIVER_CAPABILITIES_SCHEMA_VERSION
    assert capabilities.stages == frozenset(DriverStage)
    assert not capabilities.supports(DriverHook.REMOTE_SMOKE)


def test_capability_contract_rejects_incomplete_or_contradictory_facts() -> None:
    capabilities = _fixture_capabilities()

    with pytest.raises(ValueError, match="omits core stages: teardown"):
        replace(capabilities, stages=frozenset(set(DriverStage) - {DriverStage.TEARDOWN}))
    with pytest.raises(ValueError, match="engine-governed acquisition requires"):
        replace(
            capabilities,
            resources=ResourceSemantics.DRIVER_OWNED,
            acquisition=AcquisitionSemantics.ENGINE_GOVERNED,
        )
    with pytest.raises(ValueError, match="unsupported driver capability schema version"):
        replace(capabilities, schema_version="2")


def test_registry_constructs_from_immutable_typed_context() -> None:
    capabilities = _fixture_capabilities()
    supplied_configuration = {"value": 3}
    observed_contexts: list[DriverConstructionContext] = []

    class FixtureDriver:
        def __init__(self, context: DriverConstructionContext) -> None:
            self.capabilities = capabilities
            observed_contexts.append(context)

    registry = DriverRegistry(
        (
            DriverRegistration(
                name=capabilities.driver_name,
                capabilities=capabilities,
                factory=FixtureDriver,
            ),
        )
    )
    context = DriverConstructionContext(configuration=supplied_configuration)
    supplied_configuration["value"] = 4

    driver = registry.construct("fixture:driver", context)

    assert isinstance(driver, FixtureDriver)
    assert observed_contexts == [context]
    assert context.configuration == {"value": 3}
    with pytest.raises(TypeError):
        context.configuration["value"] = 5  # type: ignore[index]


def test_registry_fails_closed_and_lists_registered_drivers() -> None:
    capabilities = _fixture_capabilities()
    registry = DriverRegistry(
        (
            DriverRegistration(
                name=capabilities.driver_name,
                capabilities=capabilities,
                factory=lambda _context: object(),  # type: ignore[arg-type,return-value]
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="unknown orchestration driver 'missing'; registered drivers: 'fixture:driver'",
    ):
        registry.resolve("missing")


def test_registry_rejects_duplicate_and_mismatched_declarations() -> None:
    capabilities = _fixture_capabilities()
    registration = DriverRegistration(
        name=capabilities.driver_name,
        capabilities=capabilities,
        factory=lambda _context: object(),  # type: ignore[arg-type,return-value]
    )
    registry = DriverRegistry((registration,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(registration)
    with pytest.raises(ValueError, match="name must match"):
        DriverRegistration(
            name="different",
            capabilities=capabilities,
            factory=lambda _context: object(),  # type: ignore[arg-type,return-value]
        )


def test_registry_rejects_factory_capability_drift() -> None:
    capabilities = _fixture_capabilities()

    class DriftedDriver:
        capabilities = _fixture_capabilities("fixture:other")

    registry = DriverRegistry(
        (
            DriverRegistration(
                name=capabilities.driver_name,
                capabilities=capabilities,
                factory=lambda _context: DriftedDriver(),  # type: ignore[arg-type,return-value]
            ),
        )
    )

    with pytest.raises(ValueError, match="does not match its registry"):
        registry.construct(capabilities.driver_name, DriverConstructionContext())


def test_local_driver_declares_only_implemented_optional_hooks() -> None:
    capabilities = LocalOrchestrationDriver.capabilities

    assert capabilities.driver_name == "local"
    assert capabilities.venue is DriverVenue.LOCAL_PROCESS
    assert capabilities.spend is SpendSemantics.NONE
    assert capabilities.custody is CustodySemantics.LOCAL_RUN_SET
    assert capabilities.optional_hooks == frozenset(
        {
            DriverHook.PREFLIGHT_CHECKS,
            DriverHook.CHECKPOINT_STOP,
        }
    )


def test_worker_http_declares_external_ownership_and_no_optional_hooks() -> None:
    capabilities = WorkerHttpDriver.capabilities

    assert capabilities.driver_name == "worker-http"
    assert capabilities.venue is DriverVenue.REMOTE_SERVICE
    assert capabilities.resources is ResourceSemantics.EXTERNALLY_MANAGED
    assert capabilities.spend is SpendSemantics.EXTERNALLY_MANAGED
    assert capabilities.acquisition is AcquisitionSemantics.EXTERNALLY_PROVIDED
    assert capabilities.teardown is TeardownSemantics.EXTERNAL_RESOURCES_PRESERVED
    assert capabilities.custody is CustodySemantics.EXTERNAL_SERVICE
    assert capabilities.optional_hooks == frozenset()
