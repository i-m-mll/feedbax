from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from feedbax.orchestration.drivers.capabilities import (
    DRIVER_CAPABILITIES_SCHEMA_ID,
    DRIVER_CAPABILITIES_SCHEMA_VERSION,
    DRIVER_CAPABILITIES_SCHEMA_VERSION_V2,
    DRIVER_CAPABILITIES_SCHEMA_VERSION_V3,
    AcquisitionSemantics,
    AuthorizationSemantics,
    CustodySemantics,
    DriverCapabilityEnvelope,
    DriverCapabilityFacts,
    DriverConstructionContext,
    DriverHook,
    DriverRegistration,
    DriverRegistry,
    DriverStage,
    DriverVenue,
    EnvironmentSemantics,
    MonitoringSemantics,
    RealizedDriverCapabilities,
    RecoverySemantics,
    ResourceSemantics,
    RetrySemantics,
    SpendSemantics,
    TeardownSemantics,
)
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.orchestration.drivers.runpod import RunPodDriverConfig
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.worker_driver import WorkerHttpDriver


def _local_facts(variant_id: str = "local") -> DriverCapabilityFacts:
    return DriverCapabilityFacts(
        variant_id=variant_id,
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


def _conditional_envelope() -> DriverCapabilityEnvelope:
    external = DriverCapabilityFacts(
        variant_id="external",
        venue=DriverVenue.CLOUD_RESOURCE,
        resources=ResourceSemantics.EXTERNALLY_MANAGED,
        spend=SpendSemantics.EXTERNALLY_MANAGED,
        authorization=AuthorizationSemantics.OPTIONAL_CALLER_CREDENTIAL,
        environment=EnvironmentSemantics.REMOTE_REALIZATION,
        monitoring=MonitoringSemantics.PROVIDER_INVENTORY,
        recovery=RecoverySemantics.DURABLE_REMOTE,
        retry=RetrySemantics.NONE,
        acquisition=AcquisitionSemantics.EXTERNALLY_PROVIDED,
        teardown=TeardownSemantics.RESOURCES_PRESERVED,
        custody=CustodySemantics.EPHEMERAL_REMOTE_RESOURCE,
    )
    acquired = DriverCapabilityFacts(
        variant_id="acquired",
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
    )
    return DriverCapabilityEnvelope(
        driver_name="fixture:conditional",
        variants={external.variant_id: external, acquired.variant_id: acquired},
    )


class _CoreOnlyDriverMethods:
    def provision(self, *_args):
        return {}

    def realize_env(self, *_args):
        return "fixture"

    def stage_inputs(self, *_args):
        return {}

    def launch_row(self, *_args):
        return {}

    def probe(self, *_args):
        return object()

    def stop_row(self, *_args):
        return {}

    def collect(self, *_args):
        return {}

    def teardown(self, *_args):
        return {}


class _CoreDriverMethods(_CoreOnlyDriverMethods):

    def acquisition_candidates(self, *_args):
        return (None,)

    def acquisition_pod_name(self, *_args):
        return "fixture"

    def acquisition_config_identity(self, *_args):
        return "fixture"

    def create_pod_once(self, *_args):
        return object()

    def finish_acquired_pod(self, *_args):
        return {}

    def acquisition_failure_evidence(self, *_args):
        return {}

    def observe_pod_inventory(self, *_args, **_kwargs):
        return (), {}

    def adopt_owned_pod(self, *_args, **_kwargs):
        return None

    def adopted_provision_record(self, *_args):
        return {}


def test_capability_envelope_has_stable_v3_identity_and_core_stages() -> None:
    facts = _local_facts()
    envelope = DriverCapabilityEnvelope.single("fixture:driver", facts)
    realized = envelope.realize("local")

    assert envelope.schema_id == DRIVER_CAPABILITIES_SCHEMA_ID
    assert DRIVER_CAPABILITIES_SCHEMA_VERSION_V2 == "2"
    assert envelope.schema_version == DRIVER_CAPABILITIES_SCHEMA_VERSION_V3
    assert DRIVER_CAPABILITIES_SCHEMA_VERSION == DRIVER_CAPABILITIES_SCHEMA_VERSION_V3
    assert facts.stages == frozenset(DriverStage)
    assert realized.driver_name == "fixture:driver"
    assert realized.variant_id == "local"
    assert envelope.supports(realized)


def test_capability_facts_reject_incomplete_or_contradictory_variants() -> None:
    facts = _local_facts()

    with pytest.raises(ValueError, match="omit core stages: teardown"):
        replace(facts, stages=frozenset(set(DriverStage) - {DriverStage.TEARDOWN}))
    with pytest.raises(ValueError, match="engine-governed acquisition requires"):
        replace(
            facts,
            resources=ResourceSemantics.DRIVER_OWNED,
            acquisition=AcquisitionSemantics.ENGINE_GOVERNED,
        )
    with pytest.raises(ValueError, match="unsupported driver capability schema version"):
        RealizedDriverCapabilities(
            driver_name="fixture:driver",
            facts=facts,
            schema_version="1",
        )


def test_registry_realizes_external_and_acquired_contexts_without_name_branching() -> None:
    envelope = _conditional_envelope()

    class ConditionalDriver(_CoreDriverMethods):
        def __init__(
            self,
            _context: DriverConstructionContext,
            realized: RealizedDriverCapabilities,
        ) -> None:
            self.realized_capabilities = realized

    def resolve(context: DriverConstructionContext) -> RealizedDriverCapabilities:
        mode = context.configuration["resource_mode"]
        assert isinstance(mode, str)
        return envelope.realize(mode)

    registry = DriverRegistry(
        (
            DriverRegistration(
                name=envelope.driver_name,
                supported_capabilities=envelope,
                resolve_capabilities=resolve,
                factory=ConditionalDriver,
            ),
        )
    )

    external = registry.construct(
        envelope.driver_name,
        DriverConstructionContext(configuration={"resource_mode": "external"}),
    )
    acquired = registry.construct(
        envelope.driver_name,
        DriverConstructionContext(configuration={"resource_mode": "acquired"}),
    )

    assert external.realized_capabilities.facts.resources is ResourceSemantics.EXTERNALLY_MANAGED
    assert (
        external.realized_capabilities.facts.teardown
        is TeardownSemantics.RESOURCES_PRESERVED
    )
    assert acquired.realized_capabilities.facts.resources is ResourceSemantics.DRIVER_OWNED
    assert acquired.realized_capabilities.facts.acquisition is AcquisitionSemantics.ENGINE_GOVERNED
    assert (
        acquired.realized_capabilities.facts.teardown is TeardownSemantics.VERIFIED_RESOURCE_ABSENCE
    )


def test_registry_rejects_unsupported_context_realization() -> None:
    envelope = DriverCapabilityEnvelope.single("fixture:driver", _local_facts())
    unsupported = RealizedDriverCapabilities(
        driver_name=envelope.driver_name,
        facts=_local_facts("other"),
    )
    registry = DriverRegistry(
        (
            DriverRegistration(
                name=envelope.driver_name,
                supported_capabilities=envelope,
                resolve_capabilities=lambda _context: unsupported,
                factory=lambda _context, realized: _CoreOnlyFixtureDriver(realized),
            ),
        )
    )

    with pytest.raises(ValueError, match="selected unsupported facts"):
        registry.construct(envelope.driver_name, DriverConstructionContext())


def test_registry_rejects_factory_realization_drift() -> None:
    envelope = DriverCapabilityEnvelope.single("fixture:driver", _local_facts())
    expected = envelope.realize("local")
    drifted = RealizedDriverCapabilities(
        driver_name=envelope.driver_name,
        facts=_local_facts("other"),
    )
    registry = DriverRegistry(
        (
            DriverRegistration(
                name=envelope.driver_name,
                supported_capabilities=envelope,
                resolve_capabilities=lambda _context: expected,
                factory=lambda _context, _realized: _FixtureDriver(drifted),
            ),
        )
    )

    with pytest.raises(ValueError, match="do not match the context selection"):
        registry.construct(envelope.driver_name, DriverConstructionContext())


def test_registry_fails_closed_and_lists_registered_drivers() -> None:
    envelope = DriverCapabilityEnvelope.single("fixture:driver", _local_facts())
    registry = DriverRegistry(
        (
            DriverRegistration(
                name=envelope.driver_name,
                supported_capabilities=envelope,
                resolve_capabilities=lambda _context: envelope.realize("local"),
                factory=lambda _context, realized: _FixtureDriver(realized),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="unknown orchestration driver 'missing'; registered drivers: 'fixture:driver'",
    ):
        registry.resolve("missing")


def test_construction_context_deep_detaches_and_freezes_nested_inputs() -> None:
    configuration = {
        "nested": {"items": [1, {"enabled": True}]},
        "box": _MutableBox(values=[2, 3]),
    }
    runtime_bindings = {"roots": [{"path": "/before"}]}
    recovery_inputs = {"attempts": [{"records": ["first"]}]}
    credentials = {"token": "before"}
    context = DriverConstructionContext(
        configuration=configuration,
        runtime_bindings=runtime_bindings,
        credentials=credentials,
        recovery_inputs=recovery_inputs,
    )

    configuration["nested"]["items"][1]["enabled"] = False  # type: ignore[index]
    configuration["box"].values.append(4)  # type: ignore[union-attr]
    runtime_bindings["roots"][0]["path"] = "/after"  # type: ignore[index]
    recovery_inputs["attempts"][0]["records"].append("second")  # type: ignore[index,union-attr]
    credentials["token"] = "after"

    snapshot = context.configuration
    nested = snapshot["nested"]
    assert isinstance(nested, Mapping)
    assert nested["items"] == (1, {"enabled": True})
    assert context.runtime_bindings["roots"] == ({"path": "/before"},)
    assert context.recovery_inputs["attempts"] == ({"records": ("first",)},)
    assert context.credentials == {"token": "before"}
    box = snapshot["box"]
    assert isinstance(box, _MutableBox)
    assert box.values == [2, 3]

    box.values.append(99)
    assert context.configuration["box"].values == [2, 3]  # type: ignore[union-attr]
    with pytest.raises(TypeError):
        nested["new"] = "value"  # type: ignore[index]


def test_local_driver_default_realizes_stop_variant() -> None:
    envelope = LocalOrchestrationDriver.capability_envelope
    realized = LocalOrchestrationDriver.realized_capabilities

    assert envelope.supports(realized)
    assert realized.driver_name == "local"
    assert realized.variant_id == "local-stop"
    assert realized.facts.resources is ResourceSemantics.LOCAL_PROCESS
    assert realized.facts.optional_hooks == frozenset(
        {DriverHook.PREFLIGHT_CHECKS, DriverHook.CHECKPOINT_STOP}
    )


def test_worker_http_has_one_truthful_external_service_variant() -> None:
    envelope = WorkerHttpDriver.capability_envelope
    realized = WorkerHttpDriver.realized_capabilities

    assert envelope.supports(realized)
    assert realized.driver_name == "worker-http"
    assert realized.facts.resources is ResourceSemantics.EXTERNALLY_MANAGED
    assert realized.facts.acquisition is AcquisitionSemantics.EXTERNALLY_PROVIDED
    assert realized.facts.teardown is TeardownSemantics.RESOURCES_PRESERVED
    assert realized.facts.optional_hooks == frozenset()


def test_builtin_registry_realizes_runpod_ownership_from_construction_context() -> None:
    registry = new_application_registry_bundle(local_component_source=None).drivers

    external = registry.construct(
        "runpod",
        DriverConstructionContext(
            configuration={
                "driver_config": RunPodDriverConfig(
                    pod_id="supplied-pod",
                    ssh_host="127.0.0.1",
                    ssh_port=2222,
                )
            }
        ),
    )
    acquired = registry.construct(
        "runpod",
        DriverConstructionContext(configuration={"driver_config": RunPodDriverConfig()}),
    )

    assert external.realized_capabilities.variant_id == "externally-managed"
    assert external.realized_capabilities.facts.resources is ResourceSemantics.EXTERNALLY_MANAGED
    assert (
        external.realized_capabilities.facts.teardown
        is TeardownSemantics.RESOURCES_PRESERVED
    )
    assert external.realized_capabilities.facts.spend is SpendSemantics.EXTERNALLY_MANAGED
    assert acquired.realized_capabilities.variant_id == "engine-acquired"
    assert acquired.realized_capabilities.facts.resources is ResourceSemantics.DRIVER_OWNED
    assert acquired.realized_capabilities.facts.spend is SpendSemantics.DRIVER_OBSERVED
    assert acquired.realized_capabilities.facts.recovery is RecoverySemantics.DURABLE_REMOTE
    assert acquired.realized_capabilities.facts.supports(DriverHook.ENGINE_ACQUISITION)
    assert acquired.realized_capabilities.facts.supports(DriverHook.GLOBAL_RESOURCE_INVENTORY)


def test_runpod_auto_teardown_false_realizes_owned_resource_preservation() -> None:
    registry = new_application_registry_bundle(local_component_source=None).drivers

    driver = registry.construct(
        "runpod",
        DriverConstructionContext(
            configuration={"driver_config": RunPodDriverConfig(auto_teardown=False)}
        ),
    )

    assert driver.realized_capabilities.variant_id == "engine-acquired-preserved"
    assert driver.realized_capabilities.facts.resources is ResourceSemantics.DRIVER_OWNED
    assert driver.realized_capabilities.facts.teardown is TeardownSemantics.RESOURCES_PRESERVED
    assert not driver.realized_capabilities.facts.supports(
        DriverHook.GLOBAL_RESOURCE_INVENTORY
    )


def test_keep_alive_context_realizes_preservation_for_local_and_owned_runpod() -> None:
    registry = new_application_registry_bundle(local_component_source=None).drivers

    local = registry.construct(
        "local",
        DriverConstructionContext(configuration={"preserve_owned_resources": True}),
    )
    runpod = registry.construct(
        "runpod",
        DriverConstructionContext(
            configuration={
                "driver_config": RunPodDriverConfig(),
                "preserve_owned_resources": True,
            }
        ),
    )

    assert local.realized_capabilities.variant_id == "local-preserved"
    assert local.realized_capabilities.facts.teardown is TeardownSemantics.RESOURCES_PRESERVED
    assert runpod.realized_capabilities.variant_id == "engine-acquired-preserved"
    assert runpod.realized_capabilities.facts.teardown is TeardownSemantics.RESOURCES_PRESERVED


def test_preserved_runpod_realization_mechanically_prevents_provider_removal() -> None:
    registry = new_application_registry_bundle(local_component_source=None).drivers
    driver = registry.construct(
        "runpod",
        DriverConstructionContext(
            configuration={
                "driver_config": RunPodDriverConfig(auto_teardown=True),
                "preserve_owned_resources": True,
            },
            runtime_bindings={"transport": _RemovalCanaryTransport()},
        ),
    )
    driver.adopt_owned_pod("owned-pod")

    outputs = driver.teardown(SimpleNamespace(keep_alive=False), object())

    assert outputs["teardown"] == "skipped"
    assert outputs["skip_reason"] == "realized-capability-preserves-resources"
    assert driver.transport.calls == []


def test_preserved_local_realization_mechanically_prevents_process_stop() -> None:
    registry = new_application_registry_bundle(local_component_source=None).drivers
    driver = registry.construct(
        "local",
        DriverConstructionContext(configuration={"preserve_owned_resources": True}),
    )
    stop_calls: list[str] = []
    driver.stop_row = lambda *_args: stop_calls.append("stop")  # type: ignore[method-assign]

    outputs = driver.teardown(SimpleNamespace(keep_alive=False), object())

    assert outputs["teardown"] == "skipped"
    assert outputs["skip_reason"] == "realized-capability-preserves-resources"
    assert stop_calls == []


def test_registry_rejects_missing_core_and_advertised_hook_members() -> None:
    core_envelope = DriverCapabilityEnvelope.single("fixture:core", _local_facts())
    missing_core = DriverRegistry(
        (
            DriverRegistration(
                name="fixture:core",
                supported_capabilities=core_envelope,
                resolve_capabilities=lambda _context: core_envelope.realize("local"),
                factory=lambda _context, realized: type(
                    "MissingCore", (), {"realized_capabilities": realized}
                )(),
            ),
        )
    )
    hook_facts = replace(
        _local_facts(),
        optional_hooks=frozenset({DriverHook.PREFLIGHT_CHECKS}),
    )
    hook_envelope = DriverCapabilityEnvelope.single("fixture:hook", hook_facts)
    missing_hook = DriverRegistry(
        (
            DriverRegistration(
                name="fixture:hook",
                supported_capabilities=hook_envelope,
                resolve_capabilities=lambda _context: hook_envelope.realize("local"),
                factory=lambda _context, realized: _FixtureDriver(realized),
            ),
        )
    )

    with pytest.raises(TypeError, match="lacks callable members: 'provision'"):
        missing_core.construct("fixture:core", DriverConstructionContext())
    with pytest.raises(TypeError, match="'preflight_checks'"):
        missing_hook.construct("fixture:hook", DriverConstructionContext())


@pytest.mark.parametrize(
    ("name", "facts", "missing_member"),
    [
        (
            "fixture:acquisition-group",
            _conditional_envelope().variants["acquired"],
            "acquisition_candidates",
        ),
        (
            "fixture:inventory-group",
            replace(
                _local_facts(),
                optional_hooks=frozenset({DriverHook.GLOBAL_RESOURCE_INVENTORY}),
            ),
            "observe_global_resource_inventory",
        ),
    ],
)
def test_group_hooks_require_their_complete_callable_surface(
    name: str,
    facts: DriverCapabilityFacts,
    missing_member: str,
) -> None:
    envelope = DriverCapabilityEnvelope.single(name, facts)
    registry = DriverRegistry(
        (
            DriverRegistration(
                name=name,
                supported_capabilities=envelope,
                resolve_capabilities=lambda _context: envelope.realize(facts.variant_id),
                factory=lambda _context, realized: _CoreOnlyFixtureDriver(realized),
            ),
        )
    )

    with pytest.raises(TypeError, match=missing_member):
        registry.construct(name, DriverConstructionContext())


def test_application_driver_registry_is_fresh_and_sealed_with_builtins() -> None:
    first = new_application_registry_bundle(local_component_source=None)
    second = new_application_registry_bundle(local_component_source=None)

    assert first.drivers is not second.drivers
    assert first.drivers.registered_names() == ("local", "runpod", "worker-http")
    first.seal()
    with pytest.raises(RuntimeError, match="registry is sealed"):
        first.drivers.register(second.drivers.resolve("local"))


@dataclass
class _MutableBox:
    values: list[int]


class _FixtureDriver(_CoreDriverMethods):
    def __init__(self, realized: RealizedDriverCapabilities) -> None:
        self.realized_capabilities = realized


class _CoreOnlyFixtureDriver(_CoreOnlyDriverMethods):
    def __init__(self, realized: RealizedDriverCapabilities) -> None:
        self.realized_capabilities = realized


class _RemovalCanaryTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def runpodctl(self, *args, **kwargs):
        self.calls.append((*args, kwargs))
        raise AssertionError("preserved realization attempted provider removal")
