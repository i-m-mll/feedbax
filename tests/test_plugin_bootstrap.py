from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, dataclass
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import BaseModel

from feedbax.analysis.evaluation import EvaluationAuthoringSchema
from feedbax.component_registry import declare_component
from feedbax.plugins.application import (
    APPLICATION_REGISTRY_KEYS,
    COMPONENTS,
    EXPERIMENT_PACKAGES,
    EVALUATION_RECIPES,
    FIGURES,
    TRAINING_PROGRAMS,
    ApplicationRegistryBundle,
    new_application_registry_bundle,
)
from feedbax.plugins.bootstrap import (
    BootstrapError,
    BootstrapErrorCode,
    FamilyRequirement,
    PluginDeclaration,
    PluginDependency,
    PluginRegistration,
    RegistrationContext,
    RegistryFamilyRegistration,
    RegistryKey,
    bootstrap_application,
    discover_plugin_registrations,
)
from feedbax.plugins.composition import compose_application


class NamesRegistry:
    def __init__(self) -> None:
        self.values: list[str] = []
        self.sealed = False

    def register(self, value: str) -> None:
        if self.sealed:
            raise RuntimeError("names registry is sealed")
        if value in self.values:
            raise ValueError(f"already registered: {value}")
        self.values.append(value)

    def seal(self) -> None:
        self.sealed = True


@dataclass(frozen=True)
class ExtendedBundle(ApplicationRegistryBundle):
    names: NamesRegistry

    def seal(self) -> None:
        super().seal()
        self.names.seal()


NAMES = RegistryKey(
    "test.names", "names", NamesRegistry, registered_keys=lambda registry: registry.values
)


def _context() -> RegistrationContext:
    def factory() -> ExtendedBundle:
        base = new_application_registry_bundle(local_component_source=None)
        return ExtendedBundle(**base.__dict__, names=NamesRegistry())

    return RegistrationContext(factory, (*APPLICATION_REGISTRY_KEYS, NAMES))


def _plugin(
    plugin_id: str,
    register,
    *,
    dependencies: tuple[PluginDependency, ...] = (),
    family_version: str = "1",
    registry_families: tuple[RegistryFamilyRegistration[NamesRegistry], ...] = (),
) -> PluginRegistration:
    return PluginRegistration(
        PluginDeclaration(
            plugin_id,
            "1",
            1,
            dependencies=dependencies,
            families=(FamilyRequirement(NAMES.family, family_version),),
        ),
        register,
        registry_families,
    )


def _provider(
    *,
    key: RegistryKey = NAMES,
    factory=NamesRegistry,
) -> RegistryFamilyRegistration:
    return RegistryFamilyRegistration(key, factory, lambda registry: registry.seal())


class _EntryPoint:
    dist = None

    def __init__(self, registration: PluginRegistration, loads: list[str] | None = None) -> None:
        self.name = registration.declaration.plugin_id
        self.value = f"{self.name}:PLUGIN_REGISTRATION"
        self._registration = registration
        self._loads = loads

    def load(self) -> PluginRegistration:
        if self._loads is not None:
            self._loads.append(self.name)
        return self._registration


def test_composer_discovers_provider_and_consumer_once_with_exact_state_lookup() -> None:
    loads: list[str] = []
    provider = _plugin(
        "pkg.provider",
        lambda context: context.registry(NAMES).register("provider"),
        registry_families=(_provider(),),
    )
    consumer = _plugin(
        "pkg.consumer",
        lambda context: context.registry(NAMES).register("consumer"),
        dependencies=(PluginDependency("pkg.provider"),),
    )

    state = asyncio.run(
        compose_application(
            entry_points=(_EntryPoint(consumer, loads), _EntryPoint(provider, loads)),
            local_component_source=None,
        )
    )

    assert loads == ["pkg.consumer", "pkg.provider"]
    assert state.registry(NAMES).values == ["provider", "consumer"]
    assert [item.plugin_id for item in state.provenance] == ["pkg.provider", "pkg.consumer"]
    wrong_key = RegistryKey(
        NAMES.family,
        NAMES.attribute,
        NamesRegistry,
        registered_keys=lambda registry: registry.values,
    )
    with pytest.raises(BootstrapError, match="is unavailable"):
        state.registry(wrong_key)


@pytest.mark.parametrize("mismatch", ["duplicate", "key", "protocol", "type"])
def test_composer_rejects_conflicting_extension_family_providers_before_callbacks(
    mismatch: str,
) -> None:
    calls: list[str] = []
    first = _plugin(
        "pkg.first_provider",
        lambda _context: calls.append("first"),
        registry_families=(_provider(),),
    )
    if mismatch == "duplicate":
        key = NAMES
        family_version = "1"
        factory = NamesRegistry
    elif mismatch == "key":
        key = RegistryKey(NAMES.family, "other_names", NamesRegistry)
        family_version = "1"
        factory = NamesRegistry
    elif mismatch == "protocol":
        key = RegistryKey(NAMES.family, NAMES.attribute, NamesRegistry, protocol_version="2")
        family_version = "2"
        factory = NamesRegistry
    else:
        key = RegistryKey(NAMES.family, NAMES.attribute, dict)
        family_version = "1"
        factory = dict
    second = _plugin(
        "pkg.second_provider",
        lambda _context: calls.append("second"),
        family_version=family_version,
        registry_families=(_provider(key=key, factory=factory),),
    )

    with pytest.raises(BootstrapError) as caught:
        asyncio.run(
            compose_application(
                entry_points=(_EntryPoint(first), _EntryPoint(second)),
                local_component_source=None,
            )
        )

    expected_code = (
        BootstrapErrorCode.UNSUPPORTED_PROTOCOL
        if mismatch == "protocol"
        else BootstrapErrorCode.INVALID_REGISTRATION
    )
    assert caught.value.code is expected_code
    assert calls == []


def test_composer_rejects_extension_factory_type_mismatch_and_seals_value() -> None:
    retained: list[dict] = []
    sealed: list[dict] = []

    def factory() -> dict:
        value: dict = {}
        retained.append(value)
        return value

    provider = _plugin(
        "pkg.wrong_factory",
        lambda _context: None,
        registry_families=(
            RegistryFamilyRegistration(NAMES, factory, lambda value: sealed.append(value)),
        ),
    )

    with pytest.raises(BootstrapError) as caught:
        asyncio.run(
            compose_application(
                entry_points=(_EntryPoint(provider),),
                local_component_source=None,
            )
        )

    assert caught.value.code is BootstrapErrorCode.INVALID_REGISTRATION
    assert sealed == retained


def test_composer_extension_provider_cannot_access_undeclared_core_family() -> None:
    provider = _plugin(
        "pkg.undeclared_access",
        lambda context: context.registry(COMPONENTS),
        registry_families=(_provider(),),
    )

    with pytest.raises(BootstrapError) as caught:
        asyncio.run(
            compose_application(
                entry_points=(_EntryPoint(provider),),
                local_component_source=None,
            )
        )

    assert caught.value.code is BootstrapErrorCode.MISSING_FAMILY


def test_composer_extension_failure_rolls_back_seals_and_isolates_later_state() -> None:
    retained: list[NamesRegistry] = []

    def factory() -> NamesRegistry:
        registry = NamesRegistry()
        retained.append(registry)
        return registry

    def fail(context: RegistrationContext) -> None:
        context.registry(NAMES).register("partial")
        raise RuntimeError("boom")

    failing = _plugin(
        "pkg.failing_provider",
        fail,
        registry_families=(_provider(factory=factory),),
    )
    with pytest.raises(BootstrapError):
        asyncio.run(
            compose_application(
                entry_points=(_EntryPoint(failing),),
                local_component_source=None,
            )
        )
    assert retained[0].sealed
    with pytest.raises(RuntimeError, match="sealed"):
        retained[0].register("escaped")

    succeeding = _plugin(
        "pkg.succeeding_provider",
        lambda context: context.registry(NAMES).register("fresh"),
        registry_families=(_provider(),),
    )
    first = asyncio.run(
        compose_application(
            entry_points=(_EntryPoint(succeeding),),
            local_component_source=None,
        )
    )
    second = asyncio.run(
        compose_application(
            entry_points=(_EntryPoint(succeeding),),
            local_component_source=None,
        )
    )
    assert first.registry(NAMES) is not second.registry(NAMES)
    assert first.registry(NAMES).values == second.registry(NAMES).values == ["fresh"]


def test_bootstrap_sorts_dependencies_and_records_provenance() -> None:
    context = _context()
    first = _plugin("pkg.first", lambda ctx: ctx.registry(NAMES).register("first"))
    second = _plugin(
        "pkg.second",
        lambda ctx: ctx.registry(NAMES).register("second"),
        dependencies=(PluginDependency("pkg.first", "1"),),
    )

    state = asyncio.run(bootstrap_application(context, registrations=(second, first)))

    assert state.registry(NAMES).values == ["first", "second"]
    assert [item.plugin_id for item in state.provenance] == ["pkg.first", "pkg.second"]
    assert state.provenance[0].registered_keys == {NAMES.family: ("first",)}
    assert state.provenance[1].registration_order == 1
    with pytest.raises(BootstrapError, match="context is sealed"):
        context.registry(NAMES).register("late")
    with pytest.raises(RuntimeError, match="names registry is sealed"):
        state.registry(NAMES).register("late")
    assert state.registry(NAMES).values == ["first", "second"]


def test_context_rejects_registry_key_with_wrong_runtime_type() -> None:
    wrong_key = RegistryKey("components", "components", dict)
    with pytest.raises(BootstrapError, match="is not dict"):
        RegistrationContext(
            lambda: new_application_registry_bundle(local_component_source=None),
            (wrong_key,),
        )


def test_bootstrap_rejects_unsupported_family_protocol() -> None:
    with pytest.raises(BootstrapError) as caught:
        asyncio.run(
            bootstrap_application(
                _context(),
                registrations=(_plugin("pkg.bad", lambda _ctx: None, family_version="2"),),
            )
        )
    assert caught.value.code is BootstrapErrorCode.UNSUPPORTED_PROTOCOL


def test_failed_registration_never_publishes_partial_state() -> None:
    context = _context()

    def fail(ctx: RegistrationContext) -> None:
        ctx.registry(NAMES).register("partial")
        raise RuntimeError("boom")

    with pytest.raises(BootstrapError) as caught:
        asyncio.run(bootstrap_application(context, registrations=(_plugin("pkg.fail", fail),)))
    assert caught.value.code is BootstrapErrorCode.REGISTRATION_FAILURE
    assert not hasattr(context, "bundle")
    with pytest.raises(BootstrapError, match="context is sealed"):
        context.registry(NAMES)


def test_failed_bootstrap_seals_plugin_retained_registry() -> None:
    retained = []

    def retain(context: RegistrationContext) -> None:
        registry = context.registry(COMPONENTS)
        retained.append(registry)
        registry.register_component_type("pkg.Retained", lambda _params: object())

    retain_plugin = PluginRegistration(
        PluginDeclaration(
            "pkg.retain",
            "1",
            1,
            families=(FamilyRequirement(COMPONENTS.family),),
        ),
        retain,
    )
    failing_plugin = _plugin(
        "pkg.fail_after_retain",
        lambda _context: (_ for _ in ()).throw(RuntimeError("boom")),
        dependencies=(PluginDependency("pkg.retain"),),
    )

    with pytest.raises(BootstrapError):
        asyncio.run(
            bootstrap_application(_context(), registrations=(failing_plugin, retain_plugin))
        )

    with pytest.raises(RuntimeError, match="component registry is sealed"):
        retained[0].register_component_type("pkg.Late", lambda _params: object())


def test_plugin_can_only_access_declared_families() -> None:
    plugin = PluginRegistration(
        PluginDeclaration("pkg.undeclared", "1", 1),
        lambda context: context.registry(NAMES).register("hidden"),
    )
    with pytest.raises(BootstrapError) as caught:
        asyncio.run(bootstrap_application(_context(), registrations=(plugin,)))
    assert caught.value.code is BootstrapErrorCode.MISSING_FAMILY


def test_declaration_rejects_malformed_nested_members() -> None:
    with pytest.raises(BootstrapError) as caught:
        PluginDeclaration("pkg.bad", "1", 1, dependencies=(object(),))  # type: ignore[arg-type]
    assert caught.value.code is BootstrapErrorCode.INVALID_REGISTRATION


def test_concurrent_callers_share_one_execution_but_contexts_are_isolated() -> None:
    calls = 0

    async def register(ctx: RegistrationContext) -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        ctx.registry(NAMES).register("value")

    plugin = _plugin("pkg.once", register)

    async def scenario():
        context = _context()
        left, right = await asyncio.gather(
            bootstrap_application(context, registrations=(plugin,)),
            bootstrap_application(context, registrations=(plugin,)),
        )
        isolated = await bootstrap_application(_context(), registrations=(plugin,))
        return left, right, isolated

    left, right, isolated = asyncio.run(scenario())
    assert left is right
    assert isolated is not left
    assert isolated.registry(NAMES).values == ["value"]
    assert calls == 2


def test_same_context_reentrancy_is_typed_failure() -> None:
    context = _context()

    async def register(_ctx: RegistrationContext) -> None:
        await bootstrap_application(context, registrations=())

    with pytest.raises(BootstrapError) as caught:
        asyncio.run(
            bootstrap_application(context, registrations=(_plugin("pkg.reentrant", register),))
        )
    assert caught.value.code is BootstrapErrorCode.REENTRANCY
    assert caught.value.plugin_id == "pkg.reentrant"


def test_spawned_task_reentrancy_is_typed_failure_without_deadlock() -> None:
    context = _context()

    async def register(_ctx: RegistrationContext) -> None:
        task = asyncio.create_task(bootstrap_application(context, registrations=()))
        await asyncio.wait_for(task, timeout=0.1)

    with pytest.raises(BootstrapError) as caught:
        asyncio.run(
            bootstrap_application(
                context,
                registrations=(_plugin("pkg.spawned_reentrant", register),),
            )
        )
    assert caught.value.code is BootstrapErrorCode.REENTRANCY
    assert caught.value.plugin_id == "pkg.spawned_reentrant"


def test_discovery_rejects_legacy_entry_point_values() -> None:
    class LegacyPoint:
        name = "legacy"
        value = "legacy:register"
        dist = None

        @staticmethod
        def load():
            return lambda _registry: None

    with pytest.raises(BootstrapError) as caught:
        discover_plugin_registrations(entry_points=(LegacyPoint(),))
    assert caught.value.code is BootstrapErrorCode.INVALID_REGISTRATION


def test_installed_and_explicit_same_registration_executes_once(monkeypatch) -> None:
    calls: list[str] = []
    registration = _plugin(
        "pkg.same_source",
        lambda context: (
            calls.append("register"),
            context.registry(NAMES).register("same"),
        ),
    )
    module_name = "tests_same_source_plugin"
    module = ModuleType(module_name)
    module.PLUGIN_REGISTRATION = registration
    monkeypatch.setitem(sys.modules, module_name, module)

    class InstalledPoint:
        name = "same-source"
        value = f"{module_name}:PLUGIN_REGISTRATION"
        dist = None

        @staticmethod
        def load():
            return registration

    state = asyncio.run(
        bootstrap_application(
            _context(),
            entry_points=(InstalledPoint(),),
            modules=(module_name,),
        )
    )

    assert calls == ["register"]
    assert state.registry(NAMES).values == ["same"]
    assert len(state.provenance) == 1
    assert state.provenance[0].entry_point_name == "same-source"
    assert state.provenance[0].entry_point_value == InstalledPoint.value


def test_distinct_registrations_with_same_plugin_id_are_rejected() -> None:
    first = _plugin("pkg.conflict", lambda _context: None)
    second = _plugin("pkg.conflict", lambda _context: None)

    with pytest.raises(BootstrapError) as caught:
        asyncio.run(bootstrap_application(_context(), registrations=(first, second)))

    assert caught.value.code is BootstrapErrorCode.DUPLICATE_PLUGIN


def test_concurrent_load_failure_poisoning_executes_discovery_once() -> None:
    loads = 0

    class BrokenPoint:
        name = "broken"
        value = "broken:plugin"
        dist = None

        @staticmethod
        def load():
            nonlocal loads
            loads += 1
            raise RuntimeError("broken load")

    async def scenario():
        context = _context()
        results = await asyncio.gather(
            bootstrap_application(context, entry_points=(BrokenPoint(),)),
            bootstrap_application(context, entry_points=(BrokenPoint(),)),
            return_exceptions=True,
        )
        return context, results

    context, results = asyncio.run(scenario())
    assert loads == 1
    assert all(isinstance(item, BootstrapError) for item in results)
    assert all(item.code is BootstrapErrorCode.LOAD for item in results)
    with pytest.raises(BootstrapError, match="context is sealed"):
        context.registry(NAMES)


def test_concurrent_invalid_discovery_poisoning_executes_discovery_once() -> None:
    loads = 0

    class InvalidPoint:
        name = "invalid"
        value = "invalid:plugin"
        dist = None

        @staticmethod
        def load():
            nonlocal loads
            loads += 1
            return object()

    async def scenario():
        context = _context()
        results = await asyncio.gather(
            bootstrap_application(context, entry_points=(InvalidPoint(),)),
            bootstrap_application(context, entry_points=(InvalidPoint(),)),
            return_exceptions=True,
        )
        return context, results

    context, results = asyncio.run(scenario())
    assert loads == 1
    assert all(isinstance(item, BootstrapError) for item in results)
    assert all(item.code is BootstrapErrorCode.INVALID_REGISTRATION for item in results)
    with pytest.raises(BootstrapError, match="context is sealed"):
        context.registry(NAMES)


def test_duplicate_dependency_declaration_is_invalid() -> None:
    dependency = _plugin("pkg.base", lambda _context: None)
    duplicate = _plugin(
        "pkg.duplicate",
        lambda _context: None,
        dependencies=(PluginDependency("pkg.base"), PluginDependency("pkg.base")),
    )
    with pytest.raises(BootstrapError) as caught:
        asyncio.run(bootstrap_application(_context(), registrations=(dependency, duplicate)))
    assert caught.value.code is BootstrapErrorCode.INVALID_REGISTRATION


def test_published_concrete_registries_are_sealed() -> None:
    state = asyncio.run(bootstrap_application(_context(), registrations=()))
    with pytest.raises(RuntimeError, match="component registry is sealed"):
        state.bundle.components.register_component_type("pkg.X", lambda _params: object())
    with pytest.raises(RuntimeError, match="experiment registry is sealed"):
        state.bundle.experiment_packages.register_package(
            "pkg", __import__("types"), (), "analysis", "training", "config"
        )
    with pytest.raises(RuntimeError, match="training method registry is sealed"):
        state.bundle.training_programs.register_program(object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="analysis recipe registry is sealed"):
        state.bundle.analysis_recipes.register("pkg.analysis", lambda _context: None)
    with pytest.raises(RuntimeError, match="figure registry is sealed"):
        state.bundle.figures.register_template(type("Template", (), {"name": "pkg.template"})())


def test_published_registry_metadata_is_detached() -> None:
    class Params(BaseModel):
        value: int = 1

    component_meta = declare_component(
        name="pkg.Snapshot",
        category="Original",
        description="snapshot",
        param_schema=[],
        input_ports=[],
        output_ports=[],
        builder=lambda _params: object(),
    )
    routing = {"paths": ["original"]}
    template = SimpleNamespace(name="pkg.template", metadata={"values": ["original"]})
    piece = SimpleNamespace(name="pkg.piece", style={"values": ["original"]})
    schema = EvaluationAuthoringSchema(
        schema_id="pkg.spec.evaluation.snapshot",
        schema_version="pkg.spec.evaluation.snapshot.v1",
        params_model=Params,
        axis_profiles=({"condition": ("original",)},),
    )

    def register(context: RegistrationContext) -> None:
        context.registry(COMPONENTS).register(component_meta)
        context.registry(EXPERIMENT_PACKAGES).register_package(
            "pkg", __import__("types"), (), "analysis", "training", "config", routing
        )
        context.registry(FIGURES).register_template(template)
        context.registry(FIGURES).register_piece(piece)
        context.registry(EVALUATION_RECIPES).register_authoring_schema("pkg.snapshot", schema)

    plugin = PluginRegistration(
        PluginDeclaration(
            "pkg.snapshots",
            "1",
            1,
            families=tuple(
                FamilyRequirement(key.family)
                for key in (COMPONENTS, EXPERIMENT_PACKAGES, FIGURES, EVALUATION_RECIPES)
            ),
        ),
        register,
    )
    state = asyncio.run(bootstrap_application(_context(), registrations=(plugin,)))

    with pytest.raises(FrozenInstanceError):
        component_meta.category = "mutated-input"
    routing["paths"].append("mutated-input")
    template.metadata["values"].append("mutated-input")
    piece.style["values"].append("mutated-input")
    schema.axis_profiles[0]["condition"] = ("mutated-input",)

    returned_component = state.bundle.components.get("pkg.Snapshot")
    assert returned_component is not None
    with pytest.raises(FrozenInstanceError):
        returned_component.category = "mutated-output"
    package = state.bundle.experiment_packages.get_package_metadata("pkg")
    package.figure_routing["paths"].append("mutated-output")
    returned_template = state.bundle.figures.template("pkg.template")
    returned_template.metadata["values"].append("mutated-output")
    returned_piece = state.bundle.figures.piece("pkg.piece")
    returned_piece.style["values"].append("mutated-output")
    returned_schema = state.bundle.evaluation_recipes.authoring_schema("pkg.snapshot")
    assert returned_schema is not None
    returned_schema.axis_profiles[0]["condition"] = ("mutated-output",)

    assert state.bundle.components.get("pkg.Snapshot").category == "Original"
    assert state.bundle.experiment_packages.get_figure_routing("pkg") == {"paths": ["original"]}
    assert state.bundle.figures.template("pkg.template").metadata == {"values": ["original"]}
    assert state.bundle.figures.piece("pkg.piece").style == {"values": ["original"]}
    assert state.bundle.evaluation_recipes.authoring_schema("pkg.snapshot").axis_profiles == (
        {"condition": ("original",)},
    )


def test_component_and_plugin_namespace_collisions_are_typed() -> None:
    component_plugin = PluginRegistration(
        PluginDeclaration(
            "pkg.component",
            "1",
            1,
            families=(FamilyRequirement(COMPONENTS.family),),
        ),
        lambda context: context.registry(COMPONENTS).register_component_type(
            "Gain", lambda _params: object()
        ),
    )
    with pytest.raises(BootstrapError) as component_error:
        asyncio.run(bootstrap_application(_context(), registrations=(component_plugin,)))
    assert component_error.value.code is BootstrapErrorCode.NAMESPACE_COLLISION

    def package_plugin(plugin_id: str) -> PluginRegistration:
        return PluginRegistration(
            PluginDeclaration(
                plugin_id,
                "1",
                1,
                families=(FamilyRequirement(EXPERIMENT_PACKAGES.family),),
            ),
            lambda context: context.registry(EXPERIMENT_PACKAGES).register_package(
                "pkg", __import__("types"), (), "analysis", "training", "config"
            ),
        )

    with pytest.raises(BootstrapError) as package_error:
        asyncio.run(
            bootstrap_application(
                _context(), registrations=(package_plugin("pkg.one"), package_plugin("pkg.two"))
            )
        )
    assert package_error.value.code is BootstrapErrorCode.NAMESPACE_COLLISION


def test_runpod_smoke_fixture_derives_program_preparation_facet() -> None:
    from feedbax.training.runpod_smoke_fixture import METHOD_REF

    state = asyncio.run(
        bootstrap_application(
            _context(),
            entry_points=(),
            modules=("feedbax.training.runpod_smoke_fixture",),
        )
    )

    assert state.registry(TRAINING_PROGRAMS).program(METHOD_REF) is not None
    assert state.bundle.execution_preparations.get(METHOD_REF) is not None
