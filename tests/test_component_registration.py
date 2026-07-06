from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import pytest

from feedbax.component_registry import (
    ComponentRegistry,
    ComponentMigration,
    ComponentMigrationPack,
    get_component_registry,
    register_component_type,
)
from feedbax.runtime.channel import Channel
from feedbax.runtime.components import Gain
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.contracts.graphs.builders import build_component
from feedbax.runtime.graph import Component
from feedbax.contracts.graphs.serialization import spec_to_graph


class _PrototypeSource(Component):
    input_ports = ()
    output_ports = ("signal",)

    def __call__(self, inputs, state, *, key):
        return {"signal": jnp.ones((3,))}, state


def _single_node_spec(
    component_type: str,
    params: dict[str, Any] | None = None,
    *,
    param_schema_version: str | None = None,
) -> GraphSpec:
    return GraphSpec(
        nodes={
            "component": ComponentSpec(
                type=component_type,
                params=params or {},
                param_schema_version=param_schema_version,
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("component", "input")},
        output_bindings={"output": ("component", "output")},
    )


def test_programmatic_component_registration_materializes_via_spec_to_graph() -> None:
    register_component_type(
        "TestProgrammaticGain",
        lambda params: Gain(gain=float(params["gain"])),
        category="Test",
        description="Test-only gain.",
        param_schema=[{"name": "gain", "type": "float", "default": 3.0}],
        input_ports=["input"],
        output_ports=["output"],
        provenance="test-suite",
    )

    registry = get_component_registry()
    meta = registry.get("TestProgrammaticGain")
    assert meta is not None
    assert meta.default_params == {"gain": 3.0}
    assert meta.input_ports == ["input"]
    assert meta.output_ports == ["output"]
    assert meta.provenance == "test-suite"

    definition = next(item for item in registry.list_all() if item.name == "TestProgrammaticGain")
    assert definition.default_params == {"gain": 3.0}
    assert definition.input_ports == ["input"]
    assert definition.output_ports == ["output"]
    assert definition.provenance == "test-suite"

    graph = spec_to_graph(_single_node_spec("TestProgrammaticGain", {"gain": 4.0}))

    component = graph.nodes["component"]
    assert isinstance(component, Gain)
    assert component.gain == 4.0

    default_graph = spec_to_graph(_single_node_spec("TestProgrammaticGain"))
    default_component = default_graph.nodes["component"]
    assert isinstance(default_component, Gain)
    assert default_component.gain == 3.0


def test_registered_component_output_prototype_feeds_stateful_materialization() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_component_type(
        "TestPrototypeSource",
        lambda params: _PrototypeSource(),
        category="Test",
        description="Source with registry-owned output prototype.",
        param_schema=[{"name": "width", "type": "int", "default": 3}],
        input_ports=[],
        output_ports=["signal"],
        output_prototype_fn=lambda params, inputs: {
            "signal": jnp.zeros((int(params.get("width", 3)),))
        },
    )
    spec = GraphSpec(
        nodes={
            "source": ComponentSpec(
                type="TestPrototypeSource",
                params={},
                input_ports=[],
                output_ports=["signal"],
            ),
            "delay": ComponentSpec(
                type="Channel",
                params={"delay": 1, "add_noise": False},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="source",
                source_port="signal",
                target_node="delay",
                target_port="input",
            )
        ],
        output_ports=["output"],
        output_bindings={"output": ("delay", "output")},
    )

    graph = spec_to_graph(spec, registry)

    assert isinstance(graph.nodes["delay"], Channel)
    assert graph.nodes["delay"].input_proto.shape == (3,)


def test_feedbax_component_meta_rejects_output_prototype_mutation() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    meta = registry.get("FixedField")
    assert meta is not None
    assert meta.provenance == "feedbax"
    original = meta.output_prototype_fn

    with pytest.raises(AttributeError, match="Feedbax-owned ComponentMeta.output_prototype_fn"):
        meta.output_prototype_fn = lambda params, inputs: {"force": inputs["force"]}

    assert meta.output_prototype_fn is original


def test_builtin_registry_has_explicit_builders_and_consistent_port_metadata() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    for name in registry.names():
        meta = registry.get(name)
        assert meta is not None
        assert callable(meta.builder), f"{name} has no explicit builder contract"
        if meta.port_types is not None:
            assert set(meta.port_types.inputs) == set(meta.input_ports), name
            assert set(meta.port_types.outputs) == set(meta.output_ports), name


def test_unsupported_builtin_builder_contract_fails_clearly() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    meta = registry.get("MomentArmProjection")
    assert meta is not None
    assert callable(meta.builder)
    assert getattr(meta.builder, "_feedbax_unsupported_builder", False)
    assert "MomentArmProjection" not in registry.executable_names()

    with pytest.raises(NotImplementedError, match="display-only abstraction"):
        build_component("projection", "MomentArmProjection", {}, component_registry=registry)


def test_entry_point_component_registration_records_package_provenance() -> None:
    def registrar(component_registry: ComponentRegistry) -> None:
        component_registry.register_component_type(
            "TestEntryPointGain",
            lambda params: Gain(gain=float(params.get("gain", 1.0))),
            category="Test",
            param_schema=[{"name": "gain", "type": "float", "default": 2.0}],
            input_ports=["input"],
            output_ports=["output"],
        )

    class FakeDist:
        metadata = {"Name": "feedbax-test-plugin"}

    class FakeEntryPoint:
        name = "feedbax_test_plugin"
        dist = FakeDist()

        def load(self) -> Callable[[ComponentRegistry], None]:
            return registrar

    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.discover_entry_point_components(entry_points=[FakeEntryPoint()])

    meta = registry.get("TestEntryPointGain")
    assert meta is not None
    assert meta.provenance == "package:feedbax-test-plugin"

    definition = next(item for item in registry.list_all() if item.name == "TestEntryPointGain")
    assert definition.default_params == {"gain": 2.0}
    assert definition.input_ports == ["input"]
    assert definition.output_ports == ["output"]
    assert definition.provenance == "package:feedbax-test-plugin"

    graph = spec_to_graph(_single_node_spec("TestEntryPointGain", {"gain": 5.0}), registry)
    assert isinstance(graph.nodes["component"], Gain)
    assert graph.nodes["component"].gain == 5.0


def test_user_component_file_registers_palette_metadata_and_builder(
    tmp_path: Path,
) -> None:
    component_file = tmp_path / "custom_gain.py"
    component_file.write_text(
        textwrap.dedent(
            """
            from feedbax.runtime.components import Gain


            class UserGain(Gain):
                pass


            UserGain._feedbax_component_meta = {
                "name": "TestUserFileGain",
                "category": "User",
                "description": "Loaded from a user component file.",
                "param_schema": [{"name": "gain", "type": "float", "default": 6.0}],
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
            """
        )
    )

    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.load_user_components(tmp_path)

    definition = next(item for item in registry.list_all() if item.name == "TestUserFileGain")
    assert definition.default_params == {"gain": 6.0}
    assert definition.input_ports == ["input"]
    assert definition.output_ports == ["output"]
    assert definition.provenance == f"file:{component_file}"

    graph = spec_to_graph(_single_node_spec("TestUserFileGain", {"gain": 7.0}), registry)
    assert isinstance(graph.nodes["component"], Gain)
    assert graph.nodes["component"].gain == 7.0


def test_unknown_component_error_names_type_and_known_registry_contents() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(_single_node_spec("DefinitelyUnknownComponent"), registry)

    message = str(exc_info.value)
    assert "DefinitelyUnknownComponent" in message
    assert "Known component types:" in message
    assert "Gain" in message


def test_builtin_component_rename_migration_materializes_registered_target() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_migration(
        ComponentMigration(
            source_type="LegacyGain",
            target_type="Gain",
            owner="feedbax",
            migration_id="feedbax.component.LegacyGain-to-Gain.v1",
            target_param_schema_version="1",
            description="Test-only built-in rename edge.",
        )
    )

    graph = spec_to_graph(_single_node_spec("LegacyGain", {"gain": 4.0}), registry)

    component = graph.nodes["component"]
    assert isinstance(component, Gain)
    assert component.gain == 4.0
    gain = next(item for item in registry.list_all() if item.name == "Gain")
    assert [migration.source_type for migration in gain.migrations] == ["LegacyGain"]
    assert gain.migrations[0].owner == "feedbax"


def test_component_parameter_schema_migration_renames_required_parameter() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_migration(
        ComponentMigration(
            source_type="Gain",
            target_type="Gain",
            owner="feedbax",
            source_param_schema_version="legacy-scale",
            target_param_schema_version="1",
            migration_id="feedbax.component.Gain.params.legacy-scale-to-1",
            migrate_params=lambda params: {"gain": params["scale"]},
        )
    )

    graph = spec_to_graph(
        _single_node_spec("Gain", {"scale": 8.0}, param_schema_version="legacy-scale"),
        registry,
    )

    component = graph.nodes["component"]
    assert isinstance(component, Gain)
    assert component.gain == 8.0


def test_downstream_migration_pack_can_migrate_owned_component_id() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_component_type(
        "rlrmp.CurrentGain",
        lambda params: Gain(gain=float(params["gain"])),
        owner="rlrmp",
        provenance="package:rlrmp",
        param_schema=[{"name": "gain", "type": "float", "default": 1.0, "required": True}],
        input_ports=["input"],
        output_ports=["output"],
    )
    registry.register_migration_pack(
        ComponentMigrationPack(
            owner="rlrmp",
            package="rlrmp",
            migrations=(
                ComponentMigration(
                    source_type="rlrmp.LegacyGain",
                    target_type="rlrmp.CurrentGain",
                    owner="rlrmp",
                    migration_id="rlrmp.component.LegacyGain-to-CurrentGain.v1",
                ),
            ),
        )
    )

    graph = spec_to_graph(_single_node_spec("rlrmp.LegacyGain", {"gain": 9.0}), registry)

    component = graph.nodes["component"]
    assert isinstance(component, Gain)
    assert component.gain == 9.0
    definition = next(item for item in registry.list_all() if item.name == "rlrmp.CurrentGain")
    assert definition.owner == "rlrmp"
    assert definition.identity is not None
    assert definition.identity.provenance_kind == "package"
    assert definition.migrations[0].source_type == "rlrmp.LegacyGain"


def test_absent_downstream_owner_fails_with_actionable_message() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(_single_node_spec("rlrmp.LegacyGain", {"gain": 1.0}), registry)

    message = str(exc_info.value)
    assert "owner='rlrmp'" in message
    assert "migration pack" in message
    assert "rlrmp.LegacyGain" in message


def test_loaded_owner_without_migration_edge_fails_with_version_context() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_component_type(
        "rlrmp.CurrentGain",
        lambda params: Gain(gain=float(params["gain"])),
        owner="rlrmp",
        provenance="package:rlrmp",
        param_schema=[{"name": "gain", "type": "float", "default": 1.0}],
        input_ports=["input"],
        output_ports=["output"],
    )

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(_single_node_spec("rlrmp.UnsupportedGain", {"gain": 1.0}), registry)

    message = str(exc_info.value)
    assert "No component migration registered" in message
    assert "owner='rlrmp'" in message
    assert "rlrmp.UnsupportedGain" in message


def test_unsupported_component_parameter_schema_fails_with_current_version() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(
            _single_node_spec("Gain", {"scale": 1.0}, param_schema_version="unsupported"),
            registry,
        )

    message = str(exc_info.value)
    assert "No component migration registered" in message
    assert "source_param_schema_version='unsupported'" in message
    assert "current_param_schema_version='1'" in message


def test_elementwise_affine_modulator_is_builtin_component() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    meta = registry.get("ElementwiseAffineModulator")

    assert meta is not None
    assert meta.default_params["signal_shape"] == [1]
    assert meta.default_params["baseline"] == 1.0
    assert meta.default_params["gain_init"] == 0.0
    assert meta.default_params["bias_init"] == 0.0
    assert meta.input_ports == ["signal", "modulator", "scale", "bias"]
    assert meta.output_ports == ["output"]
