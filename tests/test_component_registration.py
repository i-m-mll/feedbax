from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Callable

import pytest

from feedbax.component_registry import ComponentRegistry, register_component_type
from feedbax.components import Gain
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.serialization import spec_to_graph


def _single_node_spec(component_type: str, params: dict[str, Any] | None = None) -> GraphSpec:
    return GraphSpec(
        nodes={
            "component": ComponentSpec(
                type=component_type,
                params=params or {},
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
        param_schema=[{"name": "gain", "type": "float", "default": 3.0, "required": True}],
        input_ports=["input"],
        output_ports=["output"],
        provenance="test-suite",
    )

    graph = spec_to_graph(_single_node_spec("TestProgrammaticGain", {"gain": 4.0}))

    component = graph.nodes["component"]
    assert isinstance(component, Gain)
    assert component.gain == 4.0


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
            from feedbax.components import Gain


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
