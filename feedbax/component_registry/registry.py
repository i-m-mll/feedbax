from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from feedbax.contracts.component import ComponentDefinition, PortType, PortTypeSpec
from feedbax.contracts.graph import ParamSchema

from .builtins import register_builtin_components
from .meta import ComponentBuilder, ComponentMeta, OutputPrototypeFn


logger = logging.getLogger(__name__)
_DEFAULT_REGISTRY: ComponentRegistry | None = None
_REGISTRATION_PROVENANCE: list[str] = []


@contextmanager
def _registration_provenance(provenance: str | None) -> Iterator[None]:
    if provenance is None:
        yield
        return
    _REGISTRATION_PROVENANCE.append(provenance)
    try:
        yield
    finally:
        _REGISTRATION_PROVENANCE.pop()


def _current_provenance() -> str | None:
    return _REGISTRATION_PROVENANCE[-1] if _REGISTRATION_PROVENANCE else None


class ComponentRegistry:
    def __init__(
        self,
        *,
        load_user_components: bool = True,
        discover_plugins: bool = True,
    ) -> None:
        self._components: Dict[str, ComponentMeta] = {}
        self._register_builtins()
        if discover_plugins:
            self.discover_entry_point_components()
        if load_user_components:
            self.load_user_components(Path.home() / ".feedbax" / "components")

    def _register_builtins(self) -> None:
        register_builtin_components(self)
        from feedbax.serialization_builders import register_builtin_component_builders

        register_builtin_component_builders(self)

    def register(self, meta: ComponentMeta) -> None:
        if meta.provenance is None:
            meta.provenance = _current_provenance()
        self._components[meta.name] = meta

    def register_component_type(
        self,
        name: str,
        builder: ComponentBuilder,
        *,
        category: str = "Custom",
        description: str = "",
        param_schema: Iterable[ParamSchema | dict[str, Any]] = (),
        input_ports: Iterable[str] = (),
        output_ports: Iterable[str] = (),
        icon: str = "box",
        port_types: PortTypeSpec | dict[str, Any] | None = None,
        is_composite: bool = False,
        template_graph: Any = None,
        template_ui_state: Any = None,
        template_id: str | None = None,
        template_kind: str | None = None,
        output_prototype_fn: OutputPrototypeFn | None = None,
        provenance: str | None = None,
    ) -> ComponentMeta:
        if not callable(builder):
            raise TypeError(f"Builder for component type {name!r} must be callable")
        if port_types is not None and not isinstance(port_types, PortTypeSpec):
            port_types = PortTypeSpec.model_validate(port_types)
        meta = ComponentMeta(
            name=name,
            category=category,
            description=description,
            param_schema=[
                schema if isinstance(schema, ParamSchema) else ParamSchema(**schema)
                for schema in param_schema
            ],
            input_ports=list(input_ports),
            output_ports=list(output_ports),
            icon=icon,
            port_types=port_types,
            is_composite=is_composite,
            template_graph=template_graph,
            template_ui_state=template_ui_state,
            template_id=template_id,
            template_kind=template_kind,
            builder=builder,
            output_prototype_fn=output_prototype_fn,
            provenance=provenance or _current_provenance(),
        )
        self.register(meta)
        return meta

    def register_builder(
        self,
        name: str,
        builder: ComponentBuilder,
        *,
        provenance: str | None = None,
    ) -> ComponentMeta:
        if not callable(builder):
            raise TypeError(f"Builder for component type {name!r} must be callable")
        meta = self._components.get(name)
        if meta is None:
            meta = ComponentMeta(
                name=name,
                category="Custom",
                description="",
                param_schema=[],
                input_ports=[],
                output_ports=[],
                builder=builder,
                provenance=provenance or _current_provenance(),
            )
        else:
            meta.builder = builder
            if meta.provenance is None:
                meta.provenance = provenance or _current_provenance()
        self.register(meta)
        return meta

    def get(self, name: str) -> Optional[ComponentMeta]:
        return self._components.get(name)

    def names(self) -> List[str]:
        return sorted(self._components)

    def executable_names(self) -> List[str]:
        return sorted(name for name, meta in self._components.items() if meta.builder is not None)

    def list_all(self) -> List[ComponentDefinition]:
        return [self._to_definition(meta) for meta in self._components.values()]

    def list_by_category(self) -> Dict[str, List[ComponentDefinition]]:
        by_category: Dict[str, List[ComponentDefinition]] = {}
        for meta in self._components.values():
            by_category.setdefault(meta.category, []).append(self._to_definition(meta))
        return by_category

    def load_user_components(self, path: Path) -> None:
        if not path.exists():
            return

        for py_file in path.glob("*.py"):
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):
                obj = getattr(module, name)
                meta = getattr(obj, "_feedbax_component_meta", None)
                if not isinstance(meta, dict):
                    continue
                builder = meta.get("builder")
                if builder is None and callable(obj):

                    def _build_user_component(params: dict[str, Any], component_type: Any = obj):
                        return component_type(**dict(params))

                    builder = _build_user_component
                port_types = meta.get("port_types")
                self.register_component_type(
                    name=meta.get("name", name),
                    builder=builder,
                    category=meta.get("category", "Custom"),
                    description=meta.get("description", ""),
                    param_schema=meta.get("param_schema", []),
                    input_ports=meta.get("input_ports", []),
                    output_ports=meta.get("output_ports", []),
                    icon=meta.get("icon", "box"),
                    is_composite=bool(meta.get("is_composite", False)),
                    port_types=(
                        PortTypeSpec(
                            inputs={
                                key: PortType(**value)
                                for key, value in port_types.get("inputs", {}).items()
                            },
                            outputs={
                                key: PortType(**value)
                                for key, value in port_types.get("outputs", {}).items()
                            },
                        )
                        if isinstance(port_types, dict)
                        else port_types
                    ),
                    output_prototype_fn=meta.get("output_prototype_fn"),
                    provenance=f"file:{py_file}",
                )

    def discover_entry_point_components(
        self,
        entry_point_group: str = "feedbax.plugins",
        entry_points: Iterable[Any] | None = None,
    ) -> None:
        if entry_points is None:
            try:
                entry_points = importlib.metadata.entry_points(group=entry_point_group)
            except TypeError:
                all_entry_points = importlib.metadata.entry_points()
                if hasattr(all_entry_points, "get"):
                    entry_points = all_entry_points.get(entry_point_group, [])
                else:
                    entry_points = [
                        entry_point
                        for entry_point in all_entry_points
                        if entry_point.group == entry_point_group
                    ]

        for entry_point in entry_points:
            provenance = self._entry_point_provenance(entry_point)
            try:
                plugin = entry_point.load()
                registrar = self._component_registrar(plugin)
                if registrar is None:
                    continue
                with _registration_provenance(provenance):
                    registrar(self)
            except Exception as exc:
                logger.warning("Failed to load component entry point %s: %s", provenance, exc)
                continue

    def _component_registrar(self, plugin: Any) -> Any:
        for attr in ("register_feedbax_components", "register_components"):
            registrar = getattr(plugin, attr, None)
            if callable(registrar):
                return registrar
        if not callable(plugin):
            return None
        try:
            signature = inspect.signature(plugin)
        except (TypeError, ValueError):
            return None
        parameter_names = set(signature.parameters)
        if parameter_names & {"component_registry", "components"}:
            return plugin
        return None

    def _entry_point_provenance(self, entry_point: Any) -> str:
        dist = getattr(entry_point, "dist", None)
        if dist is not None:
            metadata = getattr(dist, "metadata", {})
            package_name = metadata.get("Name") if hasattr(metadata, "get") else None
            if package_name:
                return f"package:{package_name}"
        return f"entry-point:{getattr(entry_point, 'name', '<unknown>')}"

    def _to_definition(self, meta: ComponentMeta) -> ComponentDefinition:
        return ComponentDefinition(
            name=meta.name,
            category=meta.category,
            description=meta.description,
            param_schema=meta.param_schema,
            input_ports=meta.input_ports,
            output_ports=meta.output_ports,
            icon=meta.icon,
            default_params=meta.default_params,
            port_types=meta.port_types,
            is_composite=meta.is_composite,
            template_graph=meta.template_graph,
            template_ui_state=meta.template_ui_state,
            template_id=meta.template_id,
            template_kind=meta.template_kind,
            provenance=meta.provenance,
        )


def get_component_registry() -> ComponentRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
        _DEFAULT_REGISTRY = registry
        registry.discover_entry_point_components()
        registry.load_user_components(Path.home() / ".feedbax" / "components")
    return _DEFAULT_REGISTRY


def register_component_type(
    name: str,
    builder: ComponentBuilder,
    *,
    category: str = "Custom",
    description: str = "",
    param_schema: Iterable[ParamSchema | dict[str, Any]] = (),
    input_ports: Iterable[str] = (),
    output_ports: Iterable[str] = (),
    icon: str = "box",
    port_types: PortTypeSpec | dict[str, Any] | None = None,
    is_composite: bool = False,
    output_prototype_fn: OutputPrototypeFn | None = None,
    provenance: str | None = None,
) -> ComponentMeta:
    """Register an executable component type in the process-wide registry."""

    return get_component_registry().register_component_type(
        name=name,
        builder=builder,
        category=category,
        description=description,
        param_schema=param_schema,
        input_ports=input_ports,
        output_ports=output_ports,
        icon=icon,
        port_types=port_types,
        is_composite=is_composite,
        output_prototype_fn=output_prototype_fn,
        provenance=provenance,
    )
