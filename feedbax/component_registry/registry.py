from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

from feedbax.contracts.component import ComponentDefinition, PortType, PortTypeSpec
from feedbax.contracts.graph import ParamSchema

from .builtins import register_builtin_components
from .meta import ComponentMeta


class ComponentRegistry:
    def __init__(self) -> None:
        self._components: Dict[str, ComponentMeta] = {}
        self._register_builtins()
        self.load_user_components(Path.home() / '.feedbax' / 'components')

    def _register_builtins(self) -> None:
        register_builtin_components(self)

    def register(self, meta: ComponentMeta) -> None:
        self._components[meta.name] = meta

    def get(self, name: str) -> Optional[ComponentMeta]:
        return self._components.get(name)

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

        for py_file in path.glob('*.py'):
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):
                obj = getattr(module, name)
                meta = getattr(obj, '_feedbax_component_meta', None)
                if not isinstance(meta, dict):
                    continue
                self.register(
                    ComponentMeta(
                        name=meta.get('name', name),
                        category=meta.get('category', 'Custom'),
                        description=meta.get('description', ''),
                        param_schema=[ParamSchema(**schema) for schema in meta.get('param_schema', [])],
                        input_ports=list(meta.get('input_ports', [])),
                        output_ports=list(meta.get('output_ports', [])),
                        icon=meta.get('icon', 'box'),
                        is_composite=bool(meta.get('is_composite', False)),
                        port_types=(
                            PortTypeSpec(
                                inputs={
                                    key: PortType(**value)
                                    for key, value in meta.get('port_types', {}).get('inputs', {}).items()
                                },
                                outputs={
                                    key: PortType(**value)
                                    for key, value in meta.get('port_types', {}).get('outputs', {}).items()
                                },
                            )
                            if meta.get('port_types')
                            else None
                        ),
                    )
                )

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
        )
