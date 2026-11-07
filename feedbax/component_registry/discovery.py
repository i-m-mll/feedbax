"""Bounded discovery and compilation of typed component owners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

from feedbax.contracts.portable import ComponentOwner, GraphTemplate


@dataclass(frozen=True)
class ComponentPack:
    """Owners and their history in one stable portable namespace."""

    namespace: str
    modules: tuple[str | ModuleType, ...] = ()
    histories: Mapping[type, Any] = field(default_factory=dict)
    presentation: Mapping[type, Any] = field(default_factory=dict)
    components: tuple[type[ComponentOwner], ...] = ()
    sources: tuple[Path, ...] = ()

    def __post_init__(self):
        if not self.namespace or self.namespace.startswith(".") or self.namespace.endswith("."):
            raise ValueError("Component pack requires a stable namespace")
        frame = inspect.currentframe().f_back.f_back
        filename = frame.f_code.co_filename
        if not filename.startswith("<"):
            object.__setattr__(self, "sources", (*self.sources, Path(filename).resolve()))


def discover_modules(*modules: str | ModuleType, namespace: str, histories=None) -> ComponentPack:
    loaded = [import_module(module) if isinstance(module, str) else module for module in modules]
    classes = {
        value
        for module in loaded
        for value in vars(module).values()
        if isinstance(value, type)
        and value.__module__ == module.__name__
        and issubclass(value, ComponentOwner)
        and value not in (ComponentOwner, GraphTemplate)
        and not inspect.isabstract(value)
    }
    return ComponentPack(
        namespace=namespace,
        histories=histories or {},
        components=tuple(sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__))),
    )


def source_path(obj: Any) -> Path | None:
    """Locate an owner-supplied implementation, including a wrapped callable."""
    while isinstance(obj, partial):
        obj = obj.func
    try:
        path = inspect.getsourcefile(inspect.unwrap(obj) if callable(obj) else obj)
    except (TypeError, OSError):
        return None
    if path is None or path.startswith("<"):
        return None
    return Path(path).resolve()


def compile_owner(cls, *, namespace, history=None, registry=None):
    from feedbax.contracts.portable import ComponentOwner, GraphTemplate
    from feedbax.contracts.portable import (
        ComponentOptions,
        configuration_model,
        configuration_paths,
    )
    from .history import History

    if not issubclass(cls, ComponentOwner) or cls in (ComponentOwner, GraphTemplate) or inspect.isabstract(cls):
        raise TypeError(f"{cls!r} must be a concrete portable ComponentOwner")
    if not namespace or namespace.startswith(".") or namespace.endswith("."):
        raise ValueError("Component pack requires a stable namespace")
    options = (cls.component_options or ComponentOptions()).as_kwargs()
    history = history or History()
    configuration_paths(cls)
    model = configuration_model(cls)
    ports = cls.port_contract(registry=registry)
    if issubclass(cls, GraphTemplate):
        options["template_graph"] = cls.expand(model())
    elif options.get("template_graph") is not None and callable(options["template_graph"]):
        options["template_graph"] = options["template_graph"]()
    input_ports, output_ports = tuple(ports.inputs), tuple(ports.outputs)
    port_types = ports.types
    if ports.dynamic is not None:
        from feedbax.contracts.component import derive_dynamic_port_layout
        names = ((*ports.dynamic.input_name_params, *ports.dynamic.output_name_params)
                 if ports.dynamic.count_mode == "names" else (ports.dynamic.count_param,))
        if all(not model.model_fields[name].is_required() for name in names):
            defaults = {name: model.model_fields[name].get_default(call_default_factory=True) for name in names}
            layout = derive_dynamic_port_layout(ports.dynamic, defaults)
            input_ports, output_ports = layout.input_ports, layout.output_ports
            port_types = type(port_types)(
                inputs={name: ports.inputs[name] if name in ports.inputs else ports.dynamic.dynamic_port_type for name in input_ports},
                outputs={name: ports.outputs[name] if name in ports.outputs else ports.dynamic.dynamic_port_type for name in output_ports},
            )
    from .declarations import (
        ComponentAuthoringFacet, ComponentCompilerFacet, ComponentDeclaration,
        ComponentParamContract, ComponentRuntimeFacet, ComponentSerializationFacet,
        ComponentStudioFacet, ComponentTrainingFacet, DeclaredComponent,
    )

    name = f"{namespace}.{cls.__name__}"
    contract = ComponentParamContract(
        component_type=name,
        model=model,
        revision=history.revision,
        runtime_type=cls,
        build_context_fields=frozenset(options.pop("build_context_fields")),
    )
    prototype_bindings = options.pop("build_output_prototypes") or {}
    if prototype_bindings:
        if options["output_prototype_fn"] is None or set(prototype_bindings.values()) - set(output_ports):
            raise ValueError(f"Component {name!r} must infer each output prototype it builds from")
        if set(prototype_bindings) & (set(model.model_fields) | contract.build_context_fields):
            raise ValueError(f"Component {name!r} cannot author compiler-provided prototypes")
    authoring = ComponentAuthoringFacet(
        options.pop("template_graph"), options.pop("template_id"), options.pop("template_kind")
    )
    training = ComponentTrainingFacet(options.pop("trainable_by_default"))
    if options["is_composite"] and options["interior_domain"] is None:
        options["interior_domain"] = options["domain"]
    group = next((
        group for prefix, group in (
            ("feedbax.tasks.", "Tasks"), ("feedbax.control.", "Control"),
            ("feedbax.acausal.", "Acausal"), ("feedbax.compiler.", "Templates"),
            ("feedbax.components.equinox", "Equinox"), ("feedbax.mechanics.", "Mechanics"),
            ("feedbax.models.", "Models"), ("feedbax.runtime.", "Runtime"),
        ) if cls.__module__.startswith(prefix)
    ), namespace)
    meta = DeclaredComponent(
        declaration=ComponentDeclaration(
            type_id=name,
            revision=history.revision,
            owner=namespace,
        ),
        params=contract,
        compiler=ComponentCompilerFacet(
            parameter_contract=contract,
            input_ports=input_ports,
            output_ports=output_ports,
            port_types=port_types,
            dynamic_port_policy=ports.dynamic,
            build_output_prototypes=prototype_bindings,
            **options,
        ),
        serialization=ComponentSerializationFacet(history.revision),
        runtime=ComponentRuntimeFacet(contract.build),
        authoring=authoring,
        studio=ComponentStudioFacet(
            group, (inspect.getdoc(cls) or "").split("\n\n", 1)[0]
        ),
        training=training,
        owner_class=cls,
    )
    sources = (
        source_path(cls),
        *((source_path(cls.config),) if cls.config else ()),
        *(
            source_path(binding.group)
            for binding in cls.config_bindings
            if binding.group is not None
        ),
        *((history.source,) if history.source is not None else ()),
    )
    return (meta,), sources, history.compile(namespace, cls.__name__)
