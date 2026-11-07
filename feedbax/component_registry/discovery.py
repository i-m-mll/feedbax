"""Explicit component ownership and bounded module discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace, field
from functools import partial
from importlib import import_module
import inspect
from pathlib import Path
from types import ModuleType, UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    TypedDict,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, create_model

from feedbax.contracts.migrations import ComponentMigration

from .declarations import DeclaredComponent, declare_component

_C = TypeVar("_C", bound=type)
_MARKER = "_feedbax_component_owner"


class ComponentAliasOptions(TypedDict, total=False):
    """Display and domain overrides for a durable alternate component ID."""

    domain: str
    category: str
    icon: str


@dataclass(frozen=True)
class _ClassDeclaration:
    type_id: str
    options: Mapping[str, Any]
    sources: tuple[Path, ...]
    configure: Callable[[type], Mapping[str, Any]] | None
    aliases: Mapping[str, ComponentAliasOptions]
    migrations: tuple[ComponentMigration, ...]


def component(
    type_id: str,
    *,
    sources: tuple[Path, ...] = (),
    configure: Callable[[type], Mapping[str, Any]] | None = None,
    aliases: Mapping[str, ComponentAliasOptions] | None = None,
    migrations: tuple[ComponentMigration, ...] = (),
    **options: Any,
) -> Callable[[_C], _C]:
    """Mark a concrete class for discovery, with optional declaration overrides.

    Ports default to class port tuples. Straightforward keyword constructor
    settings default to a model derived from their annotations and defaults.
    Other options are passed to the declaration compiler.
    """

    def mark(cls: _C) -> _C:
        if not isinstance(cls, type) or inspect.isabstract(cls):
            raise TypeError(f"Component owner {cls!r} must be a concrete class")
        if _MARKER in vars(cls):
            raise ValueError(f"Component owner {cls.__qualname__} already has a declaration")
        setattr(
            cls,
            _MARKER,
            _ClassDeclaration(
                type_id,
                options,
                sources,
                aliases=aliases or {},
                configure=configure,
                migrations=migrations,
            ),
        )
        return cls

    return mark


DefinitionFactory = Callable[[], DeclaredComponent]
PackComponent = type | DeclaredComponent | DefinitionFactory


@dataclass(frozen=True)
class ComponentPack:
    """One batch of component definitions and historical migration routes.

    Extra sources name absolute paths that feed this batch, such as a generator
    or a plugin entry point that composes owner modules.
    """

    namespace: str | None = None
    modules: tuple[str | ModuleType, ...] = ()
    histories: Mapping[type, Any] = field(default_factory=dict)
    presentation: Mapping[type, Any] = field(default_factory=dict)
    components: tuple[PackComponent, ...] = ()
    migrations: tuple[ComponentMigration, ...] = ()
    sources: tuple[Path, ...] = ()

    def __post_init__(self):
        frame = inspect.currentframe().f_back.f_back
        filename = frame.f_code.co_filename
        if not filename.startswith("<"):
            object.__setattr__(self, "sources", (*self.sources, Path(filename).resolve()))


def discover_modules(
    *modules: str | ModuleType,
    namespace: str | None = None,
    histories: Mapping[type, Any] | None = None,
) -> ComponentPack:
    from feedbax.contracts.portable import ComponentOwner, GraphTemplate

    loaded = [import_module(module) if isinstance(module, str) else module for module in modules]
    classes = {
        value
        for module in loaded
        for value in vars(module).values()
        if isinstance(value, type)
        and value.__module__ == module.__name__
        and ((_MARKER in vars(value)) if namespace is None else issubclass(value, ComponentOwner) and value not in (ComponentOwner, GraphTemplate))
        and not inspect.isabstract(value)
    }
    return ComponentPack(
        namespace=namespace,
        histories=histories or {},
        components=tuple(sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__))),
    )


def _simple_annotation(annotation: Any) -> bool:
    if annotation in (bool, int, float, str):
        return True
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _simple_annotation(args[0])
    if origin is Literal:
        return all(isinstance(arg, str) for arg in args)
    if origin in (Union, UnionType):
        present = [arg for arg in args if arg is not type(None)]
        return len(args) == 2 and len(present) == 1 and _simple_annotation(present[0])
    return False


def _constructor_model(cls: type, type_id: str, options: Mapping[str, Any]) -> type[BaseModel]:
    context = set(options.get("build_context_fields", ()))
    context.update(options.get("build_output_prototypes") or ())
    signature = inspect.signature(cls.__init__)
    params = tuple(signature.parameters.values())[1:]
    # Resolve annotations only when there are authored fields. Default
    # object.__init__ accepts no authored arguments.
    hints = (
        get_type_hints(cls.__init__, include_extras=True)
        if cls.__init__ is not object.__init__
        else {}
    )
    fields: dict[str, Any] = {}
    for param in params if cls.__init__ is not object.__init__ else ():
        if param.name in context:
            continue
        annotation = hints.get(param.name, param.annotation)
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY) or not (
            _simple_annotation(annotation)
        ):
            raise TypeError(
                f"Component {type_id!r} needs an explicit parameter model "
                f"for constructor parameter {param.name!r}"
            )
        fields[param.name] = (annotation, ... if param.default is param.empty else param.default)
    return create_model(
        f"{type_id.replace('.', '_')}Params",
        __config__=ConfigDict(extra="forbid", strict=True),
        **fields,
    )


def _port_names(cls: type, attribute: str) -> tuple[str, ...]:
    ports = getattr(cls, attribute, ())
    if isinstance(ports, tuple) and all(isinstance(port, str) for port in ports):
        return ports
    raise TypeError(f"Component {cls.__qualname__} needs explicit {attribute}")


def compile_owner(
    cls: type,
    *,
    namespace: str | None = None,
    history: Any = None,
    registry: Any = None,
) -> tuple[tuple[DeclaredComponent, ...], tuple[Path | None, ...], tuple[ComponentMigration, ...]]:
    if namespace is not None:
        return _compile_inferred_owner(cls, namespace, history, registry)
    declaration = vars(cls).get(_MARKER)
    if not isinstance(declaration, _ClassDeclaration) or inspect.isabstract(cls):
        raise TypeError(f"Component owner {cls.__qualname__} must declare @component directly")
    options = dict(declaration.options)
    sources: list[Path | None] = [source_path(cls), *declaration.sources]
    if declaration.configure is not None:
        sources.append(source_path(declaration.configure))
        configured = declaration.configure(cls)
        if duplicate := options.keys() & configured.keys():
            raise ValueError(
                f"Component {declaration.type_id!r} configures fields twice: {sorted(duplicate)}"
            )
        options.update(configured)
    for direction in ("input_ports", "output_ports"):
        if direction not in options:
            options[direction] = _port_names(cls, direction)
    options.setdefault("runtime_type", cls)
    configured_migrations = options.pop("migrations", ())
    if configured_migrations and declaration.migrations:
        raise ValueError(f"Component {declaration.type_id!r} configures migrations twice")
    derived = all(
        options.get(key) is None for key in ("param_model", "parameter_fields", "param_schema")
    )
    if derived:
        options["param_model"] = _constructor_model(cls, declaration.type_id, options)
    definitions = [declare_component(name=declaration.type_id, **options)]
    for alias, overrides in sorted(declaration.aliases.items()):
        if extra := overrides.keys() - ComponentAliasOptions.__annotations__.keys():
            raise ValueError(f"Component alias {alias!r} cannot override fields {sorted(extra)}")
        alias_options: dict[str, Any] = dict(options)
        alias_options.update(overrides)
        definitions.append(
            replace(declare_component(name=alias, **alias_options), alias_of=declaration.type_id)
        )
    definitions = [
        replace(meta, owner_class=cls, source_paths=() if derived else meta.source_paths)
        for meta in definitions
    ]
    return (
        tuple(definitions),
        tuple(sources),
        declaration.migrations or tuple(configured_migrations),
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


def _compile_inferred_owner(cls, namespace, history, registry):
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
    forbidden = {
        "name",
        "runtime_type",
        "param_model",
        "parameter_fields",
        "param_schema",
        "constructor",
        "builder",
        "extractor",
        "attribute_paths",
        "param_schema_version",
        "supported_param_schema_versions",
        "migrations",
        "input_ports",
        "output_ports",
        "port_types",
        "dynamic_port_policy",
        "description",
        "category",
        "owner",
        "identity",
        "override_reason",
    }
    if conflict := forbidden & options.keys():
        raise ValueError(f"{cls.__name__} has competing declarations: {sorted(conflict)}")
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
    meta = declare_component(
        name=f"{namespace}.{cls.__name__}",
        runtime_type=cls,
        param_model=model,
        input_ports=input_ports,
        output_ports=output_ports,
        port_types=port_types,
        dynamic_port_policy=ports.dynamic,
        param_schema_version=history.revision,
        supported_param_schema_versions=tuple(range(1, history.revision + 1)),
        owner=namespace,
        description=(inspect.getdoc(cls) or "").split("\n\n", 1)[0],
        category=namespace,
        **options,
    )
    contract = replace(meta.params, portable=True)
    meta = replace(
        meta,
        params=contract,
        compiler=replace(meta.compiler, parameter_contract=contract),
        runtime=replace(meta.runtime, builder=contract.build),
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
