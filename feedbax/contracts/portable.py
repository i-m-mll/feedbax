"""Typed authoring and runtime storage bindings, independent of registry assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from typing import Any, ClassVar, get_type_hints
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, create_model

from feedbax.contracts.component import DynamicPortPolicy, PortType, PortTypeSpec


@dataclass(frozen=True)
class Ports:
    """One authority for fixed port names and their types."""

    inputs: Mapping[str, PortType]
    outputs: Mapping[str, PortType]
    dynamic: DynamicPortPolicy | None = None

    def __post_init__(self):
        for names in (self.inputs, self.outputs):
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError("Port names must be nonempty strings")
        if self.dynamic is not None and self.dynamic.count_mode != "names":
            if (
                tuple(self.inputs) != self.dynamic.fixed_input_ports
                or tuple(self.outputs) != self.dynamic.fixed_output_ports
            ):
                raise ValueError("Dynamic port fixed names must agree with the typed ports")

    @property
    def types(self):
        return PortTypeSpec(inputs=dict(self.inputs), outputs=dict(self.outputs))


@dataclass(frozen=True)
class ConfigBinding:
    """Bind a field, or all fields of a typed group, to actual runtime storage.

    A group stored in external state uses a path such as ``params_index.init``.
    Group member names are derived from its typed fields.
    """

    path: str
    name: str | None = None
    group: type | None = None
    argument: str | None = None

    def __post_init__(self):
        if (self.name is None) == (self.group is None):
            raise ValueError("A configuration binding names either one field or one typed group")

    @property
    def projections(self):
        if self.name is not None:
            return {self.name: self.path}
        import dataclasses

        names = (
            self.group.model_fields
            if issubclass(self.group, BaseModel)
            else (item.name for item in dataclasses.fields(self.group))
        )
        return {name: f"{self.path}.{name}" for name in names}

    def materialize(self, config):
        if self.group is None:
            return getattr(config, self.name)
        return self.group(**{name: getattr(config, name) for name in self.projections})


def default_configuration(cls, **values):
    """Apply the single host authoring contract to direct constructor settings."""
    return validate_authoring(configuration_model(cls), values)


def materialize_configuration(cls, config):
    values = {name: getattr(config, name) for name in type(config).model_fields}
    for binding in cls.config_bindings:
        if binding.group is not None and binding.argument is not None:
            for name in binding.projections:
                values.pop(name)
            values[binding.argument] = binding.materialize(config)
    return values


def install_ports(cls):
    ports = vars(cls).get("ports")
    if ports is None:
        return
    if not isinstance(ports, Ports):
        raise TypeError("Component.ports must be a Ports declaration")
    if {"input_ports", "output_ports"} & vars(cls).keys():
        raise ValueError(f"{cls.__name__} declares port names twice")
    cls.input_ports = tuple(ports.inputs)
    cls.output_ports = tuple(ports.outputs)
    if ports.dynamic is not None:
        from equinox import field
        cls.__annotations__ = {**vars(cls).get("__annotations__", {}),
                               "input_ports": tuple[str, ...], "output_ports": tuple[str, ...]}
        cls.input_ports = field(default=tuple(ports.inputs), static=True, kw_only=True)
        cls.output_ports = field(default=tuple(ports.outputs), static=True, kw_only=True)


def configuration_model(cls):
    if cls.config is not None:
        if not isinstance(cls.config, type) or not issubclass(cls.config, BaseModel):
            raise TypeError(f"{cls.__name__}.config must be a typed Pydantic model")
        return cls.config
    signature = inspect.signature(cls)
    hints = get_type_hints(cls.__init__, include_extras=True)
    fields = {}
    options = cls.component_options or ComponentOptions()
    context = set(options.build_context_fields)
    context.update(options.build_output_prototypes or ())
    for name, parameter in signature.parameters.items():
        if name in context:
            continue
        if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            raise TypeError(f"{cls.__name__} needs a typed configuration for {name!r}")
        annotation = hints.get(name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"{cls.__name__} configuration field {name!r} needs a type")
        fields[name] = (
            annotation,
            ... if parameter.default is parameter.empty else parameter.default,
        )
    return create_model(
        f"{cls.__name__}Config", __config__=ConfigDict(extra="forbid", strict=True), **fields
    )


def configuration_paths(cls):
    paths = {}
    for binding in cls.config_bindings:
        for name, path in binding.projections.items():
            if name in paths:
                raise ValueError(f"{cls.__name__} binds configuration field {name!r} twice")
            paths[name] = path
    if unknown := paths.keys() - configuration_model(cls).model_fields.keys():
        raise ValueError(f"{cls.__name__} binds unknown fields: {sorted(unknown)}")
    return paths


def json_value(value):
    if hasattr(value, "to_param"):
        return json_value(value.to_param())
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {name: json_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def configuration_values(component):
    cls = type(component)
    paths = configuration_paths(cls)
    values = {}
    for name in configuration_model(cls).model_fields:
        value = component
        path = paths.get(name, name)
        for part in path.split("."):
            if not hasattr(value, part):
                if ".init" in path:
                    raise ValueError(
                        f"{cls.__name__} reset payload at {path!r} was stripped by make_with_state; preserve initial payloads for export"
                    )
                raise AttributeError(f"{cls.__name__} has no configuration storage path {path!r}")
            value = getattr(value, part)
            if part == "init":
                from equinox.nn._stateful import _Sentinel
                if isinstance(value, _Sentinel):
                    raise ValueError(
                        f"{cls.__name__} reset payload at {path!r} was stripped by make_with_state; preserve initial payloads for export"
                    )
        values[name] = value
    return values


def validate_authoring(model, values):
    """Validate portable JSON structure, including JSON arrays for typed tuples."""
    import json

    return model.model_validate_json(json.dumps(json_value(values)))


def extract_configuration(component):
    return validate_authoring(
        configuration_model(type(component)), component.configuration_values()
    )


class MissingPrototypeInput(ValueError):
    """An output prototype needs a connected input not yet known."""


@dataclass(frozen=True)
class InputPrototypeParameter:
    port: str
    parameter: str
    resolve: Any


@dataclass(frozen=True)
class ComponentOptions:
    """Execution/lowering capabilities; configuration and identity are derived."""

    domain: str = "feedbax.domain.causal"
    interior_domain: str | None = None
    is_composite: bool = False
    output_prototype_fn: Any = None
    prototype_depends_on_connected_inputs: bool = False
    input_prototype_parameter: InputPrototypeParameter | None = None
    build_output_prototypes: Mapping[str, str] | None = None
    build_context_fields: tuple[str, ...] = ()
    trainable_by_default: bool = False
    template_graph: Any = None
    template_id: str | None = None
    template_kind: str | None = None

    def as_kwargs(self):
        import dataclasses

        return {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}


def validate_configuration(component):
    """Validate host-visible construction values without materializing traced arrays."""
    import jax
    from pydantic import TypeAdapter

    cls = type(component)
    model = configuration_model(cls)
    values = {}
    traced = False
    raw = component.configuration_values()
    import json

    for name, field in model.model_fields.items():
        value = raw[name]
        if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value)):
            traced = True
            continue
        values[name] = json_value(value)
        TypeAdapter(field.rebuild_annotation()).validate_json(json.dumps(values[name]), strict=True)
    if not traced:
        validate_authoring(model, values)


class ComponentOwner:
    """A reconstructible typed authoring contract, independent of execution."""

    config: ClassVar[type | None] = None
    config_bindings: ClassVar[tuple[ConfigBinding, ...]] = ()
    ports: ClassVar[Ports | None] = None
    component_options: ClassVar[ComponentOptions | None] = None

    @classmethod
    def from_config(cls, config, **context):
        return cls(**materialize_configuration(cls, config), **context)

    @classmethod
    def port_contract(cls, *, registry=None) -> Ports:
        if not isinstance(cls.ports, Ports):
            raise TypeError(f"{cls.__name__} needs one typed Ports declaration")
        return cls.ports

    def configuration_values(self):
        return configuration_values(self)

    def configuration(self):
        return extract_configuration(self)


class GraphTemplate(ComponentOwner, ABC):
    """Expand one canonical typed configuration into an editable graph."""

    @classmethod
    @abstractmethod
    def expand(cls, config):
        """Return the graph whose expanded contents are the save authority."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, **context):
        return cls.expand(config)

    @classmethod
    def port_contract(cls, *, registry=None) -> Ports:
        return graph_port_contract(cls.expand(configuration_model(cls)()), registry=registry)


def graph_port_contract(graph, *, registry=None) -> Ports:
    """Project a graph boundary from existing compiler metadata."""
    from feedbax.contracts.acausal import AcausalGraphSpec
    from feedbax.contracts.acausal_interface import derive_acausal_interface, signal_port_type

    if isinstance(graph, AcausalGraphSpec):
        interface = derive_acausal_interface(graph)
        return Ports(
            {name: payload.port_type() for name, payload in interface.input_payloads.items()},
            {name: signal_port_type() for name in interface.output_ports},
        )
    if not graph.input_bindings and not graph.output_bindings:
        return Ports({}, {})
    if registry is None:
        raise ValueError("Causal template port projection requires its compiled registry")

    def boundary(bindings, direction):
        result = {}
        for public, (node_id, port) in bindings.items():
            if node_id in (graph.subgraphs or {}):
                nested = graph_port_contract(graph.subgraphs[node_id], registry=registry)
                result[public] = getattr(nested, direction)[port]
            else:
                node = graph.nodes[node_id]
                resolved = registry.resolve_component_spec(node.type, node.params, revision=node.revision, complete_params=False)
                types = getattr(resolved.meta.port_types, direction)
                if port in types:
                    result[public] = types[port]
                elif resolved.meta.dynamic_port_policy is not None:
                    result[public] = resolved.meta.dynamic_port_policy.dynamic_port_type
                else:
                    raise ValueError(f"Template boundary {public!r} references an untyped port {node.type}.{port}")
        return result

    return Ports(boundary(graph.input_bindings, "inputs"), boundary(graph.output_bindings, "outputs"))


def expansion_configuration(expand, *, exclude=()):
    """Derive a template config from its typed graph-building operation."""
    hints = get_type_hints(expand, include_extras=True)
    fields = {
        name: (hints[name], ... if parameter.default is parameter.empty else parameter.default)
        for name, parameter in inspect.signature(expand).parameters.items()
        if name not in exclude
    }
    return create_model(
        f"{expand.__name__}Config", __config__=ConfigDict(extra="forbid", strict=True), **fields
    )
