"""Bounded discovery and compilation of typed component owners."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
import inspect
from pathlib import Path
from typing import Any

from feedbax.contracts.portable import ComponentOwner, GraphTemplate
from feedbax.plugins.declarations import DistributionSource


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


def compile_owner(
    cls: type[ComponentOwner],
    *,
    key: str,
    source: DistributionSource,
    registry,
):
    from feedbax.contracts.portable import ComponentOwner, GraphTemplate
    from feedbax.contracts.portable import (
        ComponentOptions,
        configuration_model,
        configuration_paths,
    )
    from .history import History

    if (
        not issubclass(cls, ComponentOwner)
        or cls in (ComponentOwner, GraphTemplate)
        or inspect.isabstract(cls)
    ):
        raise TypeError(f"{cls!r} must be a concrete portable ComponentOwner")
    namespace, _, local_name = key.rpartition(".")
    options = (cls.component_options or ComponentOptions()).as_kwargs()
    history = cls.component_history or History()
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

        names = (
            (*ports.dynamic.input_name_params, *ports.dynamic.output_name_params)
            if ports.dynamic.count_mode == "names"
            else (ports.dynamic.count_param,)
        )
        if all(not model.model_fields[name].is_required() for name in names):
            defaults = {
                name: model.model_fields[name].get_default(call_default_factory=True)
                for name in names
            }
            layout = derive_dynamic_port_layout(ports.dynamic, defaults)
            input_ports, output_ports = layout.input_ports, layout.output_ports
            port_types = type(port_types)(
                inputs={
                    name: ports.inputs[name]
                    if name in ports.inputs
                    else ports.dynamic.dynamic_port_type
                    for name in input_ports
                },
                outputs={
                    name: ports.outputs[name]
                    if name in ports.outputs
                    else ports.dynamic.dynamic_port_type
                    for name in output_ports
                },
            )
    from .declarations import (
        ComponentAuthoringFacet,
        ComponentCompilerFacet,
        ComponentDeclaration,
        ComponentParamContract,
        ComponentRuntimeFacet,
        ComponentSerializationFacet,
        ComponentStudioFacet,
        ComponentTrainingFacet,
        DeclaredComponent,
    )
    from feedbax.contracts.component import ComponentIdentity

    name = key
    explicit_context = frozenset(options.pop("build_context_fields"))
    prototype_bindings = options.pop("build_output_prototypes") or {}
    if explicit_context & set(model.model_fields):
        raise ValueError(f"Component {name!r} cannot author runtime context fields")
    if prototype_bindings:
        if options["output_prototype_fn"] is None or set(prototype_bindings.values()) - set(
            output_ports
        ):
            raise ValueError(f"Component {name!r} must infer each output prototype it builds from")
        if set(prototype_bindings) & (set(model.model_fields) | explicit_context):
            raise ValueError(f"Component {name!r} cannot author compiler-provided prototypes")
    contract = ComponentParamContract(
        component_type=name,
        model=model,
        revision=history.revision,
        runtime_type=cls,
        build_context_fields=explicit_context | frozenset(prototype_bindings),
    )
    authoring = ComponentAuthoringFacet(
        options.pop("template_graph"), options.pop("template_id"), options.pop("template_kind")
    )
    training = ComponentTrainingFacet(options.pop("trainable_by_default"))
    if options["is_composite"] and options["interior_domain"] is None:
        options["interior_domain"] = options["domain"]
    group = namespace
    if cls.__module__.startswith("feedbax.builtins."):
        group = cls.__module__.split(".")[2].capitalize()
        if ".equinox" in cls.__module__:
            group = "Equinox"
    studio = ComponentStudioFacet(group, (inspect.getdoc(cls) or "").split("\n\n", 1)[0])
    if presentation := cls.component_presentation:
        studio = replace(studio, **presentation.as_kwargs())
    core = source.distribution == "feedbax"
    provenance = "feedbax" if core else f"package:{source.distribution}"
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
            required_inputs=ports.required_inputs,
            port_types=port_types,
            dynamic_port_policy=ports.dynamic,
            build_output_prototypes=prototype_bindings,
            **options,
        ),
        serialization=ComponentSerializationFacet(history.revision),
        runtime=ComponentRuntimeFacet(contract.build),
        authoring=authoring,
        studio=studio,
        training=training,
        provenance=provenance,
        identity=ComponentIdentity(
            type_id=name, owner=namespace, provenance=provenance,
            provenance_kind="feedbax" if core else "package", package=source.package,
            import_path=f"{cls.__module__}.{cls.__qualname__}", stable=True,
        ),
        owner_class=cls,
    )
    sources = (
        source_path(cls),
        *cls.component_sources,
        source_path(configuration_model),
        *((source_path(cls.config),) if cls.config else ()),
        *(
            source_path(binding.group)
            for binding in cls.config_bindings
            if binding.group is not None
        ),
        *((history.source,) if history.source is not None else ()),
        *((studio.source,) if studio.source is not None else ()),
    )
    return meta, sources, history.compile(namespace, local_name)
