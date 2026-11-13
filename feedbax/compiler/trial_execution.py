"""Compile a model, task document and objective through their shared boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from pydantic import BaseModel, ValidationError

from feedbax.compiler.graph import (
    ExecutableGraph,
    GraphCompilationError,
    GraphDocument,
    compile_graph,
)
from feedbax.compiler.model_document import (
    LoweredModelDocument, ModelCompilationProblem, ResolvedModelInterface, lower_model_document,
)
from feedbax.compiler.resolution import ParameterResolutionError
from feedbax.contracts.graph import GraphSpec
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.artifact_custody import ArtifactBlobProvider
from feedbax.contracts.expression_values import Problem, ProblemCode, Site, ValueBase
from feedbax.contracts.model_document import GraphInterior, ModelDocument
from feedbax.contracts.project import json_pointer_parts
from feedbax.contracts.task_document import PythonTaskSelection
from feedbax.expression_registry import FunctionRegistry
from feedbax.objectives.document import ObjectiveDocument, ObjectiveExecutor, ObjectiveUse, compile_objective
from feedbax.runtime.graph import GraphTraceRequest
from feedbax.runtime.trial_execution import TrialExecution
from feedbax.tasks.document import TaskDocument, TrialSource, bind_task_inputs, compile_task
from feedbax.tasks.reaching import builtin_python_tasks
from feedbax.tasks.registry import PythonTaskDefinition, PythonTaskRegistry, PythonTaskTemplate


@dataclass(frozen=True)
class PreparedTrialExecution:
    executable: ExecutableGraph
    execution: TrialExecution
    objective: ObjectiveExecutor | None
    lowered: LoweredModelDocument


_DTYPES = {ValueBase.BOOLEAN: jnp.bool_, ValueBase.INTEGER: jnp.int32, ValueBase.REAL: jnp.float32}


@dataclass(frozen=True)
class TrialPair:
    model: ResolvedModelInterface
    task: TrialSource
    lowered: LoweredModelDocument
    prototypes: dict
    input_bindings: tuple[tuple[str, str], ...]
    problems: tuple[Problem, ...]


def validate_python_task(
    selection: PythonTaskSelection, registry: PythonTaskRegistry,
) -> tuple[PythonTaskDefinition, BaseModel] | tuple[Problem, ...]:
    definition = registry.resolve(selection.type)
    if definition is None:
        return (Problem(code=ProblemCode.UNKNOWN_NAME, path=("python_task", "type"),
                        reason=f"Python task {selection.type!r} is not registered."),)
    try:
        params = definition.parameters.model_validate(selection.params)
    except ValidationError as error:
        return tuple(Problem(code=ProblemCode.TYPE_MISMATCH,
                             path=("python_task", "params", *item["loc"]), reason=item["msg"])
                     for item in error.errors(include_url=False))
    return definition, params


def resolve_python_task(
    selection: PythonTaskSelection, registry: PythonTaskRegistry,
) -> PythonTaskTemplate | tuple[Problem, ...]:
    resolved = validate_python_task(selection, registry)
    if isinstance(resolved[0], Problem):
        return resolved
    definition, params = resolved
    return definition.template(params)


def evaluation_conditions(task: TaskDocument | PythonTaskSelection, registry: PythonTaskRegistry | None) -> tuple[str, ...]:
    if isinstance(task, TaskDocument):
        return task.condition_ids
    template = resolve_python_task(task, registry or builtin_python_tasks())
    if isinstance(template, tuple):
        raise ValueError(template[0].reason)
    return template.condition_ids


def pair_trial_source(
    model_name: str,
    model: ModelDocument,
    task: TaskDocument | PythonTaskSelection,
    components: ComponentRegistry,
    functions: FunctionRegistry,
    *,
    python_tasks: PythonTaskRegistry | None = None,
    input_overrides: Mapping[str, str] | None = None,
    graph_override: GraphSpec | None = None,
    task_sites: dict[tuple[str | int, ...], Site] | None = None,
) -> TrialPair | tuple[Problem, ...]:
    overrides = input_overrides or {}
    if isinstance(task, PythonTaskSelection):
        template = resolve_python_task(task, python_tasks or builtin_python_tasks())
        if isinstance(template, tuple):
            return template
        signal_types = template.signals
    else:
        template = None
        signal_types = {signal.name: signal.type for signal in task.signals}
    supplied = frozenset(
        port.name for port in model.ports.inputs
        if overrides.get(port.name, port.name) in signal_types
    )
    declared_prototypes = {}
    for port in model.ports.inputs:
        shape = port.type.shape
        if shape is None or any(type(size) is not int or size <= 0 for size in shape):
            continue
        base = port.type.base
        dtype = (
            jnp.bool_ if base in ("bool", "boolean") else
            jnp.int32 if base in ("int", "int32", "integer") else jnp.float32
        )
        declared_prototypes[port.name] = jax.ShapeDtypeStruct(shape, dtype)
    try:
        lowered = lower_model_document(model_name, model, components, supplied_inputs=supplied)
        prototypes = {
            target: (
                jax.ShapeDtypeStruct(value.shape, _DTYPES[value.base])
                if (value := signal_types.get(overrides.get(port, port))) is not None
                else declared_prototypes[port]
            )
            for port, target in lowered.graph.input_bindings.items()
            if port in supplied or port in declared_prototypes
        }
        if graph_override is not None:
            if graph_override.input_bindings != lowered.graph.input_bindings:
                return (Problem(
                    code=ProblemCode.STRUCTURE, path=("model", "inputs"),
                    reason="The selected task changes which optional inputs use the trained model's defaults.",
                ),)
            lowered = replace(lowered, graph=graph_override)
        interface = lowered.interface(
            model_name, model, components, input_prototypes=prototypes,
        )
    except ModelCompilationProblem as error:
        return (Problem(code=ProblemCode.STRUCTURE, reason=str(error),
                        path=("model", *error.address.stored_path())),)
    except ParameterResolutionError as error:
        address = lowered.source_address(error.authored_path)
        path = address.stored_path() if address else json_pointer_parts(error.authored_path)
        return (Problem(code=ProblemCode.STRUCTURE, reason=str(error), path=("model", *path)),)
    if isinstance(task, TaskDocument):
        compiled = compile_task(task, functions, model=replace(interface, inputs=()), sites=task_sites)
    else:
        assert template is not None
        compiled = template.prepare(interface, functions)
    if isinstance(compiled, tuple):
        return compiled
    if template is not None and template.condition_ids != compiled.condition_ids:
        return (Problem(
            code=ProblemCode.STRUCTURE, path=("python_task", "type"),
            reason="The prepared Task's conditions differ from its registry declaration.",
        ),)
    bindings, problems = bind_task_inputs(interface, compiled.interface.signals, overrides)
    return TrialPair(interface, compiled, lowered, prototypes, bindings, problems)


def _missing_required_input(
    graph: GraphSpec, model: ModelDocument, components: ComponentRegistry,
    bound_ports: frozenset[str],
) -> Problem | None:
    def check(
        graph: GraphSpec, interior: GraphInterior, path: tuple[str, ...], supplied: set[str],
    ) -> Problem | None:
        connected = {(wire.target_node, wire.target_port) for wire in graph.wires}
        connected.update(graph.input_bindings[name] for name in supplied if name in graph.input_bindings)
        for name, component in interior.components.items():
            here = (*path, "interior", "components", name)
            child = (graph.subgraphs or {}).get(name)
            if component.kind == "composite":
                if isinstance(child, GraphSpec) and isinstance(component.interior, GraphInterior):
                    ports = {port for node, port in connected if node == name}
                    if problem := check(child, component.interior, here, ports):
                        return problem
            elif (owner := components.get(component.type)) is not None:
                for port in owner.compiler.required_inputs:
                    if (name, port) not in connected:
                        return Problem(
                            code=ProblemCode.STRUCTURE, path=("model", *here),
                            reason=f"{component.name} input {port!r} needs a connection.",
                        )
        return None

    return check(graph, model.interior, (), set(bound_ports))


def compile_trial_execution(
    model_name: str,
    model: ModelDocument,
    task: TaskDocument | PythonTaskSelection,
    components: ComponentRegistry,
    functions: FunctionRegistry,
    *,
    python_tasks: PythonTaskRegistry | None = None,
    objective: ObjectiveDocument | None = None,
    objective_use: ObjectiveUse = ObjectiveUse.TRAIN,
    input_overrides: Mapping[str, str] | None = None,
    initialization_key: jax.Array | None = None,
    artifact_provider: ArtifactBlobProvider | None = None,
    graph_override: GraphSpec | None = None,
) -> PreparedTrialExecution | tuple[Problem, ...]:
    """Bind a trial source and compile the paired model and optional objective."""
    try:
        pairing = pair_trial_source(
            model_name, model, task, components, functions, python_tasks=python_tasks,
            input_overrides=input_overrides, graph_override=graph_override,
        )
        if isinstance(pairing, tuple):
            return pairing
        if pairing.problems:
            return pairing.problems
        if problem := _missing_required_input(
            pairing.lowered.graph, model, components, frozenset(port for port, _ in pairing.input_bindings),
        ):
            return (problem,)
        interface, compiled_task = pairing.model, pairing.task
        scorer = None
        observations = ()
        if objective is not None:
            names = (
                *interface.outputs,
                *compiled_task.interface.variables,
                *compiled_task.interface.signals,
            )
            scorer = compile_objective(objective, names, functions, task=compiled_task, use=objective_use)
            if not isinstance(scorer, ObjectiveExecutor):
                return scorer
            referenced = {name.id for name in scorer.names}
            observations = tuple(
                GraphTraceRequest("graph_output", output.id, port=output.name)
                for output in interface.outputs
                if output.id in referenced
            )
        executable = compile_graph(
            GraphDocument(graph=pairing.lowered.graph),
            components,
            input_prototypes=pairing.prototypes,
            initialization_key=initialization_key,
            artifact_provider=artifact_provider,
        )
    except ModelCompilationProblem as error:
        return (
            Problem(
                code=ProblemCode.STRUCTURE,
                reason=str(error),
                path=("model", *error.address.stored_path()),
            ),
        )
    except GraphCompilationError as error:
        return tuple(
            Problem(
                code=ProblemCode.STRUCTURE,
                reason=item.observed_condition,
                path=("model", *json_pointer_parts(item.source_anchor.authored_path)),
            )
            for item in error.record.diagnostics
        )
    return PreparedTrialExecution(
        executable,
        TrialExecution(compiled_task, interface.reset, pairing.input_bindings, observations),
        scorer,
        pairing.lowered,
    )
