"""Compile a model, task document and objective through their shared boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from feedbax.compiler.graph import (
    ExecutableGraph,
    GraphCompilationError,
    GraphDocument,
    GraphResolution,
    realize_graph,
    resolve_graph,
)
from feedbax.compiler.model_document import (
    LoweredModelDocument, ModelCompilationProblem, ResolvedModelInterface, lower_model_document,
)
from feedbax.compiler.model_reset import ModelResetField
from feedbax.compiler.resolution import ParameterResolutionError, ParameterResolutionErrors
from feedbax.contracts.graph import GraphSpec
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.artifact_custody import ArtifactBlobProvider
from feedbax.contracts.expression_values import Level, Problem, ProblemCode, ScopedValue, Site, ValueBase, ValueType
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import GraphInterior, ModelDocument
from feedbax.contracts.project import json_pointer_parts
from feedbax.contracts.task_document import PythonTaskSelection
from feedbax.contracts.trial_bindings import selected_signal_id
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.objectives.document import ObjectiveDocument, ObjectiveExecutor, ObjectiveUse, compile_objective
from feedbax.runtime.graph import GraphTraceRequest
from feedbax.execution.trial_execution import TrialExecution
from feedbax.tasks.document import TaskDocument, TrialSource, bind_task_inputs, compile_task
from feedbax.tasks.registry import (
    PythonTaskRegistry,
    PythonTaskTemplate,
    resolve_python_task_instance,
)


@dataclass(frozen=True)
class PreparedTrialExecution:
    executable: ExecutableGraph
    execution: TrialExecution
    objective: ObjectiveExecutor | None
    lowered: LoweredModelDocument
    task_source: TaskDocument | PythonTaskSelection
    resolution: GraphResolution


@dataclass(frozen=True)
class ResolvedTrialExecution:
    graph: GraphResolution
    execution: TrialExecution
    objective: ObjectiveExecutor | None
    lowered: LoweredModelDocument
    task_source: TaskDocument | PythonTaskSelection

    def realize(
        self,
        *,
        initialization_key: jax.Array | None = None,
        artifact_provider: ArtifactBlobProvider | None = None,
    ) -> PreparedTrialExecution | tuple[Problem, ...]:
        try:
            executable = realize_graph(
                self.graph, initialization_key=initialization_key, artifact_provider=artifact_provider,
            )
        except GraphCompilationError as error:
            return _graph_problems(error)
        return PreparedTrialExecution(
            executable, self.execution, self.objective, self.lowered, self.task_source, self.graph,
        )


_DTYPES = {ValueBase.BOOLEAN: jnp.bool_, ValueBase.INTEGER: jnp.int32, ValueBase.REAL: jnp.float32}


@dataclass(frozen=True)
class TrialPair:
    model: ResolvedModelInterface
    task: TrialSource
    lowered: LoweredModelDocument
    prototypes: dict
    input_bindings: tuple[tuple[str, str], ...]
    problems: tuple[Problem, ...]
    task_source: TaskDocument | PythonTaskSelection


def resolve_python_task(
    selection: PythonTaskSelection, registry: PythonTaskRegistry,
) -> PythonTaskTemplate | tuple[Problem, ...]:
    resolved = resolve_python_task_instance(selection, registry)
    return resolved if isinstance(resolved, tuple) else resolved.template()


def evaluation_conditions(task: TaskDocument | PythonTaskSelection, registry: PythonTaskRegistry) -> tuple[str, ...]:
    if isinstance(task, TaskDocument):
        return task.condition_ids
    template = resolve_python_task(task, registry)
    if isinstance(template, tuple):
        raise ValueError(template[0].reason)
    return template.condition_ids


def declared_model_input_prototypes(model: ModelDocument) -> dict[str, jax.ShapeDtypeStruct]:
    prototypes = {}
    for port in model.ports.inputs:
        shape = port.type.shape
        if shape is None or any(type(size) is not int or size <= 0 for size in shape):
            continue
        base = port.type.base
        dtype = (
            jnp.bool_ if base in ("bool", "boolean") else
            jnp.int32 if base in ("int", "int32", "integer") else jnp.float32
        )
        prototypes[port.name] = jax.ShapeDtypeStruct(shape, dtype)
    return prototypes


def pair_trial_source(
    model: ModelDocument,
    task: TaskDocument | PythonTaskSelection,
    components: ComponentRegistry,
    functions: FunctionRegistry,
    *,
    python_tasks: PythonTaskRegistry,
    input_bindings: Mapping[str, str | None] | None = None,
    graph_override: GraphSpec | None = None,
    task_sites: dict[tuple[str | int, ...], Site] | None = None,
    reset_fields: list[ModelResetField] | None = None,
    nominal: bool = False,
) -> TrialPair | tuple[Problem, ...]:
    """The Model and Task paired for execution, or the Problems that refuse them. ``task_sites`` receives the Task's
    field sites, and ``reset_fields`` the Model fields a trial starts from, once the Model resolves."""
    authored = input_bindings or {}
    if isinstance(task, PythonTaskSelection):
        resolved = resolve_python_task_instance(task, python_tasks)
        if isinstance(resolved, tuple):
            return resolved
        template = resolved.template()
        task = resolved.selection()
        signal_types = template.signals
        signal_ids = {name: name for name in signal_types}
    else:
        template = None
        signal_types = {signal.id: signal.type for signal in task.signals}
        signal_ids = {signal.name: signal.id for signal in task.signals}
    supplied_types = {}
    for port in model.ports.inputs:
        signal_id = selected_signal_id(port.id, port.name, signal_ids, authored)
        if signal_id is not None and signal_id in signal_types:
            supplied_types[port.name] = signal_types[signal_id]
    supplied = frozenset(supplied_types)
    declared_prototypes = declared_model_input_prototypes(model)
    try:
        lowered = lower_model_document(model, components, supplied_inputs=supplied)
        prototypes = {
            target: (
                jax.ShapeDtypeStruct(value.shape, _DTYPES[value.base])
                if (value := supplied_types.get(port)) is not None
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
        interface = lowered.interface(model, components, input_prototypes=prototypes, task_inputs=supplied_types)
    except ModelCompilationProblem as error:
        return model_compilation_problems(error)
    except ParameterResolutionError as error:
        return tuple(_parameter_problem(lowered, item) for item in (
            error.errors if isinstance(error, ParameterResolutionErrors) else (error,)
        ))
    if reset_fields is not None:
        reset_fields.extend(interface.reset.fields)
    links: list[Problem] = []
    if isinstance(task, TaskDocument):
        compiled = compile_task(
            task, functions, model=replace(interface, inputs=()), sites=task_sites, link_problems=links,
            nominal=nominal,
        )
    else:
        assert template is not None
        compiled = template.prepare(interface, functions)
    if isinstance(compiled, tuple):
        return (*compiled, *links)
    if template is not None and template.condition_ids != compiled.condition_ids:
        return (Problem(
            code=ProblemCode.STRUCTURE, path=("python_task", "type"),
            reason="The prepared Task's conditions differ from its registry declaration.",
        ),)
    bindings, problems = bind_task_inputs(interface, compiled.interface.signals, authored)
    graph_problems = tuple(Problem(
        code=ProblemCode.TYPE_MISMATCH,
        reason=issue.reason,
        path=("model", *issue.address.stored_path()),
        model_address=issue.address,
    ) for issue in interface.signals.problems)
    return TrialPair(interface, compiled, lowered, prototypes, bindings, (*links, *problems, *graph_problems), task)


def declared_model_outputs(model: ModelDocument) -> tuple[ScopedValue, ...]:
    """Outputs whose authored types are known without resolving the model's graph."""
    outputs = []
    for port in model.ports.outputs:
        shape = port.type.shape
        unit = UNITS.get(port.type.unit or "")
        if shape is None or unit is None or any(type(size) is not int or size <= 0 for size in shape):
            continue
        base = port.type.base
        kind = (
            ValueBase.BOOLEAN if base in ("bool", "boolean") else
            ValueBase.INTEGER if base in ("int", "int32", "integer") else ValueBase.REAL
        )
        address = ModelAddress.model_validate({
            "composites": [], "element": {"kind": "port", "end": {"owner": "outputs", "port": port.id}},
        })
        outputs.append(ScopedValue(
            id=address.to_text(), name=port.name, level=Level.STEP,
            type=ValueType(base=kind, shape=shape, dimensions=unit.dimensions, space=port.type.space),
        ))
    return tuple(outputs)


@dataclass(frozen=True)
class PreviewTrialSource:
    task: TrialSource
    outputs: tuple[ScopedValue, ...]


def preview_trial_source(
    model: ModelDocument,
    task: TaskDocument | PythonTaskSelection,
    components: ComponentRegistry,
    functions: FunctionRegistry,
    *,
    python_tasks: PythonTaskRegistry,
    input_bindings: Mapping[str, str | None],
    nominal: bool = False,
) -> PreviewTrialSource | tuple[Problem, ...]:
    """Compile the Task's trial, including each Model link that can be resolved."""
    paired = pair_trial_source(
        model, task, components, functions, python_tasks=python_tasks, input_bindings=input_bindings,
        nominal=nominal,
    )
    if isinstance(paired, TrialPair):
        return PreviewTrialSource(paired.task, paired.model.outputs)
    if any(problem.path[:1] != ("model",) for problem in paired):
        return tuple(problem for problem in paired if problem.path[:1] != ("links",))
    return compile_unpaired_task(model, task, functions, python_tasks=python_tasks, nominal=nominal)


def compile_unpaired_task(
    model: ModelDocument, task: TaskDocument | PythonTaskSelection, functions: FunctionRegistry,
    *, python_tasks: PythonTaskRegistry, sites: dict[tuple[str | int, ...], Site] | None = None,
    nominal: bool = False,
) -> PreviewTrialSource | tuple[Problem, ...]:
    """Compile values that need only the authored Model period and output declarations."""
    outputs = declared_model_outputs(model)
    interface = ResolvedModelInterface(model.step_seconds, (), outputs, (), ())
    if isinstance(task, TaskDocument):
        compiled = compile_task(task, functions, model=interface, sites=sites, link_problems=[], nominal=nominal)
    else:
        template = resolve_python_task(task, python_tasks)
        if isinstance(template, tuple):
            return template
        compiled = template.prepare(interface, functions)
    return compiled if isinstance(compiled, tuple) else PreviewTrialSource(compiled, outputs)


def _parameter_problem(lowered: LoweredModelDocument, error: ParameterResolutionError) -> Problem:
    address = lowered.source_address(error.authored_path)
    path = address.stored_path() if address else json_pointer_parts(error.authored_path)
    return Problem(code=ProblemCode.STRUCTURE, reason=str(error), path=("model", *path), model_address=address)


def _missing_required_inputs(
    graph: GraphSpec, model: ModelDocument, components: ComponentRegistry,
    bound_ports: frozenset[str],
) -> tuple[Problem, ...]:
    def check(
        graph: GraphSpec, interior: GraphInterior, path: tuple[str, ...], supplied: set[str],
    ) -> Iterator[Problem]:
        connected = {(wire.target_node, wire.target_port) for wire in graph.wires}
        connected.update(graph.input_bindings[name] for name in supplied if name in graph.input_bindings)
        for name, component in interior.components.items():
            here = (*path, "interior", "components", name)
            child = (graph.subgraphs or {}).get(name)
            if component.kind == "composite":
                if isinstance(child, GraphSpec) and isinstance(component.interior, GraphInterior):
                    yield from check(child, component.interior, here, {port for node, port in connected if node == name})
            elif (owner := components.get(component.type)) is not None:
                for port in owner.compiler.required_inputs:
                    if (name, port) not in connected:
                        yield Problem(
                            code=ProblemCode.STRUCTURE, path=("model", *here),
                            reason=f"{component.name} input {port!r} needs a connection.",
                        )

    return tuple(check(graph, model.interior, (), set(bound_ports)))


def compile_trial_execution(
    model: ModelDocument,
    task: TaskDocument | PythonTaskSelection,
    components: ComponentRegistry,
    functions: FunctionRegistry,
    *,
    python_tasks: PythonTaskRegistry,
    objective: ObjectiveDocument | None = None,
    objective_use: ObjectiveUse = ObjectiveUse.TRAIN,
    input_bindings: Mapping[str, str | None] | None = None,
    initialization_key: jax.Array | None = None,
    artifact_provider: ArtifactBlobProvider | None = None,
    graph_override: GraphSpec | None = None,
) -> PreparedTrialExecution | tuple[Problem, ...]:
    """Bind a trial source and compile the paired model and optional objective."""
    trial = resolve_trial_execution(
        model, task, components, functions, python_tasks=python_tasks,
        objective=objective, objective_use=objective_use, input_bindings=input_bindings,
        graph_override=graph_override,
    )
    if isinstance(trial, tuple):
        return trial
    return trial.realize(initialization_key=initialization_key, artifact_provider=artifact_provider)


def resolve_trial_execution(
    model: ModelDocument,
    task: TaskDocument | PythonTaskSelection,
    components: ComponentRegistry,
    functions: FunctionRegistry,
    *,
    python_tasks: PythonTaskRegistry,
    objective: ObjectiveDocument | None = None,
    objective_use: ObjectiveUse = ObjectiveUse.TRAIN,
    input_bindings: Mapping[str, str | None] | None = None,
    graph_override: GraphSpec | None = None,
) -> ResolvedTrialExecution | tuple[Problem, ...]:
    """Validate the paired Model, Task and Objective before runtime construction."""
    try:
        pairing = pair_trial_source(
            model, task, components, functions, python_tasks=python_tasks,
            input_bindings=input_bindings, graph_override=graph_override,
        )
        if isinstance(pairing, tuple):
            return pairing
        if pairing.problems:
            return pairing.problems
        if problems := _missing_required_inputs(
            pairing.lowered.graph, model, components, frozenset(port for port, _ in pairing.input_bindings),
        ):
            return problems
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
        graph = resolve_graph(
            GraphDocument(graph=pairing.lowered.graph), components, pairing.prototypes,
        )
    except ModelCompilationProblem as error:
        return model_compilation_problems(error)
    except GraphCompilationError as error:
        return _graph_problems(error)
    return ResolvedTrialExecution(
        graph,
        TrialExecution(compiled_task, interface.reset, pairing.input_bindings, observations),
        scorer,
        pairing.lowered,
        pairing.task_source,
    )


def model_compilation_problems(error: ModelCompilationProblem) -> tuple[Problem, ...]:
    return tuple(
        Problem(
            code=ProblemCode.STRUCTURE,
            reason=problem.reason,
            path=("model", *problem.address.stored_path()),
            model_address=problem.address,
        )
        for problem in error.problems
    )


def _graph_problems(error: GraphCompilationError) -> tuple[Problem, ...]:
    return tuple(
        Problem(
            code=ProblemCode.STRUCTURE,
            reason=item.observed_condition,
            path=("model", *json_pointer_parts(item.source_anchor.authored_path)),
        )
        for item in error.record.diagnostics
    )
