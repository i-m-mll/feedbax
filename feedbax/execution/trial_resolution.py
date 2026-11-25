"""Compile a model, task document and objective through their shared boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

import jax

from feedbax.compiler.graph import (
    ExecutableGraph,
    GraphCompilationError,
    GraphDocument,
    GraphResolution,
    realize_graph,
    resolve_graph,
)
from feedbax.compiler.model_document import (
    LoweredModelDocument, ModelCompilationProblem, ResolvedModelInterface, lower_model_document, model_input_prototypes,
)
from feedbax.compiler.model_document import declared_model_input_prototypes as declared_model_input_prototypes
from feedbax.compiler.model_reset import ModelResetField
from feedbax.compiler.model_signals import ModelMeasureScope, ResolvedSignalLeaf, model_measure_scope
from feedbax.compiler.resolution import ParameterResolutionError, ParameterResolutionErrors
from feedbax.contracts.graph import GraphSpec
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.artifact_custody import ArtifactBlobProvider
from feedbax.contracts.expression_values import Level, Problem, ProblemCode, ScopedValue, Site, ValueBase, ValueType
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.model_address import (
    ComponentElement, ComponentPortAddress, FieldElement, ModelAddress, OutputBoundaryEnd, PortElement,
)
from feedbax.contracts.model_attachment import is_trial_binding
from feedbax.contracts.model_document import GraphInterior, ModelDocument, SignalInput, SignalType
from feedbax.contracts.model_values import component_model_facts
from feedbax.contracts.project import json_pointer_parts
from feedbax.contracts.task_document import PythonTaskSelection
from feedbax.contracts.trial_bindings import selected_signal_id
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.objectives.document import ObjectiveDocument, ObjectiveExecutor, ObjectiveUse, compile_objective
from feedbax.runtime.graph import GraphTraceRequest
from feedbax.execution.trial_execution import TrialExecution, TrialModelOutput
from feedbax.tasks.document import TaskDocument, TrialSource, bind_task_inputs, compile_task, delay_initial_problems
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
    signal_scope: ModelMeasureScope


@dataclass(frozen=True)
class ResolvedTrialExecution:
    graph: GraphResolution
    execution: TrialExecution
    objective: ObjectiveExecutor | None
    lowered: LoweredModelDocument
    task_source: TaskDocument | PythonTaskSelection
    signal_scope: ModelMeasureScope

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
            executable, self.execution, self.objective, self.lowered, self.task_source, self.graph, self.signal_scope,
        )


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
        from feedbax.compiler.set_fields import lower_trial_interventions

        recorded = {}
        if graph_override is not None:
            recorded = {signal.id: signal for signal in task.signals if is_trial_binding(signal.id)}
            if recorded:
                task = task.model_copy(update={"signals": tuple(signal for signal in task.signals if signal.id not in recorded)})
        try:
            model, task = lower_trial_interventions(model, task, components)
        except ModelCompilationProblem as error:
            return model_compilation_problems(error)
        except ValueError as error:
            return (Problem(code=ProblemCode.STRUCTURE, path=("model",), reason=str(error)),)
        if graph_override is not None:
            signals = {signal.id: signal for signal in task.signals} | recorded
            task = task.model_copy(update={"signals": tuple(
                signal for id, signal in signals.items() if not is_trial_binding(id) or id in graph_override.input_bindings
            )})
        signal_types = {signal.id: signal.type for signal in task.signals}
        signal_ids = {signal.name: signal.id for signal in task.signals}
    supplied_types = {}
    for port in model.ports.inputs:
        signal_id = selected_signal_id(port.id, port.name, signal_ids, authored)
        if signal_id is not None and signal_id in signal_types:
            supplied_types[port.name] = signal_types[signal_id]
    supplied = frozenset(supplied_types)
    try:
        lowered = lower_model_document(model, components, supplied_inputs=supplied)
        if graph_override is not None:
            def public(bindings):
                return {name: slot for name, slot in bindings.items() if not is_trial_binding(name)}

            if public(graph_override.input_bindings) != public(lowered.graph.input_bindings):
                return (Problem(
                    code=ProblemCode.STRUCTURE, path=("model", "inputs"),
                    reason="The selected task changes which optional inputs use the trained model's defaults.",
                ),)
            generated = tuple(name for name in graph_override.input_ports if is_trial_binding(name))
            if missing := next((name for name in generated if name not in signal_types), None):
                return (Problem(
                    code=ProblemCode.STRUCTURE, path=("task", "signals", missing),
                    reason="The generated intervention input has no Task signal.",
                ),)
            supplied_types.update({name: signal_types[name] for name in generated})
            model = model.model_copy(update={"ports": model.ports.model_copy(update={
                "inputs": (
                    *(port for port in model.ports.inputs if not is_trial_binding(port.name)),
                    *(SignalInput(id=name, name=name, type=SignalType(
                        base=signal_types[name].base.value, shape=signal_types[name].shape,
                    )) for name in generated),
                ),
            })})
            lowered = replace(lowered, graph=graph_override)
        prototypes = model_input_prototypes(lowered.graph, model, supplied_types)
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
        scope = model_measure_scope(model, interface.signals)
        outputs = tuple(name for name in scope.names if _task_model_output(scope.leaves[name.id]))
        compiled = compile_task(
            task, functions, model=replace(interface, inputs=(), outputs=outputs), sites=task_sites, link_problems=links,
            nominal=nominal,
        )
    else:
        assert template is not None
        compiled = template.prepare(interface, functions)
        if not isinstance(compiled, tuple):
            links.extend(delay_initial_problems(interface, {item.id: item for item in compiled.interface.variables}))
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


def _task_model_output(leaf: ResolvedSignalLeaf) -> bool:
    port = leaf.port.element
    return not leaf.port.composites and isinstance(port, PortElement) and isinstance(port.end, OutputBoundaryEnd)


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
    signal_scope: ModelMeasureScope | None = None
    document: TaskDocument | None = None


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
        scope = model_measure_scope(model, paired.model.signals)
        document = paired.task_source if isinstance(paired.task_source, TaskDocument) else None
        return PreviewTrialSource(paired.task, scope.names, scope, document)
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


def _state_trace_path(origin: ModelAddress, graph: GraphSpec, components: ComponentRegistry) -> str | None:
    state = origin.element
    if not isinstance(state, FieldElement) or not isinstance(state.parent, ComponentElement):
        return None
    if len(state.path) != 2 or state.path[0] != "state":
        return None
    owner = graph
    for node in origin.composites:
        child = (owner.subgraphs or {}).get(node)
        if not isinstance(child, GraphSpec):
            return None
        owner = child
    node = owner.nodes[state.parent.id]
    facts = component_model_facts(node, components)
    if facts is None:
        return None
    return next((
        ".".join((*origin.composites, state.parent.id, *slot.fields))
        for slot in facts.states or () if slot.symbol == state.path[1] and not slot.subtree
    ), None)


def _signal_trace(
    leaf: ResolvedSignalLeaf, selector: str, graph: GraphSpec, components: ComponentRegistry,
) -> GraphTraceRequest:
    if leaf.origin is not None and (state_path := _state_trace_path(leaf.origin, graph, components)) is not None:
        return GraphTraceRequest("state_path", selector, path=state_path)
    port = leaf.port.element
    assert isinstance(port, PortElement)
    scope, end = leaf.port.composites, port.end
    if isinstance(end, ComponentPortAddress):
        return GraphTraceRequest("port", selector, node=end.component, port=end.port, scope=scope, leaf=leaf.path)
    if scope:
        return GraphTraceRequest("port", selector, node=scope[-1], port=end.port, scope=scope[:-1], leaf=leaf.path)
    return GraphTraceRequest("graph_output", selector, port=leaf.name, leaf=leaf.path)


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
    nominal: bool = False,
) -> ResolvedTrialExecution | tuple[Problem, ...]:
    """Validate the paired Model, Task and Objective before runtime construction."""
    try:
        pairing = pair_trial_source(
            model, task, components, functions, python_tasks=python_tasks,
            input_bindings=input_bindings, graph_override=graph_override, nominal=nominal,
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
        signal_scope = model_measure_scope(model, interface.signals)
        scorer = None
        observations: dict[str, GraphTraceRequest] = {}
        origins: list[tuple[str, str]] = []
        if objective is not None:
            names = (
                *signal_scope.names,
                *compiled_task.interface.variables,
                *compiled_task.interface.signals,
            )
            scorer = compile_objective(objective, names, functions, task=compiled_task, use=objective_use, nominal=nominal)
            if not isinstance(scorer, ObjectiveExecutor):
                return scorer
            for operand in scorer.names:
                leaf = signal_scope.leaves.get(operand.id)
                if leaf is None:
                    continue
                origin = (leaf.origin or leaf.address).to_text()
                origins.append((operand.id, origin))
                if origin not in observations:
                    observations[origin] = _signal_trace(leaf, origin, pairing.lowered.graph, components)
        graph = resolve_graph(
            GraphDocument(graph=pairing.lowered.graph), components, pairing.prototypes,
        )
    except ModelCompilationProblem as error:
        return model_compilation_problems(error)
    except GraphCompilationError as error:
        return _graph_problems(error)
    task_reads = tuple(
        TrialModelOutput(name.id, leaf.name, leaf.path, name.type)
        for name in compiled_task.interface.model_outputs
        if (leaf := signal_scope.leaves.get(name.id)) is not None
    )
    return ResolvedTrialExecution(
        graph,
        TrialExecution(
            compiled_task, interface.reset, pairing.input_bindings, tuple(observations.values()), tuple(origins),
            task_reads,
        ),
        scorer,
        pairing.lowered,
        pairing.task_source,
        signal_scope,
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
