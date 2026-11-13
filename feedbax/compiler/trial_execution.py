"""Compile a model, task document and objective through their shared boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from feedbax.compiler.graph import (
    ExecutableGraph,
    GraphCompilationError,
    GraphDocument,
    compile_graph,
)
from feedbax.compiler.model_document import ModelCompilationProblem, lower_model_document
from feedbax.contracts.graph import GraphSpec
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.artifact_custody import ArtifactBlobProvider
from feedbax.contracts.expression_values import Problem, ProblemCode, ValueBase
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.project import json_pointer_parts
from feedbax.expression_registry import FunctionRegistry
from feedbax.objectives.document import ObjectiveDocument, ObjectiveExecutor, compile_objective
from feedbax.runtime.graph import GraphTraceRequest
from feedbax.runtime.trial_execution import TrialExecution
from feedbax.tasks.document import DocumentTask, TaskDocument, compile_task


@dataclass(frozen=True)
class PreparedTrialExecution:
    executable: ExecutableGraph
    execution: TrialExecution
    objective: ObjectiveExecutor | None


_DTYPES = {ValueBase.BOOLEAN: jnp.bool_, ValueBase.INTEGER: jnp.int32, ValueBase.REAL: jnp.float32}


def compile_trial_execution(
    model_name: str,
    model: ModelDocument,
    task: TaskDocument,
    components: ComponentRegistry,
    functions: FunctionRegistry,
    *,
    objective: ObjectiveDocument | None = None,
    input_overrides: Mapping[str, str] | None = None,
    initialization_key: jax.Array | None = None,
    artifact_provider: ArtifactBlobProvider | None = None,
    graph_override: GraphSpec | None = None,
) -> PreparedTrialExecution | tuple[Problem, ...]:
    """Use authored task signal types to resolve the model's numeric sample ports."""
    overrides = input_overrides or {}
    signal_types = {signal.name: signal.type for signal in task.signals}
    try:
        lowered = lower_model_document(model_name, model, components)
        prototypes = {
            target: jax.ShapeDtypeStruct(value.shape, _DTYPES[value.base])
            for port, target in lowered.graph.input_bindings.items()
            if (value := signal_types.get(overrides.get(port, port))) is not None
        }
        if graph_override is not None:
            from dataclasses import replace

            lowered = replace(lowered, graph=graph_override)
        interface = lowered.interface(
            model_name,
            model,
            components,
            input_prototypes=prototypes,
        )
        compiled_task = compile_task(task, functions, model=interface, input_overrides=overrides)
        if not isinstance(compiled_task, DocumentTask):
            return compiled_task
        scorer = None
        observations = ()
        if objective is not None:
            names = (
                *interface.outputs,
                *compiled_task.interface.variables,
                *compiled_task.interface.signals,
            )
            scorer = compile_objective(objective, names, functions, task=compiled_task)
            if not isinstance(scorer, ObjectiveExecutor):
                return scorer
            referenced = {name.id for name in scorer.names}
            observations = tuple(
                GraphTraceRequest("graph_output", output.id, port=output.name)
                for output in interface.outputs
                if output.id in referenced
            )
        executable = compile_graph(
            GraphDocument(graph=lowered.graph),
            components,
            input_prototypes=prototypes,
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
        TrialExecution(compiled_task, interface.reset, compiled_task.input_bindings, observations),
        scorer,
    )
