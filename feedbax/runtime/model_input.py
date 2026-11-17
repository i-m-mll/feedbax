"""Resolve exact model inputs against the running application's declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from feedbax.compiler.graph import ExecutableGraph, GraphCompilationError, GraphDocument, compile_graph
from feedbax.compiler.model_document import ModelCompilationProblem, lower_model_document
from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution, declared_model_input_prototypes
from feedbax.contracts.expression_values import Problem, ProblemCode
from feedbax.contracts.model_input import CheckpointModelSource, InitialModelSource, ModelSource
from feedbax.contracts.project import json_pointer_parts
from feedbax.objectives.document import ObjectiveUse
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.training.model_checkpoint import ModelCheckpointError, ModelRestoreTarget, SemanticModelSlot

if TYPE_CHECKING:
    from feedbax.compiler.graph import ResolvedGraph
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.objective_document import ObjectiveDocument
    from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.plugins.application import ApplicationRegistryBundle
    from feedbax.runtime.graph import Graph


@dataclass(frozen=True)
class ReconstructedModel:
    executable: ExecutableGraph
    source: ModelSource
    replica_ids: tuple[str, ...]
    graphs: tuple[Graph, ...]

    @property
    def graph(self) -> Graph:
        if len(self.graphs) != 1:
            raise ValueError("Model contains several replicas")
        return self.graphs[0]

    @property
    def resolved(self) -> ResolvedGraph:
        return self.executable.resolved


@dataclass(frozen=True)
class PreparedModelTask:
    model: ReconstructedModel
    paired: PreparedTrialExecution


def reconstruct_model(
    source: ModelSource,
    *,
    registry: ApplicationRegistryBundle,
    artifacts: ImmutableArtifactBlobProvider,
    template: ExecutableGraph | None = None,
    variant: ExecutableGraph | None = None,
    source_overrides: Mapping[str, float] | None = None,
) -> ReconstructedModel | tuple[Problem, ...]:
    """Build an initial model or restore learned values onto a compiled template."""
    try:
        if isinstance(source, InitialModelSource):
            if template is None:
                lowered = lower_model_document(source.model_name, source.document, registry.components)
                prototypes = declared_model_input_prototypes(source.document)
                template = compile_graph(
                    GraphDocument(graph=lowered.graph), registry.components,
                    {
                        target: prototypes[name]
                        for name, target in lowered.graph.input_bindings.items()
                        if name in prototypes
                    },
                    initialization_key=model_initialization_key(source.seed), artifact_provider=artifacts,
                )
            executable = variant or template
            return ReconstructedModel(executable, source, ("0",), (executable.graph,))

        slot = SemanticModelSlot.open_store(artifacts, source.tree, source.encoding, components=registry.components)
        if template is None:
            prototypes = {} if source.initial is None else declared_model_input_prototypes(source.initial.document)
            template = compile_graph(
                GraphDocument(graph=slot.graph), registry.components,
                {
                    target: prototypes[name]
                    for name, target in slot.graph.input_bindings.items()
                    if name in prototypes
                },
                artifact_provider=artifacts,
            )
        target = None if variant is None else ModelRestoreTarget(
            variant.resolved.graph, variant.graph, source_overrides or {},
        )
        count = len(slot.replica_ids)
        graphs = slot.restore_all((template.graph,) * count, targets=None if target is None else (target,) * count)
        return ReconstructedModel(
            replace(variant or template, graph=graphs[0]), source, slot.replica_ids, graphs,
        )
    except ModelCompilationProblem as error:
        return (Problem(
            code=ProblemCode.STRUCTURE, path=("model", *error.address.stored_path()),
            reason=str(error), model_address=error.address,
        ),)
    except GraphCompilationError as error:
        return tuple(
            Problem(
                code=ProblemCode.STRUCTURE,
                path=("model", *json_pointer_parts(item.source_anchor.authored_path)),
                reason=item.observed_condition,
            )
            for item in error.record.diagnostics
        )
    except ModelCheckpointError as error:
        return (Problem(code=ProblemCode.STRUCTURE, path=json_pointer_parts(error.path), reason=str(error)),)


def prepare_model_task(
    source: ModelSource,
    task: TaskDocument | PythonTaskSelection,
    *,
    registry: ApplicationRegistryBundle,
    artifacts: ImmutableArtifactBlobProvider,
    objective: ObjectiveDocument | None = None,
    objective_use: ObjectiveUse = ObjectiveUse.TRAIN,
    input_overrides: Mapping[str, str] | None = None,
    graph_override: GraphSpec | None = None,
) -> PreparedModelTask | tuple[Problem, ...]:
    """Pair a source with task-derived prototypes and its optional objective."""
    definition = source.initial if isinstance(source, CheckpointModelSource) else source
    if definition is None:
        return (Problem(
            code=ProblemCode.STRUCTURE, path=("model",),
            reason="Task pairing requires the checkpoint's authored Model definition.",
        ),)
    paired = compile_trial_execution(
        definition.model_name, definition.document, task, registry.components, registry.expression_functions,
        python_tasks=registry.python_tasks, objective=objective, objective_use=objective_use,
        input_overrides=input_overrides, graph_override=graph_override,
        initialization_key=model_initialization_key(definition.seed), artifact_provider=artifacts,
    )
    if isinstance(paired, tuple):
        return paired
    model = reconstruct_model(source, registry=registry, artifacts=artifacts, template=paired.executable)
    return model if isinstance(model, tuple) else PreparedModelTask(model, paired)
