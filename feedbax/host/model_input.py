"""Resolve exact model inputs against the running application's declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from feedbax.compiler.graph import ExecutableGraph, GraphCompilationError, GraphDocument, GraphResolution, realize_graph, resolve_graph
from feedbax.compiler.model_document import ModelCompilationProblem, lower_model_document
from feedbax.execution.trial_resolution import PreparedTrialExecution, declared_model_input_prototypes, resolve_trial_execution
from feedbax.contracts.expression_values import Problem, ProblemCode
from feedbax.contracts.model_input import InitialModelSource, ModelSource
from feedbax.contracts.project import json_pointer_parts
from feedbax.objectives.document import ObjectiveUse
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.host.model_checkpoint import ModelCheckpointError, ModelRestoreTarget, SemanticModelSlot

if TYPE_CHECKING:
    from feedbax.compiler.graph import ResolvedGraph
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.objective_document import ObjectiveDocument
    from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.application.bundle import ApplicationRegistryBundle
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
    template: ExecutableGraph | GraphResolution | None = None,
    variant: ExecutableGraph | GraphResolution | None = None,
    variant_seed: int | None = None,
    source_overrides: Mapping[str, float] | None = None,
    slot: SemanticModelSlot | None = None,
) -> ReconstructedModel | tuple[Problem, ...]:
    """Build an initial model or restore learned values onto a compiled template."""
    if variant_seed is not None and not isinstance(variant, GraphResolution):
        return (Problem(
            code=ProblemCode.STRUCTURE, path=("model",),
            reason="A variant seed requires an unrealized variant template.",
        ),)

    def materialize(value, seed, replica):
        if isinstance(value, ExecutableGraph):
            return value
        return realize_graph(value, initialization_key=model_initialization_key(seed, replica), artifact_provider=artifacts)

    try:
        if isinstance(source, InitialModelSource):
            if template is None:
                lowered = lower_model_document(source.document, registry.components)
                prototypes = declared_model_input_prototypes(source.document)
                template = resolve_graph(
                    GraphDocument(graph=lowered.graph), registry.components,
                    {
                        target: prototypes[name]
                        for name, target in lowered.graph.input_bindings.items()
                        if name in prototypes
                    },
                )
            executable = materialize(variant or template, source.seed if variant_seed is None else variant_seed, "0")
            return ReconstructedModel(executable, source, ("0",), (executable.graph,))

        if slot is None:
            slot = SemanticModelSlot.open_store(artifacts, source.tree, source.encoding, components=registry.components)
        elif (slot.tree, slot.encoding) != (source.tree, source.encoding):
            raise ModelCheckpointError("/model", "model slot differs from its authenticated source")
        definition = source.initial
        if template is None:
            prototypes = declared_model_input_prototypes(definition.document)
            template = resolve_graph(
                GraphDocument(graph=slot.graph), registry.components,
                {
                    target: prototypes[name]
                    for name, target in slot.graph.input_bindings.items()
                    if name in prototypes
                },
            )
        ids = slot.replica_ids
        if len(ids) != 1 and (isinstance(template, ExecutableGraph) or isinstance(variant, ExecutableGraph)):
            raise ModelCheckpointError("/model/replicas", "replica restore requires an unrealized template")
        sources = tuple(materialize(template, definition.seed, replica) for replica in ids)
        targets = sources if variant is None else tuple(
            materialize(variant, definition.seed if variant_seed is None else variant_seed, replica) for replica in ids
        )
        graphs = slot.restore_all(
            tuple(source.graph for source in sources),
            targets=None if variant is None else tuple(
                ModelRestoreTarget(target.resolved.graph, target.graph, source_overrides or {}) for target in targets
            ),
        )
        return ReconstructedModel(
            replace(targets[0], graph=graphs[0]), source, ids, graphs,
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
    input_bindings: Mapping[str, str | None] | None = None,
    graph_override: GraphSpec | None = None,
) -> PreparedModelTask | tuple[Problem, ...]:
    """Pair a source with task-derived prototypes and its optional objective."""
    definition = source if isinstance(source, InitialModelSource) else source.initial
    resolved = resolve_trial_execution(
        definition.document, task, registry.components, registry.expression_functions,
        python_tasks=registry.python_tasks, objective=objective, objective_use=objective_use,
        input_bindings=input_bindings, graph_override=graph_override,
    )
    if isinstance(resolved, tuple):
        return resolved
    model = reconstruct_model(source, registry=registry, artifacts=artifacts, template=resolved.graph)
    if isinstance(model, tuple):
        return model
    return PreparedModelTask(model, PreparedTrialExecution(
        model.executable, resolved.execution, resolved.objective, resolved.lowered, resolved.task_source, resolved.graph,
        resolved.signal_scope,
    ))
