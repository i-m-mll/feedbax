"""Execute the isolated saved rollout evaluation on the authenticated Train model."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
import os
from collections.abc import Callable, Iterator, Sequence
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord, ContentTreeRef
from feedbax.contracts.base import ArtifactRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.code_validity import ComputationUnavailable
from feedbax.contracts.effective_success import (
    Product, RequestObject, Success, TerminalTrainSubject, read_effective_request, read_success_request,
    success_content_ref,
)
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.live_evaluate import BoundLiveEvaluationPoint, ScoredEvaluationPoint
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    SpecPayload,
    normalize_manifest_spec_payloads,
)
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.loaded_model import ImportedModelSource, decode_imported_model_document
from feedbax.contracts.model_input import CheckpointModelSource, InitialModelSource, SynthesizedModelSource, read_document_input
from feedbax.contracts.model_origin import InitialModelOrigin, LoadedModelOrigin, SynthesizedModelOrigin
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.contracts.project_sources import pointer as json_pointer
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import ContentRef, LastMember, ProductRef, UpdateSeriesMember, is_synthesized_model_product
from feedbax.contracts.rollout import (
    ResolvedInitialRollout, ResolvedRolloutBinding, ResolvedRolloutMember, ResolvedSynthesizedRollout,
    rollout_objective_sha256, rollout_semantics,
)
from feedbax.contracts.rollout_dataset import UpdateAxis, rollout_coordinates, rollout_leading, rollout_update_axis
from feedbax.contracts.strict_json import strict_model_validate_json
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.persistence.completion import CompletionFileError, read_completion_bindings
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset, publish_dataset
from feedbax.persistence.evaluation_subject import read_terminal_train_subject, terminal_train_provenance
from feedbax.application.composition import load_application
from feedbax.host.model_input import ImportedModelWeights, prepare_model_task, reconstruct_model
from feedbax.persistence.canonical_evaluate import (
    ROLLOUT_TYPE,
    ROLLOUT_JAX_BACKEND,
    ROLLOUT_JAX_ENABLE_X64,
    STATES_MEDIA_TYPE,
    BoundEvaluationJobInput,
    RolloutParameters,
    SynthesizedSubjectBinding,
    decode_rollout_request,
    evaluation_checkpoint_coordinates,
    exact_evaluation_subject,
    evaluation_manifest_id,
    evaluation_subject_provenance,
    initial_evaluation_provenance,
    initial_evaluation_references,
    verified_imported_subject,
)
from feedbax.persistence.model_input import read_synthesized_model
from feedbax.contracts.rollout import CannotRun
from feedbax.persistence.canonical_train import BoundStudioTrainingJobInput
from feedbax.host.training_worker import _Events, _Result, _error
from feedbax.host.job_completion import read_host_completion_binding
from feedbax.host.evaluate_computation import EvaluationComputation, RolloutExecutables, drive_replicas
from feedbax.validity.recording import numerical_work, record_computation, set_execution_prefix


if TYPE_CHECKING:
    from feedbax.execution.trial_resolution import PreparedTrialExecution
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.receipt_validity import ValidityCheckDelivery, ValidityCheckEvidence
    from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
    from feedbax.contracts.training import TrialDocuments
    from feedbax.dataset import Dataset
    from feedbax.runtime.graph import Graph
    from feedbax.runtime.raw_state import RawStateLayout
    from feedbax.execution.trial_execution import TrialEvaluation
    from feedbax.application.bundle import ApplicationRegistryBundle
    from feedbax.training.evaluation_variant import EvaluationVariant
    from feedbax.host.model_checkpoint import SemanticModelSlot
    from feedbax.host.rollout_scoring import LoweredRollout, RolloutKernel, RolloutScorer


@dataclass(frozen=True)
class ResolvedRollout:
    paired: PreparedTrialExecution
    variant: EvaluationVariant
    source_pair: PreparedTrialExecution
    scorer: RolloutScorer


@dataclass(frozen=True)
class PreparedRollout:
    paired: PreparedTrialExecution
    variant: EvaluationVariant
    graph: Graph
    layout: RawStateLayout
    scorer: RolloutScorer
    slot: SemanticModelSlot | None
    member: ResolvedRolloutMember | ResolvedInitialRollout | ResolvedSynthesizedRollout
    objective: str | None
    varied: bool
    replicas: tuple[tuple[str, Graph], ...] = ()


def _bound_native_subject(
    request: BoundEvaluationJobInput, binding: dict,
) -> tuple[tuple[PublishedTrainingCheckpoint, BoundStudioTrainingJobInput], ...]:
    from feedbax.host.job_completion import (
        load_bound_checkpoint_selection,
        verify_bound_training_completion,
    )

    assert request.canonical_request.checkpoint is not None
    subject = exact_evaluation_subject(request.canonical_request.checkpoint, binding, verify_metrics=True)
    if (coordinates := evaluation_checkpoint_coordinates(subject)) is not None:
        return load_bound_checkpoint_selection(
            binding, root=request.manifest_root, producer_request=subject.request_id,
            coordinates=coordinates, project=request.project_id,
            allow_gaps=isinstance(subject.select, UpdateSeriesMember),
        )
    completion = binding["completion"]
    if (
        completion["project_id"] != request.project_id
        or completion["request_id"] != subject.request_id
        or completion["manifest_root"] != str(request.manifest_root)
    ):
        raise CompletionFileError("Pinned completion differs from the authored Train subject.")
    source = verify_bound_training_completion(completion)
    if source.canonical_request is None:
        raise CompletionFileError("Pinned subject is missing its canonical Train request.")
    checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
    return ((checkpoint, source),)



def score_paired_validation(
    graph, task_execution, objective, parameters, *, key,
):
    """Score the fixed evaluation trials through the same objective as training."""
    return task_execution.validation(graph, objective, parameters, key=key)


def resolve_rollout(
    initial: InitialModelSource, task: TaskDocument | PythonTaskSelection,
    objective: ObjectiveDocument | None, source_graph: GraphSpec | None, params: RolloutParameters,
    *, registry: ApplicationRegistryBundle, model_assets: ImmutableArtifactBlobProvider,
    input_bindings: dict[str, str | None],
) -> ResolvedRollout:
    import jax

    from feedbax.objectives.document import ObjectiveUse
    from feedbax.compiler.model_anchors import anchored_outputs
    from feedbax.execution.trial_places import bind_trial_places
    from feedbax.training.evaluation_variant import resolve_evaluation_variant, validate_evaluation_source
    from feedbax.host.rollout_scoring import RolloutScorer

    if jax.default_backend() != ROLLOUT_JAX_BACKEND or jax.config.jax_enable_x64 != ROLLOUT_JAX_ENABLE_X64:
        raise CannotRun("/numerical_settings", "requires CPU execution with JAX x64 disabled")
    base = prepare_model_task(
        initial, task, objective=objective, registry=registry, artifacts=model_assets,
        input_bindings=input_bindings,
    )
    if isinstance(base, tuple):
        problem = base[0]
        raise CannotRun(json_pointer(("inputs", "subject", *problem.path)), problem.reason)
    source_pair = base.paired
    if source_graph is None:
        source_graph = source_pair.executable.resolved.graph
    validate_evaluation_source(source_graph, registry.components)
    variant = resolve_evaluation_variant(source_graph, task, params.variation, components=registry.components)
    paired = source_pair
    varied_bindings = input_bindings | params.input_bindings
    if variant.graph != source_graph or variant.task != task or varied_bindings != input_bindings:
        varied = prepare_model_task(
            initial, variant.task, objective=objective, objective_use=ObjectiveUse.INHERITED,
            input_bindings=varied_bindings, registry=registry, artifacts=model_assets, graph_override=variant.graph,
        )
        if isinstance(varied, tuple):
            raise variant.task_refusal(varied[0])
        paired = varied.paired
    try:
        places = bind_trial_places(
            paired.objective, paired.execution.task,
            anchored_outputs(initial.document, registry.components, paired.signal_scope.leaves),
        )
        scorer = RolloutScorer(paired.execution, paired.objective, params.seed, params.trials, places=places)
    except ValueError as error:
        raise CannotRun("/params/trials", str(error)) from error
    return ResolvedRollout(paired, variant, source_pair, scorer)


def prepare_rollout(
    source: BoundStudioTrainingJobInput, checkpoint: PublishedTrainingCheckpoint, params: RolloutParameters,
) -> PreparedRollout:
    from feedbax.host.method_checkpoint import read_method_checkpoint
    from feedbax.contracts.request_specs import DocumentTrainingRequestSpec

    documents = source.spec.trial_documents
    request = source.canonical_request
    if request is None:
        raise CannotRun("/inputs/subject", "Managed Evaluate needs an authored Train checkpoint.")
    registry = load_application().bundle
    slot = read_method_checkpoint(checkpoint.checkpoint_root, checkpoint.pointer, components=registry.components).model
    store = ImmutableArtifactBlobProvider(source.manifest_root)
    authored = read_success_request(
        RequestObject(id=source.request_id, object=request.request, documents=request.documents), store.get_object,
    ).spec
    if not isinstance(authored, DocumentTrainingRequestSpec) or authored != source.spec:
        raise CannotRun("/inputs/subject", "Managed Evaluate needs an authored Train checkpoint.")
    return _prepare_model_rollout(
        documents, params, slot, checkpoint.completed_updates,
        seed=authored.seed or 0, registry=registry,
    )


def _prepare_model_rollout(
    documents: TrialDocuments, params: RolloutParameters, slot: SemanticModelSlot, coordinate: int,
    *, seed: int, registry: ApplicationRegistryBundle,
) -> PreparedRollout:
    initial = InitialModelSource(document=documents.model, seed=seed)
    return _prepare_weighted_rollout(
        initial, documents.task_source, documents.objective, documents.input_bindings,
        params, slot, registry=registry, coordinate=coordinate,
    )


def _prepare_weighted_rollout(
    initial: InitialModelSource, task_source: TaskDocument | PythonTaskSelection,
    objective: ObjectiveDocument | None, input_bindings: dict[str, str | None],
    params: RolloutParameters, slot: SemanticModelSlot, *, registry: ApplicationRegistryBundle,
    coordinate: int | None = None, imported_subject_sha256: str | None = None,
    synthesized_source: SynthesizedModelSource | None = None,
) -> PreparedRollout:
    imported = imported_subject_sha256 is not None
    resolved = resolve_rollout(
        initial, task_source, objective, None if imported else slot.graph, params, registry=registry,
        model_assets=slot.provider, input_bindings=input_bindings,
    )
    paired, variant, source_pair = resolved.paired, resolved.variant, resolved.source_pair
    source_type = ImportedModelWeights if imported else CheckpointModelSource
    source = synthesized_source or source_type(tree=slot.tree, encoding=slot.encoding, initial=initial)
    restored = reconstruct_model(
        source, registry=registry, artifacts=slot.provider, template=source_pair.resolution,
        variant=None if paired is source_pair else paired.resolution, source_overrides=variant.source_overrides,
        slot=slot,
    )
    if isinstance(restored, tuple):
        problem = restored[0]
        subject_path = (
            ("inputs", "subject", "checkpoint") if coordinate is not None else ("inputs", "subject")
        )
        raise CannotRun(json_pointer((*subject_path, *problem.path)), problem.reason)
    model, task = rollout_semantics(
        restored.resolved.graph, restored.resolved.key_schedule, paired.task_source,
        paired.execution.input_bindings, variant.source_overrides,
    )
    replicas = ()
    if len(restored.replica_ids) > 1 or imported_subject_sha256 is not None:
        replicas = tuple(zip(restored.replica_ids, restored.graphs, strict=True))
    if imported_subject_sha256 is not None:
        member = ResolvedInitialRollout(
            initial_model=imported_subject_sha256, model=model, task=task, trials=params.trials,
            replica_ids=restored.replica_ids,
        )
    elif synthesized_source is not None:
        member = ResolvedSynthesizedRollout(
            model_tree=source.tree.descriptor.sha256, model=model, task=task, trials=params.trials,
        )
    else:
        assert coordinate is not None
        member = ResolvedRolloutMember(
            coordinate=coordinate, model_tree=slot.tree.descriptor.sha256, model=model, task=task, trials=params.trials,
            replica_ids=restored.replica_ids if replicas else (),
        )
    return _prepared_rollout(
        resolved, restored.graphs[0], member, objective, slot=slot, replicas=replicas,
    )


def _prepare_initial_rollout(
    refs: dict[str, ContentRef], params: RolloutParameters,
    *, registry: ApplicationRegistryBundle, store: ImmutableArtifactBlobProvider,
) -> PreparedRollout:
    from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument

    source = read_document_input(refs["subject"], store)
    task = read_document_input(refs["task"], store)
    objective = read_document_input(refs["objective"], store) if "objective" in refs else None
    assert isinstance(task, (TaskDocument, PythonTaskSelection))
    assert objective is None or isinstance(objective, ObjectiveDocument)
    if isinstance(source, ImportedModelSource):
        from feedbax.host.model_checkpoint import SemanticModelSlot

        model = decode_imported_model_document(store.get_object(source.model))
        slot = SemanticModelSlot.open_store(store, source.weights, source.encoding, components=registry.components)
        return _prepare_weighted_rollout(
            InitialModelSource(document=model), task, objective, params.input_bindings,
            params, slot, registry=registry, imported_subject_sha256=refs["subject"].sha256,
        )
    assert isinstance(source, InitialModelSource)
    resolved = resolve_rollout(
        source, task, objective, None, params, registry=registry, model_assets=store,
        input_bindings=params.input_bindings,
    )
    paired, variant = resolved.paired, resolved.variant
    executable = paired.executable
    model, effective_task = rollout_semantics(
        executable.resolved.graph, executable.resolved.key_schedule, paired.task_source,
        paired.execution.input_bindings, variant.source_overrides,
    )
    member = ResolvedInitialRollout(
        initial_model=refs["subject"].sha256, model=model, task=effective_task, trials=params.trials,
    )
    return _prepared_rollout(resolved, executable.graph, member, objective)


def _prepared_rollout(
    resolved: ResolvedRollout, graph: Graph,
    member: ResolvedRolloutMember | ResolvedInitialRollout | ResolvedSynthesizedRollout,
    objective: ObjectiveDocument | None, *, slot: SemanticModelSlot | None = None,
    replicas: tuple[tuple[str, Graph], ...] = (),
) -> PreparedRollout:
    from feedbax.runtime.raw_state import StateLayoutError, bind_raw_state

    paired, variant = resolved.paired, resolved.variant
    try:
        layout = bind_raw_state(graph, paired.executable.resolved.graph)
    except StateLayoutError as error:
        raise CannotRun(error.path, str(error)) from error
    return PreparedRollout(
        paired, variant, graph, layout, resolved.scorer, slot, member,
        None if objective is None else rollout_objective_sha256(objective),
        paired is not resolved.source_pair, replicas,
    )


def _prepare_stored_rollout(
    store: ImmutableArtifactBlobProvider, model: Product, documents: TrialDocuments, params: RolloutParameters,
    coordinate: int, *, registry: ApplicationRegistryBundle, seed: int,
) -> PreparedRollout:
    from feedbax.host.model_checkpoint import SemanticModelSlot

    tree = success_content_ref(model.content)
    if not isinstance(tree, ContentTreeRef):
        raise ValueError("Evaluate checkpoint model requires a retained content tree")
    slot = SemanticModelSlot.open_store(
        store, tree, SemanticModelEncoding.model_validate(model.descriptor), components=registry.components,
    )
    return _prepare_model_rollout(documents, params, slot, coordinate, seed=seed, registry=registry)


def _prepare_synthesized_rollout(
    store: ImmutableArtifactBlobProvider, source: SynthesizedModelSource, documents: TrialDocuments,
    params: RolloutParameters, *, registry: ApplicationRegistryBundle,
) -> PreparedRollout:
    from feedbax.host.model_checkpoint import SemanticModelSlot

    slot = SemanticModelSlot.open_store(store, source.tree, source.encoding, components=registry.components)
    return _prepare_weighted_rollout(
        source.initial, documents.task_source, documents.objective, documents.input_bindings, params, slot,
        registry=registry, synthesized_source=source,
    )


def rollout_dataset(
    prepared: PreparedRollout, evaluated: TrialEvaluation, params: RolloutParameters,
    *, update_axis: UpdateAxis | None = "checkpoint",
) -> Dataset:
    import jax
    import numpy as np

    from feedbax.contracts.authored_canonical import canonical_sha256
    from feedbax.dataset import Dataset, Variable
    from feedbax.persistence.evaluation_trial_records import trial_dataset_fields
    from feedbax.contracts.task_document import TaskDocument

    paired, variant, graph = prepared.paired, prepared.variant, prepared.graph
    effective_graph = paired.executable.resolved.graph
    slot, layout = prepared.slot, prepared.layout
    assert evaluated.history is not None
    history = jax.tree.map(np.asarray, evaluated.history)
    n_trials = evaluated.trials.epochs.starts.shape[0]
    trials = [jax.tree.map(lambda value: value[i], history) for i in range(n_trials)]
    leading = rollout_leading(update_axis)

    def expand(value):
        array = np.asarray(value)
        return array.reshape((1,) * len(leading) + array.shape)

    variables, state_coordinates = layout.dataset_fields(
        trials, axes=(*leading, "trial"),
    )
    if evaluated.objective is not None:
        for name, term in evaluated.objective.terms.items():
            if not np.isfinite(term.value):
                raise FloatingPointError(f"Nonfinite objective value at {name}")
            path = json_pointer((name,))
            variables["objective/terms" + path] = Variable(expand(term.value), leading)
            variables["objective/coverage" + path] = Variable(expand(term.covered), leading)
        variables["objective/total"] = Variable(expand(evaluated.objective.loss), leading)
        variables["objective/trial_loss"] = Variable(expand(evaluated.objective.trial_losses), (*leading, "trial"))
    document = variant.task if isinstance(variant.task, TaskDocument) else None
    labels = {epoch.id: epoch.name for epoch in document.epochs} if document is not None else {}
    trial_fields, trial_coordinates, record_layout = trial_dataset_fields(
        paired.execution.task, evaluated, layout, leading=leading, labels=labels, document=document,
        observation_origins=paired.execution.observation_origins,
        places=prepared.scorer.places,
    )
    variables.update(trial_fields)
    state_coordinates.update(trial_coordinates)
    time = np.arange(paired.execution.task.n_steps + 1, dtype=np.float64) * graph.period_seconds
    variation = params.condition_id()
    attrs = {
        "state_sampling": "reset_and_post_transition",
        "state_lifetime": "trial",
        "trial_record": record_layout.model_dump(mode="json"),
    }
    if params.retention == "raw_states":
        attrs["source_graph_sha256"] = sha256(
            canonical_json_v2_bytes((slot.graph if slot is not None else effective_graph).model_dump(mode="json"))
        ).hexdigest()
        if prepared.varied:
            attrs.update({
                "effective_graph_sha256": canonical_sha256({
                    "resolved_graph": paired.executable.resolved.resolved_sha256,
                    "model_values": slot.tree.descriptor.sha256 if slot is not None else prepared.member.initial_model,
                    "source_overrides": variant.source_overrides,
                }),
                "effective_task_sha256": sha256(
                    canonical_json_v2_bytes(variant.task.model_dump(mode="json"))
                ).hexdigest(),
                "variation_sha256": variation,
            })
    coordinate_values = {
        "condition": np.asarray([variation]),
        "trial": np.asarray(params.trials),
        "state_time": time,
        "transition_start": time[:-1],
        "transition_end": time[1:],
    }
    if update_axis is not None:
        assert isinstance(prepared.member, ResolvedRolloutMember)
        coordinate_values[update_axis] = np.asarray([prepared.member.coordinate])
    coordinates = {
        name: Variable.from_field(coordinate_values[name], descriptor, descriptor.dims)
        for name, descriptor in rollout_coordinates(update_axis).items()
    }
    return Dataset.from_arrays(
        variables,
        coords=coordinates | state_coordinates,
        attrs=attrs,
        schema_id=params.states_schema,
    )


def _lower_rollouts[T](
    prepared: PreparedRollout, stopped: Callable[[], bool], *,
    consume: Callable[[Graph, LoweredRollout], T], kernel: RolloutKernel | None = None,
) -> tuple[T, ...] | None:
    replicas = prepared.replicas or (("0", prepared.graph),)
    if kernel is None:
        with numerical_work():
            kernel = prepared.scorer.kernel(capture_state=prepared.layout.capture)
    coordinate = prepared.member.coordinate if isinstance(prepared.member, ResolvedRolloutMember) else None

    def lower(slot, graph, seed):
        with numerical_work():
            lowered = kernel.lower(graph, slot=slot, seed=seed)
        return consume(graph, lowered)

    return drive_replicas(coordinate, replicas, seed=prepared.scorer.seed, call=lower, stopped=stopped)


def _rollouts(
    prepared: PreparedRollout, params: RolloutParameters, stopped: Callable[[], bool],
    *, update_axis: str | None, computation: EvaluationComputation, executables: RolloutExecutables,
    kernel: RolloutKernel | None = None,
):
    def run(graph: Graph, lowered: LoweredRollout):
        with numerical_work():
            evaluated = executables.run(lowered)
        if stopped():
            return None
        computation.observe(lowered)
        computation.observe_output(lowered.slot, evaluated)
        return _replica_product(prepared, graph, evaluated, params, update_axis)

    results = _lower_rollouts(prepared, stopped, consume=run, kernel=kernel)
    if results is None or any(result is None for result in results):
        return None
    return _rollout_product(prepared, results, update_axis)


def _replica_product(prepared, graph, evaluated, params, update_axis):
    dataset = rollout_dataset(replace(prepared, graph=graph), evaluated, params, update_axis=update_axis)
    return dataset, None if evaluated.objective is None else float(evaluated.objective.loss)


def _rollout_product(prepared, results, update_axis):
    datasets, totals = zip(*results, strict=True)
    if len(datasets) > 1:
        from feedbax.dataset_replicates import stack_replicates

        dataset = stack_replicates(tuple(replica for replica, _ in prepared.replicas), datasets, after=update_axis)
    else:
        (dataset,) = datasets
    summaries = [total for total in totals if total is not None]
    return dataset, sum(summaries) / len(summaries) if summaries else None, prepared.member, prepared.objective


def _checkpoint_dataset(datasets, *, axis="checkpoint"):
    """Concatenate singleton rollout datasets along their existing checkpoint axis."""
    if len(datasets) == 1:
        return datasets[0]
    import numpy as np

    from feedbax.dataset import Dataset, Variable

    descriptors = [dataset.descriptor.model_copy(deep=True) for dataset in datasets]
    base = descriptors[0]
    member_graphs = [descriptor.attrs.pop("effective_graph_sha256", None) for descriptor in descriptors]
    if any(descriptor != base for descriptor in descriptors[1:]):
        raise CannotRun("/inputs/subject/checkpoint", "selected checkpoints have different dataset layouts")

    def join(name, field):
        values = [dataset.data(name) for dataset in datasets]
        if axis in field.dims:
            if any(value.dtype != values[0].dtype for value in values[1:]):
                raise CannotRun("/inputs/subject/checkpoint", f"selected checkpoints differ in dtype for {name}")
            array = np.concatenate(values, axis=field.dims.index(axis))
        else:
            if any(not np.array_equal(value, values[0]) for value in values[1:]):
                raise CannotRun("/inputs/subject/checkpoint", f"selected checkpoints differ in coordinate {name}")
            array = values[0]
        return Variable.from_field(array, field, field.dims)

    coords = {name: join(name, field) for name, field in base.coords.items()}
    if any(graph is not None for graph in member_graphs):
        coords["effective_graph_sha256"] = Variable(np.asarray(member_graphs), (axis,))
    return Dataset.from_arrays(
        {name: join(name, field) for name, field in base.variables.items()},
        coords=coords, attrs=base.attrs, schema_id=base.schema_id,
    )


def _empty_update_dataset(params: RolloutParameters) -> Dataset:
    import numpy as np

    from feedbax.dataset import Dataset, Variable

    return Dataset.from_arrays({}, coords={
        "update": Variable(np.empty(0, dtype=np.int64), ("update",)),
    }, schema_id=params.states_schema)


def _series_dataset(
    members, binding, params, stop, *, update_axis, job_id, store, opened: ExitStack,
    computation: EvaluationComputation,
):
    points = {}
    if "live_evaluation_points" in binding:
        pinned = {
            member["publication_cursor"]: member["checkpoint"]
            for member in binding["checkpoint_selection"]["members"]
        }
        for raw in binding["live_evaluation_points"]:
            point = BoundLiveEvaluationPoint.model_validate(raw)
            if (
                point.evaluation_job_id != job_id
                or pinned.get(point.publication_cursor) != point.checkpoint.model_dump(mode="json")
                or point.checkpoint.completed_updates in points
            ):
                raise ValueError("Live Evaluate point differs from its settled checkpoint binding.")
            points[point.checkpoint.completed_updates] = point

    datasets, summaries, resolved_members, objectives = [], [], [], set()
    executables = RolloutExecutables()
    for coordinate, prepare in members:
        if stop.exists():
            return None
        with numerical_work():
            prepared = prepare()
        delivery = points.get(coordinate)
        if delivery is not None and isinstance(point := delivery.result, ScoredEvaluationPoint):
            from feedbax.persistence.live_evaluate import read_live_evaluation_evidence

            if point.member != prepared.member or point.objective != prepared.objective:
                raise ValueError("Live Evaluate point differs from its resolved scientific binding.")
            evidence = read_live_evaluation_evidence(store, delivery)
            recorded = evidence.trace.computation or evidence.trace.computation_unavailable
            assert recorded is not None
            computation.inherit(recorded)
            set_execution_prefix(point.evidence)
            dataset, summary = _scored_product(opened, store, point, coordinate, params)
            member, objective = point.member, point.objective
        else:
            result = _rollouts(
                prepared, params, stop.exists, update_axis=update_axis,
                computation=computation, executables=executables,
            )
            if result is None:
                return None
            dataset, summary, member, objective = result
        datasets.append(dataset)
        if summary is not None:
            summaries.append(summary)
        resolved_members.append(member)
        objectives.add(objective)
    if stop.exists():
        return None
    return _series_product(datasets, summaries, resolved_members, objectives, params, update_axis)


def _scored_product(opened, store, point, coordinate, params):
    from feedbax.dataset_replicates import replicate_ids

    store.verify_object(point.states.object_ref)
    dataset = opened.enter_context(open_stored_dataset(store, point.states))
    if (
        point.member.coordinate != coordinate
        or point.member.trials != params.trials
        or dataset.descriptor.schema_id != params.states_schema
        or dataset.data("update").tolist() != [coordinate]
        or dataset.data("trial").tolist() != list(params.trials)
        or (replicate_ids(dataset) or ()) != point.member.replica_ids
    ):
        raise ValueError("Live Evaluate dataset differs from its resolved inputs.")
    totals = dataset.data("objective/total").reshape(-1).tolist()
    if len(totals) != (len(point.member.replica_ids) or 1):
        raise ValueError("Live Evaluate dataset differs from its expected replica count.")
    summary = sum(totals) / len(totals)
    if summary != point.total:
        raise ValueError("Live Evaluate summary differs from its durable dataset.")
    return dataset, summary


def _series_product(datasets, summaries, resolved_members, objectives, params, update_axis):
    if len(objectives) > 1:
        raise CannotRun("/inputs/subject/checkpoint", "selected checkpoints have different Objectives")
    dataset = _checkpoint_dataset(datasets, axis=update_axis) if datasets else _empty_update_dataset(params)
    summary = {"objective/total": sum(summaries) / len(summaries)} if summaries else {}
    effective = ResolvedRolloutBinding(
        members=tuple(resolved_members), seed=params.seed, trial_policy=params.trial_policy,
        objective=objectives.pop(),
    ) if resolved_members else None
    return dataset, summary, effective


def _replay_preparations(
    receipt: Success, store: ImmutableArtifactBlobProvider,
) -> Iterator[PreparedRollout]:
    from feedbax.persistence.checkpoint_publication import model_product, read_retained_checkpoint
    from feedbax.persistence.evaluation_subject import terminal_train_subject_at

    decoded = read_effective_request(receipt, store.get_object)
    _, params, subject = decode_rollout_request(decoded.id, store.get_object(receipt.request.object))
    registry = load_application().bundle
    if isinstance(subject, ContentRef):
        refs = initial_evaluation_references(decoded)
        yield _prepare_initial_rollout(refs, params, registry=registry, store=store)
        return
    (bound,) = receipt.inputs["subject"]
    if is_synthesized_model_product(subject):
        if bound.selection is None:
            raise ValueError("Evaluate receipt lacks its exact Synthesize model selection.")
        synthesized, producer, _ = read_synthesized_model(store, bound.selection.receipt, subject)
        yield _prepare_synthesized_rollout(store, synthesized, producer.trial_documents, params, registry=registry)
        return
    source = bound.evaluation_subject
    if source is None:
        raise ValueError("Evaluate receipt lacks its exact retained checkpoint subject")
    if source.terminal_checkpoint is not None:
        if source.terminal_receipt is None:
            raise ValueError("Terminal Evaluate subject lacks its exact Train success")
        pin = terminal_train_subject_at(store, source.terminal_receipt, source.producer.id)
        train, spec = read_terminal_train_subject(store, pin, source.producer.id)
        if train.continuation is None or train.continuation.provenance != source.terminal_checkpoint:
            raise ValueError("Retained Train success names a different checkpoint")
        yield _prepare_stored_rollout(
            store, pin.model, spec.trial_documents, params, train.continuation.clock.completed_updates,
            registry=registry, seed=spec.seed or 0,
        )
        return
    for publication in source.checkpoints:
        checkpoint, spec = read_retained_checkpoint(store, publication, source.producer)
        yield _prepare_stored_rollout(
            store, model_product(checkpoint), spec.trial_documents, params, checkpoint.clock.completed_updates,
            registry=registry, seed=spec.seed or 0,
        )


def check_evaluate(delivery: ValidityCheckDelivery) -> ValidityCheckEvidence:
    """Re-lower the successful receipt's exact frozen rollout inputs."""
    from feedbax.contracts.computation_validity import ComputationDifferent, ComputationUnconfirmable
    from feedbax.contracts.receipt_validity import CheckOutcome, ValidityCheckEvidence
    from feedbax.persistence.effective_success import SuccessView
    from feedbax.validity.computation import UnrecordableArguments, compare_computation

    code = None
    if "code" in delivery.checks:
        store = ImmutableArtifactBlobProvider(delivery.store_root)
        receipt = SuccessView(store).read_owned_success(delivery.receipt).receipt
        if receipt.request.id != delivery.request_id or not delivery.request_id.startswith("evaluate:"):
            raise ValueError("Validity check names a different Evaluate receipt")
        former = receipt.trace.computation if receipt.trace is not None else None
        present = EvaluationComputation(store)
        try:
            for prepared in _replay_preparations(receipt, store):
                _lower_rollouts(prepared, lambda: False, consume=lambda _, call: present.observe(call))
        except CannotRun as error:
            code = CheckOutcome(outcome="different", detail=str(error))
        except UnrecordableArguments as error:
            code = CheckOutcome(outcome="unavailable", detail=str(error))
        else:
            replayed = present.snapshot()
            if isinstance(replayed, ComputationUnavailable):
                code = CheckOutcome(outcome="unavailable", detail=replayed.detail)
            elif former is None or replayed is None:
                code = CheckOutcome(outcome="unavailable", detail="Evaluate has no comparable lowered execution")
            else:
                comparison = compare_computation(former, replayed, read_blob=store.get_object)
                if isinstance(comparison, ComputationDifferent):
                    code = CheckOutcome(outcome="different", changes=comparison.changes)
                elif isinstance(comparison, ComputationUnconfirmable):
                    code = CheckOutcome(outcome="unavailable", detail=comparison.reason)
                else:
                    code = CheckOutcome(outcome="equal")
    environment = (
        CheckOutcome(outcome="unavailable", detail="Evaluate has no qualified environment comparison profile")
        if "environment" in delivery.checks else None
    )
    return ValidityCheckEvidence(code=code, environment=environment)


def main(argv: Sequence[str]) -> int:
    events = _Events(Path(os.environ["FEEDBAX_HOST_EVENT_LOG"]))
    try:
        events.phase("binding")
        if argv:
            raise ValueError("Evaluate accepts the authored Host input document.")
        job_id = os.environ["FEEDBAX_HOST_JOB_ID"]
        stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
        request = strict_model_validate_json(
            BoundEvaluationJobInput,
            Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes()
        )
        store = ImmutableArtifactBlobProvider(request.manifest_root)
        computation = EvaluationComputation(store)
        record_computation(computation.snapshot)
        spec, params, subject = decode_rollout_request(
            request.request_id, store.get_object(request.canonical_request.request)
        )
        if subject != request.canonical_request.subject:
            raise ValueError("Launch subject differs from the canonical Evaluate request.")
        events.phase("restoring")
        launch = request.canonical_request
        registry = load_application().bundle
        loaded_model = None
        model_origin = None
        if isinstance(subject, ContentRef):
            binding = {}
            decoded = decode_request(store.get_object(launch.request), store.get_object)
            refs = initial_evaluation_references(decoded)
            loaded_model, _ = verified_imported_subject(store, subject)
            model_origin = LoadedModelOrigin(kind="loaded") if loaded_model is not None else InitialModelOrigin(kind="initial")
            members = ((None, partial(_prepare_initial_rollout, refs, params, registry=registry, store=store)),)
            train, provenance = [], initial_evaluation_provenance(refs)
        elif is_synthesized_model_product(subject):
            binding = read_completion_bindings()
            anchored = SynthesizedSubjectBinding.model_validate(binding["model_subject"])
            source, producer, success = read_synthesized_model(store, anchored.receipt, subject)
            model_origin = SynthesizedModelOrigin(
                kind="synthesized", key=anchored.key, job_id=success.attempt, request_id=subject.request_id,
            )
            members = ((None, partial(
                _prepare_synthesized_rollout, store, source, producer.trial_documents, params, registry=registry,
            )),)
            train, provenance = [], anchored.provenance(model_origin)
        elif launch.terminal is not None or "terminal" in (binding := read_host_completion_binding()):
            if not isinstance(subject, ProductRef) or not isinstance(subject.select, LastMember):
                raise ValueError("Store terminal subject requires an authored last checkpoint")
            pin = launch.terminal or TerminalTrainSubject.model_validate(binding["terminal"])
            binding = {"terminal": pin.model_dump(mode="json")}
            source, authored = read_terminal_train_subject(store, pin, subject.request_id)
            occurrence = source.continuation
            assert occurrence is not None
            coordinate = occurrence.clock.completed_updates
            members = ((coordinate, partial(
                _prepare_stored_rollout, store, pin.model, authored.trial_documents, params, coordinate,
                seed=authored.seed or 0, registry=registry,
            )),)
            train, provenance = terminal_train_provenance(pin, source)
        else:
            checkpoints = _bound_native_subject(request, binding)
            members = tuple(
                (point.completed_updates, partial(prepare_rollout, source, point, params))
                for point, source in checkpoints
            )
            train, provenance = evaluation_subject_provenance(binding)
        events.phase("running")
        is_series = isinstance(subject, ProductRef) and isinstance(subject.select, UpdateSeriesMember)
        with ExitStack() as opened:
            result = _series_dataset(
                members, binding, params, stop,
                update_axis=rollout_update_axis(subject),
                job_id=job_id, store=store, opened=opened, computation=computation,
            )
            if result is None or stop.exists():
                return events.result(_Result("stopped"))
            dataset, summary, effective = result
            effective_metadata = {
                "effective_binding": effective.model_dump(mode="json"),
                "effective_binding_sha256": effective.digest(),
            } if effective is not None else {}
            if is_series:
                effective_metadata["placement"] = "live" if "live_evaluation_points" in binding else "ordinary"
            events.phase("finalizing")
            projection = EvaluationRunSpec(
                evaluation_type=ROLLOUT_TYPE, training_run_ids=[parent.id for parent in train],
                inputs=provenance.parents, params=spec.params,
            )
            with store._exclusive_lock():
                ref = publish_dataset(store, dataset)
                manifest = EvaluationRunManifest(
                    id=evaluation_manifest_id(job_id),
                    status="completed",
                    evaluation_spec=SpecPayload(
                        kind="EvaluationRunSpec", inline=projection.model_dump(mode="json"),
                    ),
                    input_training_runs=train,
                    loaded_model=loaded_model,
                    model_origin=model_origin,
                    provenance=provenance,
                    artifacts=[
                        ArtifactRef(
                            role="states", logical_name="states", sha256=ref.content.sha256,
                            size_bytes=ref.content.size, media_type=STATES_MEDIA_TYPE,
                            storage_backend="feedbax-cas",
                            metadata={"descriptor_sha256": ref.descriptor_sha256},
                        ),
                    ],
                    summary_metrics=summary,
                    metadata={
                        "request_id": request.request_id,
                        "canonical_request": request.canonical_request.request.model_dump(mode="json"),
                        **effective_metadata,
                    },
                )
                normalized = normalize_manifest_spec_payloads(manifest)
                manifest_ref = store.bind_manifest(
                    manifest.kind, manifest.id, normalized.model_dump_json(exclude_none=True).encode(),
                )
                store.register_root(ContentRootRecord(
                    root_id=manifest.id,
                    root_kind="run",
                    targets=(
                        ContentRetentionTarget(ref_kind="manifest", manifest=manifest_ref),
                        ContentRetentionTarget(ref_kind="object", object=ref.object_ref),
                    ),
                ))
        receipt = JobReceipt(manifest_id=manifest.id, manifest_sha256=manifest_ref.object.sha256)
        return events.result(_Result("succeeded", receipt=receipt.model_dump(mode="json")))
    except CannotRun as error:
        return events.result(_Result("failed", error={
            "code": "cannot_run", "path": error.path, "phase": events.current_phase,
            "message": str(error),
        }))
    except Exception as error:
        return events.result(_Result("failed", error=_error(error, events.current_phase)))
