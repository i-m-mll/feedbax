"""Execute the isolated saved rollout evaluation on the authenticated Train model."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import lru_cache, partial
import os
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord, ContentTreeRef
from feedbax.contracts.base import ArtifactRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.effective_success import RequestObject, TerminalTrainSubject, read_success_request, success_content_ref
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.live_evaluate import BoundLiveEvaluationPoint, ScoredEvaluationPoint
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    SpecPayload,
    normalize_manifest_spec_payloads,
)
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.model_input import CheckpointModelSource, InitialModelSource, read_document_input
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.project_sources import pointer as json_pointer
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import ContentRef, LastMember, ProductRef, UpdateSeriesMember
from feedbax.contracts.rollout import (
    ResolvedInitialRollout, ResolvedRolloutBinding, ResolvedRolloutMember, rollout_objective_sha256, rollout_semantics,
)
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host.completion import CompletionFileError
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset, publish_dataset
from feedbax.persistence.evaluation_subject import read_terminal_train_subject, terminal_train_provenance
from feedbax.runtime.model_input import prepare_model_task, reconstruct_model
from feedbax.studio.canonical_evaluate import (
    ROLLOUT_TYPE,
    ROLLOUT_JAX_BACKEND,
    ROLLOUT_JAX_ENABLE_X64,
    STATES_MEDIA_TYPE,
    BoundEvaluationJobInput,
    CannotRun,
    RolloutParameters,
    decode_rollout_request,
    evaluation_checkpoint_coordinates,
    exact_evaluation_subject,
    evaluation_manifest_id,
    evaluation_subject_provenance,
    initial_evaluation_provenance,
    initial_evaluation_references,
)
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.training_host_worker import _Events, _Result, _error
from feedbax.job_completion import read_host_completion_binding


if TYPE_CHECKING:
    from feedbax.compiler.trial_execution import PreparedTrialExecution
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
    from feedbax.contracts.training import TrialDocuments
    from feedbax.dataset import Dataset
    from feedbax.runtime.graph import Graph
    from feedbax.runtime.raw_state import RawStateLayout
    from feedbax.runtime.trial_execution import TrialEvaluation
    from feedbax.plugins.application import ApplicationRegistryBundle
    from feedbax.training.evaluation_variant import EvaluationVariant
    from feedbax.training.model_checkpoint import SemanticModelSlot
    from feedbax.training.rollout_scoring import RolloutScorer


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
    member: ResolvedRolloutMember | ResolvedInitialRollout
    objective: str | None
    varied: bool
    replicas: tuple[tuple[str, Graph], ...] = ()


def _bound_native_subject(
    request: BoundEvaluationJobInput, binding: dict,
) -> tuple[tuple[PublishedTrainingCheckpoint, BoundStudioTrainingJobInput], ...]:
    from feedbax.job_completion import (
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


@lru_cache(maxsize=1)
def _registry(bootstrap_modules: tuple[str, ...]) -> ApplicationRegistryBundle:
    from feedbax.plugins.composition import compose_application

    return asyncio.run(compose_application(modules=bootstrap_modules)).bundle


def resolve_rollout(
    initial: InitialModelSource, task: TaskDocument | PythonTaskSelection,
    objective: ObjectiveDocument | None, source_graph: GraphSpec | None, params: RolloutParameters,
    *, registry: ApplicationRegistryBundle, model_assets: ImmutableArtifactBlobProvider,
    input_overrides: dict[str, str],
) -> ResolvedRollout:
    import jax

    from feedbax.objectives.document import ObjectiveUse
    from feedbax.training.evaluation_variant import resolve_evaluation_variant, validate_evaluation_source
    from feedbax.training.rollout_scoring import RolloutScorer

    if jax.default_backend() != ROLLOUT_JAX_BACKEND or jax.config.jax_enable_x64 != ROLLOUT_JAX_ENABLE_X64:
        raise CannotRun("/numerical_settings", "requires CPU execution with JAX x64 disabled")
    base = prepare_model_task(
        initial, task, objective=objective, registry=registry, artifacts=model_assets,
        input_overrides=input_overrides,
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
    if variant.graph != source_graph or variant.task != task:
        varied = prepare_model_task(
            initial, variant.task, objective=objective, objective_use=ObjectiveUse.INHERITED,
            input_overrides=input_overrides, registry=registry, artifacts=model_assets, graph_override=variant.graph,
        )
        if isinstance(varied, tuple):
            raise variant.task_refusal(varied[0])
        paired = varied.paired
    try:
        scorer = RolloutScorer(paired.execution, paired.objective, params.seed, params.trials)
    except ValueError as error:
        raise CannotRun("/params/trials", str(error)) from error
    return ResolvedRollout(paired, variant, source_pair, scorer)


def prepare_rollout(
    source: BoundStudioTrainingJobInput, checkpoint: PublishedTrainingCheckpoint, params: RolloutParameters,
) -> PreparedRollout:
    from feedbax.training.checkpoint_custody import _published_model_slot
    from feedbax.training.model_checkpoint import SemanticModelSlot
    from feedbax.contracts.request_specs import DocumentTrainingRequestSpec

    documents = source.spec.trial_documents
    request = source.canonical_request
    if request is None:
        raise CannotRun("/inputs/subject", "Managed Evaluate needs an authored Train checkpoint.")
    transaction, model_slot = _published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
    if model_slot is None:
        raise CannotRun("/inputs/subject/checkpoint/model", "requires a semantic model")
    registry = _registry(source.bootstrap_modules)
    slot = SemanticModelSlot.open(transaction, model_slot, components=registry.components)
    store = ImmutableArtifactBlobProvider(source.manifest_root)
    authored = read_success_request(
        RequestObject(id=source.request_id, object=request.request, documents=request.documents), store.get_object,
    ).spec
    if not isinstance(authored, DocumentTrainingRequestSpec):
        raise CannotRun("/inputs/subject", "Managed Evaluate needs an authored Train checkpoint.")
    return _prepare_model_rollout(
        documents, params, slot, checkpoint.completed_updates,
        seed=authored.seed or 0, registry=registry,
    )


def _prepare_model_rollout(
    documents: TrialDocuments, params: RolloutParameters, slot: SemanticModelSlot, coordinate: int,
    *, seed: int, registry: ApplicationRegistryBundle,
) -> PreparedRollout:
    initial = InitialModelSource(model_name=documents.model_name, document=documents.model, seed=seed)
    resolved = resolve_rollout(
        initial, documents.task_source, documents.objective, slot.graph, params, registry=registry,
        model_assets=slot.provider, input_overrides=documents.input_overrides,
    )
    paired, variant, source_pair = resolved.paired, resolved.variant, resolved.source_pair
    source = CheckpointModelSource(tree=slot.tree, encoding=slot.encoding, initial=initial)
    restored = reconstruct_model(
        source, registry=registry, artifacts=slot.provider, template=source_pair.resolution,
        variant=None if paired is source_pair else paired.resolution, source_overrides=variant.source_overrides,
        slot=slot,
    )
    if isinstance(restored, tuple):
        problem = restored[0]
        raise CannotRun(json_pointer(("inputs", "subject", "checkpoint", *problem.path)), problem.reason)
    model, task = rollout_semantics(
        restored.resolved.graph, restored.resolved.key_schedule, paired.task_source,
        paired.execution.input_bindings, variant.source_overrides,
    )
    member = ResolvedRolloutMember(
        coordinate=coordinate, model_tree=slot.tree.descriptor.sha256, model=model, task=task, trials=params.trials,
        replica_ids=restored.replica_ids if len(restored.replica_ids) > 1 else (),
    )
    return _prepared_rollout(
        resolved, restored.graphs[0], member, documents.objective, slot=slot,
        replicas=tuple(zip(restored.replica_ids, restored.graphs, strict=True)) if len(restored.graphs) > 1 else (),
    )


def _prepare_initial_rollout(
    refs: dict[str, ContentRef], params: RolloutParameters,
    *, registry: ApplicationRegistryBundle, store: ImmutableArtifactBlobProvider,
) -> PreparedRollout:
    from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument

    source = read_document_input(refs["subject"], store)
    task = read_document_input(refs["task"], store)
    objective = read_document_input(refs["objective"], store) if "objective" in refs else None
    assert isinstance(source, InitialModelSource) and isinstance(task, (TaskDocument, PythonTaskSelection))
    assert objective is None or isinstance(objective, ObjectiveDocument)
    resolved = resolve_rollout(
        source, task, objective, None, params, registry=registry, model_assets=store,
        input_overrides=params.input_overrides,
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
    resolved: ResolvedRollout, graph: Graph, member: ResolvedRolloutMember | ResolvedInitialRollout,
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


def rollout_dataset(
    prepared: PreparedRollout, evaluated: TrialEvaluation, params: RolloutParameters,
    *, update_axis: str | None = "checkpoint",
) -> Dataset:
    import jax
    import numpy as np

    from feedbax.contracts.authored_canonical import canonical_sha256
    from feedbax.dataset import Dataset, Variable
    from feedbax.runtime.raw_state import scene_position_key
    from feedbax.builtins.reaching import is_world_target

    paired, variant, graph = prepared.paired, prepared.variant, prepared.graph
    effective_graph = paired.executable.resolved.graph
    slot, layout = prepared.slot, prepared.layout
    history = evaluated.history
    assert history is not None
    n_trials = evaluated.trials.epochs.starts.shape[0]
    trials = [jax.tree.map(lambda value: value[i], history) for i in range(n_trials)]
    leading = ("condition",) if update_axis is None else (update_axis, "condition")

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
    for signal in paired.execution.task.interface.signals:
        samples = evaluated.trials.inputs[signal.name]
        base_axes = (*leading, "trial", "transition")
        if is_world_target(variant.task, signal):
            state_coordinates.setdefault("xy", Variable(np.asarray(["x", "y"]), ("xy",)))
            variable = Variable(expand(samples), (*base_axes, "xy"), units="m", frame="world")
        else:
            axes = (*base_axes, *(f"{signal.name}_component_{i}" for i in range(samples.ndim - 2)))
            variable = Variable(expand(samples), axes)
        variables[semantic_key("task", "output", (), signal.name)] = variable
    time = np.arange(paired.execution.task.n_steps + 1, dtype=np.float64) * graph.period_seconds
    variation = params.condition_id()
    attrs = {"state_sampling": "reset_and_post_transition", "state_lifetime": "trial"}
    if (position := scene_position_key(effective_graph, graph)) is not None and position in variables:
        attrs["scene_position"] = position
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
    coordinates = {}
    if update_axis is not None:
        assert isinstance(prepared.member, ResolvedRolloutMember)
        coordinates[update_axis] = Variable(np.asarray([prepared.member.coordinate]), (update_axis,))
    return Dataset.from_arrays(
        variables,
        coords={
            **coordinates,
            "condition": Variable(np.asarray([variation]), ("condition",)),
            "trial": Variable(np.asarray(params.trials), ("trial",)),
            **state_coordinates,
            "state_time": Variable(time, ("state_time",), units="s"),
            "transition_start": Variable(time[:-1], ("transition",), units="s"),
            "transition_end": Variable(time[1:], ("transition",), units="s"),
        },
        attrs=attrs,
        schema_id=params.states_schema,
    )


def _rollouts(prepared, params, stop, *, update_axis="checkpoint"):
    if stop.exists():
        return None
    datasets = []
    summaries = []
    replicas = prepared.replicas or (("0", prepared.graph),)
    for _, graph in replicas:
        evaluated = _execute_rollout(prepared, graph, update_axis)
        if stop.exists():
            return None
        datasets.append(rollout_dataset(replace(prepared, graph=graph), evaluated, params, update_axis=update_axis))
        if evaluated.objective is not None:
            summaries.append(float(evaluated.objective.loss))
    if len(datasets) > 1:
        from feedbax.dataset_replicates import stack_replicates

        dataset = stack_replicates(tuple(replica for replica, _ in replicas), datasets, after=update_axis)
    else:
        (dataset,) = datasets
    return dataset, sum(summaries) / len(summaries) if summaries else None, prepared.member, prepared.objective


def _execute_rollout(prepared, graph, update_axis):
    if update_axis == "update":
        import jax

        compiled = prepared.scorer.lower(graph, capture_state=prepared.layout.capture).compile()
        return jax.block_until_ready(compiled(graph))
    return prepared.scorer.evaluate(graph, capture_state=prepared.layout.capture)


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
        return Variable(array, field.dims, units=field.units, quantity=field.quantity, frame=field.frame)

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


def _series_dataset(members, binding, params, stop, *, update_axis, job_id, store, opened: ExitStack):
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
            points[point.checkpoint.completed_updates] = point.result

    datasets, summaries, resolved_members, objectives = [], [], [], set()
    for coordinate, prepare in members:
        if stop.exists():
            return None
        prepared = prepare()
        point = points.get(coordinate)
        if isinstance(point, ScoredEvaluationPoint):
            if point.member != prepared.member or point.objective != prepared.objective:
                raise ValueError("Live Evaluate point differs from its resolved scientific binding.")
            store.verify_object(point.states.object_ref)
            dataset = opened.enter_context(open_stored_dataset(store, point.states))
            if (
                point.member.coordinate != coordinate
                or point.member.trials != params.trials
                or dataset.descriptor.schema_id != params.states_schema
                or dataset.data("update").tolist() != [coordinate]
                or dataset.data("trial").tolist() != list(params.trials)
            ):
                raise ValueError("Live Evaluate dataset differs from its resolved inputs.")
            summary = float(dataset.data("objective/total").item())
            if summary != point.total:
                raise ValueError("Live Evaluate summary differs from its durable dataset.")
            member, objective = point.member, point.objective
        else:
            result = _rollouts(prepared, params, stop, update_axis=update_axis)
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
    if len(objectives) > 1:
        raise CannotRun("/inputs/subject/checkpoint", "selected checkpoints have different Objectives")
    dataset = _checkpoint_dataset(datasets, axis=update_axis) if datasets else _empty_update_dataset(params)
    summary = {"objective/total": sum(summaries) / len(summaries)} if summaries else {}
    effective = ResolvedRolloutBinding(
        members=tuple(resolved_members), seed=params.seed, trial_policy=params.trial_policy,
        objective=objectives.pop(),
    ) if resolved_members else None
    return dataset, summary, effective


def main(argv: Sequence[str]) -> int:
    events = _Events(Path(os.environ["FEEDBAX_HOST_EVENT_LOG"]))
    try:
        events.phase("binding")
        if argv:
            raise ValueError("Evaluate accepts the authored Host input document.")
        job_id = os.environ["FEEDBAX_HOST_JOB_ID"]
        stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
        request = BoundEvaluationJobInput.model_validate_json(
            Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes()
        )
        store = ImmutableArtifactBlobProvider(request.manifest_root)
        spec, params, subject = decode_rollout_request(
            request.request_id, store.get_object(request.canonical_request.request)
        )
        if subject != request.canonical_request.subject:
            raise ValueError("Launch subject differs from the canonical Evaluate request.")
        events.phase("restoring")
        launch = request.canonical_request
        registry = _registry(launch.bootstrap_modules)
        if isinstance(subject, ContentRef):
            binding = {}
            decoded = decode_request(store.get_object(launch.request), store.get_object)
            refs = initial_evaluation_references(decoded)
            members = ((None, partial(_prepare_initial_rollout, refs, params, registry=registry, store=store)),)
            train, provenance = [], initial_evaluation_provenance(refs)
        elif launch.terminal is not None or "terminal" in (binding := read_host_completion_binding()):
            if not isinstance(subject, ProductRef) or not isinstance(subject.select, LastMember):
                raise ValueError("Store terminal subject requires an authored last checkpoint")
            pin = launch.terminal or TerminalTrainSubject.model_validate(binding["terminal"])
            binding = {"terminal": pin.model_dump(mode="json")}
            source, authored = read_terminal_train_subject(store, pin, subject.request_id)
            occurrence = source.continuation
            assert occurrence is not None
            model = pin.model
            tree = success_content_ref(model.content)
            assert isinstance(tree, ContentTreeRef)
            from feedbax.training.model_checkpoint import SemanticModelSlot

            slot = SemanticModelSlot.open_store(
                store, tree, SemanticModelEncoding.model_validate(model.descriptor), components=registry.components,
            )
            coordinate = occurrence.clock.completed_batches
            members = ((coordinate, partial(
                _prepare_model_rollout, authored.trial_documents, params, slot, coordinate,
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
                update_axis=None if isinstance(subject, ContentRef) else "update" if is_series else "checkpoint",
                job_id=job_id, store=store, opened=opened,
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
