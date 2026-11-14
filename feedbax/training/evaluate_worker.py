"""Execute the isolated saved rollout evaluation on the authenticated Train model."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import dataclass
import os
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.base import ArtifactRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.live_evaluate import BoundLiveEvaluationPoint, ScoredEvaluationPoint
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    SpecPayload,
    normalize_manifest_spec_payloads,
)
from feedbax.contracts.project_sources import pointer as json_pointer
from feedbax.contracts.request_refs import ProductRef, UpdateSeriesMember
from feedbax.contracts.rollout import ResolvedRolloutBinding, ResolvedRolloutMember
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host.completion import CompletionFileError
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset, publish_dataset
from feedbax.studio.canonical_evaluate import (
    ROLLOUT_TYPE,
    STATES_MEDIA_TYPE,
    BoundEvaluationJobInput,
    CannotRun,
    RolloutParameters,
    decode_rollout_request,
    evaluation_checkpoint_coordinates,
    exact_evaluation_subject,
    evaluation_manifest_id,
    evaluation_subject_provenance,
)
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.training_host_worker import _Events, _Result, _error


if TYPE_CHECKING:
    from feedbax.compiler.trial_execution import PreparedTrialExecution
    from feedbax.contracts.artifact_custody import ArtifactBlobProvider
    from feedbax.contracts.graph import GraphSpec
    from feedbax.dataset import Dataset
    from feedbax.runtime.graph import Graph
    from feedbax.runtime.raw_state import RawStateLayout
    from feedbax.runtime.trial_execution import TrialEvaluation
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
    slot: SemanticModelSlot
    member: ResolvedRolloutMember
    objective: str
    varied: bool


def _bound_subject(
    request: BoundEvaluationJobInput,
) -> tuple[tuple[tuple[PublishedTrainingCheckpoint, BoundStudioTrainingJobInput], ...], dict]:
    from feedbax.job_completion import (
        load_bound_checkpoint_selection,
        read_host_completion_binding,
        verify_bound_training_completion,
    )

    binding = read_host_completion_binding()
    subject = exact_evaluation_subject(request.canonical_request.checkpoint, binding, verify_metrics=True)
    if (coordinates := evaluation_checkpoint_coordinates(subject)) is not None:
        return load_bound_checkpoint_selection(
            binding, root=request.manifest_root, producer_request=subject.request_id,
            coordinates=coordinates, project=request.project_id,
            allow_gaps=isinstance(subject.select, UpdateSeriesMember),
        ), binding
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
    return ((checkpoint, source),), binding



def score_paired_validation(
    graph, task_execution, objective, parameters, *, key,
):
    """Score the fixed evaluation trials through the same objective as training."""
    return task_execution.validation(graph, objective, parameters, key=key)


def resolve_rollout(
    source: BoundStudioTrainingJobInput, source_graph: GraphSpec, params: RolloutParameters,
    *, model_assets: ArtifactBlobProvider,
) -> ResolvedRollout:
    import jax

    from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
    from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload
    from feedbax.objectives.document import ObjectiveUse
    from feedbax.plugins.composition import compose_application
    from feedbax.training.evaluation_variant import resolve_evaluation_variant, validate_evaluation_source
    from feedbax.training.rollout_scoring import RolloutScorer
    from feedbax.training.studio_executor import studio_training_key

    spec = source.spec
    documents = spec.trial_documents
    if documents is None or spec.method_ref != STUDIO_GRAPH_METHOD:
        raise CannotRun("/inputs/subject", "Managed Evaluate needs a document Train checkpoint.")
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise CannotRun("/numerical_settings", "requires CPU execution with JAX x64 disabled")
    validate_evaluation_source(source_graph)
    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    registry = asyncio.run(compose_application(modules=source.bootstrap_modules)).bundle
    variant = resolve_evaluation_variant(source_graph, documents.task_source, params.variation)
    key = studio_training_key(method.seed, "model")
    source_pair = compile_trial_execution(
        documents.model_name, documents.model, documents.task_source,
        registry.components, registry.expression_functions, python_tasks=registry.python_tasks, objective=documents.objective,
        input_overrides=documents.input_overrides, artifact_provider=model_assets,
        initialization_key=key,
    )
    if not isinstance(source_pair, PreparedTrialExecution):
        problem = source_pair[0]
        raise CannotRun(json_pointer(("inputs", "subject", *problem.path)), problem.reason)
    paired = source_pair
    if variant.graph != source_graph or variant.task != documents.task_source:
        paired = compile_trial_execution(
            documents.model_name, documents.model, variant.task,
            registry.components, registry.expression_functions, python_tasks=registry.python_tasks, objective=documents.objective,
            objective_use=ObjectiveUse.INHERITED,
            input_overrides=documents.input_overrides, artifact_provider=model_assets,
            initialization_key=key, graph_override=variant.graph,
        )
    if not isinstance(paired, PreparedTrialExecution):
        raise variant.task_refusal(paired[0])
    assert paired.objective is not None
    try:
        scorer = RolloutScorer(paired.execution, paired.objective, params.seed, params.trials)
    except ValueError as error:
        raise CannotRun("/params/trials", str(error)) from error
    return ResolvedRollout(paired, variant, source_pair, scorer)


def prepare_rollout(
    source: BoundStudioTrainingJobInput, checkpoint: PublishedTrainingCheckpoint, params: RolloutParameters,
) -> PreparedRollout:
    from feedbax.runtime.raw_state import StateLayoutError, bind_raw_state
    from feedbax.training.checkpoint_custody import open_published_model_slot
    from feedbax.training.model_checkpoint import ModelCheckpointError, ModelRestoreTarget

    slot = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
    if slot is None:
        raise CannotRun("/inputs/subject/checkpoint/model", "requires a typed model slot")
    resolved = resolve_rollout(source, slot.graph, params, model_assets=slot.provider)
    paired, variant, source_pair = resolved.paired, resolved.variant, resolved.source_pair
    try:
        target = None if paired is source_pair else ModelRestoreTarget(
            paired.executable.resolved.graph, paired.executable.graph, variant.source_overrides,
        )
        graph = slot.restore(source_pair.executable.graph, target=target)
        effective_graph = paired.executable.resolved.graph
        layout = bind_raw_state(graph, effective_graph)
    except (ModelCheckpointError, StateLayoutError) as error:
        raise CannotRun("/inputs/subject/checkpoint/model" + error.path, str(error)) from error
    model = sha256(canonical_json_v2_bytes({
        "graph": effective_graph.model_dump(mode="json"),
        "key_schedule": paired.executable.resolved.key_schedule.model_dump(mode="json"),
        "source_overrides": variant.source_overrides,
    })).hexdigest()
    task = sha256(canonical_json_v2_bytes({
        "source": paired.task_source.model_dump(mode="json"),
        "input_bindings": dict(paired.execution.input_bindings),
    })).hexdigest()
    member = ResolvedRolloutMember(
        coordinate=checkpoint.completed_updates, model_tree=slot.tree.descriptor.sha256,
        model=model, task=task, trials=params.trials,
    )
    documents = source.spec.trial_documents
    assert documents is not None
    objective = sha256(canonical_json_v2_bytes(documents.objective.model_dump(mode="json"))).hexdigest()
    return PreparedRollout(
        paired, variant, graph, layout, resolved.scorer, slot, member, objective, paired is not source_pair,
    )


def rollout_dataset(
    prepared: PreparedRollout, evaluated: TrialEvaluation, params: RolloutParameters,
    *, update_axis: str = "checkpoint",
) -> Dataset:
    import jax
    import numpy as np

    from feedbax.contracts.authored_canonical import canonical_sha256
    from feedbax.dataset import Dataset, Variable
    from feedbax.runtime.raw_state import scene_position_key
    from feedbax.tasks.reaching import is_world_target

    paired, variant, graph = prepared.paired, prepared.variant, prepared.graph
    effective_graph = paired.executable.resolved.graph
    slot, layout = prepared.slot, prepared.layout
    history = evaluated.history
    assert history is not None
    n_trials = evaluated.trials.epochs.starts.shape[0]
    trials = [jax.tree.map(lambda value: value[i], history) for i in range(n_trials)]
    variables, state_coordinates = layout.dataset_fields(
        trials, axes=(update_axis, "condition", "trial"),
    )
    terms = evaluated.objective.terms
    for name, term in terms.items():
        if not np.isfinite(term.value):
            raise FloatingPointError(f"Nonfinite objective value at {name}")
        path = json_pointer((name,))
        variables["objective/terms" + path] = Variable(np.asarray(term.value)[None, None], (update_axis, "condition"))
        variables["objective/coverage" + path] = Variable(np.asarray(term.covered)[None, None], (update_axis, "condition"))
    variables["objective/total"] = Variable(np.asarray(evaluated.objective.loss)[None, None], (update_axis, "condition"))
    for signal in paired.execution.task.interface.signals:
        samples = evaluated.trials.inputs[signal.name]
        base_axes = (update_axis, "condition", "trial", "transition")
        if is_world_target(variant.task, signal):
            state_coordinates.setdefault("xy", Variable(np.asarray(["x", "y"]), ("xy",)))
            variable = Variable(np.asarray(samples)[None, None], (*base_axes, "xy"), units="m", frame="world")
        else:
            axes = (*base_axes, *(f"{signal.name}_component_{i}" for i in range(samples.ndim - 2)))
            variable = Variable(np.asarray(samples)[None, None], axes)
        variables[semantic_key("task", "output", (), signal.name)] = variable
    time = np.arange(paired.execution.task.n_steps + 1, dtype=np.float64) * graph.period_seconds
    variation = canonical_sha256(params.model_dump(mode="json")["variation"])
    attrs = {"state_sampling": "reset_and_post_transition", "state_lifetime": "trial"}
    if (position := scene_position_key(effective_graph)) is not None and position in variables:
        attrs["scene_position"] = position
    if params.retention == "raw_states":
        attrs["source_graph_sha256"] = sha256(
            canonical_json_v2_bytes(slot.graph.model_dump(mode="json"))
        ).hexdigest()
        if prepared.varied:
            attrs.update({
                "effective_graph_sha256": canonical_sha256({
                    "resolved_graph": paired.executable.resolved.resolved_sha256,
                    "checkpoint_values": slot.arrays.sha256,
                    "source_overrides": variant.source_overrides,
                }),
                "effective_task_sha256": sha256(
                    canonical_json_v2_bytes(variant.task.model_dump(mode="json"))
                ).hexdigest(),
                "variation_sha256": variation,
            })
    return Dataset.from_arrays(
        variables,
        coords={
            update_axis: Variable(np.asarray([prepared.member.coordinate]), (update_axis,)),
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


def _rollouts(source, checkpoint, params, stop, *, update_axis="checkpoint"):
    if stop.exists():
        return None
    prepared = prepare_rollout(source, checkpoint, params)
    if stop.exists():
        return None
    if update_axis == "update":
        import jax

        compiled = prepared.scorer.lower(prepared.graph, capture_state=prepared.layout.capture).compile()
        evaluated = jax.block_until_ready(compiled(prepared.graph))
    else:
        evaluated = prepared.scorer.evaluate(prepared.graph, capture_state=prepared.layout.capture)
    if stop.exists():
        return None
    return (
        rollout_dataset(prepared, evaluated, params, update_axis=update_axis), float(evaluated.objective.loss),
        prepared.member, prepared.objective,
    )


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
    for checkpoint, source in members:
        if stop.exists():
            return None
        point = points.get(checkpoint.completed_updates)
        if isinstance(point, ScoredEvaluationPoint):
            prepared = prepare_rollout(source, checkpoint, params)
            if point.member != prepared.member or point.objective != prepared.objective:
                raise ValueError("Live Evaluate point differs from its resolved scientific binding.")
            store.verify_object(point.states.object_ref)
            dataset = opened.enter_context(open_stored_dataset(store, point.states))
            if (
                point.member.coordinate != checkpoint.completed_updates
                or point.member.trials != params.trials
                or dataset.descriptor.schema_id != params.states_schema
                or dataset.data("update").tolist() != [checkpoint.completed_updates]
                or dataset.data("trial").tolist() != list(params.trials)
            ):
                raise ValueError("Live Evaluate dataset differs from its resolved inputs.")
            summary = float(dataset.data("objective/total").item())
            if summary != point.total:
                raise ValueError("Live Evaluate summary differs from its durable dataset.")
            member, objective = point.member, point.objective
        else:
            result = _rollouts(source, checkpoint, params, stop, update_axis=update_axis)
            if result is None:
                return None
            dataset, summary, member, objective = result
        datasets.append(dataset)
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
        if subject != request.canonical_request.checkpoint:
            raise ValueError("Launch subject differs from the canonical Evaluate request.")
        events.phase("restoring")
        members, binding = _bound_subject(request)
        events.phase("running")
        is_series = isinstance(subject, ProductRef) and isinstance(subject.select, UpdateSeriesMember)
        with ExitStack() as opened:
            result = _series_dataset(
                members, binding, params, stop, update_axis="update" if is_series else "checkpoint",
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
            train, provenance = evaluation_subject_provenance(binding)
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
