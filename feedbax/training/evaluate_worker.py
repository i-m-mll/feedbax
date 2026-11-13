"""Execute the isolated saved rollout evaluation on the authenticated Train model."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.base import ArtifactRef
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    SpecPayload,
    normalize_manifest_spec_payloads,
)
from feedbax.contracts.project_sources import pointer as json_pointer
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.request_refs import BatchSetMember
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host.completion import CompletionFileError
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import publish_dataset
from feedbax.studio.canonical_evaluate import (
    ROLLOUT_TYPE,
    STATES_MEDIA_TYPE,
    BoundEvaluationJobInput,
    CannotRun,
    RolloutParameters,
    decode_rollout_request,
    evaluation_manifest_id,
    evaluation_subject_provenance,
)
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.training_host_worker import _Events, _Result, _error


def _bound_subject(
    request: BoundEvaluationJobInput,
) -> tuple[tuple[tuple[PublishedTrainingCheckpoint, BoundStudioTrainingJobInput], ...], dict]:
    from feedbax.job_completion import (
        load_bound_checkpoint_selection,
        read_host_completion_binding,
        verify_bound_training_completion,
    )

    subject = request.canonical_request.checkpoint
    binding = read_host_completion_binding()
    if isinstance(subject.select, BatchSetMember):
        return load_bound_checkpoint_selection(
            binding, root=request.manifest_root, producer_request=subject.request_id,
            coordinates=subject.select.batches, project=request.project_id,
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


def _rollouts(
    source: BoundStudioTrainingJobInput,
    checkpoint: PublishedTrainingCheckpoint,
    params: RolloutParameters,
    stop: Path,
):
    import jax
    import numpy as np

    from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
    from feedbax.contracts.authored_canonical import canonical_sha256
    from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload
    from feedbax.dataset import Dataset, Variable
    from feedbax.plugins.composition import compose_application
    from feedbax.runtime.raw_state import StateLayoutError, bind_raw_state
    from feedbax.tasks.document import TaskParameters
    from feedbax.training.checkpoint_custody import open_published_model_slot
    from feedbax.training.model_checkpoint import ModelCheckpointError, ModelRestoreTarget
    from feedbax.training.evaluation_variant import resolve_evaluation_variant
    from feedbax.training.studio_executor import studio_training_key

    spec = source.spec
    documents = spec.trial_documents
    if documents is None or spec.method_ref != STUDIO_GRAPH_METHOD:
        raise CannotRun("/inputs/subject", "Managed Evaluate needs a document Train checkpoint.")
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise CannotRun("/numerical_settings", "requires CPU execution with JAX x64 disabled")
    slot = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
    if slot is None:
        raise CannotRun("/inputs/subject/checkpoint/model", "requires a typed model slot")
    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    registry = asyncio.run(compose_application(modules=source.bootstrap_modules)).bundle
    variant = resolve_evaluation_variant(slot.graph, documents.task, params.variation)
    key = studio_training_key(method.seed, "model")
    source_pair = compile_trial_execution(
        documents.model_name, documents.model, documents.task,
        registry.components, registry.expression_functions, objective=documents.objective,
        input_overrides=documents.input_overrides, artifact_provider=slot.provider,
        initialization_key=key,
    )
    if not isinstance(source_pair, PreparedTrialExecution):
        problem = source_pair[0]
        raise CannotRun(json_pointer(("inputs", "subject", *problem.path)), problem.reason)
    paired = source_pair
    if variant.graph != slot.graph or variant.task != documents.task:
        paired = compile_trial_execution(
            documents.model_name, documents.model, variant.task,
            registry.components, registry.expression_functions, objective=documents.objective,
            input_overrides=documents.input_overrides, artifact_provider=slot.provider,
            initialization_key=key, graph_override=variant.graph,
        )
    if not isinstance(paired, PreparedTrialExecution):
        problem = paired[0]
        raise CannotRun(json_pointer(("inputs", "subject", *problem.path)), problem.reason)
    assert paired.objective is not None
    try:
        target = None if paired is source_pair else ModelRestoreTarget(
            paired.executable.resolved.graph, paired.executable.graph, variant.source_overrides,
        )
        graph = slot.restore(source_pair.executable.graph, target=target)
        layout = bind_raw_state(graph, paired.executable.resolved.graph)
    except (ModelCheckpointError, StateLayoutError) as error:
        raise CannotRun("/inputs/subject/checkpoint/model" + error.path, str(error)) from error
    if stop.exists():
        return None
    key = jax.random.PRNGKey(params.seed)
    evaluated = paired.execution.validation(
        graph, paired.objective, TaskParameters(), key=key, capture_state=layout.capture,
    )
    if stop.exists():
        return None
    history = evaluated.history
    assert history is not None
    n_trials = evaluated.trials.epochs.starts.shape[0]
    trials = [jax.tree.map(lambda value: value[i], history) for i in range(n_trials)]
    variables, state_coordinates = layout.dataset_fields(
        trials, axes=("checkpoint", "condition", "trial"),
    )
    terms = evaluated.objective.terms
    for name, term in terms.items():
        if not np.isfinite(term.value):
            raise FloatingPointError(f"Nonfinite objective value at {name}")
        path = json_pointer((name,))
        variables["objective/terms" + path] = Variable(np.asarray(term.value)[None, None], ("checkpoint", "condition"))
        variables["objective/coverage" + path] = Variable(np.asarray(term.covered)[None, None], ("checkpoint", "condition"))
    variables["objective/total"] = Variable(np.asarray(evaluated.objective.loss)[None, None], ("checkpoint", "condition"))
    for name, samples in evaluated.trials.inputs.items():
        axes = ("checkpoint", "condition", "trial", "transition", *(
            f"{name}_component_{i}" for i in range(samples.ndim - 2)
        ))
        variables[semantic_key("task", "output", (), name)] = Variable(np.asarray(samples)[None, None], axes)
    time = np.arange(paired.execution.task.n_steps + 1, dtype=np.float64) * graph.period_seconds
    variation = canonical_sha256(params.model_dump(mode="json")["variation"])
    return Dataset.from_arrays(
        variables,
        coords={
            "checkpoint": Variable(np.asarray([checkpoint.completed_updates]), ("checkpoint",)),
            "condition": Variable(np.asarray([variation]), ("condition",)),
            "trial": Variable(np.arange(n_trials), ("trial",)),
            **state_coordinates,
            "state_time": Variable(time, ("state_time",), units="s"),
            "transition_start": Variable(time[:-1], ("transition",), units="s"),
            "transition_end": Variable(time[1:], ("transition",), units="s"),
        },
        attrs={"state_sampling": "reset_and_post_transition", "state_lifetime": "trial"},
        schema_id=params.states_schema,
    ), {"objective/total": float(evaluated.objective.loss)}


def _checkpoint_dataset(datasets):
    """Concatenate singleton rollout datasets along their existing checkpoint axis."""
    if len(datasets) == 1:
        return datasets[0]
    import numpy as np

    from feedbax.dataset import Dataset, Variable

    descriptors = [dataset.descriptor for dataset in datasets]
    base = descriptors[0]
    member_graphs = [descriptor.attrs.pop("effective_graph_sha256", None) for descriptor in descriptors]
    if any(descriptor != base for descriptor in descriptors[1:]):
        raise CannotRun("/inputs/subject/checkpoint", "selected checkpoints have different dataset layouts")

    def join(name, field):
        values = [dataset.data(name) for dataset in datasets]
        if "checkpoint" in field.dims:
            if any(value.dtype != values[0].dtype for value in values[1:]):
                raise CannotRun("/inputs/subject/checkpoint", f"selected checkpoints differ in dtype for {name}")
            array = np.concatenate(values, axis=field.dims.index("checkpoint"))
        else:
            if any(not np.array_equal(value, values[0]) for value in values[1:]):
                raise CannotRun("/inputs/subject/checkpoint", f"selected checkpoints differ in coordinate {name}")
            array = values[0]
        return Variable(array, field.dims, units=field.units, quantity=field.quantity, frame=field.frame)

    coords = {name: join(name, field) for name, field in base.coords.items()}
    if any(graph is not None for graph in member_graphs):
        coords["effective_graph_sha256"] = Variable(np.asarray(member_graphs), ("checkpoint",))
    return Dataset.from_arrays(
        {name: join(name, field) for name, field in base.variables.items()},
        coords=coords, attrs=base.attrs, schema_id=base.schema_id,
    )


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
        datasets, summaries = [], []
        for checkpoint, source in members:
            result = _rollouts(source, checkpoint, params, stop)
            if result is None or stop.exists():
                return events.result(_Result("stopped"))
            dataset, summary = result
            datasets.append(dataset)
            summaries.append(summary["objective/total"])
        dataset = _checkpoint_dataset(datasets)
        summary = {"objective/total": sum(summaries) / len(summaries)}
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
