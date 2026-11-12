"""Execute the isolated saved rollout evaluation on the authenticated Train model."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.base import ArtifactRef, ParentRef, Provenance
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    SpecPayload,
    normalize_manifest_spec_payloads,
)
from feedbax.contracts.project_sources import pointer as json_pointer
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host.completion import CompletionFileError
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import publish_dataset
from feedbax.studio.canonical_evaluate import (
    RAW_STATES_SCHEMA,
    ROLLOUT_TYPE,
    STATES_MEDIA_TYPE,
    STATES_SCHEMA,
    BoundEvaluationJobInput,
    CannotRun,
    RolloutParameters,
    decode_rollout_request,
    evaluation_manifest_id,
    state_key,
)
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.training_host_worker import _Events, _Result, _error


def _bound_subject(
    request: BoundEvaluationJobInput,
) -> tuple[BoundStudioTrainingJobInput, PublishedTrainingCheckpoint, dict]:
    from feedbax.job_completion import verify_bound_training_completion

    path = Path(os.environ["FEEDBAX_HOST_COMPLETION_BINDINGS"])
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != os.environ["FEEDBAX_HOST_COMPLETION_BINDINGS_SHA256"]:
        raise CompletionFileError("Host completion binding changed after assignment.")
    (binding,) = strict_json_loads(raw)["bindings"]
    completion = binding["completion"]
    if (
        completion["project_id"] != request.project_id
        or completion["request_id"] != request.canonical_request.checkpoint.request_id
        or completion["manifest_root"] != str(request.manifest_root)
    ):
        raise CompletionFileError("Pinned completion differs from the authored Train subject.")
    source = verify_bound_training_completion(completion)
    if source.canonical_request is None:
        raise CompletionFileError("Pinned subject is missing its canonical Train request.")
    return source, PublishedTrainingCheckpoint.model_validate(completion["checkpoint"]), completion


def _rollouts(
    source: BoundStudioTrainingJobInput,
    checkpoint: PublishedTrainingCheckpoint,
    params: RolloutParameters,
    stop: Path,
):
    import jax
    import numpy as np

    from feedbax.compiler import GraphCompilationError, GraphDocument, compile_graph
    from feedbax.contracts.authored_canonical import canonical_sha256
    from feedbax.contracts.canonical_json import canonical_json_v2_bytes
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload
    from feedbax.contracts.training import TrainingSpec
    from feedbax.dataset import Dataset, Variable
    from feedbax.mechanics.direct_force_components import PointMass
    from feedbax.plugins.composition import compose_application
    from feedbax.runtime._graph_key_schedule import derive_key
    from feedbax.runtime.model_values import MODEL_OWNERS
    from feedbax.runtime.raw_state import StateLayoutError, bind_raw_state
    from feedbax.training.evaluation_variant import resolve_evaluation_variant
    from feedbax.training.model_checkpoint import ModelCheckpointError, ModelRestoreTarget
    from feedbax.training.checkpoint_custody import open_published_model_slot
    from feedbax.training.studio_executor import studio_training_key
    from feedbax.web.worker.diagnostics import GraphCompilationError as WorkerCompilationError
    from feedbax.web.worker.execution import (
        _evaluate_loss,
        compile_training_run,
        prototypes_from_task_bindings,
        rollout_graph,
    )

    spec = source.spec
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise CannotRun("/numerical_settings", "requires CPU execution with JAX x64 disabled")
    if spec.method_ref != STUDIO_GRAPH_METHOD:
        raise CannotRun("/inputs/subject/method_ref", "requires the shared Studio graph executor")
    if spec.task_binding_spec is None:
        raise CannotRun("/inputs/subject/task_binding_spec", "requires declared task bindings")
    if spec.objective.loss is None:
        raise CannotRun("/inputs/subject/objective", "requires a shared loss term plan")
    authored_graph = GraphSpec.model_validate(spec.graph.inline)
    if authored_graph.subgraphs:
        raise CannotRun("/inputs/subject/graph/subgraphs", "nested state owners are unavailable")
    for name, node in authored_graph.nodes.items():
        declaration = MODEL_OWNERS.get(node.type)
        if (
            declaration is None or not declaration.checkpointable or declaration.states is None
            or node.revision not in {None, declaration.revision}
        ):
            raise CannotRun(
                json_pointer(("inputs", "subject", "graph", "nodes", name)),
                "owner revision has no declared raw-state layout",
            )
    slot = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
    if slot is None:
        raise CannotRun("/inputs/subject/checkpoint/model", "requires a typed model slot")
    variant = resolve_evaluation_variant(slot.graph, spec.task, params.variation)
    outputs = variant.graph.output_bindings
    point_mass = [
        (name, node)
        for name, (node, port) in outputs.items()
        if port == "effector" and node in variant.graph.nodes
        and variant.graph.nodes[node].type == "feedbax.PointMass"
    ]
    if len(point_mass) != 1 or variant.graph.subgraphs:
        raise CannotRun(
            "/inputs/subject/graph/output_bindings",
            "requires one declared root PointMass effector output and a single-clock graph",
        )
    _, owner = point_mass[0]
    target_spec = next(
        (item for item in spec.task_binding_spec.exposed_data if item.id == "reach_target"), None,
    )
    if target_spec is None or target_spec.units not in {None, "m"} or target_spec.frame not in {
        None, "world",
    }:
        raise CannotRun(
            "/inputs/subject/task_binding_spec/exposed_data",
            "requires reach_target in the world Cartesian frame in metres",
        )
    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    registry = asyncio.run(compose_application(modules=source.bootstrap_modules)).bundle
    prototypes = prototypes_from_task_bindings(spec.task_binding_spec)
    key = studio_training_key(method.seed, "model")
    source_executable = compile_graph(
        GraphDocument(graph=slot.graph), registry.components,
        input_prototypes=prototypes, initialization_key=key, artifact_provider=slot.provider,
    )
    executable = source_executable
    if variant.graph != slot.graph:
        try:
            executable = compile_graph(
                GraphDocument(graph=variant.graph), registry.components,
                input_prototypes=prototypes, initialization_key=key, artifact_provider=slot.provider,
            )
        except GraphCompilationError as error:
            path = error.record.diagnostics[0].source_anchor.authored_path
            raise CannotRun("/params/variation" + path, str(error)) from error
    target = None
    if executable is not source_executable or variant.source_overrides:
        target = ModelRestoreTarget(executable.resolved.graph, executable.graph, variant.source_overrides)
    try:
        graph = slot.restore(source_executable.graph, target=target)
    except ModelCheckpointError as error:
        raise CannotRun("/inputs/subject/checkpoint/model" + error.path, str(error)) from error

    plant = graph.nodes[owner]
    if not isinstance(plant, PointMass):
        raise CannotRun(f"/inputs/subject/graph/nodes/{owner}", "requires the PointMass owner")
    for name, component in graph.nodes.items():
        clock = MODEL_OWNERS[executable.resolved.graph.nodes[name].type].clock
        if clock is not None and getattr(component, clock) != plant.dt:
            raise CannotRun(
                json_pointer(("inputs", "subject", "graph", "nodes", name, "params", clock)),
                "declared owner clock differs from the PointMass control period",
            )
    try:
        compiled = compile_training_run(
            graph_spec=variant.graph,
            training_spec=TrainingSpec(
                optimizer=method.optimizer,
                loss=spec.objective.loss,
                n_batches=spec.training_config.n_batches,
                batch_size=1,
            ),
            task_spec=variant.task.model_dump(mode="json", exclude_none=True),
            task_binding_spec=spec.task_binding_spec,
            cfg=spec.training_config,
            component_registry=registry.components,
            executable_graph=replace(executable, graph=graph),
        )
    except WorkerCompilationError as error:
        location = error.diagnostics[0].location if error.diagnostics else None
        path = location.get("path", "") if location else ""
        raise CannotRun("/params/variation" + path, str(error)) from error
    graph = compiled.graph
    position = state_key(owner, "effector/position")
    velocity = state_key(owner, "effector/velocity")
    raw = params.retention == "raw_states"
    try:
        layout = bind_raw_state(
            graph, executable.resolved.graph,
            selected=None if raw else frozenset({position, velocity}),
        )
    except StateLayoutError as error:
        raise CannotRun(error.path, str(error)) from error
    histories, targets, totals = [], [], []
    terms_by_id = {}
    for trial in params.trials:
        if stop.exists():
            return None
        key = derive_key(jax.random.PRNGKey(params.seed), "evaluate-trial", trial)
        rollout = rollout_graph(graph, compiled, key=key, capture_state=layout.capture)
        target = np.asarray(rollout["task_data"]["reach_target"])
        if target.shape != (compiled.n_steps, 2):
            raise CannotRun("/inputs/subject/task_binding_spec/exposed_data", "requires 2D targets")
        histories.append(rollout["state_history"])
        targets.append(target)
        total, terms = _evaluate_loss(compiled, rollout)
        totals.append(float(total))
        for name, value in terms.items():
            terms_by_id.setdefault(name, []).append(float(value))
    fields = (*targets, totals, *terms_by_id.values())
    if any(not np.isfinite(field).all() for field in fields):
        raise FloatingPointError("Evaluation produced a nonfinite observation or objective.")

    axes = ("checkpoint", "condition", "trial")
    time = np.arange(compiled.n_steps + 1, dtype=np.float64) * float(plant.dt)

    def field(values, *dims, units=None, frame=None):
        return Variable(np.asarray(values)[None, None], (*axes, *dims), units=units, frame=frame)

    variables, state_coordinates = layout.dataset_fields(histories, axes=axes)
    target_key = semantic_key("task", "output", (), "reach_target")
    variables.update({
        target_key: field(targets, "transition", "xy", units="m", frame="world"),
        "objective/total": field(totals),
    })
    variables.update({
        "objective/terms" + json_pointer((name,)): field(values)
        for name, values in terms_by_id.items()
    })
    variation_sha256 = canonical_sha256(params.model_dump(mode="json")["variation"])
    attrs = {"scene_position": position}
    if raw:
        attrs.update({
            "state_sampling": "reset_and_post_transition",
            "state_lifetime": "trial",
            "source_graph_sha256": sha256(
                canonical_json_v2_bytes(slot.graph.model_dump(mode="json"))
            ).hexdigest(),
        })
        if variant.task != spec.task or any(patch.path.startswith("graph.") for patch in params.variation):
            attrs.update({
                "effective_graph_sha256": canonical_sha256({
                    "resolved_graph": executable.resolved.resolved_sha256,
                    "checkpoint_values": slot.arrays.sha256,
                    "source_overrides": variant.source_overrides,
                }),
                "effective_task_sha256": sha256(
                    canonical_json_v2_bytes(variant.task.model_dump(mode="json"))
                ).hexdigest(),
                "variation_sha256": variation_sha256,
            })
        if any(name.startswith("/v1/model/carry/") for name in variables):
            attrs["carry_address_migration"] = "legacy-gru-hidden-self-loop.v1"
    return Dataset.from_arrays(
        variables,
        coords={
            "checkpoint": Variable(np.asarray([checkpoint.completed_updates]), ("checkpoint",)),
            "condition": Variable(
                np.asarray([variation_sha256]),
                ("condition",),
            ),
            "trial": Variable(np.asarray(params.trials), ("trial",)),
            **state_coordinates,
            "state_time": Variable(time, ("state_time",), units="s"),
            "transition_start": Variable(time[:-1], ("transition",), units="s"),
            "transition_end": Variable(time[1:], ("transition",), units="s"),
        },
        attrs=attrs,
        schema_id=RAW_STATES_SCHEMA if raw else STATES_SCHEMA,
    ), {"objective/total": float(np.mean(totals))}


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
        source, checkpoint, completion = _bound_subject(request)
        events.phase("running")
        result = _rollouts(source, checkpoint, params, stop)
        if result is None or stop.exists():
            return events.result(_Result("stopped"))
        dataset, summary = result
        events.phase("finalizing")
        source_receipt = JobReceipt.model_validate(completion["receipt"])
        train = ParentRef(
            kind="TrainingRunManifest",
            id=source_receipt.manifest_id,
            role="subject",
            metadata={"manifest_sha256": source_receipt.manifest_sha256},
        )
        pointer = checkpoint.pointer
        model = ParentRef(
            kind="TrainingCheckpointTransactionManifest",
            id=pointer.transaction_id,
            role="checkpoint",
            metadata={
                "manifest_sha256": pointer.manifest_sha256,
                "completed_updates": checkpoint.completed_updates,
                "producer_job_id": pointer.run_id,
            },
        )
        projection = EvaluationRunSpec(
            evaluation_type=ROLLOUT_TYPE, training_run_ids=[train.id], inputs=[model],
            params=spec.params,
        )
        with store._exclusive_lock():
            ref = publish_dataset(store, dataset)
            manifest = EvaluationRunManifest(
                id=evaluation_manifest_id(job_id),
                status="completed",
                evaluation_spec=SpecPayload(
                    kind="EvaluationRunSpec", inline=projection.model_dump(mode="json"),
                ),
                input_training_runs=[train],
                provenance=Provenance(parents=[model]),
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
