"""Host entry point for one Studio training attempt."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
import re
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
    from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
    from feedbax.plugins.application import ApplicationRegistryBundle
    from feedbax.contracts.method_checkpoint import MethodCheckpointClock
    from feedbax.contracts.receipt_validity import ValidityCheckDelivery, ValidityCheckEvidence
    from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
    from feedbax.runtime.graph import Graph
    from feedbax.runtime.trial_execution import TrialExecution
    from feedbax.studio.canonical_train import BoundStudioTrainingJobInput, VerifiedTrainBinding
    from feedbax.training.method_runtime import TrainingLoop
    from feedbax.training.methods import TrainingState
    from feedbax.training.model_checkpoint import SemanticModelWriter
    from feedbax.training.train_driver import TrainSchedule


_warm_registries: ApplicationRegistryBundle | None = None


def prewarm() -> None:
    """Discover installed declarations before accepting a delivery."""
    global _warm_registries

    from feedbax.plugins.composition import load_application

    _warm_registries = load_application().bundle


@dataclass(frozen=True)
class _Result:
    outcome: Literal["succeeded", "stopped", "failed"]
    receipt: Mapping[str, str] | None = None
    error: Mapping[str, str] | None = None
    completion: Mapping[str, Any] | None = None
    model_product: Mapping[str, Any] | None = None
    store_publication_error: Mapping[str, str] | None = None
    stopped_checkpoints: ContentObjectRef | None = None


@dataclass
class _MetricReduction:
    count: int
    mean: float
    min: float
    max: float
    last: float

    def add(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.last = value


_MetricReductions = dict[str, Any]
"""Each metric's reduction: a `_MetricReduction` per number, mirroring the metric's mappings and lists."""


def _accumulate_metric(reduction: Any, value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if reduction is None:
            reduction = {}
        if not isinstance(reduction, dict):
            raise ValueError(f"Studio metric {name!r} changed structure")
        for key, item in value.items():
            reduction[key] = _accumulate_metric(reduction.get(key), item, f"{name}.{key}")
    elif isinstance(value, list):
        if reduction is None:
            reduction = [None] * len(value)
        if not isinstance(reduction, list) or len(reduction) != len(value):
            raise ValueError(f"Studio metric {name!r} changed structure")
        for index, item in enumerate(value):
            reduction[index] = _accumulate_metric(reduction[index], item, f"{name}[{index}]")
    elif type(value) in (int, float):
        if reduction is None:
            return _MetricReduction(1, value, value, value, value)
        if not isinstance(reduction, _MetricReduction):
            raise ValueError(f"Studio metric {name!r} changed structure")
        reduction.add(value)
    return reduction


def _accumulate_metrics(reductions: _MetricReductions, metrics: Mapping[str, Any]) -> None:
    for name, value in metrics.items():
        reductions[name] = _accumulate_metric(reductions.get(name), value, name)


def _serialize_reduction(reduction: Any) -> Any:
    if isinstance(reduction, _MetricReduction):
        return asdict(reduction)
    if isinstance(reduction, dict):
        return {name: _serialize_reduction(item) for name, item in reduction.items()}
    if isinstance(reduction, list):
        return [_serialize_reduction(item) for item in reduction]
    return None


def _serialize_reductions(reductions: _MetricReductions) -> dict[str, Any]:
    return {name: _serialize_reduction(reduction) for name, reduction in reductions.items()}


class _Events:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()
        self.current_phase = "importing"
        self.completed_updates = 0
        self.checkpoint: PublishedTrainingCheckpoint | None = None

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def append(self, kind: str, data: Mapping[str, Any], *, sync: bool = False) -> None:
        line = (json.dumps({"kind": kind, "data": data}, allow_nan=False) + "\n").encode()
        with self.path.open("ab", buffering=0) as stream:
            if stream.write(line) != len(line):
                raise OSError(f"incomplete host event write: {kind}")
            if sync:
                os.fsync(stream.fileno())

    def phase(self, phase: str) -> None:
        self.current_phase = phase
        self.append("phase", {"phase": phase, "elapsed_seconds": self.elapsed()})

    def result(self, result: _Result) -> int:
        data: dict[str, Any] = {
            "outcome": result.outcome,
            "completed_updates": self.completed_updates,
            "elapsed_seconds": self.elapsed(),
        }
        if self.checkpoint is not None:
            data["checkpoint"] = self.checkpoint.model_dump(mode="json")
        if result.receipt is not None:
            data["receipt"] = result.receipt
        if result.error is not None:
            data["error"] = result.error
        if result.completion is not None:
            data["completion"] = result.completion
        if result.model_product is not None:
            data["model_product"] = result.model_product
        if result.store_publication_error is not None:
            data["store_publication_error"] = result.store_publication_error
        if result.stopped_checkpoints is not None:
            data["stopped_checkpoints"] = result.stopped_checkpoints.model_dump(mode="json")
        self.append("result", data, sync=True)
        return int(result.outcome == "failed")


def _attempt_root(manifest_root: Path, job_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9_-]+", job_id) is None:
        raise ValueError("job id must be a single identifier")
    return manifest_root / "checkpoints" / job_id


def _error(exc: Exception, phase: str) -> dict[str, str]:
    path = getattr(exc, "path", None)
    diagnostics = getattr(getattr(exc, "record", None), "diagnostics", ())
    if diagnostics:
        path = diagnostics[0].source_anchor.authored_path
    elif diagnostics := getattr(exc, "diagnostics", ()):
        path = (diagnostics[0].location or {}).get("path")
    from feedbax.host.completion import CompletionFileError

    if isinstance(exc, CompletionFileError):
        code = "dependency_corrupt"
    elif isinstance(exc, FloatingPointError):
        code = "numerical_failure"
        phase = "running"
    elif isinstance(exc, OSError) and phase in {"binding", "running", "finalizing"}:
        code = "publication_failed"
    else:
        code = {
            "binding": "invalid_request",
            "building": "preparation_failed",
            "compiling": "compilation_failed",
            "restoring": "incompatible_checkpoint",
            "finalizing": "publication_failed",
        }.get(phase, "executor_failed")
    result = {"code": code, "phase": phase, "message": str(exc) or type(exc).__name__}
    if isinstance(path, str):
        result["path"] = path
    return result


def main(argv: Sequence[str]) -> int:
    """Run one accepted delivery. The launcher owns the terminal event."""
    events = _Events(Path(os.environ["FEEDBAX_HOST_EVENT_LOG"]))
    try:
        if argv:
            events.current_phase = "binding"
            raise ValueError("Studio training accepts its request from the host input document")
        if _warm_registries is None:
            events.phase("importing")
        with ExitStack() as recovery_views:
            result = _run(
                events,
                job_id=os.environ["FEEDBAX_HOST_JOB_ID"],
                input_path=Path(os.environ["FEEDBAX_HOST_INPUT"]),
                stop_file=Path(os.environ["FEEDBAX_HOST_STOP_FILE"]),
                recovery_views=recovery_views,
            )
    except Exception as exc:
        traceback.print_exc()
        result = _Result("failed", error=_error(exc, events.current_phase))
    return events.result(result)


@dataclass(frozen=True)
class _PreparedTrain:
    state: TrainingState
    schedule: TrainSchedule
    clock: MethodCheckpointClock
    loop: TrainingLoop
    execution: TrialExecution
    model_writer: SemanticModelWriter
    templates: tuple[Graph, ...]
    starting_validity: ContentTreeRef | None
    starting_metrics: ContentObjectRef | None


def _prepare_train(
    launch: BoundStudioTrainingJobInput, *, binding: VerifiedTrainBinding,
    registries: ApplicationRegistryBundle, producer: PublishedTrainingCheckpoint | None,
    resumed: bool, source_spec: DocumentTrainingRequestSpec | None, phase: Callable[[str], None],
) -> _PreparedTrain:
    from feedbax.contracts.method_checkpoint import MethodCheckpointIdentity
    from feedbax.contracts.model_input import CheckpointModelSource, InitialModelSource
    from feedbax.persistence.model_assets import model_artifact_provider
    from feedbax.runtime._graph_key_schedule import model_initialization_key
    from feedbax.runtime.model_input import reconstruct_model
    from feedbax.training.method_checkpoint import read_method_checkpoint, validate_method_extra
    from feedbax.training.method_runtime import TrainingLoop, initialize_method, stack_replicas
    from feedbax.training.model_checkpoint import SemanticModelWriter
    from feedbax.training.train_driver import TrainSchedule, carries_method_state, train_start_clock
    from feedbax.validity.recording import numerical_work

    trial, method, settings = binding.trial, binding.method, binding.settings
    identity = MethodCheckpointIdentity(request_id=launch.request_id, method_key=method.key)
    spec = launch.spec
    policy = spec.continuation
    config = spec.training_config
    replica_ids = tuple(map(str, range(config.replicates)))
    restored = None
    if producer is not None:
        phase("restoring")
        with numerical_work():
            restored = read_method_checkpoint(producer.checkpoint_root, producer.pointer, components=registries.components)
        if restored.clock.completed_updates != producer.completed_updates:
            raise ValueError("Checkpoint coordinate differs from its pinned binding.")
        if restored.model.replica_ids != replica_ids:
            raise ValueError("/training_config/replicates: checkpoint replica IDs differ from the target.")
        if resumed and restored.identity != identity:
            raise ValueError("Resume requires the exact admitted Train request and method.")
        if not resumed and policy is not None and policy.state == "preserve" and restored.identity.method_key != identity.method_key:
            raise ValueError("/checkpoint_initialization/state: carrying state requires the same method.")
    clock = train_start_clock(
        spec.seed, source=restored.clock if restored is not None else None, resume=resumed, continuation=policy,
    )

    phase("building")
    graph = trial.graph.graph
    model_assets = restored.model.provider if restored is not None else model_artifact_provider(graph)
    paired = []
    for replica_id in replica_ids:
        with numerical_work():
            result = trial.realize(
                initialization_key=model_initialization_key(spec.seed, replica_id),
                artifact_provider=model_assets,
            )
        if isinstance(result, tuple):
            problem = result[0]
            raise ValueError("/trial_documents/" + "/".join(map(str, problem.path)) + ": " + problem.reason)
        paired.append(result)
    templates = tuple(result.executable.graph for result in paired)
    models = templates
    if restored is not None:
        source_spec = source_spec or spec
        documents = source_spec.trial_documents
        source_graph = trial.graph
        variant = None
        if documents != spec.trial_documents:
            from feedbax.studio.canonical_train import _resolve_trial

            with numerical_work():
                source_graph = _resolve_trial(source_spec, registries).graph
            if source_graph.graph != trial.graph.graph:
                variant = trial.graph
        model_source = CheckpointModelSource(
            tree=restored.model.tree,
            encoding=restored.model.encoding,
            initial=InitialModelSource(document=documents.model, seed=source_spec.seed),
        )
        with numerical_work():
            reconstructed = reconstruct_model(
                model_source, registry=registries, artifacts=model_assets,
                template=source_graph, variant=variant, variant_seed=spec.seed if variant is not None else None,
                slot=restored.model,
            )
        if isinstance(reconstructed, tuple):
            problem = reconstructed[0]
            raise ValueError("/inputs/checkpoint/" + "/".join(map(str, problem.path)) + ": " + problem.reason)
        models = reconstructed.graphs
    writer = SemanticModelWriter.for_graph(graph, model_assets, registries.components)
    if writer is None:
        raise ValueError("Managed Train requires a graph with semantic checkpoint model owners.")
    execution, objective = paired[0].execution, paired[0].objective
    assert objective is not None
    with numerical_work():
        state = initialize_method(method, settings, models=models, seed=spec.seed, replica_ids=replica_ids)
        if restored is not None and carries_method_state(policy, resume=resumed):
            state = state._replace(model=stack_replicas(models),
                                   extra=restored.restore_extra(state.extra))
        validate_method_extra(state.extra, replicas=len(replica_ids))
    phase("compiling")
    schedule = TrainSchedule(
        start=clock.completed_updates, target=spec.target_total, segment_start=clock.segment_start,
        checkpoint_interval=config.snapshot_interval, checkpoints=launch.checkpoint_catalog,
    )
    with numerical_work():
        loop = TrainingLoop(
            method, settings, execution=execution, objective=objective, batch_size=config.batch_size,
            seed=clock.seed, replica_ids=replica_ids, start_step=clock.schedule_origin, sample_at=schedule.prefix_end,
        )

    return _PreparedTrain(
        state=state,
        schedule=schedule,
        clock=clock,
        loop=loop, execution=execution, model_writer=writer, templates=templates,
        starting_validity=restored.manifest.validity if restored is not None else None,
        starting_metrics=restored.manifest.metrics_prefix if restored is not None else None,
    )


def _run(
    events: _Events, *, job_id: str, input_path: Path, stop_file: Path, recovery_views: ExitStack,
) -> _Result:
    from math import isfinite

    import jax.random as jr

    from feedbax._atomic import write_bytes_atomic
    from feedbax.contracts.base import ArtifactRef, Provenance, sha256_file, utc_now
    from feedbax.contracts.canonical_json import canonical_json_v2_bytes
    from feedbax.contracts.effective_success import RequestObject
    from feedbax.contracts.manifest import (
        InheritedTerminalCheckpoint, TrainingRunCertification, TrainingRunManifest,
        canonical_manifest_path, load_manifest_bytes, training_run_manifest_id, write_manifest,
    )
    from feedbax.contracts.method_checkpoint import MethodCheckpointIdentity
    from feedbax.contracts.strict_json import strict_json_loads
    from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
    from feedbax.host.completion import CompletionFileError, read_completion_bindings
    from feedbax.job_completion import (
        attest_checkpoint_publication, checkpoint_reference, training_binding_path,
        verify_bound_training_completion, verify_training_completion, verify_training_model_product,
    )
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.method_checkpoint import (
        bind_method_checkpoint_content, materialize_checkpoint_content, retain_checkpoint_evidence,
    )
    from feedbax.persistence.train_metrics import MetricDelta, latest_metrics as read_latest_metrics, materialize_metrics, read_metric_prefix
    from feedbax.runtime._graph_key_schedule import derive_key
    from feedbax.studio.canonical_train import (
        BoundStudioTrainingJobInput, PinnedTrainCheckpoint, read_collected_train_job, resolve_training_request_binding,
    )
    from feedbax.tasks.document import TaskParameters
    from feedbax.training.method_checkpoint import write_method_checkpoint
    from feedbax.training.method_runtime import replica_value
    from feedbax.training.train_driver import drive_train
    from feedbax.validity.evidence import stage_checkpoint_evidence
    from feedbax.validity.recording import numerical_work, record_computation, set_execution_prefix
    from feedbax.validity.train import (
        CheckpointStart, TrainCapture, TrainReplay, archive_checkpoint_validity, checkpoint_evidence,
    )

    events.current_phase = "binding"
    launch = BoundStudioTrainingJobInput.model_validate(strict_json_loads(input_path.read_bytes()))
    manifest_root = launch.manifest_root
    if not manifest_root.is_absolute() or manifest_root != manifest_root.resolve():
        raise ValueError("Canonical Train store must be absolute and canonical")
    store = ImmutableArtifactBlobProvider(manifest_root)
    if _warm_registries is None:
        prewarm()
    assert _warm_registries is not None
    registries = _warm_registries
    with numerical_work():
        trial_binding = resolve_training_request_binding(launch, registries=registries)
    method = trial_binding.method
    identity = MethodCheckpointIdentity(request_id=launch.request_id, method_key=method.key)
    spec = launch.spec
    policy = spec.continuation
    config = spec.training_config
    target = spec.target_total
    replica_ids = tuple(map(str, range(config.replicates)))
    checkpoint_root = _attempt_root(manifest_root, job_id)
    binding_path = training_binding_path(manifest_root, job_id)
    if launch.checkpoint_input is not None:
        raise ValueError("The Host delivery owns the resolved Train checkpoint input.")

    dependencies = (
        read_completion_bindings()["bindings"] if "FEEDBAX_HOST_COMPLETION_BINDINGS" in os.environ else []
    )
    resume = launch.resume
    source_launch = None
    collected = None
    checkpoint_ref = launch.canonical_request.checkpoint
    if launch.collected_resume is not None:
        if dependencies or launch.collected_resume == job_id:
            raise CompletionFileError("Collected Resume has conflicting recovery authority.")
        events.current_phase = "restoring"
        collected = read_collected_train_job(launch)
        pointer = collected.recovery.pointer
        updates = pointer.completed_updates
        if updates is None or updates >= target or pointer.run_id == job_id:
            raise CompletionFileError("Collected Resume requires a checkpoint before the authored target.")
        source = recovery_views.enter_context(materialize_checkpoint_content(
            collected.recovery.content, store, owner=job_id,
        ))
        resume = PublishedTrainingCheckpoint(checkpoint_root=source, completed_updates=updates, pointer=pointer)
    elif checkpoint_ref is not None:
        if len(dependencies) != 1:
            raise CompletionFileError("Train requires one bound predecessor checkpoint.")
        product = dependencies[0]["completion"]
        if (
            product["project_id"] != launch.project_id
            or product["request_id"] != checkpoint_ref.request_id
            or product["manifest_root"] != str(manifest_root)
        ):
            raise CompletionFileError("Bound Train checkpoint differs from its canonical input.")
        if resume is None:
            scope = "full" if policy is not None and policy.state == "preserve" else "model_only"
            source_launch = verify_bound_training_completion(product, scope=scope)
        launch = launch.model_copy(update={"checkpoint_input": PinnedTrainCheckpoint(
            producer_job_id=product["producer_job_id"],
            receipt=product["receipt"],
            checkpoint=PublishedTrainingCheckpoint.model_validate(product["checkpoint"]),
        )})
    elif dependencies:
        raise CompletionFileError("Train delivery has an undeclared predecessor.")

    if resume is not None and launch.collected_resume is None:
        events.current_phase = "restoring"
        parent_root = _attempt_root(manifest_root, resume.pointer.run_id)
        if resume.checkpoint_root != parent_root or parent_root == checkpoint_root:
            raise ValueError("Resume source lies outside the original attempt's custody.")
        parent_binding = BoundStudioTrainingJobInput.model_validate_json((parent_root / "binding.json").read_bytes())
        if launch.model_dump(exclude={"resume"}) != parent_binding.model_dump(exclude={"resume"}):
            raise ValueError("Resume inputs differ from the original attempt's binding.")

    events.current_phase = "binding"
    write_bytes_atomic(binding_path, launch.model_dump_json(exclude_none=True).encode(), replace=False)
    binding_sha256 = sha256_file(binding_path)
    events.append("binding", {
        "binding_path": str(binding_path), "binding_sha256": binding_sha256,
        "resume": resume.model_dump(mode="json") if resume is not None else None,
    }, sync=True)

    manifest_id = training_run_manifest_id(job_id)
    manifest_path = canonical_manifest_path("TrainingRunManifest", manifest_id, root=manifest_root)
    checkpoints = []
    checkpoint_proofs: dict[str, tuple[ContentTreeRef | None, ContentObjectRef | None]] = {}
    latest_metrics = {}
    history = checkpoint_root / "metrics.jsonl"
    method_provenance = asdict(method.source) | {"key": method.key}
    provenance = {"method": method_provenance, "request_id": launch.request_id, "binding_sha256": binding_sha256}
    if launch.checkpoint_input is not None:
        provenance["checkpoint_initialization"] = checkpoint_reference(launch.checkpoint_input.checkpoint).model_dump(mode="json")
    if resume is not None:
        provenance["recovery"] = checkpoint_reference(resume).model_dump(mode="json")

    def finish_manifest(status: Literal["completed", "cancelled", "failed"], *, error: dict | None = None) -> dict:
        kind = "diverged" if error and error["code"] == "numerical_failure" else "crashed" if error else status
        inherited = None
        terminal = None
        if status == "completed" and not checkpoints and resume is not None:
            terminal = checkpoint_reference(resume)
            inherited = InheritedTerminalCheckpoint(checkpoint=terminal)
        committed = [checkpoint_reference(checkpoint) for checkpoint in checkpoints]
        metrics = []
        if history.is_file() and history.stat().st_size:
            metrics = [ArtifactRef(
                role="training_metrics", logical_name="updates", media_type="application/x-ndjson",
                sha256=sha256_file(history), size_bytes=history.stat().st_size,
                uri=history.relative_to(manifest_root).as_posix(),
                metadata={"replicas": replica_ids},
            )]
        write_manifest(TrainingRunManifest(
            id=manifest_id, job_id=job_id, status=status, completed_batches=events.completed_updates,
            completed_at=utc_now(), stopped=status != "completed",
            failure_kind="nan_guard" if kind == "diverged" else None,
            checkpoint_custody=committed, artifacts=metrics,
            summary_metrics=latest_metrics, metadata={"error": error} if error else {},
            provenance=Provenance(metadata=provenance),
            inherited_terminal_checkpoint=inherited,
            terminal_certification=TrainingRunCertification(
                schema_id="feedbax.manifest.training_run_certification",
                schema_version="feedbax.manifest.training_run_certification.v1",
                termination_reason=kind, certified_artifacts=committed if terminal is None else [terminal],
            ),
        ), root=manifest_root)
        raw = manifest_path.read_bytes()
        manifest = load_manifest_bytes(raw)
        if not isinstance(manifest, TrainingRunManifest) or manifest.id != manifest_id or manifest.status != status:
            raise ValueError("Published Train receipt has the wrong identity or status.")
        return {"manifest_id": manifest_id, "manifest_sha256": sha256(raw).hexdigest()}

    try:
        producer = resume or (launch.checkpoint_input.checkpoint if launch.checkpoint_input is not None else None)
        prepared = _prepare_train(
            launch, binding=trial_binding, registries=registries, producer=producer, resumed=resume is not None,
            source_spec=source_launch.spec if source_launch is not None else None, phase=events.phase,
        )
        state, loop = prepared.state, prepared.loop
        execution, writer, templates = prepared.execution, prepared.model_writer, prepared.templates
        completed = prepared.clock.completed_updates
        seed = prepared.clock.seed
        metric_prefix = prepared.starting_metrics if resume is not None else None
        inherited_metrics = read_metric_prefix(
            store, metric_prefix, completed=completed, origin=prepared.clock.segment_start,
        )
        inherited_refs = inherited_metrics.references
        latest_metrics = read_latest_metrics(store, inherited_metrics)
        replay_checkpoint = None
        if producer is not None:
            if resume is not None:
                if prepared.starting_validity is None:
                    raise ValueError("Resume checkpoint does not retain its executed prefix evidence.")
                set_execution_prefix(checkpoint_evidence(store, prepared.starting_validity))
            source_request = None
            if source_launch is not None:
                source_request = RequestObject(
                    id=source_launch.request_id, object=source_launch.canonical_request.request,
                    documents=source_launch.canonical_request.documents,
                )
            replay_checkpoint = CheckpointStart(
                kind="resume" if resume is not None else "predecessor",
                pointer=producer.pointer,
                content=(collected.recovery.content if collected is not None else
                         bind_method_checkpoint_content(producer.checkpoint_root, producer.pointer, store)),
                source_request=source_request,
            )
        capture = TrainCapture(loop, state, prepared.schedule, TrainReplay(
            request=RequestObject(
                id=launch.request_id, object=launch.canonical_request.request,
                documents=launch.canonical_request.documents,
            ),
            checkpoint_catalog=launch.checkpoint_catalog, checkpoint=replay_checkpoint,
        ), store)
        record_computation(lambda: capture.snapshot(completed))
        loop.observe_lowering = capture.observe
        observation_key = derive_key(jr.PRNGKey(seed), "training-observation")
        start_boundary = completed
        events.completed_updates = completed
        if resume is not None and launch.collected_resume is None:
            events.checkpoint = resume
        events.phase("running")
        events.append("progress", {
            "completed_updates": completed, "total_updates": target, "elapsed_seconds": events.elapsed(),
            "metrics": latest_metrics,
        })
        evaluate_checkpoint = None
        if "FEEDBAX_HOST_LIVE_EVALUATE" in os.environ:
            from feedbax.training.live_evaluate import live_evaluation_callback

            try:
                evaluate_checkpoint = live_evaluation_callback(launch, events, registry=registries)
            except (OSError, ValueError) as error:
                events.append("live_evaluate_error", _error(error, "live_evaluate"), sync=True)

        reduction_start = start_boundary
        reductions: _MetricReductions = {}
        last_progress = float("-inf")
        last_frame = -1
        pending = False

        def publish_progress() -> None:
            nonlocal pending, last_progress, reduction_start
            if pending:
                last_progress = events.elapsed()
                events.append("progress", {
                    "completed_updates": completed, "total_updates": target,
                    "elapsed_seconds": last_progress, "metrics": latest_metrics,
                    "metric_interval": {
                        "start_exclusive": reduction_start, "end_inclusive": completed,
                        "metrics": _serialize_reductions(reductions),
                    },
                })
                pending = False
                reductions.clear()
                reduction_start = completed

        def publish_checkpoint(delta: MetricDelta, *, final: bool) -> None:
            nonlocal inherited_refs, metric_prefix
            with store._exclusive_lock():
                metric_prefix, delta_refs = delta.checkpoint(store, completed)
                validity = archive_checkpoint_validity(store)
                evidence = (*inherited_refs, *delta_refs, *((validity,) if validity is not None else ()))
                if evidence:
                    stage_checkpoint_evidence(store, job_id, *evidence)
            write = write_method_checkpoint(
                checkpoint_root, run_id=job_id, identity=identity,
                clock=prepared.clock.model_copy(update={"completed_updates": completed}),
                state=state, model_writer=writer, model_templates=templates, replica_ids=replica_ids, final=final,
                validity=validity, metrics_prefix=metric_prefix,
            )
            retain_checkpoint_evidence(store, write.root, write.latest_pointer, evidence=evidence)
            inherited_refs = ()
            checkpoint_proofs[write.latest_pointer.transaction_id] = validity, metric_prefix
            checkpoint = PublishedTrainingCheckpoint(
                checkpoint_root=write.root, completed_updates=completed, pointer=write.latest_pointer,
            )
            publication = {
                "key": f"checkpoint:{completed}", "checkpoint": checkpoint.model_dump(mode="json"),
                "elapsed_seconds": events.elapsed(),
                "checkpoint_product": attest_checkpoint_publication(
                    launch=launch, binding_sha256=binding_sha256, checkpoint=checkpoint,
                ),
            }
            events.append("publication", publication, sync=True)
            events.checkpoint = checkpoint
            checkpoints.append(checkpoint)
            if evaluate_checkpoint is not None:
                evaluate_checkpoint(checkpoint, tuple(replica_value(state.model, i) for i in range(len(replica_ids))))

        def publish_frame() -> None:
            nonlocal last_frame
            if completed == last_frame or completed == start_boundary:
                return
            frame_key = derive_key(observation_key, "trajectory", index=completed)
            trial_sample = execution.task.ahead(TaskParameters(), frame_key, frame_key)
            retained, _ = execution.rollout(replica_value(state.model, 0), trial_sample, frame_key)
            events.append("frame", {"completed_updates": completed, "trajectory": {
                "n_steps": execution.task.n_steps, "outputs": {name: value.tolist() for name, value in retained.items()},
            }})
            last_frame = completed

        def metric_value(value):
            if isinstance(value, list):
                return [metric_value(item) for item in value]
            return None if type(value) is float and not isfinite(value) else value

        def metrics_at(metrics, index):
            if isinstance(metrics, Mapping):
                return {name: metrics_at(value, index) for name, value in metrics.items()}
            value = metrics[index, 0] if config.replicates == 1 else metrics[index]
            return metric_value(value.tolist())

        with TemporaryDirectory(prefix=".metrics-", dir=checkpoint_root) as scratch, (
            chunk_path := Path(scratch) / "updates.jsonl"
        ).open("xb") as stream:
            delta = MetricDelta(stream, chunk_path, start=completed, previous=metric_prefix)

            def record_block(block, batch) -> None:
                nonlocal state, completed, latest_metrics, pending
                state, completed = batch.state, batch.completed
                events.completed_updates = completed
                capture.prefix.record(block, batch)
                for offset in range(completed - block.completed):
                    latest_metrics = metrics_at(batch.metrics, offset)
                    _accumulate_metrics(reductions, latest_metrics)
                    delta.write(block.completed + offset + 1, latest_metrics)
                    pending = True
                if pending and events.elapsed() - last_progress >= 0.4:
                    publish_progress()

            def commit_checkpoint(final: bool) -> None:
                publish_checkpoint(delta, final=final)
                publish_frame()

            def run_numerical_block(current, block):
                with numerical_work():
                    return loop.run_block(current, completed=block.completed, count=block.count)

            progress = drive_train(
                prepared.schedule, state,
                step=run_numerical_block,
                record=record_block, checkpoint=commit_checkpoint, should_stop=stop_file.exists,
            )

        events.phase("finalizing")
        publish_progress()
        if progress.rejected:
            raise FloatingPointError("A training update produced non-finite state and was not committed.")
        metric_history = read_metric_prefix(
            store, metric_prefix, completed=completed, origin=prepared.clock.segment_start,
        )
        materialize_metrics(store, metric_history, history)
        if progress.stopped:
            stopped_receipt = finish_manifest("cancelled")
            try:
                from feedbax.persistence.train_job import TrainJobCheckpoint

                custody = []
                for checkpoint in checkpoints:
                    validity, metrics_prefix = checkpoint_proofs[checkpoint.pointer.transaction_id]
                    custody.append(TrainJobCheckpoint(
                        pointer=checkpoint.pointer, validity=validity, metrics_prefix=metrics_prefix,
                        content=bind_method_checkpoint_content(checkpoint.checkpoint_root, checkpoint.pointer, store),
                    ).model_dump(mode="json"))
                staged = store.store_object(canonical_json_v2_bytes(custody))
            except (OSError, ValueError) as error:
                return _Result("stopped", receipt=stopped_receipt, store_publication_error={
                    "code": "publication_failed", "message": str(error),
                })
            return _Result("stopped", receipt=stopped_receipt, stopped_checkpoints=staged)

        receipt = finish_manifest("completed")
        completion = verify_training_completion(
            producer_job_id=job_id, launch=launch, binding_sha256=binding_sha256,
            receipt=receipt, checkpoint=events.checkpoint,
        )
        product, _ = verify_training_model_product(completion)
        return _Result("succeeded", receipt=receipt, completion=completion, model_product=product.model_dump(mode="json"))
    except Exception as exc:
        traceback.print_exc()
        error = _error(exc, events.current_phase)
        try:
            receipt = finish_manifest("failed", error=error) if not manifest_path.exists() else None
        except Exception:
            traceback.print_exc()
            receipt = None
        return _Result("failed", receipt=receipt, error=error)


def check_train(delivery: ValidityCheckDelivery) -> ValidityCheckEvidence:
    """Re-prepare the exact recorded Train start for code or environment confirmation."""
    from feedbax.validity.train import recheck_train

    if _warm_registries is None:
        prewarm()
    assert _warm_registries is not None
    return recheck_train(delivery, _warm_registries)
