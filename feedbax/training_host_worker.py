"""Host entry point for one Studio training attempt."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
    from feedbax.contracts.worker import ProgressCoordinate
    from feedbax.training.checkpoint_custody import CheckpointWriteResult


@dataclass(frozen=True)
class _Result:
    outcome: Literal["succeeded", "stopped", "failed"]
    receipt: Mapping[str, str] | None = None
    error: Mapping[str, str] | None = None
    completion: Mapping[str, Any] | None = None
    completion_error: Mapping[str, str] | None = None
    model_product: Mapping[str, Any] | None = None
    store_publication_error: Mapping[str, str] | None = None


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


_MetricReductions = dict[str, "_MetricReduction | _MetricReductions"]


def _accumulate_metrics(reductions: _MetricReductions, metrics: Mapping[str, Any]) -> None:
    for name, value in metrics.items():
        if isinstance(value, Mapping):
            nested = reductions.setdefault(name, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Studio metric {name!r} changed structure")
            _accumulate_metrics(nested, value)
        elif type(value) in (int, float):
            reduction = reductions.get(name)
            if reduction is None:
                reductions[name] = _MetricReduction(1, value, value, value, value)
            elif isinstance(reduction, _MetricReduction):
                reduction.add(value)
            else:
                raise ValueError(f"Studio metric {name!r} changed structure")


def _serialize_reductions(reductions: _MetricReductions) -> dict[str, Any]:
    return {
        name: asdict(reduction) if isinstance(reduction, _MetricReduction)
        else _serialize_reductions(reduction)
        for name, reduction in reductions.items()
    }


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
        if result.completion_error is not None:
            data["completion_error"] = result.completion_error
        if result.model_product is not None:
            data["model_product"] = result.model_product
        if result.store_publication_error is not None:
            data["store_publication_error"] = result.store_publication_error
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
        if "feedbax.training.managed_studio" not in sys.modules:
            events.phase("importing")
        result = _run(
            events,
            job_id=os.environ["FEEDBAX_HOST_JOB_ID"],
            input_path=Path(os.environ["FEEDBAX_HOST_INPUT"]),
            stop_file=Path(os.environ["FEEDBAX_HOST_STOP_FILE"]),
        )
    except Exception as exc:
        traceback.print_exc()
        result = _Result("failed", error=_error(exc, events.current_phase))
    return events.result(result)


def _run(events: _Events, *, job_id: str, input_path: Path, stop_file: Path) -> _Result:
    from feedbax._atomic import write_bytes_atomic
    from feedbax.contracts.base import ParentRef, sha256_file
    from feedbax.contracts.checkpoint_initialization import CheckpointInitializationRequest
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.manifest import (
        TrainingRunManifest,
        canonical_manifest_path,
        load_manifest_bytes,
        training_run_manifest_id,
        write_manifest,
    )
    from feedbax.contracts.strict_json import strict_json_loads
    from feedbax.contracts.studio_training import (
        STUDIO_GRAPH_METHOD,
        PublishedTrainingCheckpoint,
        StudioGraphMethodPayload,
    )
    from feedbax.host.completion import CompletionFileError
    from feedbax.job_completion import (
        CompletionUnavailable,
        attest_checkpoint_publication,
        training_binding_path,
        verify_bound_training_completion,
        verify_training_completion,
        verify_training_model_product,
    )
    from feedbax.orchestration.events import normalize_serialized_metrics
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.plugins.composition import compose_application
    from feedbax.runtime._graph_key_schedule import derive_key
    from feedbax.studio.canonical_train import (
        BoundStudioTrainingJobInput,
        PinnedTrainCheckpoint,
        verify_training_request_binding,
    )
    from feedbax.training.checkpoint_custody import (
        CheckpointCustodyError,
        bind_published_checkpoint_content,
        open_published_model_slot,
    )
    from feedbax.training.executor import execute_managed_training_run
    from feedbax.training.manifest_preflight import build_training_run_manifest_spec_payloads
    from feedbax.training.model_checkpoint import (
        ModelCheckpointError,
        SemanticModelWriter,
        effective_model_graph,
    )
    from feedbax.training.preparation import ExecutionPreparationResult
    from feedbax.training.studio_checkpoint_fork import plan_studio_checkpoint_fork
    from feedbax.training.managed_studio import prepare_managed_training
    from feedbax.training.studio_executor import StudioUpdateKernel, studio_training_key
    from feedbax.web.model_assets import model_artifact_provider

    events.current_phase = "binding"
    launch = BoundStudioTrainingJobInput.model_validate(strict_json_loads(input_path.read_text()))
    if launch.spec.trial_documents is None:
        raise ValueError("/trial_documents: Managed Train requires authored documents.")
    if launch.spec.method_ref != STUDIO_GRAPH_METHOD:
        raise ValueError("Studio training jobs require the registered Studio graph method")
    method = StudioGraphMethodPayload.model_validate(launch.spec.method_payload.payload)
    method.require_supported_launch(resume=launch.resume is not None)
    if not launch.manifest_root.is_absolute():
        raise ValueError("manifest_root must be absolute")
    manifest_root = launch.manifest_root.resolve()
    if launch.canonical_request is not None:
        if launch.manifest_root != manifest_root:
            raise ValueError("Canonical Train store must be absolute and canonical")
        verify_training_request_binding(launch)
    if launch.checkpoint_input is not None:
        raise ValueError("The Host delivery owns the resolved Train checkpoint input.")
    checkpoint_root = _attempt_root(manifest_root, job_id)
    binding_path = training_binding_path(manifest_root, job_id)
    dependencies = []
    if sidecar := os.environ.get("FEEDBAX_HOST_COMPLETION_BINDINGS"):
        try:
            raw_dependencies = Path(sidecar).read_bytes()
        except OSError as error:
            raise CompletionFileError("Host completion bindings are unavailable.") from error
        if sha256(raw_dependencies).hexdigest() != os.environ.get(
            "FEEDBAX_HOST_COMPLETION_BINDINGS_SHA256"
        ):
            raise CompletionFileError("Host completion bindings changed after assignment.")
        dependencies = strict_json_loads(raw_dependencies)["bindings"]
    checkpoint_ref = launch.canonical_request.checkpoint if launch.canonical_request else None
    stateful = method.continuation
    source_scope = stateful.source_slots if stateful is not None else "model_only"
    if launch.resume is None or stateful is None:
        for dependency in dependencies:
            verify_bound_training_completion(dependency["completion"], scope=source_scope)
    if checkpoint_ref is not None:
        if len(dependencies) != 1:
            raise CompletionFileError("Train requires one bound predecessor checkpoint.")
        product = dependencies[0]["completion"]
        if (
            product["project_id"] != launch.project_id
            or product["request_id"] != checkpoint_ref.request_id
            or product["manifest_root"] != str(manifest_root)
        ):
            raise CompletionFileError("Bound Train checkpoint differs from its canonical input.")
        launch = launch.model_copy(update={"checkpoint_input": PinnedTrainCheckpoint(
            producer_job_id=product["producer_job_id"],
            receipt=product["receipt"],
            checkpoint=PublishedTrainingCheckpoint.model_validate(product["checkpoint"]),
        )})
    elif method.checkpoint_initialization is not None:
        raise ValueError("Checkpoint initialization requires a canonical Train input.")
    resume = launch.resume
    if resume is not None:
        events.current_phase = "restoring"
        parent_root = _attempt_root(manifest_root, resume.pointer.run_id)
        if resume.checkpoint_root != parent_root or parent_root == checkpoint_root:
            raise ValueError("resume source is outside the original attempt's custody")
        if resume.completed_updates != resume.pointer.completed_training_batches:
            raise ValueError("resume coordinate disagrees with its checkpoint pointer")
        parent_binding = BoundStudioTrainingJobInput.model_validate_json(
            (parent_root / "binding.json").read_bytes()
        )
        if launch.model_dump(exclude={"resume"}) != parent_binding.model_dump(exclude={"resume"}):
            raise ValueError("resume inputs differ from the original attempt's binding")
        events.completed_updates = resume.completed_updates

    start_boundary = stateful.progress.source_completed_batches if stateful and resume is None else 0

    events.current_phase = "binding"
    write_bytes_atomic(binding_path, launch.model_dump_json(exclude_none=True).encode(), replace=False)
    binding_sha256 = sha256_file(binding_path)
    events.append(
        "binding",
        {
            "binding_path": str(binding_path),
            "binding_sha256": binding_sha256,
            "resume": resume.model_dump(mode="json") if resume is not None else None,
        },
        sync=True,
    )
    manifest_id = training_run_manifest_id(job_id)
    manifest_path = canonical_manifest_path("TrainingRunManifest", manifest_id, root=manifest_root)

    def receipt(expected_status: str) -> dict[str, str]:
        raw = manifest_path.read_bytes()
        manifest = load_manifest_bytes(raw)
        if (
            not isinstance(manifest, TrainingRunManifest)
            or manifest.id != manifest_id
            or manifest.status != expected_status
        ):
            raise ValueError("published training receipt has the wrong identity or status")
        return {"manifest_id": manifest_id, "manifest_sha256": sha256(raw).hexdigest()}

    try:
        graph = GraphSpec.model_validate(launch.spec.graph.inline)
        if resume is not None:
            events.current_phase = "restoring"
        source_model = (
            open_published_model_slot(resume.checkpoint_root, resume.pointer)
            if resume is not None else None
        )
        model_assets = (
            source_model.provider if source_model is not None else model_artifact_provider(graph)
        )
        events.phase("building")
        registries = asyncio.run(compose_application(modules=launch.bootstrap_modules)).bundle
        events.phase("compiling")
        preparation = prepare_managed_training(launch.spec, registries, model_assets=model_assets)
        if not isinstance(preparation, ExecutionPreparationResult):
            raise ValueError("Studio graph training requires scalar execution preparation")
        if source_model is not None and source_model.graph != effective_model_graph(
            graph, preparation.initial_slots["model"],
        ):
            events.current_phase = "restoring"
            raise ModelCheckpointError("/graph", "Exact Resume graph differs from the bound model graph")
        kernel = preparation.kernel_context["studio_graph_update"]
        if not isinstance(kernel, StudioUpdateKernel):
            raise ValueError("Studio graph training requires the registered numerical update")
        spec = launch.spec
        target = method.target_total
        snapshot_interval = spec.training_config.snapshot_interval
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        observation_key = studio_training_key(method.seed, "observation")
        last_progress = float("-inf")
        interval_start = start_boundary or events.completed_updates
        reductions: _MetricReductions = {}
        pending: tuple[dict[str, Any], float] | None = None
        last_frame = -1
        latest_model: Any = None
        stop_requested = False

        def publish_progress() -> None:
            nonlocal pending, interval_start, last_progress
            if pending is None:
                return
            metrics, observed_at = pending
            events.append(
                "progress",
                {
                    "completed_updates": events.completed_updates,
                    "total_updates": target,
                    "elapsed_seconds": observed_at,
                    "metrics": metrics,
                    "metric_interval": {
                        "start_exclusive": interval_start,
                        "end_inclusive": events.completed_updates,
                        "metrics": _serialize_reductions(reductions),
                    },
                },
            )
            pending = None
            interval_start = events.completed_updates
            reductions.clear()
            last_progress = observed_at

        def publish_frame() -> None:
            nonlocal last_frame
            completed = events.completed_updates
            if latest_model is not None and completed != last_frame:
                frame_key = derive_key(observation_key, "trajectory", index=completed)
                events.append(
                    "frame",
                    {
                        "completed_updates": completed,
                        "trajectory": kernel.trajectory_snapshot(latest_model, frame_key),
                    },
                )
                last_frame = completed

        def on_start(completed: int) -> None:
            if completed != (start_boundary or events.completed_updates):
                raise ValueError("restored state disagrees with the bound checkpoint coordinate")
            events.completed_updates = completed
            events.checkpoint = resume
            events.phase("running")
            events.append(
                "progress",
                {
                    "completed_updates": completed,
                    "total_updates": target,
                    "elapsed_seconds": events.elapsed(),
                    "metrics": {},
                },
            )

        def on_update(
            coordinate: ProgressCoordinate, completed: int, slots: Mapping[str, Any]
        ) -> None:
            nonlocal pending, latest_model
            events.completed_updates = completed
            latest_model = slots["model"]
            now = events.elapsed()
            _, metrics = normalize_serialized_metrics(coordinate, coordinate.metrics, {})
            _accumulate_metrics(reductions, metrics)
            pending = metrics, now
            if now - last_progress >= 0.4:
                publish_progress()
            if completed % snapshot_interval == 0:
                publish_frame()

        def on_checkpoint(write: CheckpointWriteResult) -> None:
            completed = write.latest_pointer.completed_training_batches
            if completed is None:
                raise ValueError("published checkpoint does not carry its completed-update count")
            checkpoint = PublishedTrainingCheckpoint(
                checkpoint_root=write.root,
                completed_updates=completed,
                pointer=write.latest_pointer,
            )
            publication = {
                "key": f"checkpoint:{completed}",
                "checkpoint": checkpoint.model_dump(mode="json"),
                "elapsed_seconds": events.elapsed(),
            }
            if (
                launch.canonical_request is not None
                and checkpoint.pointer.completed_coordinate.completed_barrier == "after_train_batch"
            ):
                try:
                    publication["checkpoint_product"] = attest_checkpoint_publication(
                        launch=launch, binding_sha256=binding_sha256, checkpoint=checkpoint, write=write,
                    )
                except (ValueError, OSError) as exc:
                    publication["checkpoint_product_error"] = str(exc)
            events.append("publication", publication, sync=True)
            events.checkpoint = checkpoint

        def stop_before_update() -> bool:
            nonlocal stop_requested
            stop_requested = stop_requested or stop_file.exists()
            return stop_requested and events.completed_updates > start_boundary

        def finalize() -> None:
            events.phase("finalizing")
            publish_progress()
            publish_frame()

        initialization = None
        fork_plan = None
        fork_bindings = None
        source_content = None
        semantic_model = SemanticModelWriter.for_graph(graph, model_assets)
        provider = None
        source_root = resume.checkpoint_root if resume is not None else None
        if resume is not None or launch.checkpoint_input is not None:
            events.phase("restoring")
        if resume is None and launch.checkpoint_input is not None:
            checkpoint = launch.checkpoint_input.checkpoint
            source_root = checkpoint.checkpoint_root
            provider = ImmutableArtifactBlobProvider(manifest_root)
            source, source_content = bind_published_checkpoint_content(
                source_root, checkpoint.pointer, provider, scope=source_scope,
            )
            if stateful is None:
                initialization = CheckpointInitializationRequest(mode="initialize_from", source=source)
            else:
                if semantic_model is None:
                    raise ModelCheckpointError("/graph", "Stateful Train requires a typed model")
                fork_plan, fork_bindings = plan_studio_checkpoint_fork(
                    source_root=source_root,
                    pointer=checkpoint.pointer,
                    source=source,
                    target_root=checkpoint_root,
                    job_id=job_id,
                    spec=spec,
                    method=method,
                    preparation=preparation,
                    registry=registries.training_programs,
                    writer=semantic_model,
                )
        result = execute_managed_training_run(
            spec,
            run_id=job_id,
            preparation=preparation,
            manifest_root=manifest_root,
            checkpoint_root=checkpoint_root,
            checkpoint_source_root=source_root,
            checkpoint_source_pointer=resume.pointer if resume is not None else None,
            checkpoint_initialization_request=initialization,
            checkpoint_fork_plan=fork_plan,
            checkpoint_fork_plan_bindings=fork_bindings,
            checkpoint_fork_target_id="successor" if fork_plan is not None else None,
            checkpoint_content_provider=provider,
            checkpoint_source_content_binding=source_content,
            semantic_model=semantic_model,
            train_state_codec="standard_train" if semantic_model is not None else "legacy",
            resume=resume is not None,
            registry=registries.training_programs,
            stop_before_update=stop_before_update,
            start_callback=on_start,
            committed_update_callback=on_update,
            checkpoint_published=on_checkpoint,
            before_finalize=finalize,
            retain_history=False,
            execution_started_at_monotonic=events.started,
            execution_start_semantics="host_worker_entry",
        )
        if result.status == "completed" and result.end_completed_batches == target:
            terminal_receipt = receipt("completed")
            try:
                completion = verify_training_completion(
                    producer_job_id=job_id, launch=launch, binding_sha256=binding_sha256,
                    receipt=terminal_receipt, checkpoint=events.checkpoint,
                )
            except (CompletionFileError, CompletionUnavailable) as error:
                code = "input_unavailable" if isinstance(error, CompletionUnavailable) else "dependency_corrupt"
                return _Result(
                    "succeeded", receipt=terminal_receipt,
                    completion_error={"code": code, "message": str(error)},
                )
            model_product = None
            unavailable = None
            if launch.canonical_request is not None:
                if semantic_model is None:
                    unavailable = "The terminal checkpoint has no declared semantic model tree"
                else:
                    product, _ = verify_training_model_product(completion)
                    model_product = product.model_dump(mode="json")
            return _Result(
                "succeeded", receipt=terminal_receipt, completion=completion, model_product=model_product,
                store_publication_error={"code": "input_unavailable", "message": unavailable} if unavailable else None,
            )
        if result.status == "cancelled" and stop_requested and events.checkpoint is not None:
            return _Result("stopped", receipt=receipt("cancelled"))
        if result.status == "failed" and result.manifest.failure_kind == "nan_guard":
            raise FloatingPointError(result.manifest.stop_reason or "non-finite training update")
        raise RuntimeError(f"unexpected training outcome {result.status!r}")
    except Exception as exc:
        traceback.print_exc()
        error = _error(exc, events.current_phase)
        if isinstance(exc, CheckpointCustodyError) and events.current_phase != "restoring":
            error["code"] = "publication_failed"
        try:
            if not manifest_path.exists():
                payloads = build_training_run_manifest_spec_payloads(
                    launch.spec,
                    task_binding_spec=(
                        launch.spec.task_binding_spec.model_dump(mode="json", exclude_none=True)
                        if launch.spec.task_binding_spec is not None
                        else None
                    ),
                )
                checkpoint = events.checkpoint
                write_manifest(
                    TrainingRunManifest(
                        id=manifest_id,
                        job_id=job_id,
                        status="failed",
                        completed_batches=events.completed_updates,
                        training_spec=payloads.training_spec,
                        graph_spec=payloads.graph_spec,
                        task_spec=payloads.task_spec,
                        task_binding_spec=payloads.task_binding_spec,
                        checkpoint_custody=(
                            [
                                ParentRef(
                                    kind="TrainingCheckpointTransactionManifest",
                                    id=checkpoint.pointer.transaction_id,
                                    uri=checkpoint.pointer.manifest_relative_path,
                                    role="training_checkpoint_custody",
                                    metadata={"manifest_sha256": checkpoint.pointer.manifest_sha256},
                                )
                            ]
                            if checkpoint is not None and checkpoint.pointer.run_id == job_id
                            else []
                        ),
                        metadata={"error": error},
                    ),
                    root=manifest_root,
                )
            reference = receipt("failed")
        except Exception:
            traceback.print_exc()
            reference = None
        return _Result("failed", receipt=reference, error=error)
