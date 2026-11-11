"""Host entry point for one Studio training attempt."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    if isinstance(exc, FloatingPointError):
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
        if "feedbax.training.invocation" not in sys.modules:
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
        StudioTrainingJobInput,
    )
    from feedbax.orchestration.events import normalize_serialized_metrics
    from feedbax.plugins.composition import compose_application
    from feedbax.runtime._graph_key_schedule import derive_key
    from feedbax.training.checkpoint_custody import CheckpointCustodyError
    from feedbax.training.executor import execute_training_run_spec
    from feedbax.training.invocation import prepare_training_invocation
    from feedbax.training.manifest_preflight import build_training_run_manifest_spec_payloads
    from feedbax.training.preparation import ExecutionPreparationResult
    from feedbax.training.studio_executor import StudioUpdateKernel, studio_training_key

    events.current_phase = "binding"
    launch = StudioTrainingJobInput.model_validate(strict_json_loads(input_path.read_text()))
    if launch.spec.method_ref != STUDIO_GRAPH_METHOD:
        raise ValueError("Studio training jobs require the registered Studio graph method")
    if not launch.manifest_root.is_absolute():
        raise ValueError("manifest_root must be absolute")
    manifest_root = launch.manifest_root.resolve()
    checkpoint_root = _attempt_root(manifest_root, job_id)
    binding_path = checkpoint_root / "binding.json"
    resume = launch.resume
    if resume is not None:
        events.current_phase = "restoring"
        parent_root = _attempt_root(manifest_root, resume.pointer.run_id)
        if resume.checkpoint_root != parent_root or parent_root == checkpoint_root:
            raise ValueError("resume source is outside the original attempt's custody")
        if resume.completed_updates != resume.pointer.completed_training_batches:
            raise ValueError("resume coordinate disagrees with its checkpoint pointer")
        parent_binding = StudioTrainingJobInput.model_validate_json(
            (parent_root / "binding.json").read_bytes()
        )
        if launch.model_dump(exclude={"resume"}) != parent_binding.model_dump(exclude={"resume"}):
            raise ValueError("resume inputs differ from the original attempt's binding")
        events.completed_updates = resume.completed_updates

    events.current_phase = "binding"
    write_bytes_atomic(binding_path, launch.model_dump_json(exclude_none=True).encode(), replace=False)
    events.append(
        "binding",
        {
            "binding_path": str(binding_path),
            "binding_sha256": sha256_file(binding_path),
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
        events.phase("building")
        registries = asyncio.run(compose_application(modules=launch.bootstrap_modules)).bundle
        preparation = prepare_training_invocation(
            launch.spec,
            registries,
            run_id=job_id,
            resume=resume is not None,
            phase=lambda phase: events.phase(phase) if phase != events.current_phase else None,
        )
        if not isinstance(preparation, ExecutionPreparationResult):
            raise ValueError("Studio graph training requires scalar execution preparation")
        kernel = preparation.kernel_context["studio_graph_update"]
        if not isinstance(kernel, StudioUpdateKernel):
            raise ValueError("Studio graph training requires the registered numerical update")
        spec = launch.spec
        method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
        target = method.target_total
        snapshot_interval = spec.training_config.snapshot_interval
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        observation_key = studio_training_key(method.seed, "observation")
        last_progress = float("-inf")
        pending: tuple[ProgressCoordinate, float] | None = None
        last_frame = -1
        latest_model: Any = None
        stop_requested = False

        def publish_progress() -> None:
            nonlocal pending, last_progress
            if pending is None:
                return
            coordinate, observed_at = pending
            _, metrics = normalize_serialized_metrics(coordinate, coordinate.metrics, {})
            events.append(
                "progress",
                {
                    "completed_updates": events.completed_updates,
                    "total_updates": target,
                    "elapsed_seconds": observed_at,
                    "metrics": metrics,
                },
            )
            pending = None
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
            if completed != events.completed_updates:
                raise ValueError("restored state disagrees with the bound checkpoint coordinate")
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
            pending = coordinate, now
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
            events.append(
                "publication",
                {
                    "key": f"checkpoint:{completed}",
                    "checkpoint": checkpoint.model_dump(mode="json"),
                    "elapsed_seconds": events.elapsed(),
                },
                sync=True,
            )
            events.checkpoint = checkpoint

        def stop_before_update() -> bool:
            nonlocal stop_requested
            stop_requested = stop_requested or stop_file.exists()
            return stop_requested and events.completed_updates > 0

        def finalize() -> None:
            events.phase("finalizing")
            publish_progress()
            publish_frame()

        if resume is not None:
            events.phase("restoring")
        result = execute_training_run_spec(
            spec,
            run_id=job_id,
            preparation=preparation,
            manifest_root=manifest_root,
            checkpoint_root=checkpoint_root,
            checkpoint_source_root=resume.checkpoint_root if resume is not None else None,
            checkpoint_source_pointer=resume.pointer if resume is not None else None,
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
            return _Result("succeeded", receipt=receipt("completed"))
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
