"""Admission and activity for local Studio training requests."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from hashlib import sha256
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from uuid import uuid4

from feedbax._secure_fs import (
    SecurePathRigor,
    close_directory_chain,
    open_directory_chain,
    open_existing_file,
    recheck_directory_chain,
)
from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.artifact_custody import ContentObjectRef, StoreTransferProgress
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.code_validity import AttemptProvenance
from feedbax.contracts.execution_target import (
    LOCAL_RUN_KINDS, LOCAL_TARGET, LOCAL_TARGET_ID, AttemptTarget, ExecutionTargets,
    SessionTarget, SessionTargetProfile, TargetRunKind, attempt_target,
)
from feedbax.contracts.intent_diff import first_request_change
from feedbax.contracts.jobs import (
    JobError,
    JobOutcome,
    JobProgress,
    JobReceipt,
    JobStatus,
    StopMode,
    StudioActivityJob,
    StudioBoundBestSubject,
    StudioEvaluationCheckpointAttempt,
    StudioEvaluationCheckpointGap,
    StudioJobEvent,
    StudioJobEventHistory,
    StudioJobSnapshot,
    StudioJobStopItem,
    StudioJobSubmission,
    StudioLocalSession,
    StudioSubmissionActor,
    StudioSubmissionItem,
    StudioTrainCheckpointSource,
    StudioVenueStream,
)
from feedbax.contracts.live_evaluate import LIVE_EVALUATE_POINT, LiveEvaluationObservation
from feedbax.contracts.manifest import (
    TrainingRunManifest,
    canonical_manifest_path,
    load_manifest_bytes,
    training_run_manifest_id,
)
from feedbax.contracts.project_plan import (
    ElementKey, PlannedAnalyzeRun, PlannedElement, PlannedRow, PlannedRun, ProjectPlan, RunKey,
    planned_input_groups,
)
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.receipt_validity import CodeDifference, InputDifference, ReceiptValidity, ValidityJudgement
from feedbax.contracts.effective_success import EvaluationSubjectValue, TerminalTrainSubject
from feedbax.contracts.request_refs import BatchMember, BatchSetMember, BestCandidate, BestRank, ContentRef, LastMember, ProductRef, RequestId, UpdateSeriesMember
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.validation_placement import ValidationPlacement
from feedbax.host import Admission, AdmissionItem, Edge, HostClient, HostError, Job, Launch
from feedbax.host.live_evaluate import DELEGATION_EVENT
from feedbax.host.plan_history import AuthoredTrainPlanRef, plan_ref_key, plan_run_key
from feedbax.host.protocol import LIGHT_ENTRYPOINT, JobsSnapshot, TerminalReuse, can_resume_attempt
from feedbax.orchestration.session import SessionError
from feedbax.orchestration.targets import TargetRegistry
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.store_damage import inspect_without_recording
from feedbax.plugins.application import ApplicationRegistryBundle
from feedbax.validity.admission import QUICK_RECEIPT_DURATION, validity_reason
from feedbax.studio.canonical_evaluate import ROLLOUT_JAX_BACKEND, ROLLOUT_JAX_ENABLE_X64
from feedbax.studio.canonical_train import (
    BoundStudioTrainingJobInput,
    CanonicalTrainRequestBinding,
    bind_canonical_train,
)

if TYPE_CHECKING:
    from feedbax.contracts.artifact_custody import ContentObjectRef
    from feedbax.contracts.effective_success import RequestObject, Success
    from feedbax.persistence.effective_success import PublishedSuccess, SuccessView
    from feedbax.validity.project_admission import ObservedReceiptValidity, ProjectValidity
    from feedbax.persistence.store_sync import StoreEndpoint
    from feedbax.training.validation_placement import ValidationQualification
    from feedbax.component_registry import ComponentRegistry
    from feedbax.studio.canonical_evaluate import EvaluationResult, RolloutParameters
    from feedbax.project_folder import ProjectFolder, SavedProjectPlans, SettledProjectPlan


_LIVE_STATES = frozenset({"queued", "assigned", "running", "stopping"})
_HOST_ONLY_EVENTS = frozenset({"dependency_bound", "dependency_rejected", "continued"})
_CHECKPOINT_ACTIVITY_EVENTS = frozenset({LIVE_EVALUATE_POINT, DELEGATION_EVENT})
_ACTIVITY_EVENTS = frozenset(
    {
        "queued",
        "continued",
        "assigned",
        "accepted",
        "phase",
        "progress",
        "publication",
        "stopping",
        "blocked",
        "terminal",
    }
) | _CHECKPOINT_ACTIVITY_EVENTS
_ACTIVE_PLAN_STATUSES = frozenset({
    JobStatus.QUEUED, JobStatus.AWAITING_AUTHORIZATION, JobStatus.STARTING,
    JobStatus.RUNNING, JobStatus.STOPPING, JobStatus.COLLECTING,
})
EVALUATION_SUBJECT_REQUIRED = "select an unambiguous admitted Train subject"
_LIGHT_KINDS = frozenset({"analyze", "render", "report"})


class JobNotFound(LookupError):
    pass


class JobConflict(ValueError):
    pass


@dataclass(frozen=True)
class PlanJobAttempt:
    job_id: str
    status: JobStatus
    phase: str | None
    progress: JobProgress | None
    reason: str | None
    successful_receipt: JobReceipt | None
    binding_sha256: str | None = None
    continues: str | None = None
    checkpoint_source: StudioTrainCheckpointSource | None = None
    output_size_bytes: int | None = None
    checkpoint_sources: tuple[StudioTrainCheckpointSource, ...] = ()
    stopped_checkpoint: int | None = None
    checkpoint_gaps: tuple[StudioEvaluationCheckpointGap, ...] | None = None
    evaluation_placement: Literal["ordinary", "live"] | None = None
    bound_subject: StudioBoundBestSubject | None = None
    checkpoint_attempt: StudioEvaluationCheckpointAttempt | None = None
    light_success: Success | None = None
    train_request: RequestObject | None = None
    collected_evaluate: bool = False
    collection_progress: StoreTransferProgress | None = None
    validity: ReceiptValidity | None = None
    provenance: AttemptProvenance | None = None


@dataclass(frozen=True)
class PlanJobAttempts:
    """Attempt evidence at one Host cursor, keyed by canonical request ID."""

    cursor: int
    by_request: Mapping[RequestId, tuple[PlanJobAttempt, ...]]
    plan_ref: AuthoredTrainPlanRef | None = None
    checkpoint_producers: frozenset[RequestId] = frozenset()
    checkpoint_catalogs: Mapping[RequestId, tuple[int, ...]] = field(default_factory=dict)
    light_by_address: Mapping[str, tuple[PlanJobAttempt, ...]] = field(default_factory=dict)
    validity_checks: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _VerifiedEvaluation:
    receipt: JobReceipt
    size: int
    checkpoint_gaps: tuple[StudioEvaluationCheckpointGap, ...] | None
    placement: Literal["ordinary", "live"]
    manifest_object: ContentObjectRef
    checkpoint_source: StudioTrainCheckpointSource | None = None
    checkpoint_sources: tuple[StudioTrainCheckpointSource, ...] = ()
    bound_subject: StudioBoundBestSubject | None = None


@dataclass(frozen=True)
class CollectedSessionAttempt:
    activity: StudioActivityJob
    evidence_sha256: str


class PlanJobReader:
    """Read the existing Host snapshot and authenticate its saved result evidence."""

    def __init__(
        self, host: HostClient, *, components: ComponentRegistry | None = None,
        registries: ApplicationRegistryBundle | None = None,
    ) -> None:
        self.host = host
        self._evaluation_components = components
        self._validity_registries = registries
        self._quick_duration = QUICK_RECEIPT_DURATION

    def _validity_observer(self, root: Path) -> ProjectValidity:
        from feedbax.validity.project_admission import ProjectValidity

        assert self._validity_registries is not None
        return ProjectValidity(root, self._validity_registries, quick_duration=self._quick_duration)

    def evaluation_parameters(
        self, plan: ProjectPlan, run: PlannedRun,
    ) -> tuple[RolloutParameters, ProductRef | BestRank | ContentRef]:
        """Resolve the saved request capability shared by Plan and admission."""
        from feedbax.studio.canonical_evaluate import CannotRun, decode_rollout_request

        if run.requires:
            raise CannotRun("/requires", "Evaluate has unresolved capabilities.")
        _, params, checkpoint = decode_rollout_request(run.request_id, plan.requests[run.request_id])
        if isinstance(checkpoint, ProductRef) and (source := plan.requests.get(checkpoint.request_id)) is not None:
            from feedbax.contracts.request_specs import DocumentTrainingRequestSpec

            spec = decode_request(source, plan.documents.__getitem__).spec
            if not isinstance(spec, DocumentTrainingRequestSpec):
                raise CannotRun("/inputs/subject", "Managed Evaluate needs a document Train checkpoint.")
        return params, checkpoint

    @inspect_without_recording()
    def read_plan_attempts(
        self, *, project: str, store: Path, requests: Collection[RequestId],
        stopped_checkpoint_jobs: Sequence[str] = (),
        light_addresses: Collection[str] = (), project_root: Path | None = None,
    ) -> PlanJobAttempts:
        """Read saved Plan attempts and authenticate any selected stopped checkpoints."""
        manifest_root = str(store)
        key = plan_ref_key(project, manifest_root)
        include_producers = any(request.startswith("evaluate:") for request in requests)
        snapshot = self.host.jobs_snapshot(
            project=project, start_host=False, record_keys=(key,), include_launch=include_producers,
            checkpoint_progress_root=manifest_root if include_producers else None,
            checkpoint_admission=(manifest_root, sorted(requests)),
        )
        plan_ref = cast(AuthoredTrainPlanRef | None, snapshot.get("records", {}).get(key))
        wanted = set(requests)
        if plan_ref is not None:
            for row in plan_ref["rows"].values():
                wanted.add(row["request_id"])
                wanted.update(result["request_id"] for result in (row["last"], row["prior"]) if result)
        stopped_jobs = set(stopped_checkpoint_jobs)
        attempts: dict[RequestId, list[PlanJobAttempt]] = {}
        light_attempts: dict[str, list[PlanJobAttempt]] = {}
        by_id = {
            job["id"]: job for job in cast(list[dict[str, Any]], snapshot["jobs"])
            if job["data"].get("manifest_root") == manifest_root
        }
        catalogs = {
            request: tuple(by_id[anchor]["launch"]["payload"].get("checkpoint_catalog", ()))
            for request, anchor in snapshot.get("checkpoint_producers", {}).items()
        }
        from feedbax.persistence.effective_success import SuccessView

        success_view = None
        observer = (
            self._validity_observer(project_root)
            if project_root is not None and self._validity_registries is not None else None
        )
        for job in by_id.values():
            request_id = job["request"]
            light = _is_light_request(request_id)
            address = job["data"].get("run_key")
            if request_id not in wanted and not (light and address in light_addresses):
                continue
            evaluate = request_id.startswith("evaluate:")
            collected_train = _is_collected_train(job)
            collected_evaluate = evaluate and job["data"].get("collected_evaluate") is True
            succeeded = job["state"] == "succeeded"
            size = None
            result = None
            store_success = None
            checkpoint_source = None if collected_train else _train_checkpoint_source(job, by_id)
            damage = None
            if succeeded:
                if success_view is None:
                    success_view = SuccessView(ImmutableArtifactBlobProvider(store))
                damage = _owned_result_damage(success_view, job)
            if damage is not None:
                receipt = None
            elif evaluate:
                result = _verified_evaluation_result(job, store) if succeeded else None
                receipt, size = (result.receipt, result.size) if result is not None else (None, None)
                if result is not None and result.checkpoint_source is not None:
                    checkpoint_source = result.checkpoint_source
            elif succeeded and (light or collected_train):
                assert success_view is not None
                verified = _verified_store_result(success_view, job)
                receipt, store_success = verified if verified is not None else (None, None)
                if collected_train and store_success is not None:
                    checkpoint_source = _collected_train_checkpoint_source(success_view, store_success)
            else:
                receipt = _verified_training_receipt(job, store) if succeeded else None
            validity = provenance = None
            if observer is not None and receipt is not None and success_view is not None:
                from feedbax.persistence.effective_success import PublishedSuccess

                ref = success_view.attempt_ref(job["id"])
                if ref is not None:
                    try:
                        published = PublishedSuccess(ref, store_success or success_view.read_receipt(ref))
                        provenance = published.receipt.provenance
                        validity = observer.inspect(
                            success_view, published, checkpoint_differences=_checkpoint_differences(snapshot, job["id"]),
                        ).decision
                    except (KeyError, OSError, ValueError) as error:
                        damage, receipt = f"Cannot assess the committed result: {error}", None
            completion = (job.get("outcome") or {}).get("completion") or {}
            binding_sha256 = (
                completion.get("binding_sha256")
                if receipt is not None
                and completion.get("request_id") == request_id
                and completion.get("project_id") == project
                and completion.get("producer_job_id") == job["id"]
                and completion.get("manifest_root") == manifest_root
                and completion.get("receipt") == receipt.model_dump(mode="json")
                else None
            )
            progress = _job_progress(job)
            status, reason = _job_status(
                job, progress=progress, receipt_intact=not succeeded or receipt is not None
            )
            if damage is not None:
                reason = damage
            stopped_checkpoint = (
                self.host.stopped_checkpoint(job["id"], project=project, manifest_root=manifest_root)
                if request_id.startswith("train:") and status == JobStatus.STOPPED and job["id"] in stopped_jobs
                else None
            )
            attempt = PlanJobAttempt(
                job["id"], status, (job.get("phase") or {}).get("phase"),
                progress, reason, receipt, binding_sha256,
                continues=job["continues"],
                checkpoint_source=checkpoint_source,
                output_size_bytes=size,
                checkpoint_sources=result.checkpoint_sources if result is not None else (),
                stopped_checkpoint=stopped_checkpoint,
                checkpoint_gaps=result.checkpoint_gaps if result is not None else None,
                evaluation_placement=result.placement if result is not None else None,
                bound_subject=result.bound_subject if result is not None else (
                    _best_subject(job) if evaluate and not succeeded else None
                ),
                checkpoint_attempt=_evaluation_checkpoint_attempt(job) if evaluate else None,
                light_success=store_success if light else None,
                train_request=store_success.request if collected_train and store_success is not None else None,
                collected_evaluate=collected_evaluate, validity=validity, provenance=provenance,
            )
            attempts.setdefault(request_id, []).append(attempt)
            if light and address in light_addresses:
                light_attempts.setdefault(address, []).append(attempt)
        return PlanJobAttempts(
            snapshot["cursor"],
            MappingProxyType({request: tuple(rows) for request, rows in attempts.items()}),
            plan_ref,
            frozenset(catalogs),
            MappingProxyType(catalogs),
            MappingProxyType({address: tuple(rows) for address, rows in light_attempts.items()}),
            tuple(job for job in by_id.values() if "validity_check" in job["data"]),
        )


def _checkpoint_differences(snapshot: JobsSnapshot, attempt: str) -> tuple[InputDifference, ...]:
    difference = snapshot.get("checkpoint_differences", {}).get(attempt)
    return (InputDifference.model_validate(difference),) if difference is not None else ()


class SessionSubmission(Protocol):
    def __call__(
        self, profile: SessionTargetProfile, folders: ProjectFolder, reviewed: SettledProjectPlan, *,
        run_keys: Sequence[RunKey], token: str, reuse_completed: bool,
    ) -> StudioJobSubmission: ...


class JobService(PlanJobReader):
    """Bind authored requests to immutable worker launches over the durable host."""

    def __init__(
        self,
        host: HostClient,
        registries: ApplicationRegistryBundle,
        *,
        live_validation_qualifications: Sequence[ValidationQualification] = (),
        targets: TargetRegistry | None = None,
        submit_session: SessionSubmission | None = None,
        automatic_run: Literal["all", "no_train", "off"] = "all",
        quick_duration: timedelta = QUICK_RECEIPT_DURATION,
    ) -> None:
        super().__init__(host, components=registries.components, registries=registries)
        self._automatic_run = automatic_run
        self._quick_duration = quick_duration
        self.registries = registries
        self.targets = targets if targets is not None else TargetRegistry()
        self._submit_session = submit_session
        self._live_validation_qualifications = {
            (entry.train_request, entry.evaluate_request): entry for entry in live_validation_qualifications
        }
        base_worker = Launch(
            argv=(sys.executable,),
            env={
                key: os.environ[key]
                for key in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS")
                if key in os.environ
            },
            cwd=str(Path.cwd()),
            preload=("jax",),
        )
        self._training_worker = replace(
            base_worker,
            warm_entrypoint="feedbax.training_host_worker:main",
            preload=(*base_worker.preload, "feedbax.training.host_prewarm"),
        )
        self._evaluation_worker = replace(
            base_worker,
            env=base_worker.env | {
                "JAX_PLATFORMS": ROLLOUT_JAX_BACKEND,
                "JAX_ENABLE_X64": str(int(ROLLOUT_JAX_ENABLE_X64)),
            },
            warm_entrypoint="feedbax.training.evaluate_worker:main",
            preload=(*base_worker.preload, "feedbax.plugins.composition"),
        )
        self._light_worker = Launch(
            argv=(base_worker.argv[0],), cwd=base_worker.cwd,
            warm_entrypoint=LIGHT_ENTRYPOINT,
        )
        self._launch_lock = Lock()
        self._accepting = True

    def execution_targets(self, kinds: Sequence[TargetRunKind] = ()) -> ExecutionTargets:
        return self.targets.catalog(kinds=kinds, session_submission=self._submit_session is not None)

    def submit_plan(
        self, folders: ProjectFolder, reviewed: SettledProjectPlan, *,
        run_keys: Sequence[RunKey], token: str, target: str = LOCAL_TARGET_ID, reuse_completed: bool = False,
    ) -> StudioJobSubmission:
        if target == LOCAL_TARGET_ID:
            return self.admit_plan(folders, reviewed, run_keys=run_keys, token=token, reuse_completed=reuse_completed)
        selected = set(run_keys)
        kinds: dict[RunKey, TargetRunKind] = {
            row.key: cast(TargetRunKind, row.kind) for row in reviewed.plan.rows
            if row.key in selected and row.kind in LOCAL_RUN_KINDS
        }
        if not selected or kinds.keys() != selected:
            raise JobConflict("Target submission requires selected runnable saved rows.")
        try:
            profile = self.targets.require_session(target, kinds=kinds.values())
        except SessionError as error:
            raise JobConflict(str(error)) from error
        if self._submit_session is None:
            raise JobConflict("Session submission is not configured.")
        submission = self._submit_session(
            profile, folders, reviewed, run_keys=run_keys, token=token, reuse_completed=reuse_completed,
        )
        match submission.target:
            case SessionTarget(profile=selected_profile, driver=driver) if (
                selected_profile, driver
            ) == (profile.id, profile.driver):
                return submission
            case _:
                raise JobConflict("The session submission returned a different execution target.")

    def admit_plan(
        self,
        folders: SavedProjectPlans,
        reviewed: SettledProjectPlan,
        *,
        run_keys: Sequence[RunKey],
        token: str,
        checkpoint_catalog: tuple[int, ...] = (),
        actor: StudioSubmissionActor = StudioSubmissionActor(client="studio"),
        reuse_completed: bool = False,
        placement: AttemptTarget = LOCAL_TARGET,
        automatic: bool = False,
    ) -> StudioJobSubmission:
        """Admit selected rows from a server-issued saved Plan."""
        selected = {
            plan_run_key(key.variant, key.step, key.row): key
            for key in run_keys
        }
        if not selected or not token:
            raise JobConflict("Plan admission requires selected runs and a submission token.")
        project = reviewed.project_id
        before = {run.key: run for run in reviewed.plan.runs}
        directly_selected = _selected_requests(reviewed.plan, set(run_keys))
        light = _planned_light_runs(reviewed.plan, selected.values())
        light_rows = {key.run for key in light if isinstance(key, ElementKey)}
        kinds = {before[key].kind for key in selected.values() if key in before}
        evaluate = "evaluate" in kinds
        train = "train" in kinds
        command = _saved_plan_command(
            "canonical_train" if train or not evaluate else "canonical_evaluate",
            reviewed.revision, reviewed.semantic_root, str(reviewed.store_path), {
                name: before[key].request_id if key in before else None
                for name, key in selected.items() if key not in light_rows and key not in light
            } | {plan_attempt_key(key): node.request_id for key, node in light.items()},
        )
        if reuse_completed:
            command["reuse_completed"] = True
        if automatic:
            command["automatic"] = True
        if light:
            command["selected_runs"] = list(selected)
        if checkpoint_catalog:
            if evaluate and not train:
                raise JobConflict("/inputs/subject/checkpoint: checkpoint catalogue belongs to Train.")
            checkpoint_catalog = finite_coordinates(checkpoint_catalog)
            command["checkpoint_catalog"] = list(checkpoint_catalog)
        if previous := self._retry(token, project=project, continues=None, command=command, placement=placement):
            return previous

        if reuse_completed and not automatic:
            for request in directly_selected:
                self._check_before_run(reviewed, request)

        items = {}
        runs = {}
        checkpoints = {}
        names_by_key = {key: name for name, key in selected.items()}
        reviewed_catalogs = _planned_checkpoint_catalogs(reviewed.plan) if train else {}
        with folders.admission_plan(project) as current:
            if current.store_path != reviewed.store_path:
                raise JobConflict("The saved project's store changed since the reviewed Plan.")
            after = {run.key: run for run in current.plan.runs}
            current_catalogs = _planned_checkpoint_catalogs(current.plan) if train else {}
            store = ImmutableArtifactBlobProvider(current.store_path)
            heavy_selection = {
                name: key for name, key in selected.items() if key not in light_rows and key not in light
            }
            for name, key in heavy_selection.items():
                old, run = before.get(key), after.get(key)
                if old is not None and old.kind == "evaluate":
                    if run is None:
                        items[name] = _rejected(name, "/runs: the reviewed Evaluate row is no longer runnable.")
                        continue
                    if old.request_id != run.request_id:
                        path = _changed_request_path(
                            reviewed.plan, old.request_id, current.plan, run.request_id,
                        )
                        items[name] = _rejected(name, f"The selected Evaluate request changed at {path}.")
                        continue
                    data = _plan_run_data(name, run)
                    try:
                        params, checkpoint = self.evaluation_parameters(current.plan, run)
                        planned = run.inputs["subject"]
                        if planned.member.reference != checkpoint:
                            raise ValueError("The saved Plan disagrees with the encoded Evaluate subject.")
                        terminal = planned.terminal
                        reviewed_terminal = old.inputs["subject"].terminal
                        if reviewed_terminal is not None and reviewed_terminal != terminal:
                            raise ValueError("/inputs/subject: the reviewed terminal Train receipt changed")
                        item = self._evaluation_item(
                            project, run.request_id, current.plan.requests[run.request_id], data, store,
                            checkpoint, params, terminal=terminal, documents=current.plan.documents,
                        )
                    except (ValueError, KeyError, OSError) as error:
                        items[name] = _rejected(name, f"cannot_run {error}")
                        continue
                    if terminal is None and not isinstance(checkpoint, ContentRef):
                        checkpoints[name] = (
                            tuple(
                                (key, candidate.reference.request_id)
                                for key, candidate in zip(planned.candidates, checkpoint.candidates, strict=True)
                            )
                            if isinstance(checkpoint, BestRank) else ((planned.producer, checkpoint.request_id),)
                        )
                    runs[name] = data | {"request_id": run.request_id}
                    items[name] = item
                    continue
                if old is None or old.kind != "train":
                    items[name] = _rejected(name, "The reviewed row has no runnable saved request.")
                    continue
                if run is None:
                    items[name] = _rejected(name, "The reviewed Train row is no longer runnable.")
                    continue
                if old.request_id != run.request_id:
                    path = _changed_request_path(reviewed.plan, old.request_id, current.plan, run.request_id)
                    items[name] = _rejected(name, f"The selected Train request changed at {path}.")
                    continue
                if problem := run.admission_problem:
                    items[name] = _rejected(name, f"{problem.path}: {problem.detail}")
                    continue
                if run.requires:
                    items[name] = _rejected(name, "/requires: Train has unresolved capabilities.")
                    continue
                try:
                    spec, binding = bind_canonical_train(
                        run.request_id,
                        current.plan.requests[run.request_id],
                        current.plan.documents,
                        store, registries=self.registries,
                    )
                except (ValueError, KeyError) as error:
                    items[name] = _rejected(name, str(error))
                    continue
                catalog = checkpoint_catalog
                requested = current_catalogs.get(run.request_id)
                if requested != reviewed_catalogs.get(old.request_id):
                    items[name] = _rejected(name, "/checkpoint_catalog: the reviewed publication demand changed.")
                    continue
                if requested:
                    try:
                        catalog = finite_coordinates(tuple(sorted(set(catalog) | requested)))
                    except ValueError as error:
                        items[name] = _rejected(name, f"/checkpoint_catalog: {error}")
                        continue
                if binding.checkpoint is not None:
                    planned = run.inputs.get("checkpoint")
                    if planned is None or planned.member.reference != binding.checkpoint:
                        raise ValueError("The saved Plan disagrees with the encoded checkpoint input.")
                    checkpoints[name] = ((planned.producer, binding.checkpoint.request_id),)
                data = _plan_run_data(name, run)
                runs[name] = data | {"request_id": run.request_id}
                items[name] = self._training_item(
                    project, run.request_id, spec, data, current.store_path,
                    canonical_request=binding, checkpoint_catalog=catalog,
                )
            order = list(heavy_selection)
            if checkpoints:
                order = [names_by_key[run.key] for run in current.plan.runs if names_by_key.get(run.key) in items]
                order.extend(name for name, key in heavy_selection.items() if key not in after)
                for name in order:
                    if name not in checkpoints:
                        continue
                    anchors = {}
                    for producer, request_id in checkpoints[name]:
                        if request_id in anchors:
                            continue
                        if producer in names_by_key:
                            source = items[names_by_key[producer]].job
                            anchor = source.id if source is not None else None
                        else:
                            anchor = self._checkpoint_producer(project, current.store_path, request_id)
                        anchors[request_id] = anchor
                    job = items[name].job
                    assert job is not None
                    if any(anchor is None for anchor in anchors.values()):
                        items[name] = _rejected(
                            name,
                            f"cannot_run /inputs/subject: {EVALUATION_SUBJECT_REQUIRED}"
                            if job.request.startswith("evaluate:") else
                            "/inputs/checkpoint: select an unambiguous admitted Train predecessor",
                        )
                        runs.pop(name)
                    else:
                        items[name] = replace(
                            items[name], job=replace(job, edges=tuple(
                                Edge(anchor, authenticated_completion=True) for anchor in anchors.values()
                                if anchor is not None
                            ))
                        )
            if light:
                current_nodes = _planned_light_runs(current.plan, selected.values())
                row_index = {row.key: row for row in current.plan.rows}
                for key, old in light.items():
                    name = plan_attempt_key(key)
                    order.append(name)
                    node = current_nodes.get(key)
                    if node is None:
                        items[name] = _rejected(name, "The reviewed request is no longer runnable.")
                        continue
                    if old.request_id != node.request_id:
                        path = _changed_request_path(reviewed.plan, old.request_id, current.plan, node.request_id)
                        items[name] = _rejected(name, f"The selected request changed at {path}.")
                        continue
                    row = row_index[key.run if isinstance(key, ElementKey) else key]
                    try:
                        item = self._light_item(project, current.plan, store, row, node, items)
                    except (ValueError, KeyError) as error:
                        items[name] = _rejected(name, f"cannot_run {error}")
                        continue
                    items[name] = item
                    assert item.job is not None
                    runs[name] = item.job.data | {"request_id": node.request_id}
            record = {
                "project": project, "continues": None, "runs": runs, "command": command,
                "actor": actor.model_dump(mode="json"),
                "target": placement.model_dump(mode="json"),
            }
            if train and any(item.job and item.job.request.startswith("train:") for item in items.values()):
                requests = {
                    plan_run_key(key.variant, key.step, key.row): run.request_id
                    for key, run in after.items()
                    if run.kind == "train"
                }
                plan = store.store_rooted_object(
                    canonical_json_v2_bytes({
                        "revision": current.revision,
                        "semantic_root": current.semantic_root,
                        "requests": requests,
                    }),
                    namespace="studio-train-plan",
                )
                record["authored_train_plan"] = {
                    "object": plan.model_dump(mode="json"), "requests": requests,
                }
            candidates = [_with_project_root(items[name], current.project_root) for name in order]
            admitted = (
                self._admit_with_reuse(
                    token, candidates, record, reuse_completed=reuse_completed,
                    current=current, automatic=automatic, directly_selected=directly_selected,
                )
                if light or reuse_completed else self._admit(token, candidates, record)
            )
        return self._submission_for_action(
            admitted, project=project, continues=None, command=command, placement=placement,
        )

    def _checkpoint_producer(self, project: str, root: Path, request: RequestId) -> str | None:
        return self.host.checkpoint_producer(project=project, manifest_root=str(root), request=request)

    def _light_item(
        self, project: str, plan: ProjectPlan, store: ImmutableArtifactBlobProvider,
        row: PlannedRow, node: PlannedRun | PlannedElement, items: Mapping[str, AdmissionItem],
    ) -> AdmissionItem:
        from feedbax.light_jobs import LightDelivery, validate_light_request
        from feedbax.contracts.model_input import decode_document_input
        from feedbax.persistence.effective_success import store_document_inputs, store_request

        if isinstance(row, (PlannedRun, PlannedAnalyzeRun)) and row.requires:
            raise ValueError("/requires: unresolved capabilities.")
        if isinstance(row, PlannedRun) and (problem := row.admission_problem) is not None:
            raise ValueError(f"{problem.path}: {problem.detail}")
        raw = plan.requests[node.request_id]
        decoded = decode_request(raw, plan.documents.__getitem__)
        _, operation = validate_light_request(
            decoded, registry=self.registries,
            read_document=lambda content: decode_document_input(content, plan.documents[content.sha256]),
        )
        encoded = encode_request(decoded.spec, decoded.inputs)
        if encoded.id != node.request_id or encoded.data != raw:
            raise ValueError("The saved request bytes do not match the Plan identity.")
        parents = {}
        for role, bindings in planned_input_groups(node).items():
            for binding in bindings:
                reference = binding.member.reference
                if not isinstance(reference, ProductRef):
                    continue
                source = binding.element or binding.producer
                candidate = items.get(plan_attempt_key(source)) if source is not None else None
                exact_checkpoint = reference.role == "checkpoint" and isinstance(reference.select, BatchMember)
                if candidate is None and exact_checkpoint:
                    anchor = self._checkpoint_producer(project, store.root, reference.request_id)
                elif candidate is None:
                    anchors = cast(list[dict[str, Any]], self.host.jobs(project=project, request=reference.request_id))
                    anchor = next((job["id"] for job in reversed(anchors) if (
                        job["data"].get("manifest_root") == str(store.root)
                        and (job["state"] in _LIVE_STATES or (job["state"] == "succeeded" and self._receipt_matches(job)))
                    )), None)
                else:
                    anchor = (
                        candidate.job.id
                        if candidate.job is not None and candidate.job.request == reference.request_id else None
                    )
                if anchor is None:
                    raise ValueError(f"/inputs/{role}: no admitted exact producer for {reference.request_id}.")
                parents.setdefault(reference.request_id, Edge(anchor, authenticated_completion=exact_checkpoint))

        name = plan_attempt_key(node.key)
        data = _plan_run_data(name, row)
        if isinstance(node, PlannedElement):
            data["label"] = f"{data['label']} · {node.key.name}"
        store_document_inputs(store, decoded.inputs, plan.documents.__getitem__)
        payload = LightDelivery(
            project_id=project, manifest_root=Path(store.root), request=store_request(store, encoded),
        )
        profile = self._light_worker
        slot = "light"
        if operation is not None and (numerical := operation.value.definition.numerical) is not None:
            profile = replace(profile, env=numerical.environment(), preload=("jax",))
            slot = "compute"
        launch = replace(profile, payload=payload.model_dump(mode="json"))
        return AdmissionItem(run_key=name, job=Job(
            id=uuid4().hex, project=project, request=node.request_id, launch=launch, slot=slot,
            edges=tuple(parents.values()),
            data=data | {"manifest_root": str(store.root), "runtime": launch.argv[0]},
        ))

    def _evaluation_item(
        self, project: str, request_id: str, raw: bytes, data: dict[str, Any],
        store: ImmutableArtifactBlobProvider, subject: ProductRef | BestRank | ContentRef, params: RolloutParameters,
        *, terminal: TerminalTrainSubject | None, documents: Mapping[str, bytes],
    ) -> AdmissionItem:
        from feedbax.studio.canonical_evaluate import (
            BoundEvaluationJobInput,
            EvaluationRequestBinding,
        )
        from feedbax.persistence.effective_success import store_document_inputs

        root = store.root
        if isinstance(subject, ContentRef):
            decoded = decode_request(raw, documents.__getitem__)
            store_document_inputs(store, decoded.inputs, documents.__getitem__)
        payload = BoundEvaluationJobInput(
            project_id=project,
            request_id=request_id,
            manifest_root=root,
            canonical_request=EvaluationRequestBinding(
                request=store.store_rooted_object(raw, namespace="studio-evaluate-request"),
                checkpoint=subject if not isinstance(subject, ContentRef) else None,
                initial_model=subject if isinstance(subject, ContentRef) else None,
                terminal=terminal,
            ),
        )
        launch = replace(self._evaluation_worker, payload=payload.model_dump(mode="json", by_alias=True))
        if isinstance(subject, ProductRef) and isinstance(subject.select, UpdateSeriesMember) and not params.variation:
            qualification = self._live_validation_qualifications.get((subject.request_id, request_id))
            if qualification is not None:
                data = data | {"validation_qualification": qualification.model_dump(mode="json")}
        return AdmissionItem(
            run_key=data["run_key"],
            job=Job(
                id=uuid4().hex, project=project, request=request_id, launch=launch,
                data=data | {"manifest_root": str(root), "runtime": launch.argv[0]},
            ),
        )

    def read_evaluation_result(self, *, project: str, store: Path, job_id: str) -> EvaluationResult:
        """Locate an authenticated Evaluate result by exact project and attempt."""
        rows = self.host.jobs_snapshot(job_id=job_id, project=project, start_host=False)["jobs"]
        if not rows:
            raise JobNotFound(job_id)
        (job,) = rows
        if (
            job["project"] != project or job["data"].get("manifest_root") != str(store)
            or not job["request"].startswith("evaluate:") or job["state"] != "succeeded"
        ):
            raise JobNotFound(job_id)
        return self._retained_evaluation(job, store)

    @inspect_without_recording()
    def read_evaluation_objective_attempt(
        self, *, project: str, store: Path, job_id: str, request_id: str,
    ) -> EvaluationResult | LiveEvaluationObservation:
        rows = self.host.jobs_snapshot(
            job_id=job_id, project=project, start_host=False, include_launch=True,
            checkpoint_objective_root=str(store),
        )["jobs"]
        if not rows:
            raise JobNotFound(job_id)
        (job,) = rows
        if job["request"] != request_id or job["data"].get("manifest_root") != str(store):
            raise JobNotFound(job_id)
        if job["state"] == "succeeded":
            return self._retained_evaluation(job, store)
        if job.get("checkpoint_objective_error"):
            raise JobConflict("The live Evaluate checkpoint evidence could not be authenticated.")
        evidence = job.get("checkpoint_objective")
        if evidence is None:
            raise JobNotFound(job_id)
        return LiveEvaluationObservation.model_validate(evidence)

    @staticmethod
    def _retained_evaluation(job: dict[str, Any], store: Path) -> EvaluationResult:
        from feedbax.persistence.effective_success import SuccessView
        from feedbax.studio.canonical_evaluate import EvaluationResult

        damage = _owned_result_damage(SuccessView(ImmutableArtifactBlobProvider(store)), job)
        if damage is not None:
            raise JobConflict(damage)
        result = _verified_evaluation_result(job, store)
        if result is None:
            raise JobConflict("The completed Evaluate receipt cannot be authenticated.")
        return EvaluationResult(
            store, job["id"], job["request"], result.receipt, manifest_object=result.manifest_object,
        )

    def _training_item(
        self,
        project: str,
        request_id: str,
        spec: DocumentTrainingRequestSpec,
        data: dict[str, Any],
        root: Path,
        *,
        canonical_request: CanonicalTrainRequestBinding,
        checkpoint_catalog: tuple[int, ...] = (),
    ) -> AdmissionItem:
        payload = BoundStudioTrainingJobInput(
            spec=spec,
            project_id=project,
            request_id=request_id,
            manifest_root=root,
            canonical_request=canonical_request,
            checkpoint_catalog=checkpoint_catalog,
        ).model_dump(mode="json", exclude_none=True)
        launch = replace(self._training_worker, payload=payload)
        return AdmissionItem(
            run_key=data["run_key"],
            job=Job(
                id=uuid4().hex,
                project=project,
                request=request_id,
                launch=launch,
                data=data | {"manifest_root": str(root), "runtime": launch.argv[0]},
            ),
        )

    def _check_before_run(self, plan: SettledProjectPlan, request: RequestId) -> None:
        from feedbax.validity.checks import settle_check, wait_for_check

        current = self._result_observation(plan, request)
        if current is None or current[3].decision.state != "unchecked":
            return
        _, _, published, observation = current
        check = self._submit_check(plan, published, observation)
        result = wait_for_check(self.host, project=plan.project_id, job_id=check)
        fresh = self._result_observation(plan, request)
        if fresh is not None:
            settle_check(fresh[0], fresh[1], result, current_checkpoints=partial(self._result_checkpoints, plan))

    def _result_observation(
        self, plan: SettledProjectPlan, request: RequestId,
    ) -> tuple[SuccessView, ProjectValidity, PublishedSuccess, ObservedReceiptValidity] | None:
        from feedbax.persistence.effective_success import SuccessView
        from feedbax.persistence.store_damage import KnownDamage

        view = SuccessView(ImmutableArtifactBlobProvider(plan.store_path))
        try:
            published = view.authored(request)
        except (KnownDamage, FileNotFoundError):
            return None
        if published is None:
            return None
        observer = self._validity_observer(plan.project_root)
        differences = self._result_checkpoints(plan, request, published.receipt.attempt)
        return view, observer, published, observer.inspect(view, published, checkpoint_differences=differences)

    def _result_checkpoints(
        self, plan: SettledProjectPlan, request: RequestId, attempt: str,
    ) -> tuple[InputDifference, ...]:
        if not request.startswith(("analyze:", "evaluate:")):
            return ()
        snapshot = self.host.jobs_snapshot(
            project=plan.project_id, job_id=attempt, start_host=False,
            checkpoint_admission=(str(plan.store_path), [request]),
        )
        return _checkpoint_differences(snapshot, attempt)

    def _submit_check(
        self, plan: SettledProjectPlan, result: PublishedSuccess, observation: ObservedReceiptValidity,
    ) -> str:
        from feedbax.validity.checks import schedule_check

        request_id = result.receipt.request.id
        prior = cast(list[dict[str, Any]], self.host.jobs(
            project=plan.project_id, job_id=result.receipt.attempt, include_launch=True,
        ))
        producer = prior[0] if prior else None
        kind = request_id.partition(":")[0]
        worker = {
            "train": self._training_worker,
            "evaluate": self._evaluation_worker,
        }.get(kind, self._light_worker)
        if producer is not None and (original := producer.get("launch")):
            if _is_light_request(request_id):
                worker = replace(worker, env=original["env"], preload=tuple(original["preload"]))
            else:
                observed = original["env"]
                numerical = {key: observed[key] for key in ("JAX_ENABLE_X64", "XLA_FLAGS") if key in observed}
                worker = replace(worker, env=worker.env | numerical)
        return schedule_check(
            self.host, project=plan.project_id, project_root=plan.project_root, store=plan.store_path,
            observation=observation, request_id=request_id, producer=producer, worker=worker,
            slot="light" if _is_light_request(request_id) and not worker.preload else "compute",
        )

    def result_validity_action(
        self, plan: SettledProjectPlan, request: RequestId, expected_receipt: ContentObjectRef, *,
        judgement: tuple[tuple[CodeDifference, ...], str, str] | None = None,
    ) -> ReceiptValidity:
        from feedbax.persistence.receipt_validity import record_validity

        visible = {run.request_id for run in plan.plan.runs} | {
            node.request_id for node in plan.plan.light_requests.values()
        }
        if request not in visible or (current := self._result_observation(plan, request)) is None:
            raise JobNotFound(request)
        view, observer, published, observation = current
        if published.ref != expected_receipt:
            raise JobConflict("The latest successful receipt changed.")
        if judgement is None:
            if observation.decision.checks:
                from feedbax.validity.checks import settle_check

                check = self._submit_check(plan, published, observation)
                (row,) = self.host.jobs(project=plan.project_id, job_id=check)
                decided = settle_check(view, observer, row, current_checkpoints=partial(self._result_checkpoints, plan))
                if decided is not None:
                    return decided
            return observation.decision
        differences, who, why = judgement
        pending = {
            difference
            for reason in observation.decision.reasons if reason.kind == "code"
            for difference in reason.differences
        }
        if not differences or not set(differences) <= pending:
            raise JobConflict("Judgement must name the exact current code differences.")
        record_validity(view.store, request, ValidityJudgement(
            receipt=published.ref, differences=differences, who=who, when=datetime.now(UTC), why=why,
        ))
        return observer.refresh().inspect(
            view, published, checkpoint_differences=self._result_checkpoints(plan, request, published.receipt.attempt),
        ).decision

    def maintain_plan_validity(
        self, folders: SavedProjectPlans, plan: SettledProjectPlan, attempts: PlanJobAttempts,
    ) -> bool:
        from feedbax.persistence.effective_success import PublishedSuccess, SuccessView
        from feedbax.validity.checks import settle_check
        from feedbax.validity.project_admission import is_quick

        view = SuccessView(ImmutableArtifactBlobProvider(plan.store_path))
        observer = self._validity_observer(plan.project_root)
        available = 2 - sum(row["state"] in _LIVE_STATES for row in attempts.validity_checks)
        checked = False
        results = {
            request: result for request, history in attempts.by_request.items()
            if (result := next((item for item in reversed(history) if item.validity is not None), None)) is not None
        }
        for row in attempts.validity_checks:
            scope = row["data"]["validity_check"]
            attempt = results.get(scope["request_id"])
            if attempt is not None and attempt.validity is not None and attempt.validity.checks:
                decided = settle_check(view, observer, row, current_checkpoints=partial(self._result_checkpoints, plan))
                checked |= decided is not None and decided != attempt.validity
        if checked or self._automatic_run == "off":
            return checked

        holds = plan.configuration.settings.get("run_only_on_request", [])
        held_steps = set(holds) if isinstance(holds, list) and all(isinstance(name, str) for name in holds) else set()
        automatic = []
        sources = {}

        def authorized_success(request: RequestId) -> PublishedSuccess | None:
            if (current := results.get(request)) is None or current.validity is None:
                return None
            ref = current.validity.receipt
            if view.authored_ref(request) != ref:
                return None
            return PublishedSuccess(ref, view.read_receipt(ref))

        for row in plan.plan.rows:
            requests = _selected_requests(plan.plan, {row.key})
            step = plan.plan.steps[row.key.variant, row.key.step]
            allow_run = row.key.step not in held_steps and (row.kind != "train" or self._automatic_run != "no_train")
            needed = []
            for request in requests:
                attempt = results.get(request)
                validity = attempt.validity if attempt is not None else None
                if validity is None or validity.state != "current":
                    needed.append(request)
                if available > 0 and step.sweep is None and validity is not None and validity.checks:
                    if any(
                        prior["data"]["validity_check"]["receipt"]["sha256"] == validity.receipt.sha256
                        and prior["state"] in _LIVE_STATES for prior in attempts.validity_checks
                    ):
                        continue
                    if (result := authorized_success(request)) is not None:
                        quick_run = allow_run and validity.state != "current" and is_quick(result.receipt, self._quick_duration, view.store)
                        if not quick_run:
                            differences = self._result_checkpoints(plan, request, result.receipt.attempt)
                            observation = observer.inspect(view, result, checkpoint_differences=differences)
                            if observation.decision.checks:
                                self._submit_check(plan, result, observation)
                                available -= 1
            if allow_run and needed:
                candidates = [authorized_success(request) for request in needed]
                if all(
                    candidate is not None and is_quick(candidate.receipt, self._quick_duration, view.store)
                    and attempts.by_request[request][-1] is results.get(request)
                    for request, candidate in zip(needed, candidates, strict=True)
                ):
                    automatic.append(row.key)
                    sources.update({
                        request: results[request].validity.receipt.sha256 for request in needed
                    })
        if automatic:
            identity = sha256(canonical_json_v2_bytes({
                "revision": plan.revision, "receipts": sources,
                "reasons": {
                    request: results[request].validity.model_dump(mode="json") for request in sources
                },
            })).hexdigest()
            self.admit_plan(
                folders, plan, run_keys=automatic, token=f"automatic:{identity}", reuse_completed=True, automatic=True,
            )
        return False

    def _admit_with_reuse(
        self,
        token: str,
        candidates: list[AdmissionItem],
        record: dict[str, Any],
        *, reuse_completed: bool, current: SettledProjectPlan, automatic: bool,
        directly_selected: Collection[RequestId],
    ) -> Admission:
        from feedbax.persistence.effective_success import SuccessView
        from feedbax.studio.canonical_evaluate import EvaluationRequestBinding

        terminal_inputs = {}
        for item in candidates if reuse_completed else ():
            job = item.job
            if job is not None and job.request.startswith("evaluate:"):
                binding = EvaluationRequestBinding.model_validate(job.launch.payload["canonical_request"])
                checkpoint = binding.checkpoint
                if isinstance(checkpoint, ProductRef) and isinstance(checkpoint.select, LastMember):
                    terminal_inputs[job.id] = (checkpoint.request_id, binding.terminal)
        native_requests = sorted({request for request, pin in terminal_inputs.values() if pin is None})
        terminal_train = (record["command"]["store"], native_requests) if native_requests else None

        requested = sorted({item.job.request for item in candidates if item.job is not None})
        for attempt in range(3):
            snapshot = self.host.jobs_snapshot(
                project=record["project"], terminal_train=terminal_train,
                checkpoint_admission=(str(current.store_path), requested),
            )
            view = SuccessView(ImmutableArtifactBlobProvider(current.store_path))
            observer = self._validity_observer(current.project_root)
            terminal_receipts = snapshot.get("terminal_train_receipts", {})
            history: dict[str, list[dict[str, Any]]] = {}
            attempts: dict[str, dict[str, Any]] = {}
            for row in cast(list[dict[str, Any]], snapshot["jobs"]):
                if _is_studio_job(row) and row["data"].get("manifest_root") == record["command"]["store"]:
                    history.setdefault(row["request"], []).append(row)
                    attempts[row["id"]] = row
            chosen = []
            reused = {}
            checkpoint_roots = {}
            runs = {key: dict(run) for key, run in record["runs"].items()}
            for item in candidates:
                if item.job is None:
                    chosen.append(item)
                    continue
                job = replace(item.job, edges=tuple(
                    replace(edge, job_id=(checkpoint_roots if edge.authenticated_completion else reused).get(
                        edge.job_id, edge.job_id,
                    ))
                    for edge in item.job.edges
                ))
                selected = job.request in directly_selected
                eligible = reuse_completed or not selected
                terminal = terminal_inputs.get(job.id)
                receipt = None
                anchor = None
                if terminal is not None:
                    _, pin = terminal
                    if pin is not None:
                        receipt = pin.receipt.model_dump(mode="json")
                    else:
                        (edge,) = job.edges
                        anchor = edge.job_id
                        receipt = terminal_receipts.get(anchor)
                reusable = None
                if eligible and (terminal is None or receipt is not None):
                    reusable = self._reusable(
                        job, history.get(job.request, ()), view=view, observer=observer,
                        attempts=attempts, snapshot=snapshot, automatic=automatic, upstream=not selected,
                        terminal_receipt=ContentObjectRef.model_validate(receipt) if receipt is not None else None,
                    )
                if automatic and selected and reusable is None:
                    from feedbax.validity.project_admission import is_quick

                    latest = view.authored_ref(job.request)
                    if latest is None or not is_quick(view.read_receipt(latest), self._quick_duration, view.store):
                        chosen.append(_rejected(item.run_key, "This request requires explicit Run."))
                        continue
                if reusable is not None:
                    reused[job.id] = reusable
                    if job.request.startswith("train:"):
                        # The Host authenticates checkpoint selections through the original Train attempt.
                        root = attempts[reusable]
                        while root["continues"] is not None:
                            root = attempts[root["continues"]]
                        checkpoint_roots[job.id] = root["id"]
                    chosen.append(AdmissionItem(
                        run_key=item.run_key, disposition="reused", checked_through=snapshot["cursor"],
                        terminal_reuse=TerminalReuse(receipt, anchor) if receipt is not None else None,
                    ))
                    runs[item.run_key]["reused_job_id"] = reusable
                else:
                    chosen.append(replace(
                        item, job=job, checked_through=snapshot["cursor"] if eligible else None,
                    ))
            try:
                return self._admit(token, chosen, record | {"runs": runs})
            except HostError as exc:
                if exc.code != "admission_changed" or attempt == 2:
                    raise
        raise AssertionError("Admission retry exhausted without an outcome")

    def _admit(self, token: str, items: list[AdmissionItem], record: dict[str, Any]) -> Admission:
        with self._launch_lock:
            if not self._accepting:
                raise JobConflict("Studio is stopping and cannot start a run.")
            placement = attempt_target(record.get("target")).model_dump(mode="json")
            placed = [
                replace(item, job=replace(item.job, data=item.job.data | {"target": placement}))
                if item.job is not None else item for item in items
            ]
            return self.host.admit(token, placed, record=record | {"target": placement})

    def warm(self) -> None:
        with self._launch_lock:
            if self._accepting:
                self.host.warm(
                    self._training_worker,
                    runtime_owner=os.path.realpath(self._training_worker.argv[0]),
                )

    def retire_runtime(self, release: str, disposition: str) -> dict:
        """Quiesce installed-runtime admissions and release its owned Host."""
        with self._launch_lock:
            result = self.host.retire_runtime(
                os.path.realpath(sys.executable), release, disposition,
            )
            if result["status"] != "deferred":
                self._accepting = False
                self.host.prohibit_start()
            return result

    def submissions(self, *, project: str | None = None) -> list[StudioJobSubmission]:
        return [
            _project_submission(submission)
            for submission in self.host.submissions(project=project)
            if _is_studio_submission(submission)
        ]

    def submission(self, submission_id: str) -> StudioJobSubmission:
        records = self.host.submissions(id=submission_id)
        if not records or not _is_studio_submission(records[0]):
            raise JobNotFound(submission_id)
        return _project_submission(records[0])

    @staticmethod
    def recover_saved_submission(
        host: HostClient, *, token: str, project: str, revision: str, semantic_root: str,
        store: str, run_keys: Sequence[RunKey], request_ids: Sequence[str], placement: AttemptTarget,
    ) -> StudioJobSubmission | None:
        """Read the receipt for this exact saved Train/Evaluate admission token."""
        rows = host.submissions(token=token)
        if not rows:
            return None
        (record,) = rows
        command = _saved_plan_command(
            "canonical_train" if any(item.startswith("train:") for item in request_ids) else "canonical_evaluate",
            revision, semantic_root, store,
            {plan_attempt_key(key): request for key, request in zip(run_keys, request_ids, strict=True)},
        )
        return JobService._submission_for_action(
            dict(record), project=project, continues=None, command=command, placement=placement,
        )

    def jobs(self, *, project: str | None = None) -> StudioJobSnapshot:
        snapshot = self.host.jobs_snapshot(project=project)
        return StudioJobSnapshot(
            jobs=list(self._activity_rows(snapshot["jobs"], project=project).values()),
            cursor=snapshot["cursor"],
            oldest_cursor=snapshot["oldest_cursor"],
            sessions=[StudioLocalSession()],
        )

    def project_venue_activity(
        self, rows: Sequence[dict[str, Any]], *, pending: set[str], collected: set[str], logs: Mapping[str, list[str]],
    ) -> tuple[StudioActivityJob, ...]:
        """Use observed venue state until verified local custody is available."""
        by_id = {row["id"]: row for row in rows if _is_studio_job(row)}
        seen = self.host.record_get_many([_seen_key(job) for job in by_id])
        result = []
        for job, row in by_id.items():
            activity = _project_activity(
                row, receipt_intact=job in pending or job in collected,
                seen=_seen_key(job) in seen, earlier_definition=False, actor=None,
            )
            if row["state"] == "succeeded" and job in pending:
                activity.status = JobStatus.COLLECTING
                activity.seen = False
                activity.allowed_actions = []
            if activity.status == JobStatus.BLOCKED:
                parents = [
                    by_id[edge["job_id"]]["data"]["label"] for edge in row["edges"]
                    if not edge["satisfied"] and edge["job_id"] in by_id
                ]
                if parents:
                    activity.reason = f"Blocked by {', '.join(dict.fromkeys(parents))}."
            if activity.status == JobStatus.STARTING and activity.phase is None and activity.reason is None:
                activity.reason = "Waiting for a worker to start."
            if activity.status == JobStatus.STOPPING and activity.reason is None:
                activity.reason = (
                    "Finishing the checkpoint." if row["stop_mode"] == "checkpoint" else "Stopping the worker."
                )
            if activity.status == JobStatus.FAILED and activity.outcome and activity.outcome.error:
                error = activity.outcome.error
                if error.phase:
                    activity.reason = f"{error.phase}: {error.message}"
            activity.log_tail = logs.get(job, [])
            result.append(activity)
        return tuple(result)

    def mark_observed_job_seen(self, job: StudioActivityJob) -> StudioActivityJob:
        if job.status in _ACTIVE_PLAN_STATUSES:
            raise JobConflict("Only a finished run can be acknowledged.")
        self.host.record_put(_seen_key(job.job_id), {"seen": True})
        return job.model_copy(update={"seen": True})

    def background_activity(self, after: int | None) -> tuple[int | None, list[str], tuple[str, str, str, JobStatus] | None]:
        """Read active attempts and new terminal Activity rows without starting the Host."""
        try:
            snapshot = self.host.jobs_snapshot(start_host=False)
        except HostError as error:
            if isinstance(error.__cause__, (FileNotFoundError, ConnectionRefusedError)):
                return after, [], None
            raise
        active = [row["id"] for row in snapshot["jobs"] if _is_studio_job(row) and row["state"] in _LIVE_STATES]
        cursor = snapshot["cursor"]
        if after is None or after >= cursor:
            return cursor, active, None
        page = self.host.events(after=after, start_host=False)
        terminal = {
            row["id"]: row for row in snapshot["jobs"]
            if _is_studio_job(row) and row["state"] in {"succeeded", "failed"}
        }
        for event in page["events"]:
            if event["cursor"] > cursor:
                break
            if event["kind"] != "terminal" or (row := terminal.get(event["job_id"])) is None:
                continue
            status, _ = _job_status(
                row, progress=None, receipt_intact=row["state"] != "succeeded" or self._receipt_matches(row),
            )
            if status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
                return event["cursor"], active, (row["id"], row["project"], row["data"]["label"], status)
        return min(cursor, page["cursor"]), active, None

    def stop(self, job_ids: Sequence[str], mode: StopMode) -> list[StudioJobStopItem]:
        ids = tuple(dict.fromkeys(job_ids))
        eligible = tuple(
            row["id"] for row in self.host.jobs_snapshot(job_ids=ids)["jobs"] if _is_studio_job(row)
        )
        outcomes = (
            {result["job_id"]: result for result in self.host.stop(eligible, mode)}
            if eligible else {}
        )
        jobs = (
            self._activity_rows(self.host.jobs_snapshot(job_ids=eligible)["jobs"])
            if eligible else {}
        )
        return [
            StudioJobStopItem(
                job_id=job_id,
                job=jobs.get(job_id),
                error=(
                    outcomes[job_id].get("error")
                    if job_id in outcomes
                    else JobError(code="unknown_job", message="Job does not exist.")
                ),
            )
            for job_id in ids
        ]

    def collect_session_attempt(
        self, source: StoreEndpoint, attempt: Mapping[str, Any], *, project: str, store: Path,
        project_root: Path | None = None,
        progress: Callable[[StoreTransferProgress], None] | None = None,
    ) -> CollectedSessionAttempt:
        """Collect one terminal Jobs snapshot attempt and identify the verified custody for acknowledgement."""
        from feedbax.contracts.artifact_custody import ContentObjectRef
        from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
        from feedbax.persistence.effective_success import SuccessView
        from feedbax.persistence.train_job import read_train_job

        kind = attempt["request"].partition(":")[0]
        state = attempt["state"]
        outcome = attempt["outcome"]
        if (
            attempt["project"] != project or attempt["data"].get("manifest_root") != str(source.root)
            or not isinstance(outcome, dict) or outcome.get("outcome") != state
        ):
            raise JobConflict("Collection requires an exact terminal Jobs observation from its source Store.")
        if state == "succeeded" and kind in {"train", "evaluate"}:
            evidence_key = "store_success"
        elif state == "stopped" and kind == "train":
            evidence_key = "store_job_entry"
        else:
            raise JobConflict("Only successful Train or Evaluate and checkpoint-stopped Train can be collected.")
        ref = ContentObjectRef.model_validate(outcome.get(evidence_key))
        attempt_id = attempt["id"]
        existing = self.host.jobs(job_id=attempt_id)
        if existing:
            (local,) = existing
            local_data = local["data"]
            local_outcome = local["outcome"]
            if (
                (local["project"], local["request"], local["state"], local["continues"], local["delivery"])
                != (project, attempt["request"], state, attempt["continues"], attempt["delivery"])
                or not isinstance(local_data, dict) or local_data.get("manifest_root") != str(store)
                or local_data.get(f"collected_{kind}") is not True
                or attempt_target(local_data.get("target")) != attempt_target(attempt["data"].get("target"))
                or not isinstance(local_outcome, dict) or local_outcome.get(evidence_key) != ref.model_dump(mode="json")
                or (kind == "evaluate" and local_outcome.get("receipt") != outcome.get("receipt"))
            ):
                raise JobConflict("The local Host already has different evidence for this attempt.")
            destination = ImmutableArtifactBlobProvider(store)
            if state == "succeeded":
                SuccessView(destination).read_owned_success(ref)
            else:
                entry = read_train_job(destination, attempt_id, expected=ref)
                stopped = PublishedTrainingCheckpoint.model_validate(outcome["checkpoint"])
                if entry.recovery.pointer != stopped.pointer:
                    raise JobConflict("The local Train entry differs from the observed Stop checkpoint.")
            activity = self._activity_rows(existing)[attempt_id]
        elif kind == "train":
            activity = self.collect_train_attempt(
                source, attempt, project=project, store=store, project_root=project_root, progress=progress,
            )
        else:
            activity = self._collect_evaluate_attempt(source, attempt, project=project, store=store, ref=ref, progress=progress)
        if activity.status != JobStatus(state):
            raise JobConflict("The locally retained attempt differs from the observed terminal status.")
        return CollectedSessionAttempt(activity, ref.sha256)

    def _collect_evaluate_attempt(
        self, source: StoreEndpoint, attempt: Mapping[str, Any], *,
        project: str, store: Path, ref: ContentObjectRef,
        progress: Callable[[StoreTransferProgress], None] | None = None,
    ) -> StudioActivityJob:
        from feedbax.persistence.store_sync import collect_success
        from feedbax.studio.canonical_evaluate import (
            BoundEvaluationJobInput, EvaluationRequestBinding, decode_rollout_request, evaluation_manifest_id,
        )

        attempt_id, request_id = attempt["id"], attempt["request"]
        placement = attempt_target(attempt["data"].get("target"))
        observed = JobReceipt.model_validate(attempt["outcome"]["receipt"])

        def verify(success: Success) -> None:
            manifest = success.evaluation_manifest
            if (
                success.attempt != attempt_id or success.request.id != request_id or manifest is None
                or (observed.manifest_id, observed.manifest_sha256)
                != (evaluation_manifest_id(attempt_id), manifest.sha256)
            ):
                raise JobConflict("Evaluate observation differs from its exact Store receipt.")

        destination = ImmutableArtifactBlobProvider(store)
        success = collect_success(
            source, destination, ref, attempt=attempt_id, request_id=request_id,
            expected_target=placement, verify=verify, progress=progress,
        ).receipt
        _, _, subject = decode_rollout_request(request_id, destination.get_object(success.request.object))
        payload = BoundEvaluationJobInput(
            project_id=project, request_id=request_id, manifest_root=Path(destination.root),
            canonical_request=EvaluationRequestBinding(
                request=success.request.object,
                checkpoint=subject if not isinstance(subject, ContentRef) else None,
                initial_model=subject if isinstance(subject, ContentRef) else None,
            ),
        )
        launch = replace(self._evaluation_worker, payload=payload.model_dump(mode="json", by_alias=True))
        data = attempt["data"]
        job = Job(
            id=attempt_id, project=project, request=request_id, launch=launch,
            data={key: data[key] for key in ("run_key", "stage_id", "scenario_id", "label", "coordinates")}
            | {
                "collected_evaluate": True, "manifest_root": str(destination.root), "runtime": launch.argv[0],
                "target": placement.model_dump(mode="json"),
            },
        )
        self.host.mirror_collected_evaluate(
            job, receipt=ref.model_dump(mode="json"), progress=attempt["progress"], delivery=attempt["delivery"],
        )
        return self._activity_rows(self.host.jobs(job_id=attempt_id))[attempt_id]

    def collect_train_attempt(
        self, source: StoreEndpoint, attempt: Mapping[str, Any], *, project: str, store: Path,
        project_root: Path | None = None,
        progress: Callable[[StoreTransferProgress], None] | None = None,
    ) -> StudioActivityJob:
        """Collect an observed terminal Train and durably mirror its original attempt."""
        from feedbax.contracts.artifact_custody import ContentObjectRef
        from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
        from feedbax.persistence.store_sync import collect_success
        from feedbax.persistence.train_job import collect_train_job

        attempt_id, request_id, state = attempt["id"], attempt["request"], attempt["state"]
        data = attempt["data"]
        outcome = attempt["outcome"]
        if (
            attempt["project"] != project or data.get("manifest_root") != str(source.root)
            or state not in {"succeeded", "stopped"} or not request_id.startswith("train:")
            or not isinstance(outcome, dict) or outcome.get("outcome") != state
        ):
            raise JobConflict("Collection requires an exact terminal Train observation from its source Store.")
        destination = ImmutableArtifactBlobProvider(store)
        placement = attempt_target(data.get("target"))
        if state == "succeeded":
            receipt = ContentObjectRef.model_validate(outcome["store_success"])
            result = collect_success(
                source, destination, receipt, attempt=attempt_id, request_id=request_id,
                expected_target=placement, verify=_require_train_continuation, progress=progress,
            )
            request = result.receipt.request
        else:
            stopped = PublishedTrainingCheckpoint.model_validate(outcome["checkpoint"])
            receipt = ContentObjectRef.model_validate(raw) if (raw := outcome.get("store_job_entry")) else None
            entry = collect_train_job(
                source, destination, attempt=attempt_id, request_id=request_id, pointer=stopped.pointer,
                expected=receipt, progress=progress,
            )
            request = entry.request

        spec, binding = bind_canonical_train(
            request_id, destination.get_object(request.object),
            {ref.sha256: destination.get_object(ref) for ref in request.documents},
            destination, registries=self.registries,
        )
        item = self._training_item(
            project, request_id, spec,
            {key: data[key] for key in ("run_key", "stage_id", "scenario_id", "label", "coordinates")}
            | {"collected_train": True, "target": placement.model_dump(mode="json")}, store,
            canonical_request=binding,
            checkpoint_catalog=tuple(attempt["launch"]["payload"].get("checkpoint_catalog", ())),
        )
        if project_root is not None:
            item = _with_project_root(item, project_root)
        assert item.job is not None
        job = replace(item.job, id=attempt_id, continues=attempt["continues"])
        self.host.mirror_collected_train(
            job, state=state, receipt=receipt.model_dump(mode="json") if receipt is not None else None,
            progress=attempt["progress"], delivery=attempt["delivery"],
        )
        return self._activity_rows(self.host.jobs(job_id=attempt_id))[attempt_id]

    def resume(self, job_id: str, token: str) -> StudioActivityJob:
        previous = self._retry(token, continues=job_id)
        if previous is None:
            rows = self.host.jobs(job_id=job_id, include_launch=True)
            if not rows or not _is_studio_job(rows[0]):
                raise JobNotFound(job_id)
            row = rows[0]
            kind = row["request"].partition(":")[0]
            if kind != "train":
                label = "Analyze" if kind in {"analyze", "render"} else kind.title()
                raise JobConflict(f"/runs: resubmit the saved {label} row to run it again.")
            if not can_resume_attempt(row["state"], row.get("checkpoint"), row["delivery"]):
                raise JobConflict("This run has no checkpoint to resume.")
            launch_data = row["launch"]
            payload = dict(launch_data["payload"])
            portable = row["data"].get("collected_train") or payload.get("collected_resume") is not None
            if portable:
                payload.pop("resume", None)
                payload["collected_resume"] = job_id
            else:
                payload["resume"] = row.get("checkpoint")
            launch = Launch(**(launch_data | {"payload": payload}))
            placement = attempt_target(cast(dict[str, Any], row["data"]).get("target"))
            if isinstance(placement, SessionTarget) and self.host.session_id != str(placement.session_id):
                placement = LOCAL_TARGET
            key = row["data"]["run_key"]
            job = Job(
                id=uuid4().hex,
                project=row["project"],
                request=row["request"],
                launch=launch,
                continues=job_id,
                data={key: value for key, value in row["data"].items() if key != "collected_train"},
            )
            result = self._admit(
                token,
                [AdmissionItem(run_key=key, job=job)],
                {
                    "project": row["project"], "continues": job_id, "runs": {},
                    "actor": StudioSubmissionActor(client="studio").model_dump(mode="json"),
                    "target": placement.model_dump(mode="json"),
                },
            )
            previous = self._submission_for_action(result, project=row["project"], continues=job_id)

        item = previous.items[0]
        if item.job_id is None:
            raise JobConflict(item.reason or "The run could not resume.")
        return self._activity_rows(self.host.jobs(job_id=item.job_id))[item.job_id]

    def acknowledge(self, job_id: str) -> StudioActivityJob:
        rows = self.host.jobs(job_id=job_id)
        if not rows or not _is_studio_job(rows[0]):
            raise JobNotFound(job_id)
        if rows[0]["state"] in _LIVE_STATES:
            raise JobConflict("Only a finished run can be acknowledged.")
        self.host.record_put(_seen_key(job_id), {"seen": True})
        return self._activity_rows(rows)[job_id]

    def events(
        self,
        *,
        after: int,
        projects: tuple[str, ...],
        wait: float,
        machine: bool,
    ) -> tuple[int, StudioJobEventHistory]:
        page = self.host.events(after=after, projects=() if machine else projects, wait=wait)
        visible, rows = self._visible_events(page["events"], activity=True)
        activity_events = {
            event["job_id"]: event["cursor"]
            for event in visible
            if event["kind"] in _ACTIVITY_EVENTS
            or (event["kind"] == "dependency_bound" and rows[event["job_id"]]["data"].get("placement_detail"))
        }
        activity = self._activity_rows([rows[job_id] for job_id in activity_events])
        events = []
        for event in visible:
            job_id = event["job_id"]
            detail = not machine or event["project"] in projects
            snapshot = (
                activity.get(job_id) if activity_events.get(job_id) == event["cursor"] else None
            )
            if (detail and event["kind"] not in _HOST_ONLY_EVENTS) or snapshot is not None:
                events.append(_project_event(event, activity=snapshot, detail=detail))
        return page["head"], StudioJobEventHistory(
            cursor=page["cursor"],
            oldest_cursor=page["oldest_cursor"],
            events=events,
        )

    def event_history(self, *, after: int, project: str, through: int) -> StudioJobEventHistory:
        page = self.host.events(after=after, projects=(project,), wait=0)
        visible, _ = self._visible_events(
            [event for event in page["events"] if event["cursor"] <= through]
        )
        return StudioJobEventHistory(
            cursor=min(through, page["cursor"]),
            oldest_cursor=page["oldest_cursor"],
            events=[_project_event(event) for event in visible],
        )

    @staticmethod
    def venue_events(
        events: Sequence[Mapping[str, Any]], stream: StudioVenueStream,
    ) -> list[StudioJobEvent]:
        return [
            _project_event(event).model_copy(update={"venue": stream})
            for event in events if event["kind"] not in _HOST_ONLY_EVENTS
        ]
    def _visible_events(
        self, events: list[dict[str, Any]], *, activity: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        hidden = {"dependency_rejected"} if activity else {"dependency_rejected", "continued"}
        visible = [event for event in events if event["kind"] not in hidden]
        if not visible:
            return [], {}
        ids = tuple(dict.fromkeys(event["job_id"] for event in visible))
        rows = {
            row["id"]: row
            for row in self.host.jobs_snapshot(job_ids=ids)["jobs"]
            if _is_studio_job(row) or "validity_check" in row["data"]
        }
        projected = []
        for event in visible:
            row = rows.get(event["job_id"])
            if row is None:
                continue
            scope = row["data"].get("validity_check")
            if scope is not None:
                if event["kind"] == "terminal":
                    projected.append(event | {
                        "kind": "validity", "request": scope["request_id"],
                        "data": {"receipt": scope["receipt"], "checks": scope["checks"]},
                    })
            elif event["kind"] != "dependency_bound" or row["data"].get("placement_detail"):
                projected.append(event)
        return projected, rows

    def _retry(
        self,
        token: str,
        *,
        continues: str | None,
        project: str | None = None,
        command: dict[str, Any] | None = None,
        placement: AttemptTarget | None = None,
    ) -> StudioJobSubmission | None:
        rows = self.host.submissions(token=token)
        return (
            self._submission_for_action(
                rows[0], project=project, continues=continues, command=command, placement=placement,
            )
            if rows
            else None
        )

    @staticmethod
    def _submission_for_action(
        response: dict[str, Any],
        *,
        project: str | None,
        continues: str | None,
        command: dict[str, Any] | None = None,
        placement: AttemptTarget | None = None,
    ) -> StudioJobSubmission:
        record = response["record"]
        if not _is_studio_submission(response) or record["continues"] != continues or (
            project is not None and project != record["project"]
        ) or record.get("command") != command or (
            placement is not None and attempt_target(record.get("target")) != placement
        ):
            raise JobConflict("This submission token belongs to another action.")
        return _project_submission(response)

    def _reusable(
        self, candidate: Job, jobs: Sequence[dict[str, Any]], *, view: SuccessView, observer: ProjectValidity,
        attempts: Mapping[str, dict[str, Any]], snapshot: JobsSnapshot, automatic: bool, upstream: bool,
        terminal_receipt: ContentObjectRef | None = None,
    ) -> str | None:
        from feedbax.persistence.effective_success import PublishedSuccess

        parents = tuple(edge for edge in candidate.edges if not edge.authenticated_completion)
        parent_ids = {edge.job_id for edge in parents}
        if any(edge.job_id not in attempts for edge in candidate.edges):
            return None
        origins = {}
        for edge in parents:
            parent = attempts[edge.job_id]
            if parent["state"] == "succeeded" and (ref := view.attempt_ref(edge.job_id)) is not None:
                origins[parent["request"]] = ref
        for job in reversed(jobs):
            if job["state"] in _LIVE_STATES:
                return None
            if job["state"] != "succeeded":
                continue
            ref = view.attempt_ref(job["id"])
            if ref is None or view.authored_ref(candidate.request) != ref or not self._receipt_matches(
                job, terminal_receipt=terminal_receipt,
            ):
                return None
            old_parents = {edge["job_id"] for edge in job["edges"] if not edge.get("authenticated_completion")}
            if (
                terminal_receipt is None and old_parents != parent_ids
                and (not _is_light_request(candidate.request) or len(origins) != len(parents))
            ):
                return None
            from feedbax.validity.project_admission import is_quick

            retained = view.read_receipt(ref)
            result = observer.inspect(
                view, PublishedSuccess(ref, retained), origins=origins,
                checkpoint_differences=_checkpoint_differences(snapshot, job["id"]),
            ).decision
            if result.state == "current":
                return job["id"]
            if upstream and all(reason.kind in ("code", "environment") for reason in result.reasons) and (
                result.state == "unchecked" or not is_quick(retained, self._quick_duration, view.store)
            ):
                return job["id"]
            if result.state == "unchecked" and not automatic:
                raise JobConflict("The result changed during admission. Check the current result and retry Run.")
            return None
        return None

    def _receipt_matches(
        self, job: dict[str, Any], *, terminal_receipt: ContentObjectRef | None = None,
    ) -> bool:
        from feedbax.persistence.effective_success import SuccessView

        root = Path(job["data"]["manifest_root"])
        store = ImmutableArtifactBlobProvider(root)
        view = SuccessView(store)
        if _owned_result_damage(view, job) is not None:
            return False
        if _is_light_request(job["request"]) or _is_collected_train(job):
            return _verified_store_result(view, job) is not None
        if job["request"].startswith("evaluate:"):
            return _verified_evaluation_result(job, root, terminal_receipt=terminal_receipt) is not None
        return _verified_training_receipt(job, root) is not None

    def _activity_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        project: str | None = None,
    ) -> dict[str, StudioActivityJob]:
        rows = [row for row in rows if _is_studio_job(row)]
        if not rows:
            return {}
        seen = self.host.record_get_many(
            [_seen_key(row["id"]) for row in rows if row["state"] not in _LIVE_STATES]
        )
        current: dict[tuple[str, str], str] = {}
        job_ids = {row["id"] for row in rows}
        actors: dict[str, StudioSubmissionActor | None] = {}
        for submission in reversed(self.host.submissions(project=project)):
            if not _is_studio_submission(submission):
                continue
            record = submission["record"]
            for item in submission["items"]:
                job_id = item.get("job_id")
                if job_id in job_ids and item["disposition"] == "admitted":
                    actors[job_id] = _submission_actor(record)
            for run_key, run in record["runs"].items():
                current.setdefault((record["project"], run_key), run["request_id"])
        result = {}
        for row in rows:
            key = (row["project"], row["data"]["run_key"])
            request_id = row["request"]
            result[row["id"]] = _project_activity(
                row,
                receipt_intact=row["state"] != "succeeded" or self._receipt_matches(row),
                seen=_seen_key(row["id"]) in seen,
                earlier_definition=current.get(key, request_id) != request_id,
                actor=actors.get(row["id"]),
            )
        return result


def plan_attempt_key(key: RunKey | ElementKey) -> str:
    run = key.run if isinstance(key, ElementKey) else key
    address = plan_run_key(run.variant, run.step, run.row)
    return f"{address}/elements/{key.name}" if isinstance(key, ElementKey) else address


def _saved_plan_command(
    kind: str, revision: str, semantic_root: str, store: str, requests: Mapping[str, str | None],
) -> dict[str, Any]:
    return {"kind": kind, "reviewed_revision": revision, "semantic_root": semantic_root,
            "store": store, "requests": dict(requests)}


def _with_project_root(item: AdmissionItem, project_root: Path) -> AdmissionItem:
    if item.job is None:
        return item
    job = item.job
    return replace(item, job=replace(job, launch=replace(job.launch, project_root=str(project_root))))


def _is_light_request(request: str) -> bool:
    return request.partition(":")[0] in _LIGHT_KINDS


def _selected_requests(plan: ProjectPlan, selected: Collection[RunKey]) -> tuple[RequestId, ...]:
    return tuple(dict.fromkeys(
        request
        for row in plan.rows if row.key in selected
        for request in (
            tuple(element.request_id for element in row.elements.values())
            if isinstance(row, PlannedAnalyzeRun) else (row.request_id,) if isinstance(row, PlannedRun) else ()
        )
    ))


def _planned_light_runs(
    plan: ProjectPlan, selected: Collection[RunKey],
) -> dict[RunKey | ElementKey, PlannedRun | PlannedElement]:
    nodes = plan.light_requests
    groups = {row.key: row for row in plan.rows if isinstance(row, PlannedAnalyzeRun)}
    result: dict[RunKey | ElementKey, PlannedRun | PlannedElement] = {}

    def visit(key: RunKey | ElementKey) -> None:
        if key in result or (node := nodes.get(key)) is None:
            return
        for bindings in planned_input_groups(node).values():
            for binding in bindings:
                source = binding.element or binding.producer
                if source is not None:
                    visit(source)
        result[key] = node

    for key in selected:
        if (group := groups.get(key)) is not None:
            for element in group.elements.values():
                visit(element.key)
        else:
            visit(key)
    return result


def _plan_run_data(name: str, run: PlannedRow) -> dict[str, Any]:
    return {
        "run_key": name,
        "stage_id": run.key.step,
        "scenario_id": run.key.variant,
        "label": run.label or run.key.step,
        "coordinates": {axis: coordinate.value for axis, coordinate in run.coordinates.items()},
    }


def _changed_request_path(before: ProjectPlan, old: str, after: ProjectPlan, new: str) -> str:
    earlier = decode_request(before.requests[old], before.documents.__getitem__)
    current = decode_request(after.requests[new], after.documents.__getitem__)
    return first_request_change(earlier, current).path


def _rejected(run_key: str, reason: str) -> AdmissionItem:
    return AdmissionItem(run_key=run_key, disposition="rejected", reason=reason)


def _can_resume(job: dict[str, Any]) -> bool:
    return (
        not job["continued"]
        and job["request"].startswith("train:")
        and can_resume_attempt(job["state"], job.get("checkpoint"), job["delivery"])
    )


def _is_studio_job(row: dict[str, Any]) -> bool:
    return "run_key" in row["data"]


def _is_studio_submission(submission: dict[str, Any]) -> bool:
    record = submission["record"]
    return (
        isinstance(record.get("project"), str)
        and "continues" in record
        and (record["continues"] is None or isinstance(record["continues"], str))
        and isinstance(record.get("runs"), dict)
    )


def _submission_actor(record: dict[str, Any]) -> StudioSubmissionActor | None:
    if (actor := record.get("actor")) is not None:
        return StudioSubmissionActor.model_validate(actor)
    return None


def _project_submission(response: dict[str, Any]) -> StudioJobSubmission:
    record = response["record"]
    items = response["items"]
    if (command := record.get("command")) and command.get("kind") in {
        "canonical_train", "canonical_evaluate",
    }:
        by_key = {item["run_key"]: item for item in items}
        items = [by_key[key] for key in command["requests"]]
    return StudioJobSubmission(
        id=response["id"],
        token=response["token"],
        project_id=record["project"],
        actor=_submission_actor(record),
        target=attempt_target(record.get("target")),
        items=[
            StudioSubmissionItem(
                run_key=item["run_key"],
                job_id=item.get("job_id")
                or record["runs"].get(item["run_key"], {}).get("reused_job_id"),
                disposition=item["disposition"],
                reason=item.get("reason"),
            )
            for item in items
        ],
    )


def _project_event(
    event: Mapping[str, Any],
    *,
    activity: StudioActivityJob | None = None,
    detail: bool = True,
) -> StudioJobEvent:
    kind = event["kind"]
    if kind in _HOST_ONLY_EVENTS | _CHECKPOINT_ACTIVITY_EVENTS:
        detail = False
    data = event["data"] if detail else {}
    if kind in {"result", "terminal"}:
        data = {
            key: value for key, value in data.items()
            if key not in {"completion", "completion_error", "model_product", "store_success", "store_publication_error"}
        }
    if kind == "publication":
        data = {
            key: value for key, value in data.items()
            if key not in {"checkpoint_product", "checkpoint_product_error"}
        }
    return StudioJobEvent(
        cursor=event["cursor"],
        job_id=event["job_id"],
        project_id=event["project"],
        request_id=event["request"],
        kind=kind if detail else "activity",
        data=data,
        activity=activity,
    )


def _seen_key(job_id: str) -> str:
    return f"activity/seen/{job_id}"


def _planned_checkpoint_catalogs(plan: ProjectPlan) -> dict[RequestId, set[int]]:
    catalogs = {}
    for run in plan.runs:
        planned = run.inputs.get("subject")
        if planned is not None and isinstance(ref := planned.member.reference, ProductRef):
            if isinstance(ref.select, BatchSetMember):
                catalogs.setdefault(ref.request_id, set()).update(ref.select.batches)
    for element in plan.elements.values():
        for bindings in element.inputs.values():
            for binding in bindings:
                reference = binding.member.reference
                if (
                    isinstance(reference, ProductRef) and reference.role == "checkpoint"
                    and isinstance(reference.select, BatchMember)
                ):
                    catalogs.setdefault(reference.request_id, set()).add(reference.select.batch)
    return catalogs


def _owned_result_damage(view: SuccessView, job: dict[str, Any]) -> str | None:
    from feedbax.contracts.artifact_custody import ContentObjectRef

    try:
        ref = ContentObjectRef.model_validate(job["outcome"]["store_success"])
        return view.owned_damage(job["id"], ref)
    except (OSError, ValueError, KeyError) as error:
        return f"Cannot authenticate the committed result: {error}"


def _is_collected_train(job: dict[str, Any]) -> bool:
    return job["request"].startswith("train:") and job["data"].get("collected_train") is True


def _require_train_continuation(success: Success) -> None:
    if success.continuation is None:
        raise JobConflict("Collected Train success lacks its typed continuation.")


def _verified_store_result(view: SuccessView, job: dict[str, Any]) -> tuple[JobReceipt, Success] | None:
    from feedbax.contracts.artifact_custody import ContentObjectRef

    try:
        outcome = job["outcome"]
        receipt = JobReceipt.model_validate(outcome["receipt"])
        ref = ContentObjectRef.model_validate(outcome["store_success"])
        if (
            view.attempt_ref(job["id"]) != ref or receipt.manifest_id != job["id"]
            or receipt.manifest_sha256 != ref.sha256
        ):
            return None
        if _is_collected_train(job):
            result = view.read_owned_success(ref).receipt
            if result.continuation is None:
                return None
        else:
            result = view.read_receipt(ref)
        if (
            result.attempt != job["id"] or result.request.id != job["request"]
            or result.target != attempt_target(job["data"].get("target"))
        ):
            return None
        return receipt, result
    except (OSError, TypeError, ValueError, KeyError):
        return None


def _collected_train_checkpoint_source(view: SuccessView, result: Success) -> StudioTrainCheckpointSource | None:
    checkpoints = result.inputs.get("checkpoint", ())
    if len(checkpoints) != 1 or (selection := checkpoints[0].selection) is None:
        return None
    try:
        source = view.read_receipt(selection.receipt)
        if source.continuation is None or source.continuation.provenance != selection.occurrence:
            return None
        return StudioTrainCheckpointSource(
            job_id=source.attempt, request_id=source.request.id,
            completed_updates=source.continuation.clock.completed_updates,
        )
    except (OSError, TypeError, ValueError, KeyError):
        return None


def _verified_evaluation_result(
    job: dict[str, Any], store: Path, *, terminal_receipt: ContentObjectRef | None = None,
) -> _VerifiedEvaluation | None:
    from feedbax.contracts.artifact_custody import ContentObjectRef
    from feedbax.persistence.effective_success import SuccessView
    from feedbax.studio.canonical_evaluate import states_product, states_ref, verify_evaluation_receipt

    try:
        outcome = job.get("outcome") or {}
        receipt = JobReceipt.model_validate(outcome.get("receipt"))
        view = SuccessView(ImmutableArtifactBlobProvider(store))
        accepted = ContentObjectRef.model_validate(outcome["store_success"])
        if view.attempt_ref(job["id"]) != accepted:
            return None
        success = view.read_receipt(accepted)
        if (
            success.attempt != job["id"] or success.request.id != job["request"]
            or success.target != attempt_target(job["data"].get("target"))
            or success.evaluation_manifest is None
        ):
            return None
        manifest_object = success.evaluation_manifest
        (bound,) = success.inputs["subject"]
        subject = bound.evaluation_subject
        if terminal_receipt is not None:
            from feedbax.persistence.evaluation_subject import terminal_train_subject_at

            parsed = EvaluationSubjectValue.model_validate(bound.descriptor)
            if subject is None or not parsed.matches_terminal_model(
                terminal_train_subject_at(view.store, terminal_receipt, subject.producer.id).model.content
            ):
                return None
        terminal = subject is not None and subject.terminal_checkpoint is not None
        if job["data"].get("collected_evaluate") is True or terminal or subject is None:
            return _verify_stored_evaluation(view.store, success, receipt, manifest_object)

        binding, = _pinned_dependencies(job)
        manifest, subject = verify_evaluation_receipt(
            store, attempt=job["id"], request_id=job["request"], project=job["project"],
            receipt=receipt, binding=binding, manifest_object=manifest_object,
        )
        if success.products != {"states": (states_product(manifest),)}:
            return None
        gaps = None
        if isinstance(subject, ProductRef) and isinstance(subject.select, UpdateSeriesMember):
            selection = binding["checkpoint_selection"]
            gaps = tuple(StudioEvaluationCheckpointGap.model_validate(gap) for gap in selection.get("gaps", ()))
        return _VerifiedEvaluation(
            receipt=receipt, size=states_ref(manifest).content.size, checkpoint_gaps=gaps,
            placement="live" if "live_evaluation_points" in binding else "ordinary",
            manifest_object=manifest_object,
            checkpoint_sources=_selected_checkpoint_sources(job),
            bound_subject=_best_subject(job),
        )
    except (OSError, TypeError, ValueError, KeyError):
        return None


def _verify_stored_evaluation(
    store: ImmutableArtifactBlobProvider, success: Success, receipt: JobReceipt, manifest_object: ContentObjectRef,
) -> _VerifiedEvaluation:
    from feedbax.contracts.effective_success import EvaluationSubjectValue
    from feedbax.persistence.evaluation_subject import verify_evaluation_success_evidence
    from feedbax.studio.canonical_evaluate import (
        decode_evaluation_manifest, evaluation_checkpoint_coordinates, states_ref,
    )

    verify_evaluation_success_evidence(store, success, verify_bound_closure=False)
    manifest, subject = decode_evaluation_manifest(
        store, store.get_object(manifest_object), manifest_object, success.attempt, success.request.id, receipt,
    )
    if isinstance(subject, ContentRef):
        return _VerifiedEvaluation(receipt, states_ref(manifest).content.size, None, "ordinary", manifest_object)
    (bound,) = success.inputs["subject"]
    source = bound.evaluation_subject
    assert source is not None
    value = EvaluationSubjectValue.model_validate(bound.descriptor)
    members = value.resolved.members if value.resolved is not None else ()
    occurrences = (
        (source.terminal_checkpoint,) if source.terminal_checkpoint is not None
        else tuple(checkpoint.occurrence for checkpoint in source.checkpoints)
    )
    checkpoints = tuple(
        StudioTrainCheckpointSource(
            job_id=checkpoint.run_id, request_id=source.producer.id, completed_updates=member.coordinate,
        )
        for checkpoint, member in zip(occurrences, members, strict=True)
    )
    single = evaluation_checkpoint_coordinates(subject) is None
    checkpoint_source = None
    if single:
        (checkpoint_source,) = checkpoints
    gaps = None
    if value.axis == "update":
        gaps = tuple(StudioEvaluationCheckpointGap(update=gap.update, reason=gap.reason) for gap in value.gaps)
    best = None
    if isinstance(subject, BestRank):
        evidence = manifest.provenance.metadata["best_rank"]["candidates"]
        (checkpoint,) = checkpoints
        chosen = subject.candidates[subject.winner([candidate["score"] for candidate in evidence])]
        if chosen.reference.request_id != checkpoint.request_id:
            raise ValueError("Collected Best result differs from its selected Train subject.")
        best = _project_best_subject(subject, chosen, success.attempt, checkpoint, evidence)
    return _VerifiedEvaluation(
        receipt=receipt, size=states_ref(manifest).content.size, checkpoint_gaps=gaps,
        placement="live" if manifest.metadata.get("placement") == "live" else "ordinary",
        manifest_object=manifest_object,
        checkpoint_source=checkpoint_source,
        checkpoint_sources=() if single else checkpoints,
        bound_subject=best,
    )


def _selected_checkpoint_sources(job: dict[str, Any]) -> tuple[StudioTrainCheckpointSource, ...]:
    return tuple(
        StudioTrainCheckpointSource(
            job_id=member["producer_job_id"], request_id=member["request_id"],
            completed_updates=member["checkpoint"]["completed_updates"],
        )
        for binding in _pinned_dependencies(job)
        for member in binding.get("checkpoint_selection", {}).get("members", ())
    )


def _pinned_dependencies(job: dict[str, Any]):
    return (
        edge["binding"] for edge in job["edges"]
        if edge.get("authenticated_completion") and edge["satisfied"] and edge.get("binding") is not None
    )


def _evaluation_checkpoint_attempt(job: dict[str, Any]) -> StudioEvaluationCheckpointAttempt | None:
    detail = job["data"].get("placement_detail")
    if detail is None:
        return None
    placement = ValidationPlacement.model_validate({
        field: detail[field] for field in ValidationPlacement.model_fields if field in detail
    })
    return StudioEvaluationCheckpointAttempt.model_validate({
        "job_id": job["id"], "placement": placement, **job.get("checkpoint_progress", {}),
    })


def _best_subject(job: dict[str, Any]) -> StudioBoundBestSubject | None:
    from feedbax.studio.canonical_evaluate import best_bound_rank, evaluation_subject_provenance

    try:
        raw = job["launch"]["payload"]["canonical_request"]["checkpoint"]
        if raw.get("kind") != "best_rank":
            return None
        binding, = _pinned_dependencies(job)
        rank = BestRank.model_validate(raw)
        chosen = rank.candidates[best_bound_rank(rank, binding)]
        completion = binding["completion"]
        _, provenance = evaluation_subject_provenance(binding)
        checkpoint = StudioTrainCheckpointSource(
            job_id=completion["producer_job_id"], request_id=completion["request_id"],
            completed_updates=completion["checkpoint"]["completed_updates"],
        )
        return _project_best_subject(
            rank, chosen, job["id"], checkpoint, provenance.metadata["best_rank"]["candidates"],
        )
    except (KeyError, ValueError, TypeError):
        return None


def _project_best_subject(
    rank: BestRank, chosen: BestCandidate, attempt: str, checkpoint: StudioTrainCheckpointSource, evidence: Any,
) -> StudioBoundBestSubject:
    selection = {
        "metric": rank.metric, "order": rank.order, "count": rank.count,
        "candidates": [item.model_dump(mode="json") for item in rank.candidates],
        "evidence": evidence,
    }
    if rank.group:
        selection["group"] = rank.group
    digest = sha256(canonical_json_v2_bytes(selection)).hexdigest()
    return StudioBoundBestSubject(
        key=rank.key, variant=chosen.variant, row=chosen.row, consumer_job_id=attempt,
        producer_job_id=checkpoint.job_id, producer_request_id=checkpoint.request_id,
        completed_updates=checkpoint.completed_updates, selection_sha256=digest,
    )


def _verified_training_receipt(job: dict[str, Any], store: Path) -> JobReceipt | None:
    """Authenticate the successful manifest named by this exact Host attempt."""
    outcome = job.get("outcome") or {}
    try:
        receipt = JobReceipt.model_validate(outcome.get("receipt"))
        job_id = job["id"]
        if receipt.manifest_id != training_run_manifest_id(job_id):
            return None
        path = canonical_manifest_path("TrainingRunManifest", receipt.manifest_id, root=store)
        directories = open_directory_chain(
            path.parent, create=False, error_factory=ValueError, context="training receipt"
        )
        try:
            descriptor, _ = open_existing_file(
                path.name, dir_fd=directories[-1][1],
                rigor=SecurePathRigor.REGULAR_FILE_IDENTITY,
                error_factory=ValueError, context="training receipt",
            )
            with os.fdopen(descriptor, "rb") as stream:
                raw = stream.read()
            recheck_directory_chain(directories, error_factory=ValueError, context="training receipt")
        finally:
            close_directory_chain(directories)
        if sha256(raw).hexdigest() != receipt.manifest_sha256 or not raw.lstrip().startswith(b"{"):
            return None
        manifest = load_manifest_bytes(raw)
        if (
            isinstance(manifest, TrainingRunManifest)
            and (manifest.id, manifest.job_id, manifest.status)
            == (receipt.manifest_id, job_id, "completed")
        ):
            return receipt
    except (OSError, ValueError):
        return None
    return None


def _train_checkpoint_source(
    job: dict[str, Any], by_id: Mapping[str, dict[str, Any]]
) -> StudioTrainCheckpointSource | None:
    """Project the producer completion pinned to this attempt, including copied Resume edges."""
    for binding in _pinned_dependencies(job):
        if "completion" not in binding:
            continue
        completion = binding["completion"]
        producer = by_id.get(completion["producer_job_id"])
        if (
            producer is not None
            and producer["state"] == "succeeded"
            and producer["project"] == completion["project_id"] == job["project"]
            and producer["request"] == completion["request_id"]
            and producer["data"]["manifest_root"] == completion["manifest_root"]
            and producer["outcome"]["completion"] == completion
        ):
            return StudioTrainCheckpointSource(
                job_id=producer["id"], request_id=producer["request"],
                completed_updates=completion["checkpoint"]["completed_updates"],
            )
    return None


def _job_progress(job: dict[str, Any]) -> JobProgress | None:
    progress = job.get("progress")
    return (
        JobProgress(
            completed_updates=progress["completed_updates"], total_updates=progress["total_updates"]
        )
        if progress is not None else None
    )


def _job_status(
    job: dict[str, Any], *, progress: JobProgress | None, receipt_intact: bool
) -> tuple[JobStatus, str | None]:
    state = job["state"]
    outcome = job.get("outcome") or {}
    reason = (outcome.get("error") or {}).get("message")
    kind = job["request"].partition(":")[0]
    evaluate = kind == "evaluate"
    if state == "succeeded" and not receipt_intact:
        label = {"train": "training", "evaluate": "evaluation"}.get(kind, kind)
        return JobStatus.CORRUPT, f"The {label} receipt is missing or has changed."
    if state == "assigned" or (state == "running" and kind == "train" and not (
        progress and progress.completed_updates
    )):
        return JobStatus.STARTING, reason
    if state == "queued":
        if _is_light_request(job["request"]):
            waiting = any(not edge["satisfied"] for edge in job["edges"])
            return JobStatus.QUEUED, "Waiting for producer results." if waiting else "Waiting for a light worker slot."
        if evaluate and any(
            edge.get("authenticated_completion") and not edge["satisfied"] for edge in job["edges"]
        ):
            launch = job.get("launch") or {}
            selector = ((launch.get("payload") or {}).get("canonical_request") or {}).get("checkpoint")
            if selector and selector.get("kind") == "best_rank":
                return JobStatus.QUEUED, "Waiting for all Best candidate Train results."
            selection = (selector or {}).get("select")
            if isinstance(selection, dict) and selection.get("kind") == "update_series":
                updates = ", ".join(map(str, selection["updates"]))
                return JobStatus.QUEUED, f"Waiting for Train checkpoint selection (requested updates {updates})."
            return JobStatus.QUEUED, "Waiting for the Train checkpoint."
        return JobStatus.QUEUED, "Waiting for an evaluation slot." if evaluate else "Waiting for a training slot."
    return JobStatus(state), reason


def active_plan_attempt(attempts: Sequence[PlanJobAttempt]) -> PlanJobAttempt | None:
    return next((attempt for attempt in reversed(attempts) if attempt.status in _ACTIVE_PLAN_STATUSES), None)


def saved_request_status(
    attempts: Sequence[PlanJobAttempt],
    *,
    success: PlanJobAttempt | None,
    problem: str | None = None,
    changed_path: str | None = None,
) -> tuple[JobStatus, str | None]:
    if problem is not None:
        return JobStatus.CANNOT_RUN, problem
    live = active_plan_attempt(attempts)
    if live is not None:
        return live.status, live.reason
    if attempts:
        latest = attempts[-1]
        if success is not None and latest.status is not JobStatus.CORRUPT:
            notice = None
            if latest != success:
                notice = latest.reason or f"A later attempt was {latest.status.value.replace('_', ' ')}."
            if success.validity is not None:
                validity = validity_reason(success.validity)
                notice = " ".join(filter(None, (validity, notice))) or None
                if success.validity.state == "stale":
                    return JobStatus.STALE, notice
            return JobStatus.SUCCEEDED, notice
        return latest.status, latest.reason
    if changed_path is not None:
        return JobStatus.STALE, f"Definition changed at {changed_path}."
    return JobStatus.NOT_RUN, None


def _project_activity(
    job: dict[str, Any],
    *,
    receipt_intact: bool,
    seen: bool,
    earlier_definition: bool,
    actor: StudioSubmissionActor | None,
) -> StudioActivityJob:
    state = job["state"]
    phase = (job.get("phase") or {}).get("phase")
    outcome = job.get("outcome")
    progress = _job_progress(job)
    status, reason = _job_status(job, progress=progress, receipt_intact=receipt_intact)
    resumable = _can_resume(job)
    if state == "stopping":
        actions = ["force"]
    elif state in _LIVE_STATES:
        actions = ["checkpoint", "force"] if job["request"].startswith("train:") else ["force"]
    else:
        actions = ["resume"] if resumable else []
    return StudioActivityJob(
        job_id=job["id"],
        project_id=job["project"],
        request_id=job["request"],
        kind=job["request"].partition(":")[0],
        continues=job.get("continues"),
        status=status,
        phase=phase,
        reason=reason,
        can_resume=resumable,
        allowed_actions=actions,
        seen=seen,
        earlier_definition=earlier_definition,
        actor=actor,
        progress=progress,
        outcome=JobOutcome.model_validate(
            {key: value for key, value in outcome.items() if key in JobOutcome.model_fields}
        )
        if outcome is not None
        else None,
        **{
            key: value
            for key, value in job["data"].items()
            if key in StudioActivityJob.model_fields and key not in {"actor", "kind"}
        },
    )
