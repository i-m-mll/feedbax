"""Admission and activity for local Studio training requests."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from feedbax._secure_fs import (
    SecurePathRigor,
    close_directory_chain,
    open_directory_chain,
    open_existing_file,
    recheck_directory_chain,
)
from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
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
)
from feedbax.contracts.manifest import (
    TrainingRunManifest,
    canonical_manifest_path,
    load_manifest_bytes,
    training_run_manifest_id,
)
from feedbax.contracts.project_plan import PlannedRun, ProjectPlan, RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import BatchSetMember, BestRank, ProductRef, RequestId, UpdateSeriesMember
from feedbax.contracts.training import TrainingRunSpec
from feedbax.host import Admission, AdmissionItem, Edge, HostClient, HostError, Job, Launch
from feedbax.host.checkpoint_selection import is_checkpoint_producer
from feedbax.host.plan_history import AuthoredTrainPlanRef, plan_ref_key, plan_run_key
from feedbax.host.protocol import can_resume_attempt
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.store_damage import inspect_without_recording
from feedbax.plugins.application import ApplicationRegistryBundle
from feedbax.training_host_worker import BOOTSTRAP_ENV
from feedbax.studio.canonical_train import (
    BoundStudioTrainingJobInput,
    CanonicalTrainRequestBinding,
    bind_canonical_train,
)

if TYPE_CHECKING:
    from feedbax.contracts.artifact_custody import ContentObjectRef
    from feedbax.persistence.effective_success import SuccessView
    from feedbax.training.validation_placement import ValidationQualification
    from feedbax.component_registry import ComponentRegistry
    from feedbax.studio.canonical_evaluate import EvaluationResult, RolloutParameters
    from feedbax.project_folder import ProjectFolder, SettledProjectPlan


_LIVE_STATES = frozenset({"queued", "assigned", "running", "stopping"})
_HOST_ONLY_EVENTS = frozenset({"dependency_bound", "dependency_rejected"})
_ACTIVITY_EVENTS = frozenset(
    {
        "queued",
        "assigned",
        "accepted",
        "phase",
        "progress",
        "publication",
        "stopping",
        "blocked",
        "terminal",
    }
)
EVALUATION_SUBJECT_REQUIRED = "select an unambiguous admitted Train subject"


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


@dataclass(frozen=True)
class PlanJobAttempts:
    """Attempt evidence at one Host cursor, keyed by canonical request ID."""

    cursor: int
    by_request: Mapping[RequestId, tuple[PlanJobAttempt, ...]]
    plan_ref: AuthoredTrainPlanRef | None = None
    checkpoint_producers: frozenset[RequestId] = frozenset()
    checkpoint_catalogs: Mapping[RequestId, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class _VerifiedEvaluation:
    receipt: JobReceipt
    size: int
    checkpoint_gaps: tuple[StudioEvaluationCheckpointGap, ...] | None
    placement: Literal["ordinary", "live"]
    manifest_object: ContentObjectRef


class PlanJobReader:
    """Read the existing Host snapshot and authenticate its saved result evidence."""

    def __init__(
        self, host: HostClient, *, components: ComponentRegistry | None = None,
    ) -> None:
        self.host = host
        self._evaluation_components = components

    def evaluation_parameters(
        self, plan: ProjectPlan, run: PlannedRun,
    ) -> tuple[RolloutParameters, ProductRef | BestRank]:
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
    ) -> PlanJobAttempts:
        """Read saved Plan attempts and authenticate any selected stopped checkpoints."""
        manifest_root = str(store)
        key = plan_ref_key(project, manifest_root)
        include_producers = any(request.startswith("evaluate:") for request in requests)
        snapshot = self.host.jobs_snapshot(
            project=project, start_host=False, record_keys=(key,), include_launch=include_producers,
        )
        plan_ref = cast(AuthoredTrainPlanRef | None, snapshot.get("records", {}).get(key))
        wanted = set(requests)
        if plan_ref is not None:
            for row in plan_ref["rows"].values():
                wanted.add(row["request_id"])
                wanted.update(result["request_id"] for result in (row["last"], row["prior"]) if result)
        stopped_jobs = set(stopped_checkpoint_jobs)
        attempts: dict[RequestId, list[PlanJobAttempt]] = {}
        by_id = {
            job["id"]: job for job in snapshot["jobs"]
            if job["data"].get("manifest_root") == manifest_root
        }
        producers = Counter(
            job["request"] for job in by_id.values()
            if include_producers and job["request"].startswith("train:")
            and is_checkpoint_producer(job["launch"], continues=job["continues"], manifest_root=manifest_root)
        )
        catalogs = {
            job["request"]: tuple(job["launch"]["payload"].get("checkpoint_catalog", ()))
            for job in by_id.values()
            if producers[job["request"]] == 1
            and is_checkpoint_producer(job["launch"], continues=job["continues"], manifest_root=manifest_root)
        }
        from feedbax.persistence.effective_success import SuccessView

        success_view = None
        for job in by_id.values():
            request_id = job["request"]
            if request_id not in wanted:
                continue
            evaluate = request_id.startswith("evaluate:")
            succeeded = job["state"] == "succeeded"
            size = None
            result = None
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
            else:
                receipt = _verified_training_receipt(job, store) if succeeded else None
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
                if not evaluate and status == JobStatus.STOPPED and job["id"] in stopped_jobs
                else None
            )
            attempts.setdefault(request_id, []).append(
                PlanJobAttempt(
                    job["id"], status, (job.get("phase") or {}).get("phase"),
                    progress, reason, receipt, binding_sha256,
                    continues=job["continues"],
                    checkpoint_source=_train_checkpoint_source(job, by_id),
                    output_size_bytes=size,
                    checkpoint_sources=_selected_checkpoint_sources(job) if evaluate and receipt is not None else (),
                    stopped_checkpoint=stopped_checkpoint,
                    checkpoint_gaps=result.checkpoint_gaps if result is not None else None,
                    evaluation_placement=result.placement if result is not None else None,
                    bound_subject=_best_subject(job) if evaluate and (not succeeded or result is not None) else None,
                )
            )
        return PlanJobAttempts(
            snapshot["cursor"],
            MappingProxyType({request: tuple(rows) for request, rows in attempts.items()}),
            plan_ref,
            frozenset(catalogs),
            MappingProxyType(catalogs),
        )


class JobService(PlanJobReader):
    """Bind authored requests to immutable worker launches over the durable host."""

    def __init__(
        self,
        host: HostClient,
        registries: ApplicationRegistryBundle,
        *,
        bootstrap_modules: tuple[str, ...] = (),
        live_validation_qualifications: Sequence[ValidationQualification] = (),
    ) -> None:
        super().__init__(host, components=registries.components)
        self.registries = registries
        self.bootstrap_modules = bootstrap_modules
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
            preload=("jax", "feedbax.training.managed_studio"),
        )
        self._training_worker = replace(
            base_worker,
            env=base_worker.env | {BOOTSTRAP_ENV: json.dumps(bootstrap_modules, separators=(",", ":"))},
            warm_entrypoint="feedbax.training_host_worker:main",
            preload=(*base_worker.preload, "feedbax.training.host_prewarm"),
        )
        self._evaluation_worker = replace(
            base_worker,
            warm_entrypoint="feedbax.training.evaluate_worker:main",
            preload=(*base_worker.preload, "feedbax.plugins.composition"),
        )
        self._launch_lock = Lock()
        self._accepting = True

    def admit_plan(
        self,
        folders: ProjectFolder,
        reviewed: SettledProjectPlan,
        *,
        run_keys: Sequence[RunKey],
        token: str,
        checkpoint_catalog: tuple[int, ...] = (),
        actor: StudioSubmissionActor = StudioSubmissionActor(client="studio"),
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
        kinds = {before[key].kind for key in selected.values() if key in before}
        evaluate = "evaluate" in kinds
        train = "train" in kinds
        command = {
            "kind": "canonical_train" if train or not evaluate else "canonical_evaluate",
            "reviewed_revision": reviewed.revision,
            "semantic_root": reviewed.semantic_root,
            "store": str(reviewed.store_path),
            "requests": {
                name: before[key].request_id if key in before else None
                for name, key in selected.items()
            },
        }
        if checkpoint_catalog:
            if evaluate and not train:
                raise JobConflict("/inputs/subject/checkpoint: checkpoint catalogue belongs to Train.")
            checkpoint_catalog = finite_coordinates(checkpoint_catalog)
            command["checkpoint_catalog"] = list(checkpoint_catalog)
        if previous := self._retry(token, project=project, continues=None, command=command):
            return previous

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
            for name, key in selected.items():
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
                        item = self._evaluation_item(
                            project, run.request_id, current.plan.requests[run.request_id], data, store,
                            checkpoint, params,
                        )
                    except (ValueError, KeyError) as error:
                        items[name] = _rejected(name, f"cannot_run {error}")
                        continue
                    planned = run.inputs.get("subject")
                    if planned is None or planned.member.reference != checkpoint:
                        raise ValueError("The saved Plan disagrees with the encoded Evaluate subject.")
                    if isinstance(checkpoint, BestRank):
                        checkpoints[name] = tuple(
                            (key, candidate.reference.request_id)
                            for key, candidate in zip(planned.candidates, checkpoint.candidates, strict=True)
                        )
                    else:
                        checkpoints[name] = ((planned.producer, checkpoint.request_id),)
                    runs[name] = data | {"request_id": run.request_id}
                    items[name] = item
                    continue
                if old is None or old.kind != "train":
                    items[name] = _rejected(name, "The reviewed row is not a runnable Train request.")
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
                        self.registries.training_programs,
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
            order = list(selected)
            if checkpoints:
                order = [names_by_key[run.key] for run in current.plan.runs if run.key in names_by_key]
                order.extend(name for name, key in selected.items() if key not in after)
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
            record = {
                "project": project, "continues": None, "runs": runs, "command": command,
                "actor": actor.model_dump(mode="json"),
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
            admitted = self._admit(token, [items[name] for name in order], record)
        return self._submission_for_action(
            admitted, project=project, continues=None, command=command
        )

    def _checkpoint_producer(self, project: str, root: Path, request: RequestId) -> str | None:
        anchors = [
            row["id"]
            for row in self.host.jobs(project=project, request=request, include_launch=True)
            if is_checkpoint_producer(row["launch"], continues=row["continues"], manifest_root=str(root))
        ]
        return anchors[0] if len(anchors) == 1 else None

    def _evaluation_item(
        self, project: str, request_id: str, raw: bytes, data: dict[str, Any],
        store: ImmutableArtifactBlobProvider, checkpoint: ProductRef | BestRank, params: RolloutParameters,
    ) -> AdmissionItem:
        from feedbax.studio.canonical_evaluate import (
            BoundEvaluationJobInput,
            EvaluationRequestBinding,
        )

        root = store.root
        payload = BoundEvaluationJobInput(
            project_id=project,
            request_id=request_id,
            manifest_root=root,
            canonical_request=EvaluationRequestBinding(
                request=store.store_rooted_object(raw, namespace="studio-evaluate-request"),
                checkpoint=checkpoint,
            ),
        )
        launch = replace(self._evaluation_worker, payload=payload.model_dump(mode="json", by_alias=True))
        if isinstance(checkpoint, ProductRef) and isinstance(checkpoint.select, UpdateSeriesMember) and not params.variation:
            qualification = self._live_validation_qualifications.get((checkpoint.request_id, request_id))
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
        from feedbax.studio.canonical_evaluate import EvaluationResult
        from feedbax.persistence.effective_success import SuccessView

        rows = self.host.jobs_snapshot(job_id=job_id, project=project, start_host=False)["jobs"]
        if not rows:
            raise JobNotFound(job_id)
        (job,) = rows
        if (
            job["project"] != project or job["data"].get("manifest_root") != str(store)
            or not job["request"].startswith("evaluate:") or job["state"] != "succeeded"
        ):
            raise JobNotFound(job_id)
        damage = _owned_result_damage(SuccessView(ImmutableArtifactBlobProvider(store)), job)
        if damage is not None:
            raise JobConflict(damage)
        result = _verified_evaluation_result(job, store)
        if result is None:
            raise JobConflict("The completed Evaluate receipt cannot be authenticated.")
        return EvaluationResult(
            store, job_id, job["request"], result.receipt, manifest_object=result.manifest_object,
        )

    def _training_item(
        self,
        project: str,
        request_id: str,
        spec: TrainingRunSpec,
        data: dict[str, Any],
        root: Path,
        *,
        canonical_request: CanonicalTrainRequestBinding | None = None,
        checkpoint_catalog: tuple[int, ...] = (),
    ) -> AdmissionItem:
        payload = BoundStudioTrainingJobInput(
            spec=spec,
            project_id=project,
            request_id=request_id,
            manifest_root=root,
            bootstrap_modules=self.bootstrap_modules,
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

    def _admit_with_reuse(
        self,
        token: str,
        candidates: list[AdmissionItem],
        record: dict[str, Any],
    ) -> Admission:
        for attempt in range(3):
            snapshot = self.host.jobs_snapshot(project=record["project"])
            history: dict[str, list[dict[str, Any]]] = {}
            for row in snapshot["jobs"]:
                if _is_studio_job(row):
                    history.setdefault(row["request"], []).append(row)
            chosen = []
            runs = {key: dict(run) for key, run in record["runs"].items()}
            for item in candidates:
                if item.job is None:
                    chosen.append(item)
                elif reusable := self._reusable(history.get(item.job.request, ())):
                    chosen.append(AdmissionItem(run_key=item.run_key, disposition="reused"))
                    runs[item.run_key]["reused_job_id"] = reusable
                else:
                    chosen.append(replace(item, checked_through=snapshot["cursor"]))
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
            return self.host.admit(token, items, record=record)

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

    def jobs(self, *, project: str | None = None) -> StudioJobSnapshot:
        snapshot = self.host.jobs_snapshot(project=project)
        return StudioJobSnapshot(
            jobs=list(self._activity_rows(snapshot["jobs"], project=project).values()),
            cursor=snapshot["cursor"],
            oldest_cursor=snapshot["oldest_cursor"],
            sessions=[StudioLocalSession()],
        )

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

    def resume(self, job_id: str, token: str) -> StudioActivityJob:
        previous = self._retry(token, continues=job_id)
        if previous is None:
            rows = self.host.jobs(job_id=job_id, include_launch=True)
            if not rows or not _is_studio_job(rows[0]):
                raise JobNotFound(job_id)
            row = rows[0]
            if row["request"].startswith("evaluate:"):
                raise JobConflict("/runs: resubmit the saved Evaluate row to run it again.")
            if not _can_resume(row):
                raise JobConflict("This run has no checkpoint to resume.")
            launch_data = row["launch"]
            payload = launch_data["payload"] | {"resume": row.get("checkpoint")}
            launch = Launch(**(launch_data | {"payload": payload}))
            key = row["data"]["run_key"]
            job = Job(
                id=uuid4().hex,
                project=row["project"],
                request=row["request"],
                launch=launch,
                continues=job_id,
                data=row["data"],
            )
            result = self._admit(
                token,
                [AdmissionItem(run_key=key, job=job)],
                {
                    "project": row["project"], "continues": job_id, "runs": {},
                    "actor": StudioSubmissionActor(client="studio").model_dump(mode="json"),
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
        visible, rows = self._visible_events(page["events"])
        activity_events = {
            event["job_id"]: event["cursor"]
            for event in visible
            if event["kind"] in _ACTIVITY_EVENTS
        }
        activity = self._activity_rows([rows[job_id] for job_id in activity_events])
        events = []
        for event in visible:
            job_id = event["job_id"]
            detail = not machine or event["project"] in projects
            snapshot = (
                activity.get(job_id) if activity_events.get(job_id) == event["cursor"] else None
            )
            if detail or snapshot is not None:
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

    def _visible_events(
        self, events: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        visible = [event for event in events if event["kind"] not in _HOST_ONLY_EVENTS]
        if not visible:
            return [], {}
        ids = tuple(dict.fromkeys(event["job_id"] for event in visible))
        rows = {
            row["id"]: row
            for row in self.host.jobs_snapshot(job_ids=ids)["jobs"]
            if _is_studio_job(row)
        }
        return [event for event in visible if event["job_id"] in rows], rows

    def _retry(
        self,
        token: str,
        *,
        continues: str | None,
        project: str | None = None,
        command: dict[str, Any] | None = None,
    ) -> StudioJobSubmission | None:
        rows = self.host.submissions(token=token)
        return (
            self._submission_for_action(rows[0], project=project, continues=continues, command=command)
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
    ) -> StudioJobSubmission:
        record = response["record"]
        if not _is_studio_submission(response) or record["continues"] != continues or (
            project is not None and project != record["project"]
        ) or record.get("command") != command:
            raise JobConflict("This submission token belongs to another action.")
        return _project_submission(response)

    def _reusable(self, jobs: Sequence[dict[str, Any]]) -> str | None:
        if any(job["state"] in _LIVE_STATES for job in jobs):
            return None
        for job in reversed(jobs):
            if job["state"] == "succeeded":
                return job["id"] if self._receipt_matches(job) else None
        return None

    def _receipt_matches(self, job: dict[str, Any]) -> bool:
        from feedbax.persistence.effective_success import SuccessView

        root = Path(job["data"]["manifest_root"])
        store = ImmutableArtifactBlobProvider(root)
        if _owned_result_damage(SuccessView(store), job) is not None:
            return False
        if job["request"].startswith("evaluate:"):
            return _verified_evaluation_result(job, root) is not None
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


def _plan_run_data(name: str, run: PlannedRun) -> dict[str, Any]:
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
    return not job["request"].startswith("evaluate:") and can_resume_attempt(
        job["state"], job.get("checkpoint"), job["delivery"],
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
    event: dict[str, Any],
    *,
    activity: StudioActivityJob | None = None,
    detail: bool = True,
) -> StudioJobEvent:
    kind = event["kind"]
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
    return catalogs


def _owned_result_damage(view: SuccessView, job: dict[str, Any]) -> str | None:
    from feedbax.contracts.artifact_custody import ContentObjectRef

    try:
        ref = ContentObjectRef.model_validate(job["outcome"]["store_success"])
        return view.owned_damage(job["id"], ref)
    except (OSError, ValueError, KeyError) as error:
        return f"Cannot authenticate the committed result: {error}"


def _verified_evaluation_result(job: dict[str, Any], store: Path) -> _VerifiedEvaluation | None:
    from feedbax.contracts.artifact_custody import ContentObjectRef
    from feedbax.persistence.effective_success import SuccessView
    from feedbax.studio.canonical_evaluate import states_product, states_ref, verify_evaluation_receipt

    try:
        outcome = job.get("outcome") or {}
        receipt = JobReceipt.model_validate(outcome.get("receipt"))
        binding, = _pinned_dependencies(job)
        view = SuccessView(ImmutableArtifactBlobProvider(store))
        accepted = ContentObjectRef.model_validate(outcome["store_success"])
        if view.attempt_ref(job["id"]) != accepted:
            return None
        success = view.read_receipt(accepted)
        if success.attempt != job["id"] or success.request.id != job["request"] or success.evaluation_manifest is None:
            return None
        manifest_object = success.evaluation_manifest

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
        )
    except (OSError, TypeError, ValueError, KeyError):
        return None


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
        selection = {
            "metric": rank.metric, "order": rank.order, "count": rank.count,
            "candidates": [item.model_dump(mode="json") for item in rank.candidates],
            "evidence": provenance.metadata["best_rank"]["candidates"],
        }
        if rank.group:
            selection["group"] = rank.group
        digest = sha256(canonical_json_v2_bytes(selection)).hexdigest()
        return StudioBoundBestSubject(
            key=rank.key, variant=chosen.variant, row=chosen.row,
            consumer_job_id=job["id"],
            producer_job_id=completion["producer_job_id"], producer_request_id=completion["request_id"],
            completed_updates=completion["checkpoint"]["completed_updates"], selection_sha256=digest,
        )
    except (KeyError, ValueError, TypeError):
        return None


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
    evaluate = job["request"].startswith("evaluate:")
    if state == "succeeded" and not receipt_intact:
        return JobStatus.CORRUPT, (
            "The evaluation receipt is missing or has changed." if evaluate
            else "The training receipt is missing or has changed."
        )
    if state == "assigned" or (state == "running" and not evaluate and not (
        progress and progress.completed_updates
    )):
        return JobStatus.STARTING, reason
    if state == "queued":
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


def saved_request_status(
    attempts: Sequence[PlanJobAttempt],
    *,
    success: PlanJobAttempt | None,
    problem: str | None = None,
    changed_path: str | None = None,
) -> tuple[JobStatus, str | None]:
    if problem is not None:
        return JobStatus.CANNOT_RUN, problem
    live = next((attempt for attempt in reversed(attempts) if attempt.status in {
        JobStatus.QUEUED, JobStatus.AWAITING_AUTHORIZATION, JobStatus.STARTING,
        JobStatus.RUNNING, JobStatus.STOPPING, JobStatus.COLLECTING,
    }), None)
    if live is not None:
        return live.status, live.reason
    if attempts:
        latest = attempts[-1]
        if success is not None and latest.status is not JobStatus.CORRUPT:
            notice = None
            if latest != success:
                notice = latest.reason or f"A later attempt was {latest.status.value.replace('_', ' ')}."
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
        actions = ["checkpoint", "force"]
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
