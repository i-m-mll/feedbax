"""Root evidence for a saved project's read-only Store report."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from feedbax.compiler.model_assets import model_asset_refs
from feedbax.contracts.acausal import ACAUSAL_GRAPH_SCHEMA_ID, AcausalGraphSpec
from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentObjectRef
from feedbax.contracts.graph import GRAPH_SPEC_SCHEMA_ID, GraphSpec
from feedbax.contracts.project_io import TextProject
from feedbax.contracts.project_plan import compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host import HostClient, HostError
from feedbax.host.completion import completion_file
from feedbax.host.plan_history import plan_ref_key
from feedbax.host.protocol import LIVE_STATES, JobsSnapshot
from feedbax.jobs import _verified_evaluation_result, _verified_training_receipt
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
    content_object_relative_path,
)
from feedbax.persistence.reachability import (
    DEFAULT_SUPERSEDED_DAYS,
    Reachability,
    RootObservation,
    Size,
    measure_store,
    object_target,
    superseded_days,
)
from feedbax.studio.canonical_evaluate import BoundEvaluationJobInput
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.training.checkpoint_custody import (
    CHECKPOINT_SET_NAME,
    CheckpointCustodyError,
    _bound_checkpoint_manifest,
    checkpoint_slot_files,
    load_checkpoint_set,
)


@dataclass(frozen=True)
class CheckpointSize:
    job: str
    transaction: str
    files: int
    bytes: int


@dataclass(frozen=True)
class ProjectRetention:
    project: str
    revision: str
    store: str
    host_cursor: int | None
    objects: Reachability
    checkpoint_files: Size
    checkpoints: tuple[CheckpointSize, ...]


def _observed(name: str, *objects: ContentObjectRef) -> RootObservation:
    return RootObservation(name, tuple(object_target(ref) for ref in objects))


def _authored_roots(text: TextProject, gaps: list[str]) -> list[RootObservation]:
    roots = []
    graph_schemas = {GRAPH_SPEC_SCHEMA_ID: GraphSpec, ACAUSAL_GRAPH_SCHEMA_ID: AcausalGraphSpec}
    for name, document in text.project.documents.items():
        model = graph_schemas.get(document.schema_id)
        if model is None:
            gaps.append(f"document {name}: no Store-root traversal for {document.schema_id}")
            continue
        try:
            assets = tuple(
                ref for _, ref in model_asset_refs(model.model_validate(document.payload))
            )
        except ValueError as error:
            gaps.append(f"document {name}: {error}")
        else:
            if assets:
                roots.append(_observed(f"document:{name}", *assets))

    plan = compile_project(text.project)
    for (_, step), planned in plan.steps.items():
        if planned.kind != "train":
            gaps.append(f"step {step}: no complete Store-root traversal for {planned.kind}")
    for problem in plan.diagnostics:
        step = plan.steps.get((problem.variant, problem.step))
        if problem.step is None or (step is not None and step.kind == "train"):
            gaps.append(
                f"authored Train roots unavailable at {problem.step or 'project'}:{problem.path}"
            )
    for run in plan.runs:
        if run.kind == "train":
            decoded = decode_request(plan.requests[run.request_id], plan.documents.__getitem__)
            assert isinstance(decoded.spec, TrainingRequestSpec)
            assets = tuple(ref for _, ref in model_asset_refs(decoded.spec.graph))
            if assets:
                key = run.key
                roots.append(_observed(f"authored:{key.variant}/{key.step}/{key.row}", *assets))
    return roots


def _checkpoint_files(root: Path, checkpoint: PublishedTrainingCheckpoint) -> dict[Path, int]:
    pointer = checkpoint.pointer
    if Path(pointer.run_id).name != pointer.run_id or pointer.run_id in {"", ".", ".."}:
        raise ValueError("invalid checkpoint job ID")
    expected = root / "checkpoints" / pointer.run_id
    if checkpoint.checkpoint_root != expected:
        raise ValueError("checkpoint root differs from the exact published attempt")

    def read_metadata(path: Path) -> bytes:
        if not path.is_relative_to(expected):
            raise ValueError("checkpoint metadata is outside the exact published attempt")
        with completion_file(root, path.relative_to(root).as_posix()) as (stream, _):
            return stream.read()

    manifest_path, manifest = _bound_checkpoint_manifest(
        expected, pointer, read_bytes=read_metadata
    )
    if manifest.completed_training_batches != checkpoint.completed_updates:
        raise ValueError("checkpoint coordinate differs from the retained job")
    checkpoint_set = load_checkpoint_set(expected, pointer.transaction_id, read_bytes=read_metadata)
    if checkpoint_set.transaction.bytes.digest != pointer.manifest_sha256:
        raise ValueError("checkpoint set differs from the retained publication")
    sizes = {}
    for path in (manifest_path, manifest_path.parent / CHECKPOINT_SET_NAME):
        with completion_file(root, path.relative_to(root).as_posix()) as (_, size):
            sizes[path] = size
    for slot in manifest.slots:
        for path, (_, pinned_size) in checkpoint_slot_files(
            manifest_path.parent,
            slot,
            read_bytes=read_metadata,
        ).items():
            with completion_file(root, path.relative_to(root).as_posix()) as (_, size):
                if size != pinned_size:
                    raise ValueError(f"checkpoint file size differs: {path.relative_to(root)}")
                sizes[path] = size
    return sizes


def _receipt_objects(completion: Mapping) -> tuple[ContentObjectRef, ...]:
    return tuple(
        ContentObjectRef(
            object_id=CONTENT_OBJECT_REF_PREFIX + file["sha256"],
            sha256=file["sha256"],
            size_bytes=file["size_bytes"],
        )
        for file in completion["files"]
        if file["path"] == content_object_relative_path(file["sha256"]).as_posix()
    )


def _train_plan_root(
    store: ImmutableArtifactBlobProvider,
    record: Mapping,
    jobs: Mapping[str, dict],
    gaps: list[str],
) -> RootObservation:
    ref = ContentObjectRef.model_validate(record["plan"])
    plan = strict_json_loads(store.get_object(ref))
    rows = record["rows"]
    if plan["requests"] != {key: row["request_id"] for key, row in rows.items()}:
        raise ValueError("saved Plan request map differs from the Host record")
    for row in rows.values():
        for previous in (row["last"], row["prior"]):
            if previous is not None:
                job = jobs.get(previous["job_id"])
                if job is None or (job["request"], job["state"]) != (
                    previous["request_id"],
                    "succeeded",
                ):
                    gaps.append(f"saved Train result {previous['job_id']} is unavailable")
    return _observed("host:admitted-train-plan", ref)


def _job_roots(
    root: Path,
    job: dict,
    gaps: list[str],
) -> tuple[RootObservation | None, list[tuple[PublishedTrainingCheckpoint, dict[Path, int]]]]:
    name = f"job:{job['id']}"
    request = job["request"]
    payload = job["launch"]["payload"]
    refs = []
    pointers = []
    if payload.get("canonical_request") is not None and request.startswith("train:"):
        binding = BoundStudioTrainingJobInput.model_validate(payload)
        if (binding.manifest_root, binding.project_id, binding.request_id) != (
            root,
            job["project"],
            request,
        ):
            raise ValueError("Train launch differs from the scoped job")
        assert binding.canonical_request is not None
        refs.extend((binding.canonical_request.request, *binding.canonical_request.documents))
        if binding.spec.graph.inline is not None:
            refs.extend(
                ref
                for _, ref in model_asset_refs(GraphSpec.model_validate(binding.spec.graph.inline))
            )
        else:
            gaps.append(f"{name}: bound model assets require an external graph authority")
        if binding.resume is not None:
            pointers.append(binding.resume)
        if binding.checkpoint_input is not None:
            pointers.append(binding.checkpoint_input.checkpoint)
    elif payload.get("canonical_request") is not None and request.startswith("evaluate:"):
        binding = BoundEvaluationJobInput.model_validate(payload)
        if (binding.manifest_root, binding.project_id, binding.request_id) != (
            root,
            job["project"],
            request,
        ):
            raise ValueError("Evaluate launch differs from the scoped job")
        refs.append(binding.canonical_request.request)
    else:
        gaps.append(f"{name}: no typed canonical launch binding")

    for edge in job["edges"]:
        if edge.get("authenticated_completion") and edge["satisfied"]:
            bound = edge["binding"]
            members = (
                bound["checkpoint_selection"]["members"]
                if "checkpoint_selection" in bound
                else (bound["completion"],)
            )
            for member in members:
                if member["manifest_root"] != str(root) or member["project_id"] != job["project"]:
                    raise ValueError("checkpoint input differs from the scoped Store or project")
                refs.extend(_receipt_objects(member))
                pointers.append(PublishedTrainingCheckpoint.model_validate(member["checkpoint"]))

    completion = (job.get("outcome") or {}).get("completion")
    if completion is not None:
        if (
            completion["manifest_root"],
            completion["project_id"],
            completion["request_id"],
            completion["producer_job_id"],
        ) != (str(root), job["project"], request, job["id"]):
            raise ValueError("terminal completion differs from the scoped job")
        refs.extend(_receipt_objects(completion))
    elif job["state"] == "succeeded":
        gaps.append(f"{name}: canonical terminal product closure is unavailable")

    if job.get("checkpoint") is not None:
        pointers.append(PublishedTrainingCheckpoint.model_validate(job["checkpoint"]))
        gaps.append(f"{name}: the Host snapshot carries the latest checkpoint only")
    checkpoints = {}
    for checkpoint in pointers:
        pointer = checkpoint.pointer
        key = (pointer.run_id, pointer.transaction_id)
        if key in checkpoints:
            if checkpoints[key][0] != checkpoint:
                gaps.append(f"{name}: conflicting checkpoint evidence for {pointer.transaction_id}")
            continue
        try:
            checkpoints[key] = checkpoint, _checkpoint_files(root, checkpoint)
        except (CheckpointCustodyError, ValueError, OSError) as error:
            gaps.append(f"{name} checkpoint {pointer.transaction_id}: {error}")
    return _observed(name, *refs) if refs else None, list(checkpoints.values())


def inspect_project_retention(text: TextProject, host: HostClient) -> ProjectRetention:
    """Read scoped Host evidence, then measure the Store under its existing lock."""
    now = datetime.now(UTC)
    days = superseded_days(
        text.configuration.settings.get("keep_superseded", DEFAULT_SUPERSEDED_DAYS)
    )
    store = ImmutableArtifactBlobProvider(text.store_path)
    project = text.root.name
    key = plan_ref_key(project, str(store.root))
    gaps: list[str] = []
    roots = _authored_roots(text, gaps)
    summaries = {}
    checkpoint_files = {}
    cursor = None

    def read_snapshot() -> JobsSnapshot:
        return host.jobs_snapshot(
            project=project,
            start_host=False,
            include_launch=True,
            record_keys=(key,),
        )

    snapshot = None
    try:
        snapshot = read_snapshot()
    except HostError as error:
        gaps.append(f"Host job and admitted Plan roots unavailable: {error}")
    else:
        cursor = snapshot["cursor"]
        jobs = {
            job["id"]: job
            for job in snapshot["jobs"]
            if job["project"] == project and job["data"].get("manifest_root") == str(store.root)
        }
        for job in snapshot["jobs"]:
            if job["id"] not in jobs and job.get("launch", {}).get("payload", {}).get(
                "manifest_root"
            ) == str(store.root):
                gaps.append(f"job:{job['id']}: Store scope lacks a supported Host authority")
        record = snapshot.get("records", {}).get(key)
        if record is not None:
            try:
                roots.append(_train_plan_root(store, record, jobs, gaps))
            except (ValueError, TypeError, KeyError, OSError) as error:
                gaps.append(f"admitted Train Plan roots unavailable: {error}")
        elif any(job["request"].startswith("train:") for job in jobs.values()):
            gaps.append("admitted Train Plan root is unavailable")
        successes = set()
        for job in reversed(jobs.values()):
            request = job["request"]
            if job["state"] == "succeeded":
                if request.startswith("train:"):
                    verified = _verified_training_receipt(job, store.root)
                elif request.startswith("evaluate:"):
                    verified = _verified_evaluation_result(job, store.root)
                else:
                    verified = None
                if verified is None:
                    gaps.append(f"job:{job['id']}: successful receipt authority is unavailable")
                elif request in successes:
                    gaps.append(
                        f"request {request}: exact supersession links and timestamps are unavailable"
                    )
                else:
                    successes.add(request)
            elif job["state"] not in LIVE_STATES and request in successes:
                continue
            try:
                observation, checkpoints = _job_roots(store.root, job, gaps)
            except (ValueError, TypeError, KeyError, OSError) as error:
                gaps.append(f"job:{job['id']}: {error}")
                continue
            if observation is not None:
                roots.append(observation)
            for checkpoint, files in checkpoints:
                pointer = checkpoint.pointer
                summaries[pointer.run_id, pointer.transaction_id] = CheckpointSize(
                    pointer.run_id,
                    pointer.transaction_id,
                    len(files),
                    sum(files.values()),
                )
                checkpoint_files.update(files)
    objects = measure_store(
        store,
        now=now,
        keep_superseded_days=days,
        observations=tuple(roots),
        gaps=tuple(gaps),
    )
    if objects.eligible is not None and objects.eligible.objects:
        try:
            stable = snapshot == read_snapshot()
        except HostError:
            stable = False
        if not stable:
            objects = replace(
                objects,
                unclassified=objects.eligible,
                eligible=None,
                eligible_digests=None,
                roots=tuple(replace(root, unshared=None) for root in objects.roots),
                gaps=("Host root evidence changed or became unavailable during the Store read",),
            )
    return ProjectRetention(
        project,
        text.revision,
        str(store.root),
        cursor,
        objects,
        Size(len(checkpoint_files), sum(checkpoint_files.values())),
        tuple(summary for _, summary in sorted(summaries.items())),
    )
