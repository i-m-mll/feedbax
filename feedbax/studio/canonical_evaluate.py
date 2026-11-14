"""Request custody and dataset reads for saved Evaluate."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.base import ParentRef, Provenance, StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE as STATES_MEDIA_TYPE, DatasetProductRef
from feedbax.contracts.graph import StudioManifestRef, StudioSelectorRef
from feedbax.contracts.jobs import (
    JobReceipt, StudioEvaluationCheckpoint, StudioEvaluationCheckpointGap,
    StudioEvaluationObjectivePoint, StudioEvaluationObjectiveSeries, StudioEvaluationObjectiveTerm,
)
from feedbax.contracts.live_evaluate import LiveEvaluationObservation, ScoredEvaluationPoint
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    load_manifest_bytes,
)
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import BatchMember, BatchSetMember, BestRank, ContentRef, LastMember, ProductRef, UpdateSeriesMember
from feedbax.contracts.request_specs import EvaluationRequestSpec
from feedbax.contracts.rollout import (
    RAW_STATES_SCHEMA as RAW_STATES_SCHEMA,
    ResolvedRolloutBinding,
    ROLLOUT_TYPE as ROLLOUT_TYPE,
    STATES_SCHEMA as STATES_SCHEMA,
    RolloutParameters as RolloutParameters,
    authored_variation,
    rollout_objective_sha256,
    rollout_semantics,
)
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.workspace_replay import (
    WorkspaceReplayManifestRefs,
    WorkspaceReplaySampleAxis,
    WorkspaceReplayTrack,
    WorkspaceReplayTrial,
    WorkspaceReplayTrialIdentity,
)
from feedbax.host.checkpoint_selection import selected_coordinates
from feedbax.host.best_selection import completion_score
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
if TYPE_CHECKING:
    from feedbax.dataset import Dataset
    from feedbax.studio.canonical_train import BoundStudioTrainingJobInput

class CannotRun(ValueError):
    """A capability required by the authored request is unavailable."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.reason = message
        super().__init__(f"{path}: {message}")


class EvaluationRequestBinding(StrictModel):
    request: ContentObjectRef
    checkpoint: ProductRef | BestRank


class BoundEvaluationJobInput(StrictModel):
    project_id: str
    request_id: str
    manifest_root: Path
    canonical_request: EvaluationRequestBinding


def _unexpected_document(digest: str) -> bytes:
    raise CannotRun("/params", f"this candidate does not support a factored document {digest}")


def evaluation_checkpoint_coordinates(subject: ProductRef | BestRank) -> tuple[int, ...] | None:
    if isinstance(subject, BestRank):
        return None
    if isinstance(subject.select, BatchMember):
        return finite_coordinates((subject.select.batch,))
    if isinstance(subject.select, UpdateSeriesMember):
        return subject.select.updates
    if isinstance(subject.select, BatchSetMember):
        return subject.select.batches
    return None


def decode_rollout_request(
    request_id: str, raw: bytes,
) -> tuple[EvaluationRequestSpec, RolloutParameters, ProductRef | BestRank]:
    decoded = decode_request(raw, _unexpected_document)
    spec = decoded.spec
    if decoded.id != request_id or not isinstance(spec, EvaluationRequestSpec):
        raise ValueError("Launch differs from the canonical Evaluate request.")
    if spec.evaluation_type != ROLLOUT_TYPE:
        raise CannotRun("/evaluation_type", "requires the built-in rollout evaluation")
    if spec.numerical_settings.model_dump(exclude_none=True):
        raise CannotRun("/numerical_settings", "numerical overrides are unavailable")
    members = decoded.inputs.get("subject", ())
    if decoded.inputs.keys() != {"subject"} or len(members) != 1:
        raise CannotRun("/inputs/subject", "requires one Train checkpoint")
    subject = members[0].reference
    if not isinstance(subject, BestRank) and (
        not isinstance(subject, ProductRef)
        or subject.role != "checkpoint"
        or not subject.request_id.startswith("train:")
        or not isinstance(subject.select, (LastMember, BatchMember, BatchSetMember, UpdateSeriesMember))
    ):
        raise CannotRun(
            "/inputs/subject/checkpoint",
            "requires a Train checkpoint selected by last, an exact batch, a batch set, or an update series",
        )
    try:
        evaluation_checkpoint_coordinates(subject)
    except ValueError as error:
        raise CannotRun("/inputs/subject/checkpoint", str(error)) from error
    try:
        params = RolloutParameters.model_validate(spec.params)
    except ValidationError as error:
        first = error.errors()[0]
        path = pointer(("params", *map(str, first["loc"])))
        raise CannotRun(path, first["msg"]) from error
    for path, patch in authored_variation(params.variation):
        if "op" not in patch.model_fields_set:
            raise CannotRun(f"{path}/op", "declare the override operation explicitly")
    return spec, params, subject


def best_bound_rank(subject: BestRank, binding: dict, *, verify_metrics: bool = False) -> int:
    evidence = binding["best_rank"]
    if (evidence["key"], evidence["rank"]) != (subject.key, subject.rank):
        raise ValueError("Best binding differs from the requested rank.")
    candidates = evidence["candidates"]
    scores = []
    for candidate, claim in zip(subject.candidates, candidates, strict=True):
        completion = claim["completion"]
        score = claim["score"]
        if (
            completion["request_id"] != candidate.reference.request_id
            or type(score) not in (int, float)
            or (verify_metrics and completion_score(completion, subject.metric) != score)
        ):
            raise ValueError("Best binding differs from its authenticated candidate result.")
        scores.append(score)
    winner = subject.winner(scores)
    if binding["completion"] != candidates[winner]["completion"]:
        raise ValueError("Evaluate completion differs from the ranked Train subject.")
    return winner


def exact_evaluation_subject(
    subject: ProductRef | BestRank, binding: dict, *, verify_metrics: bool = False,
) -> ProductRef:
    if isinstance(subject, ProductRef):
        if "best_rank" in binding:
            raise ValueError("Fixed Evaluate subject received Best evidence.")
        return subject
    return subject.candidates[best_bound_rank(subject, binding, verify_metrics=verify_metrics)].reference


def state_key(owner: str, symbol: str) -> str:
    """Project a declared state symbol of a root occurrence into a dataset key."""
    return semantic_key("model", "state", (owner,), symbol)


def evaluation_manifest_id(job_id: str) -> str:
    return f"feedbax-evaluate-attempt:{job_id}"


def evaluation_subject_provenance(binding: dict) -> tuple[list[ParentRef], Provenance]:
    """Project the exact Host edge into an Evaluate receipt without reselection."""
    if selection := binding.get("checkpoint_selection"):
        members = selection["members"]
        metadata = {
            "checkpoint_selection": {
                "anchor": binding["dependency_job_id"],
                **{key: selection[key] for key in ("requested", "selector_sha256", "lineage", "lineage_tip")},
            },
        }
        if "live_evaluation_points" in binding:
            metadata["live_evaluation_points"] = binding["live_evaluation_points"]
        train = []
        if "gaps" in selection:
            closure = selection["closure"]
            product = closure["product"]
            metadata["checkpoint_selection"]["gaps"] = selection["gaps"]
            metadata["checkpoint_selection"]["closure"] = {
                "kind": closure["kind"],
                "attempt": product["producer_job_id"],
                "checkpoint": product["checkpoint"]["pointer"]["transaction_root_sha256"],
            }
            if closure["kind"] == "completion":
                receipt = JobReceipt.model_validate(product["receipt"])
                train.append(ParentRef(
                    kind="TrainingRunManifest", id=receipt.manifest_id, role="coverage_closure",
                    metadata={"manifest_sha256": receipt.manifest_sha256},
                ))
    else:
        completion = binding["completion"]
        members = [completion]
        metadata = {}
        receipt = JobReceipt.model_validate(completion["receipt"])
        train = [ParentRef(
            kind="TrainingRunManifest", id=receipt.manifest_id, role="subject",
            metadata={"manifest_sha256": receipt.manifest_sha256},
        )]
        if best := binding.get("best_rank"):
            metadata["best_rank"] = {
                "rank": best["rank"], "key": best["key"],
                "candidates": [
                    {
                        "attempt": claim["completion"]["producer_job_id"],
                        "request_id": claim["completion"]["request_id"],
                        "receipt_sha256": claim["completion"]["receipt"]["manifest_sha256"],
                        "binding_sha256": claim["completion"]["binding_sha256"],
                        "score": claim["score"],
                    }
                    for claim in best["candidates"]
                ],
            }
    parents = []
    for member in members:
        checkpoint = member["checkpoint"]
        point = checkpoint["pointer"]
        attributes = {
            "manifest_sha256": point["manifest_sha256"],
            "completed_updates": checkpoint["completed_updates"],
            "producer_job_id": point["run_id"],
        }
        if selection:
            attributes.update({
                "publication_cursor": member["publication_cursor"],
                "binding_sha256": member["binding_sha256"],
                "producer_request_id": member["request_id"],
                "transaction_root_sha256": point["transaction_root_sha256"],
            })
        parents.append(ParentRef(
            kind="TrainingCheckpointTransactionManifest", id=point["transaction_id"],
            role="checkpoint", metadata=attributes,
        ))
    return train, Provenance(parents=parents, metadata=metadata)


def decode_evaluation_manifest(
    store: ImmutableArtifactBlobProvider, raw: bytes, ref: ContentObjectRef,
    job_id: str, request_id: str, receipt: JobReceipt,
) -> tuple[EvaluationRunManifest, ProductRef | BestRank]:
    if receipt.manifest_id != evaluation_manifest_id(job_id):
        raise ValueError("Evaluation receipt names a different attempt.")
    if ref.sha256 != receipt.manifest_sha256:
        raise ValueError("Evaluation manifest does not match the Host receipt.")
    manifest = load_manifest_bytes(raw)
    if (
        not isinstance(manifest, EvaluationRunManifest)
        or manifest.id != receipt.manifest_id
        or manifest.status != "completed"
        or manifest.metadata.get("request_id") != request_id
    ):
        raise ValueError("Evaluation manifest does not certify this completed request attempt.")
    request = ContentObjectRef.model_validate(manifest.metadata["canonical_request"])
    spec, _, subject = decode_rollout_request(request_id, store.get_object(request))
    projection = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
    if (
        projection.evaluation_type != spec.evaluation_type
        or projection.params != spec.params
        or projection.inputs != manifest.provenance.parents
        or projection.training_run_ids != [item.id for item in manifest.input_training_runs]
    ):
        raise ValueError("Evaluate result differs from its canonical request and declared inputs.")
    return manifest, subject


def verify_evaluation_receipt(
    root: Path, *, attempt: str, request_id: str, project: str, receipt: JobReceipt, binding: dict,
    manifest_object: ContentObjectRef,
) -> tuple[EvaluationRunManifest, ProductRef | BestRank]:
    """Authenticate the worker receipt against the exact Host dependency."""
    store = ImmutableArtifactBlobProvider(root)
    manifest, subject = decode_evaluation_manifest(
        store, store.get_object(manifest_object), manifest_object, attempt, request_id, receipt,
    )
    selected_subject = exact_evaluation_subject(subject, binding)
    if (coordinates := evaluation_checkpoint_coordinates(selected_subject)) is not None:
        selection = binding["checkpoint_selection"]
        members = selection["members"]
        if (
            list(coordinates) != selection["requested"]
            or ("gaps" in selection and not isinstance(subject.select, UpdateSeriesMember))
            or selected_coordinates(selection) != tuple(
                member["checkpoint"]["completed_updates"] for member in members
            )
        ):
            raise ValueError("Evaluate checkpoint selection differs from its authored coordinates.")
        if "gaps" in selection:
            members = [*members, selection["closure"]["product"]]
    else:
        members = [binding["completion"]]
    if any(
        member["request_id"] != selected_subject.request_id
        or member["project_id"] != project
        or member["manifest_root"] != str(root)
        for member in members
    ):
        raise ValueError("Evaluate checkpoint evidence differs from its admitted Train subject.")
    train, provenance = evaluation_subject_provenance(binding)
    if (
        ("live_evaluation_points" in binding) != (manifest.metadata.get("placement") == "live")
        or manifest.input_training_runs != train
        or manifest.provenance.parents != provenance.parents
        or manifest.provenance.metadata != provenance.metadata
    ):
        raise ValueError("Evaluate receipt differs from its accepted checkpoint provenance.")
    return manifest, subject


def states_ref(manifest: EvaluationRunManifest) -> DatasetProductRef:
    (artifact,) = (item for item in manifest.artifacts if item.role == "states")
    if artifact.media_type != STATES_MEDIA_TYPE or artifact.storage_backend != "feedbax-cas":
        raise ValueError("Evaluation states do not name a safetensors object in custody.")
    return DatasetProductRef(
        content=ContentRef(sha256=artifact.sha256, size=artifact.size_bytes),
        descriptor_sha256=artifact.metadata["descriptor_sha256"],
    )


def states_product(manifest: EvaluationRunManifest):
    """Project the manifest's exact dataset to the shared typed Store product."""
    from feedbax.contracts.effective_success import Product

    states = states_ref(manifest)
    return Product(
        name="states", content=states.content.model_copy(update={"media_type": STATES_MEDIA_TYPE}),
        descriptor={"descriptor_sha256": states.descriptor_sha256},
    )


class EvaluationResult:
    """Open the dataset through the exact receipt of one successful Jobs attempt.

    Jobs supplies a manifest object only after authenticating the typed Store success
    and its exact states product.
    """

    def __init__(
        self, root: Path, job_id: str, request_id: str, receipt: JobReceipt,
        *, manifest_object: ContentObjectRef,
    ):
        self._root = root
        self._job_id = job_id
        self._request_id = request_id
        self.receipt = receipt
        self._manifest_object = manifest_object

    def _manifest(self) -> EvaluationRunManifest:
        store = ImmutableArtifactBlobProvider(self._root)
        manifest, _ = decode_evaluation_manifest(
            store, store.get_object(self._manifest_object), self._manifest_object,
            self._job_id, self._request_id, self.receipt,
        )
        return manifest

    @property
    def size_bytes(self) -> int:
        return states_ref(self._manifest()).content.size

    @contextmanager
    def open_states(self, *, verify: bool = False) -> Iterator[Dataset]:
        """Read Store-owned slices, or authenticate selected tensors on request."""
        from feedbax.persistence.dataset import open_stored_dataset

        manifest = self._manifest()
        ref = states_ref(manifest)
        spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
        params = RolloutParameters.model_validate(spec.params)
        store = ImmutableArtifactBlobProvider(self._root)
        with open_stored_dataset(store, ref, verify=verify) as dataset:
            if dataset.descriptor.schema_id != params.states_schema:
                raise ValueError("Evaluation receipt names an incompatible states schema.")
            yield dataset

    def replay_trial(
        self, trial: str, *, anchor_id: str, checkpoint: int | None = None,
    ) -> WorkspaceReplayTrial:
        """Project the declared position selector into the existing Scene trial contract."""
        with self.open_states() as dataset:
            descriptor = dataset.descriptor
            name = descriptor.attrs.get("scene_position")
            if not isinstance(name, str) or name not in descriptor.variables:
                raise CannotRun("/states/scene_position", "the result has no declared Scene position")
            update_axis = "update" if "update" in descriptor.dimensions else "checkpoint"
            if checkpoint is None:
                if descriptor.dimensions[update_axis] != 1:
                    raise CannotRun("/checkpoint", "choose a retained checkpoint for replay")
                checkpoint_index = 0
            else:
                checkpoints = dataset.data(update_axis).tolist()
                if checkpoint not in checkpoints:
                    raise CannotRun("/checkpoint", "the chosen checkpoint is absent from the retained set")
                checkpoint_index = checkpoints.index(checkpoint)
            trials = dataset.data("trial").tolist()
            if trial not in trials:
                raise CannotRun("/trial", "the selected trial is absent from the retained result")
            trial_index = trials.index(trial)
            field = descriptor.variables[name]
            if (
                field.dims != (update_axis, "condition", "trial", "state_time", "xy")
                or descriptor.dimensions["condition"] != 1 or descriptor.dimensions["xy"] != 2
            ):
                raise ValueError("The retained Scene position has an incompatible layout.")
            axis = dataset.data("state_time")
            values = dataset.data(name, (checkpoint_index, 0, trial_index))
        return WorkspaceReplayTrial(
            identity=WorkspaceReplayTrialIdentity(index=trial_index, stable_id=trial),
            time=WorkspaceReplaySampleAxis(length=len(axis), units="s", values=axis.tolist()),
            tracks=[
                WorkspaceReplayTrack(
                    anchor_id=anchor_id,
                    selector=StudioSelectorRef(namespace="state_path", compact=name),
                    samples=values.tolist(),
                    dim=values.shape[-1],
                    dtype=str(values.dtype),
                    units=field.units,
                    frame=field.frame,
                )
            ],
            manifest_refs=WorkspaceReplayManifestRefs(
                producer_manifest=StudioManifestRef(
                    kind="EvaluationRunManifest",
                    id=self.receipt.manifest_id,
                    metadata={"manifest_sha256": self.receipt.manifest_sha256},
                )
            ),
        )

    def objective_series(self, requested: list[int], *, objective: ObjectiveDocument) -> StudioEvaluationObjectiveSeries:
        """Read held-out values and their exact checkpoint publications."""
        manifest = self._manifest()
        selection = manifest.provenance.metadata.get("checkpoint_selection")
        if not isinstance(selection, dict) or selection["requested"] != requested:
            raise ValueError("The result does not cover the saved checkpoint rule.")
        params = RolloutParameters.model_validate(
            EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline).params,
        )
        objective_hash = rollout_objective_sha256(objective)
        lineage = selection["lineage"]
        if not lineage or len(set(lineage)) != len(lineage) or (
            lineage[0] != selection["anchor"] or lineage[-1] != selection["lineage_tip"]
        ):
            raise ValueError("Invalid Evaluate checkpoint lineage.")
        checkpoints = [_objective_checkpoint(parent) for parent in manifest.provenance.parents]
        gaps = [StudioEvaluationCheckpointGap.model_validate(gap) for gap in selection.get("gaps", ())]
        updates = [checkpoint.completed_updates for checkpoint in checkpoints]
        if (
            any(checkpoint.job_id not in lineage for checkpoint in checkpoints)
            or len(updates) + len(gaps) != len(requested)
            or set(updates) | {gap.update for gap in gaps} != set(requested)
        ):
            raise ValueError("Evaluate checkpoint publications and gaps do not cover the saved rule.")
        condition = None
        with self.open_states() as dataset:
            if dataset.data("update").tolist() != updates:
                raise ValueError("Evaluate dataset updates differ from its checkpoint publications.")
            points = []
            if updates:
                (condition,) = dataset.data("condition").tolist()
                effective = ResolvedRolloutBinding.model_validate(manifest.metadata["effective_binding"])
                if (
                    condition != params.condition_id()
                    or effective.digest() != manifest.metadata["effective_binding_sha256"]
                    or effective.objective != objective_hash
                    or effective.seed != params.seed or effective.trial_policy != params.trial_policy
                    or dataset.data("trial").tolist() != list(params.trials)
                    or [member.coordinate for member in effective.members] != updates
                    or any(member.trials != params.trials for member in effective.members)
                ):
                    raise ValueError("Evaluate dataset differs from its resolved scientific inputs.")
                points = _objective_points(dataset, checkpoints, objective, len(params.trials))
        return StudioEvaluationObjectiveSeries(
            job_id=self._job_id, request_id=self._request_id, receipt=self.receipt,
            lineage=lineage, condition=condition, trials=list(params.trials), seed=params.seed,
            objective=objective_hash, requested=requested, points=points, gaps=gaps,
        )


def live_objective_series(
    store: ImmutableArtifactBlobProvider, *, project_id: str, job_id: str, request_id: str,
    producer_request: str, params: RolloutParameters, requested: list[int],
    objective: ObjectiveDocument, observation: LiveEvaluationObservation,
) -> StudioEvaluationObjectiveSeries:
    """Authenticate a live objective from the exact datasets accepted by the Host."""
    from feedbax.job_completion import read_training_binding
    from feedbax.persistence.content_retention import read_active_roots
    from feedbax.persistence.dataset import open_stored_dataset
    from feedbax.training.checkpoint_custody import published_model_tree_sha256

    objective_hash = rollout_objective_sha256(objective)
    condition = params.condition_id()
    if params.variation or len(set(observation.lineage)) != len(observation.lineage):
        raise ValueError("Live objectives require one unchanged scientific lineage.")
    points = []
    scored = []
    seen_updates = set()
    for selected in observation.points:
        point = selected.point
        checkpoint = point.checkpoint
        update = checkpoint.completed_updates
        if (
            point.evaluation_job_id != job_id or selected.producer_request_id != producer_request
            or selected.producer_job_id != checkpoint.pointer.run_id
            or selected.producer_job_id not in observation.lineage
            or checkpoint.checkpoint_root != store.root / "checkpoints" / selected.producer_job_id
            or update not in requested or update in seen_updates
        ):
            raise ValueError("Live objective differs from its selected publication.")
        seen_updates.add(update)
        if isinstance(result := point.result, ScoredEvaluationPoint):
            scored.append((selected, result))
    if seen_updates.intersection(gap.update for gap in observation.gaps):
        raise ValueError("Live objective gaps overlap accepted observations.")

    launches: dict[str, BoundStudioTrainingJobInput] = {}
    semantics: dict[bytes, tuple[str, str]] = {}
    roots = read_active_roots(store, tuple(selected.point.root_id for selected, _ in scored)) if scored else ()
    for (selected, result), root in zip(scored, roots, strict=True):
        point = selected.point
        checkpoint = point.checkpoint
        pointer = checkpoint.pointer
        update = checkpoint.completed_updates
        if len(root.targets) != 1 or root.targets[0].object != result.states.object_ref:
            raise ValueError("Live objective lacks its exact active dataset.")
        if selected.binding_sha256 not in launches:
            launches[selected.binding_sha256] = read_training_binding(
                store.root, selected.producer_job_id, selected.binding_sha256,
            )
        source = launches[selected.binding_sha256]
        if (
            source.project_id != project_id or source.request_id != producer_request
            or source.manifest_root != store.root or source.spec.trial_documents is None
            or source.spec.trial_documents.objective != objective
        ):
            raise ValueError("Live objective differs from its bound Train request.")
        identity = canonical_json_v2_bytes(source.model_dump(mode="json", exclude={"resume"}))
        if identity not in semantics:
            semantics[identity] = _unchanged_rollout_semantics(source)
        model, task = semantics[identity]
        member = result.member
        if (
            (member.model, member.task, result.objective) != (model, task, objective_hash)
            or member.model_tree != published_model_tree_sha256(checkpoint.checkpoint_root, pointer)
            or member.coordinate != update or member.trials != params.trials
        ):
            raise ValueError("Live objective differs from its resolved scientific inputs.")
        subject = StudioEvaluationCheckpoint(
            job_id=selected.producer_job_id, request_id=selected.producer_request_id,
            completed_updates=update, publication_cursor=point.publication_cursor,
            transaction=JobReceipt(manifest_id=pointer.transaction_id, manifest_sha256=pointer.manifest_sha256),
            transaction_root_sha256=pointer.transaction_root_sha256, binding_sha256=selected.binding_sha256,
        )
        with open_stored_dataset(store, result.states, verify=True) as dataset:
            if (
                dataset.descriptor.schema_id != params.states_schema
                or dataset.data("update").tolist() != [update]
                or dataset.data("condition").tolist() != [condition]
                or dataset.data("trial").tolist() != list(params.trials)
            ):
                raise ValueError("Live objective dataset differs from its authored axes.")
            (numeric,) = _objective_points(dataset, [subject], objective, len(params.trials))
            if numeric.total != result.total:
                raise ValueError("Live objective total differs from its accepted dataset.")
            points.append(numeric)
    return StudioEvaluationObjectiveSeries(
        job_id=job_id, request_id=request_id, receipt=None, lineage=observation.lineage,
        condition=condition if points else None, trials=list(params.trials), seed=params.seed,
        objective=objective_hash, requested=requested, points=points, gaps=list(observation.gaps),
    )


def _unchanged_rollout_semantics(source: BoundStudioTrainingJobInput) -> tuple[str, str]:
    import asyncio

    from feedbax.compiler.graph import resolve_graph_schedule
    from feedbax.plugins.composition import compose_application
    from feedbax.studio.canonical_train import resolve_training_request_binding

    registry = asyncio.run(compose_application(modules=source.bootstrap_modules)).bundle
    trial = resolve_training_request_binding(source, registries=registry).trial
    graph = trial.graph.graph
    _, key_schedule = resolve_graph_schedule(graph)
    return rollout_semantics(graph, key_schedule, trial.task_source, trial.execution.input_bindings, {})


def _objective_checkpoint(parent: ParentRef) -> StudioEvaluationCheckpoint:
    if parent.kind != "TrainingCheckpointTransactionManifest" or parent.role != "checkpoint":
        raise ValueError("Evaluate provenance has an unexpected checkpoint parent.")
    meta = parent.metadata
    return StudioEvaluationCheckpoint(
        job_id=meta["producer_job_id"], request_id=meta["producer_request_id"],
        completed_updates=meta["completed_updates"],
        transaction=JobReceipt(manifest_id=parent.id, manifest_sha256=meta["manifest_sha256"]),
        transaction_root_sha256=meta["transaction_root_sha256"],
        binding_sha256=meta["binding_sha256"], publication_cursor=meta["publication_cursor"],
    )


def _objective_points(
    dataset: Dataset, checkpoints: list[StudioEvaluationCheckpoint], objective: ObjectiveDocument, trials: int,
) -> list[StudioEvaluationObjectivePoint]:
    def values(name: str):
        if dataset.descriptor.variables[name].dims != ("update", "condition"):
            raise ValueError(f"Unexpected axes for {name}.")
        return dataset.data(name, (slice(None), 0)).tolist()

    totals = values("objective/total")
    term_prefix = "objective/terms"
    coverage_prefix = "objective/coverage"
    paths = {key.removeprefix(term_prefix) for key in dataset.descriptor.variables if key.startswith(term_prefix + "/")}
    coverage_paths = {
        key.removeprefix(coverage_prefix) for key in dataset.descriptor.variables if key.startswith(coverage_prefix + "/")
    }
    labels = {pointer((term.id,)): term.name or term.id for term in objective.terms if term.enabled}
    if paths != coverage_paths or paths != labels.keys():
        raise ValueError("Evaluate objective terms and coverage differ from the authored Objective.")
    terms: list[list[StudioEvaluationObjectiveTerm]] = [[] for _ in checkpoints]
    for path in sorted(paths):
        for row, value, coverage in zip(terms, values(term_prefix + path), values(coverage_prefix + path), strict=True):
            if coverage > trials:
                raise ValueError("Evaluate term coverage exceeds the authored trial count.")
            row.append(StudioEvaluationObjectiveTerm(
                path=path, name=labels[path], value=value, coverage=coverage,
            ))
    return [
        StudioEvaluationObjectivePoint(checkpoint=checkpoint, total=total, terms=row)
        for checkpoint, total, row in zip(checkpoints, totals, terms, strict=True)
    ]
