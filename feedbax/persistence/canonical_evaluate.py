"""Request custody and dataset reads for saved Evaluate."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, ValidationError, model_validator

from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.base import ParentRef, Provenance, StrictModel
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE as STATES_MEDIA_TYPE, DatasetProductRef
from feedbax.contracts.effective_success import TerminalTrainSubject
from feedbax.contracts.jobs import (
    JobReceipt, StudioEvaluationCheckpoint, StudioEvaluationCheckpointGap,
    StudioEvaluationObjectivePoint, StudioEvaluationObjectiveSeries, StudioEvaluationObjectiveTerm,
)
from feedbax.contracts.loaded_model import IMPORTED_MODEL_MEDIA_TYPE, ImportedModelProvenance, ImportedModelSource
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    load_manifest_bytes,
)
from feedbax.contracts.model_input import (
    INITIAL_MODEL_MEDIA_TYPE, OBJECTIVE_INPUT_MEDIA_TYPE, TASK_INPUT_MEDIA_TYPE, document_input_object, read_document_input,
)
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.request_identity import DecodedRequest, decode_request
from feedbax.contracts.request_refs import BatchMember, BatchSetMember, BestRank, ContentRef, LastMember, ProductRef, UpdateSeriesMember
from feedbax.contracts.request_specs import EvaluationRequestSpec
from feedbax.contracts.rollout import (
    RAW_STATES_SCHEMA as RAW_STATES_SCHEMA,
    CannotRun,
    ResolvedInitialRollout,
    ResolvedRolloutBinding,
    ROLLOUT_TYPE as ROLLOUT_TYPE,
    STATES_SCHEMA as STATES_SCHEMA,
    RolloutParameters as RolloutParameters,
    authored_variation,
    rollout_objective_sha256,
)
from feedbax.contracts.semantic_values import model_state_address, semantic_key
from feedbax.contracts.trial_clock import TrialMeanSelection
from feedbax.contracts.trial_record import StudioTrialReplayResponse, TrialAggregate, TrialRecords
from feedbax.persistence.evaluation_trial_records import RecordedTrialLayout, read_trial_record
from feedbax.persistence.checkpoint_selection import selected_coordinates
from feedbax.persistence.best_selection import completion_score
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.project_subject import imported_model_provenance, read_imported_model

if TYPE_CHECKING:
    from feedbax.dataset import Dataset


ROLLOUT_JAX_BACKEND = "cpu"
ROLLOUT_JAX_ENABLE_X64 = False
type EvaluationSubject = ProductRef | BestRank | ContentRef


class EvaluationRequestBinding(StrictModel):
    request: ContentObjectRef
    checkpoint: ProductRef | BestRank | None = Field(default=None, exclude_if=lambda value: value is None)
    initial_model: ContentRef | None = Field(default=None, exclude_if=lambda value: value is None)
    terminal: TerminalTrainSubject | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _one_model_source(self) -> EvaluationRequestBinding:
        if self.initial_model is None:
            if self.checkpoint is None:
                raise ValueError("Evaluate requires a model subject")
        elif (
            self.checkpoint is not None or self.terminal is not None
            or self.initial_model.media_type not in {INITIAL_MODEL_MEDIA_TYPE, IMPORTED_MODEL_MEDIA_TYPE}
            or self.initial_model.tree
        ):
            raise ValueError("Initial Evaluate requires one immutable Model input")
        return self

    @property
    def subject(self) -> EvaluationSubject:
        if self.initial_model is not None:
            return self.initial_model
        assert self.checkpoint is not None
        return self.checkpoint


class BoundEvaluationJobInput(StrictModel):
    project_id: str
    request_id: str
    manifest_root: Path
    canonical_request: EvaluationRequestBinding


def _unexpected_document(digest: str) -> bytes:
    raise CannotRun("/params", f"this candidate does not support a factored document {digest}")


def evaluation_checkpoint_coordinates(subject: EvaluationSubject) -> tuple[int, ...] | None:
    if not isinstance(subject, ProductRef):
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
) -> tuple[EvaluationRequestSpec, RolloutParameters, EvaluationSubject]:
    decoded = decode_request(raw, _unexpected_document)
    spec = decoded.spec
    if decoded.id != request_id or not isinstance(spec, EvaluationRequestSpec):
        raise ValueError("Launch differs from the canonical Evaluate request.")
    if spec.evaluation_type != ROLLOUT_TYPE:
        raise CannotRun("/evaluation_type", "requires the built-in rollout evaluation")
    if spec.numerical_settings.model_dump(exclude_none=True):
        raise CannotRun("/numerical_settings", "numerical overrides are unavailable")
    members = decoded.inputs.get("subject", ())
    if len(members) != 1:
        raise CannotRun("/inputs/subject", "requires one Model subject")
    subject = members[0].reference
    if isinstance(subject, ContentRef):
        initial_evaluation_references(decoded)
    elif decoded.inputs.keys() != {"subject"}:
        raise CannotRun("/inputs", "a checkpoint rollout inherits its Task and Objective")
    elif not isinstance(subject, BestRank) and (
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


def initial_evaluation_references(decoded: DecodedRequest) -> dict[str, ContentRef]:
    required = {
        "subject": {INITIAL_MODEL_MEDIA_TYPE, IMPORTED_MODEL_MEDIA_TYPE},
        "task": {TASK_INPUT_MEDIA_TYPE},
    }
    if "objective" in decoded.inputs:
        required["objective"] = {OBJECTIVE_INPUT_MEDIA_TYPE}
    if decoded.inputs.keys() != required.keys():
        raise CannotRun("/inputs", "initial Evaluate requires a Model and Task, with an optional Objective")
    refs = {}
    for role, media_types in required.items():
        members = decoded.inputs[role]
        content = members[0].reference
        if len(members) != 1 or not isinstance(content, ContentRef) or content.tree or content.media_type not in media_types:
            raise CannotRun(f"/inputs/{role}", "requires one immutable document of the declared kind")
        refs[role] = content
    return refs


def initial_evaluation_provenance(refs: dict[str, ContentRef]) -> Provenance:
    kinds = {
        "subject": "LoadedModel" if refs["subject"].media_type == IMPORTED_MODEL_MEDIA_TYPE else "ModelDocument",
        "task": "TaskDocument", "objective": "ObjectiveDocument",
    }
    return Provenance(parents=[
        ParentRef(kind=kinds[role], id=ref.sha256, role=role, metadata={"sha256": ref.sha256, "size_bytes": ref.size})
        for role, ref in refs.items()
    ])


def verify_initial_rollout(
    store: ImmutableArtifactBlobProvider, manifest: EvaluationRunManifest, refs: dict[str, ContentRef],
) -> ResolvedInitialRollout:
    binding = ResolvedRolloutBinding.model_validate(manifest.metadata["effective_binding"])
    (member,) = binding.members
    objective = refs.get("objective")
    params = RolloutParameters.model_validate(EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline).params)
    loaded, _ = verified_imported_subject(store, refs["subject"])
    if (
        not isinstance(member, ResolvedInitialRollout) or member.initial_model != refs["subject"].sha256
        or manifest.input_training_runs or manifest.provenance != initial_evaluation_provenance(refs)
        or "placement" in manifest.metadata or binding.digest() != manifest.metadata.get("effective_binding_sha256")
        or binding.objective != (objective.sha256 if objective is not None else None)
        or manifest.loaded_model != loaded or member.replica_ids != (loaded.replicas if loaded else ())
        or (binding.seed, binding.trial_policy, member.trials) != (params.seed, params.trial_policy, params.trials)
    ):
        raise ValueError("Initial rollout manifest differs from its exact document subject")
    return member


def verified_imported_subject(
    store: ImmutableArtifactBlobProvider, subject: ContentRef, *, verify_bytes: bool = False,
) -> tuple[ImportedModelProvenance | None, tuple[ContentObjectRef, ...]]:
    if subject.media_type != IMPORTED_MODEL_MEDIA_TYPE:
        return None, ()
    source = read_document_input(subject, store)
    assert isinstance(source, ImportedModelSource)
    _, tree = read_imported_model(store, source, verify_bytes=verify_bytes)
    return imported_model_provenance(source, tree), (
        source.model, source.weights.descriptor, *tree.members.values(),
    )


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
) -> tuple[EvaluationRunManifest, EvaluationSubject]:
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
        or (manifest.loaded_model is not None) != (
            isinstance(subject, ContentRef) and subject.media_type == IMPORTED_MODEL_MEDIA_TYPE
        )
    ):
        raise ValueError("Evaluate result differs from its canonical request and declared inputs.")
    return manifest, subject


def verify_evaluation_receipt(
    root: Path, *, attempt: str, request_id: str, project: str, receipt: JobReceipt, binding: dict,
    manifest_object: ContentObjectRef,
) -> tuple[EvaluationRunManifest, EvaluationSubject]:
    """Authenticate the worker receipt against the exact Host dependency."""
    store = ImmutableArtifactBlobProvider(root)
    manifest, subject = decode_evaluation_manifest(
        store, store.get_object(manifest_object), manifest_object, attempt, request_id, receipt,
    )
    if isinstance(subject, ContentRef):
        if binding:
            raise ValueError("Initial Evaluate cannot have a Train binding")
        request = ContentObjectRef.model_validate(manifest.metadata["canonical_request"])
        refs = initial_evaluation_references(decode_request(store.get_object(request), _unexpected_document))
        for content in refs.values():
            store.verify_object(document_input_object(content))
        verify_initial_rollout(store, manifest, refs)
        return manifest, subject
    if "terminal" in binding:
        from feedbax.persistence.evaluation_subject import read_terminal_train_subject, terminal_train_provenance

        if not isinstance(subject, ProductRef) or not isinstance(subject.select, LastMember):
            raise ValueError("Store terminal subject requires an authored last checkpoint")
        pin = TerminalTrainSubject.model_validate(binding["terminal"])
        success, _ = read_terminal_train_subject(store, pin, subject.request_id)
        train, provenance = terminal_train_provenance(pin, success)
        if manifest.input_training_runs != train or manifest.provenance != provenance or "placement" in manifest.metadata:
            raise ValueError("Evaluate receipt differs from its exact Store subject provenance")
        return manifest, subject
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

    def replay(
        self, trial: str | None = None, *, checkpoint: int | None = None, aggregate: bool = False,
        mean: TrialMeanSelection | None = None,
    ) -> StudioTrialReplayResponse:
        """Read trial records at one exact preserved Evaluate checkpoint."""
        if mean is not None and trial is not None:
            raise CannotRun("/mean", "Mean requires all recorded trials")
        with self.open_states() as dataset:
            descriptor = dataset.descriptor
            update_axis = next((axis for axis in ("update", "checkpoint") if axis in descriptor.dimensions), None)
            selected = ()
            if "replicate" in descriptor.dimensions:
                raise CannotRun("/replicate", "choose a retained replicate for replay")
            if update_axis is not None:
                checkpoints = dataset.data(update_axis).tolist()
                if checkpoint is None and len(checkpoints) != 1:
                    raise CannotRun("/checkpoint", "choose a retained checkpoint for replay")
                if checkpoint is not None and checkpoint not in checkpoints:
                    raise CannotRun("/checkpoint", "the chosen checkpoint is absent from the retained set")
                selected = (0 if checkpoint is None else checkpoints.index(checkpoint),)
            elif checkpoint is not None:
                raise CannotRun("/checkpoint", "an initial model rollout has no checkpoint")
            trials = dataset.data("trial").tolist()
            if trial is not None and trial not in trials:
                raise CannotRun("/trial", "the selected trial is absent from the retained result")
            indices = range(len(trials)) if trial is None else (trials.index(trial),)
            if descriptor.dimensions["condition"] != 1:
                raise ValueError("The retained trial has more than one condition.")
            state_time = dataset.data("state_time")
            n_steps = len(state_time) - 1
            selection = (*selected, 0)
            raw_layout = descriptor.attrs.get("trial_record")
            if raw_layout is None:
                raise ValueError("The retained result has no trial record.")
            layout = RecordedTrialLayout.model_validate(raw_layout)
            position_key = next(iter(layout.positions), None)
            records = [
                read_trial_record(dataset, trials[index], (*selection, index), layout)
                for index in indices
            ]
            mean_options = layout.mean_options() if trials else None
            mean_result = None
            if mean is not None and mean_options is not None:
                from feedbax.execution.trial_clock import mean_trial_paths

                mean_result = mean_trial_paths(records, state_time, mean_options, mean)

            drawn: TrialRecords | TrialAggregate = TrialRecords(kind="trials", trials=records)
            if aggregate:
                from feedbax.execution.trial_record import aggregate_trials

                drawn = aggregate_trials(records, n_steps)
            return StudioTrialReplayResponse(
                n_steps=n_steps,
                state_time=state_time.tolist(),
                position=model_state_address(position_key).to_text() if position_key is not None else None,
                receipt=self.receipt,
                drawn=drawn,
                mean_options=mean_options,
                mean=mean_result,
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
