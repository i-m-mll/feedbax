"""Request custody and dataset reads for the isolated saved Evaluate candidate."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.base import ParentRef, Provenance, StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE as STATES_MEDIA_TYPE, DatasetProductRef
from feedbax.contracts.graph import StudioManifestRef, StudioSelectorRef
from feedbax.contracts.jobs import (
    JobReceipt, StudioEvaluationCheckpoint, StudioEvaluationCheckpointGap,
    StudioEvaluationObjectivePoint, StudioEvaluationObjectiveSeries, StudioEvaluationObjectiveTerm,
)
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    load_manifest_bytes,
)
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import BatchMember, BatchSetMember, ContentRef, LastMember, ProductRef, UpdateSeriesMember
from feedbax.contracts.request_specs import EvaluationRequestSpec
from feedbax.contracts.rollout import (
    RAW_STATES_SCHEMA as RAW_STATES_SCHEMA,
    ResolvedRolloutBinding,
    ROLLOUT_TYPE as ROLLOUT_TYPE,
    STATES_SCHEMA as STATES_SCHEMA,
    RolloutParameters as RolloutParameters,
    authored_variation,
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
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
if TYPE_CHECKING:
    from feedbax.dataset import Dataset

class CannotRun(ValueError):
    """A capability required by the authored request is unavailable."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.reason = message
        super().__init__(f"{path}: {message}")


class EvaluationRequestBinding(StrictModel):
    request: ContentObjectRef
    checkpoint: ProductRef


class BoundEvaluationJobInput(StrictModel):
    project_id: str
    request_id: str
    manifest_root: Path
    canonical_request: EvaluationRequestBinding


def _unexpected_document(digest: str) -> bytes:
    raise CannotRun("/params", f"this candidate does not support a factored document {digest}")


def evaluation_checkpoint_coordinates(subject: ProductRef) -> tuple[int, ...] | None:
    if isinstance(subject.select, BatchMember):
        return finite_coordinates((subject.select.batch,))
    if isinstance(subject.select, UpdateSeriesMember):
        return subject.select.updates
    if isinstance(subject.select, BatchSetMember):
        return subject.select.batches
    return None


def decode_rollout_request(
    request_id: str, raw: bytes,
) -> tuple[EvaluationRequestSpec, RolloutParameters, ProductRef]:
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
    if (
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


def read_evaluation_manifest(
    root: Path, job_id: str, request_id: str, receipt: JobReceipt,
) -> tuple[EvaluationRunManifest, ProductRef]:
    store = ImmutableArtifactBlobProvider(root)
    ref, raw = store.resolve_manifest("EvaluationRunManifest", receipt.manifest_id)
    return decode_evaluation_manifest(store, raw, ref.object, job_id, request_id, receipt)


def decode_evaluation_manifest(
    store: ImmutableArtifactBlobProvider, raw: bytes, ref: ContentObjectRef,
    job_id: str, request_id: str, receipt: JobReceipt,
) -> tuple[EvaluationRunManifest, ProductRef]:
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
    manifest_object: ContentObjectRef | None = None,
) -> tuple[EvaluationRunManifest, ProductRef]:
    """Authenticate the worker receipt against the exact Host dependency."""
    if manifest_object is None:
        manifest, subject = read_evaluation_manifest(root, attempt, request_id, receipt)
    else:
        store = ImmutableArtifactBlobProvider(root)
        manifest, subject = decode_evaluation_manifest(
            store, store.get_object(manifest_object), manifest_object, attempt, request_id, receipt,
        )
    if (coordinates := evaluation_checkpoint_coordinates(subject)) is not None:
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
        member["request_id"] != subject.request_id
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
        *, manifest_object: ContentObjectRef | None = None,
    ):
        self._root = root
        self._job_id = job_id
        self._request_id = request_id
        self.receipt = receipt
        self._manifest_object = manifest_object

    def _manifest(self) -> EvaluationRunManifest:
        if self._manifest_object is None:
            manifest, _ = read_evaluation_manifest(self._root, self._job_id, self._request_id, self.receipt)
        else:
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
        with open_stored_dataset(store, ref, verify=verify or self._manifest_object is None) as dataset:
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
        objective_hash = sha256(canonical_json_v2_bytes(objective.model_dump(mode="json"))).hexdigest()
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
                    condition != canonical_sha256(params.model_dump(mode="json")["variation"])
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
