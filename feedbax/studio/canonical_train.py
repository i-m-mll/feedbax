"""Authenticate the canonical Train request and resolve its numerical declaration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, field_validator, model_validator

from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.artifact_custody import ArtifactBlobCustodyError, ContentObjectRef
from feedbax.contracts.base import StrictModel, canonical_json_bytes
from feedbax.contracts.effective_success import RequestObject
from feedbax.contracts.model_values import model_owner_problem
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import LastMember, ProductRef
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.contracts.train_settings import normalize_train_method
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

if TYPE_CHECKING:
    from feedbax.compiler.trial_execution import ResolvedTrialExecution
    from feedbax.persistence.train_job import TrainJobEntry
    from feedbax.plugins.application import ApplicationRegistryBundle
    from feedbax.training.methods import RegisteredTrainingMethod
    from pydantic import BaseModel


class CanonicalTrainRequestBinding(StrictModel):
    request: ContentObjectRef
    documents: tuple[ContentObjectRef, ...]
    checkpoint: ProductRef | None = None


class PinnedTrainCheckpoint(StrictModel):
    producer_job_id: str
    receipt: dict[str, str]
    checkpoint: PublishedTrainingCheckpoint


class BoundStudioTrainingJobInput(StrictModel):
    """Internal Host payload and attempt sidecar, with canonical custody when admitted by Plan."""

    spec: DocumentTrainingRequestSpec
    project_id: str
    request_id: str
    manifest_root: Path
    resume: PublishedTrainingCheckpoint | None = None
    canonical_request: CanonicalTrainRequestBinding
    checkpoint_input: PinnedTrainCheckpoint | None = None
    checkpoint_catalog: tuple[int, ...] = Field(default=(), exclude_if=lambda value: not value)
    collected_resume: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")

    @field_validator("checkpoint_catalog", mode="before")
    @classmethod
    def _validate_catalog(cls, value):
        return finite_coordinates(value) if value else ()

    @model_validator(mode="after")
    def _one_resume_authority(self) -> BoundStudioTrainingJobInput:
        if self.collected_resume is not None and (self.resume is not None or self.checkpoint_input is not None):
            raise ValueError("Collected Train Resume requires the Store job entry as its recovery authority.")
        return self


def read_collected_train_job(launch: BoundStudioTrainingJobInput) -> TrainJobEntry:
    """Resolve the authenticated recovery entry for this canonical Train launch."""
    from feedbax.persistence.train_job import read_train_job

    if launch.collected_resume is None:
        raise ValueError("Train launch does not name a collected attempt.")
    entry = read_train_job(ImmutableArtifactBlobProvider(launch.manifest_root), launch.collected_resume)
    if entry.request != RequestObject(
        id=launch.request_id,
        object=launch.canonical_request.request,
        documents=launch.canonical_request.documents,
    ):
        raise ValueError("Collected Train request differs from its original binding.")
    return entry


def _decode_train(
    request_id: str, data: bytes, read_document: Callable[[str], bytes]
) -> tuple[DocumentTrainingRequestSpec, ProductRef | None]:
    decoded = decode_request(data, read_document)
    if decoded.id != request_id or not isinstance(
        decoded.spec, DocumentTrainingRequestSpec
    ):
        raise ValueError("Training launch does not match its canonical Train request.")
    checkpoint = None
    if decoded.inputs:
        members = decoded.inputs.get("checkpoint", ())
        if decoded.inputs.keys() != {"checkpoint"} or len(members) != 1:
            raise ValueError("/inputs/checkpoint: expected one predecessor checkpoint")
        reference = members[0].reference
        if (
            not isinstance(reference, ProductRef)
            or reference.role != "checkpoint"
            or not reference.request_id.startswith("train:")
            or not isinstance(reference.select, LastMember)
        ):
            raise ValueError("/inputs/checkpoint: expected the last checkpoint of a Train request")
        checkpoint = reference
    if (checkpoint is None) != (decoded.spec.checkpoint_initialization is None):
        raise ValueError("/inputs/checkpoint: a predecessor and a checkpoint initialization policy must be declared together")
    return decoded.spec, checkpoint


def _resolve_trial(
    request: DocumentTrainingRequestSpec, registries: ApplicationRegistryBundle,
) -> ResolvedTrialExecution:
    from feedbax.compiler.trial_execution import resolve_trial_execution

    documents = request.trial_documents
    paired = resolve_trial_execution(
        documents.model_name, documents.model, documents.task_source,
        registries.components, registries.expression_functions,
        python_tasks=registries.python_tasks, objective=documents.objective,
        input_overrides=documents.input_overrides,
    )
    if isinstance(paired, tuple):
        problem = paired[0]
        raise ValueError("/trial_documents/" + "/".join(map(str, problem.path)) + ": " + problem.reason)
    if problem := model_owner_problem(paired.graph.graph, "checkpoint", registries.components):
        address = paired.lowered.source_address(pointer(problem.path))
        path = ("trial_documents", "model", *address.stored_path()) if address is not None else problem.path
        raise ValueError(f"{pointer(path)}: {problem.reason}")
    return paired


def bind_canonical_train(
    request_id: str,
    data: bytes,
    documents: Mapping[str, bytes],
    store: ImmutableArtifactBlobProvider,
    *, registries: ApplicationRegistryBundle,
) -> tuple[DocumentTrainingRequestSpec, CanonicalTrainRequestBinding]:
    """Validate a compiler request and retain its exact factored objects in the project store."""
    closure: dict[str, bytes] = {}

    def read_document(digest: str) -> bytes:
        raw = documents[digest]
        closure[digest] = raw
        return raw

    request, checkpoint = _decode_train(request_id, data, read_document)
    spec, _, _ = normalize_train_method(request, registries.training_methods)
    if spec != request:
        raise ValueError("Train request must carry the resolved method key and normalized settings.")
    _resolve_trial(spec, registries)
    refs = tuple(
        store.store_rooted_object(raw, namespace="studio-train-document")
        for _, raw in sorted(closure.items())
    )
    return spec, CanonicalTrainRequestBinding(
        request=store.store_rooted_object(data, namespace="studio-train-request"),
        documents=refs,
        checkpoint=checkpoint,
    )


@dataclass(frozen=True)
class VerifiedTrainBinding:
    files: tuple[tuple[Path, ContentObjectRef], ...]
    trial: ResolvedTrialExecution
    method: RegisteredTrainingMethod
    settings: BaseModel


def verify_training_request_binding(
    launch: BoundStudioTrainingJobInput,
) -> tuple[tuple[Path, ContentObjectRef], ...]:
    """Verify the exact request and document closure independently of numerical code."""
    binding = launch.canonical_request
    store = ImmutableArtifactBlobProvider(launch.manifest_root)
    documents = {ref.sha256: ref for ref in binding.documents}
    if len(documents) != len(binding.documents):
        raise ValueError("Canonical Train binding repeats a document object.")
    used: set[str] = set()

    def read_object(ref: ContentObjectRef) -> bytes:
        try:
            return store.get_object(ref)
        except (ArtifactBlobCustodyError, OSError) as error:
            raise ValueError(
                f"Cannot authenticate canonical Train object {ref.object_id}: {error}"
            ) from error

    def read_document(digest: str) -> bytes:
        if digest not in documents:
            raise ValueError(f"Canonical Train binding is missing document object {digest}.")
        used.add(digest)
        return read_object(documents[digest])

    request, checkpoint = _decode_train(
        launch.request_id, read_object(binding.request), read_document
    )
    if checkpoint != binding.checkpoint:
        raise ValueError("Canonical Train binding does not match its checkpoint input.")
    if used != documents.keys():
        raise ValueError("Canonical Train binding does not name its exact document closure.")
    if request != launch.spec:
        raise ValueError("Canonical Train request does not match its admitted execution settings.")
    return tuple((store.object_path(ref), ref) for ref in (binding.request, *binding.documents))


def resolve_training_request_binding(
    launch: BoundStudioTrainingJobInput, *, registries: ApplicationRegistryBundle,
) -> VerifiedTrainBinding:
    files = verify_training_request_binding(launch)
    expected, method, settings = normalize_train_method(launch.spec, registries.training_methods)
    if canonical_json_bytes(expected.model_dump(mode="json")) != canonical_json_bytes(
        launch.spec.model_dump(mode="json")
    ):
        raise ValueError("Installed Train method key or settings differ from the bound request.")
    return VerifiedTrainBinding(files, _resolve_trial(expected, registries), method, settings)
