"""Typed records for the bounded effective-binding Store publication."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from feedbax.contracts.artifact_custody import (
    CONTENT_OBJECT_REF_PREFIX,
    CONTENT_TREE_REF_PREFIX,
    ContentObjectRef,
    ContentTreeRef,
)
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.checkpoint_projection import CheckpointOccurrenceRef, ContinuationOccurrence
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE, SemanticModelEncoding
from feedbax.contracts.rollout import ResolvedRolloutBinding
from feedbax.contracts.request_identity import DecodedRequest, decode_request, effective_binding_id
from feedbax.contracts.request_refs import (
    ContentRef,
    EffectiveBindingId,
    EffectiveInputMember,
    LastMember,
    Name,
    NamedMember,
    ProductRef,
    ReferenceValue,
    RequestId,
    Sha256,
)
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload


INDEX_REF = "proof/effective-successes-v1"


class RequestObject(ReferenceValue):
    id: RequestId
    object: ContentObjectRef
    documents: tuple[ContentObjectRef, ...]


class Product(ReferenceValue):
    name: Name
    content: ContentRef
    descriptor: JsonValue = None


type Products = dict[Name, Annotated[tuple[Product, ...], Field(min_length=1)]]


class Selection(ReferenceValue):
    receipt: ContentObjectRef
    member: Name
    occurrence: CheckpointOccurrenceRef | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )


EVALUATION_SUBJECT_MEDIA_TYPE = "application/vnd.feedbax.evaluate-subject.v1+tree"


class EvaluationGap(ReferenceValue):
    update: int = Field(gt=0)
    reason: Literal["checkpoint_unavailable"]


class EvaluationSubjectValue(ReferenceValue):
    axis: Literal["checkpoint", "update"]
    requested: Annotated[tuple[int, ...], Field(min_length=1)]
    gaps: tuple[EvaluationGap, ...]
    resolved: ResolvedRolloutBinding | None

    @model_validator(mode="after")
    def _ordered_coverage(self) -> EvaluationSubjectValue:
        missing = tuple(gap.update for gap in self.gaps)
        absent = set(missing)
        observed = tuple(member.coordinate for member in self.resolved.members) if self.resolved else ()
        if (
            len(set(self.requested)) != len(self.requested) or any(update <= 0 for update in self.requested)
            or missing != tuple(update for update in self.requested if update in absent)
            or observed != tuple(update for update in self.requested if update not in absent)
            or (self.gaps and self.axis != "update")
        ):
            raise ValueError("Evaluate subject does not account for its exact ordered coverage")
        return self


class EvaluationCheckpointSource(ReferenceValue):
    occurrence: CheckpointOccurrenceRef
    transaction: ContentObjectRef
    train_binding: ContentObjectRef

    @model_validator(mode="after")
    def _exact_transaction(self) -> EvaluationCheckpointSource:
        if self.transaction.sha256 != self.occurrence.manifest_sha256:
            raise ValueError("Evaluate checkpoint occurrence differs from its retained transaction")
        return self


class EvaluationClosure(ReferenceValue):
    kind: Literal["completion", "recovery"]
    checkpoint: EvaluationCheckpointSource
    receipt: ContentObjectRef | None = None

    @model_validator(mode="after")
    def _completion_receipt(self) -> EvaluationClosure:
        if (self.kind == "completion") != (self.receipt is not None):
            raise ValueError("Evaluate coverage must distinguish completion and recovery evidence")
        return self


class BestCandidateEvidence(ReferenceValue):
    attempt: Name
    producer: RequestObject
    receipt: ContentObjectRef
    checkpoint: EvaluationCheckpointSource


class EvaluationSubjectSource(ReferenceValue):
    producer: RequestObject
    checkpoints: tuple[EvaluationCheckpointSource, ...]
    closure: EvaluationClosure | None = None
    terminal_receipt: ContentObjectRef | None = None
    ranked_candidates: tuple[BestCandidateEvidence, ...] = Field(default=(), exclude_if=lambda items: not items)


class BoundMember(EffectiveInputMember):
    selection: Selection | None = None
    evaluation_subject: EvaluationSubjectSource | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def _one_source(self) -> BoundMember:
        if self.evaluation_subject is not None and self.selection is not None:
            raise ValueError("Evaluate subject has conflicting source evidence")
        return self


type Bindings = dict[Name, tuple[BoundMember, ...]]


class Success(ReferenceValue):
    schema_version: Literal["feedbax.proof.effective-success.v1"] = (
        "feedbax.proof.effective-success.v1"
    )
    request: RequestObject
    attempt: Name
    effective: EffectiveBindingId
    inputs: Bindings
    products: Products = Field(min_length=1)
    supersedes: ContentObjectRef | None
    evaluation_manifest: ContentObjectRef | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    continuation: ContinuationOccurrence | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    recovery: CheckpointOccurrenceRef | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_products(self) -> Success:
        projected = [
            role for role, members in self.inputs.items() for member in members
            if member.evaluation_subject is not None
        ]
        if projected or self.evaluation_manifest is not None:
            if (
                self.evaluation_manifest is None or projected != ["subject"]
                or not self.request.id.startswith("evaluate:") or self.inputs.keys() != {"subject"}
                or len(self.inputs["subject"]) != 1 or set(self.products) != {"states"}
                or len(self.products["states"]) != 1
            ):
                raise ValueError("Evaluate success requires its accepted manifest, one subject and states product")
        if self.continuation is not None and (
            not self.request.id.startswith("train:")
            or len(self.products.get("checkpoint", ())) != 1
            or self.products["checkpoint"][0].name != "model"
            or self.products["checkpoint"][0].content.media_type != MODEL_TREE_MEDIA_TYPE
            or not self.products["checkpoint"][0].content.tree
        ):
            raise ValueError("Continuation evidence requires one Train checkpoint occurrence")
        if self.recovery is not None and not self.request.id.startswith("train:"):
            raise ValueError("Only Train successes can record checkpoint recovery")
        for role, members in self.products.items():
            if len({member.name for member in members}) != len(members):
                raise ValueError(f"Repeated product member in {role!r}")
            if role == "checkpoint" and self.request.id.startswith("train:"):
                for member in members:
                    if (
                        member.name == "model"
                        and member.content.tree
                        and member.content.media_type != MODEL_TREE_MEDIA_TYPE
                    ):
                        raise ValueError(
                            "Train checkpoint/model tree requires the semantic model media type"
                        )
        return self


class SuccessIndex(ReferenceValue):
    schema_version: Literal["feedbax.proof.success-index.v1"] = "feedbax.proof.success-index.v1"
    authored: dict[RequestId, ContentObjectRef] = Field(default_factory=dict)
    effective: dict[EffectiveBindingId, ContentObjectRef] = Field(default_factory=dict)
    attempts: dict[Name, ContentObjectRef] = Field(default_factory=dict)
    supersession_times: dict[Sha256, AwareDatetime] = Field(
        default_factory=dict,
        exclude_if=lambda times: not times,
    )

    @model_validator(mode="after")
    def _committed_times(self) -> SuccessIndex:
        if self.supersession_times.keys() - {ref.sha256 for ref in self.attempts.values()}:
            raise ValueError("Supersession time names an uncommitted successor")
        return self


def decode_success_index(value: dict[str, JsonValue] | None) -> SuccessIndex:
    if value is None:
        return SuccessIndex()
    index = SuccessIndex.model_validate(value)
    if index.model_dump(mode="json") != value:
        raise ValueError("Successful-result index has an incomplete or noncanonical schema")
    return index


def success_content_ref(content: ContentRef) -> ContentObjectRef | ContentTreeRef:
    """Project receipt content to its exact canonical Store reference."""
    object_ref = ContentObjectRef(
        object_id=CONTENT_OBJECT_REF_PREFIX + content.sha256,
        sha256=content.sha256,
        size_bytes=content.size,
    )
    if content.tree:
        return ContentTreeRef(
            tree_id=CONTENT_TREE_REF_PREFIX + content.sha256, descriptor=object_ref
        )
    return object_ref


def read_committed_success(
    index: SuccessIndex,
    ref: ContentObjectRef,
    read_object: Callable[[ContentObjectRef], bytes],
) -> Success:
    receipt = read_success_object(ref, read_object)
    if index.attempts.get(receipt.attempt) != ref:
        raise ValueError("Receipt has no attempt in this committed Store generation")
    return receipt


def read_success_object(
    ref: ContentObjectRef,
    read_object: Callable[[ContentObjectRef], bytes],
) -> Success:
    """Read the immutable receipt bytes at an already authorized exact ref."""
    raw = read_object(ref)
    receipt = Success.model_validate_json(raw)
    if raw != canonical_json_v2_bytes(receipt.model_dump(mode="json")):
        raise ValueError("Successful receipt bytes are not canonical")
    return receipt


def read_success_request(
    request: RequestObject,
    read_object: Callable[[ContentObjectRef], bytes],
) -> DecodedRequest:
    documents = {ref.sha256: ref for ref in request.documents}
    used = set()

    def read_document(digest: str) -> bytes:
        if digest not in documents:
            raise ValueError(f"Request document is unavailable: {digest}")
        used.add(digest)
        return read_object(documents[digest])

    decoded = decode_request(read_object(request.object), read_document)
    if decoded.id != request.id or used != documents.keys() or len(used) != len(request.documents):
        raise ValueError("Stored request differs from its authored ID or document closure")
    return decoded


def read_effective_request(
    receipt: Success, read_object: Callable[[ContentObjectRef], bytes]
) -> DecodedRequest:
    decoded = read_success_request(receipt.request, read_object)
    if effective_binding_id(decoded, receipt.inputs) != receipt.effective:
        raise ValueError("Receipt effective identity differs from its recorded bindings")
    return decoded


def select_success_product(reference: ProductRef, receipt: Success) -> Product:
    """Resolve exactly one authored role member from this immutable producer."""
    members = receipt.products.get(reference.role)
    if receipt.request.id != reference.request_id or members is None:
        raise ValueError(f"Receipt does not supply {reference.request_id}/{reference.role}")
    match reference.select:
        case None:
            if len(members) == 1:
                return members[0]
            raise ValueError(f"Product role {reference.role!r} requires a member selector")
        case LastMember():
            return members[-1]
        case NamedMember(member=name):
            for product in members:
                if product.name == name:
                    return product
            raise ValueError(f"Product role {reference.role!r} has no member {name!r}")
        case selector:
            raise ValueError(
                f"Unsupported product selector {selector.kind!r} for {reference.role!r}"
            )


def select_bound_product(
    request: DecodedRequest,
    reference: ProductRef,
    receipt: Success,
) -> tuple[Product, CheckpointOccurrenceRef | None]:
    """Project the selected occurrence according to the consuming request."""
    product = select_success_product(reference, receipt)
    spec = request.spec
    if (
        reference.role != "checkpoint"
        or not isinstance(spec, DocumentTrainingRequestSpec)
        or spec.method_ref != STUDIO_GRAPH_METHOD
    ):
        return product, None
    if not isinstance(spec.method_payload.payload.get("checkpoint_initialization"), dict):
        occurrence = receipt.continuation
        return product, occurrence.provenance if occurrence is not None and product.name == "model" else None
    policy = StudioGraphMethodPayload.model_validate(spec.method_payload.payload).continuation
    assert policy is not None
    occurrence = receipt.continuation
    if occurrence is None or product.name != "model":
        raise ValueError("Selected checkpoint has no typed continuation occurrence")
    if occurrence.clock.completed_batches != policy.progress.source_completed_batches:
        raise ValueError("Selected checkpoint differs from the authored continuation boundary")
    encoding = occurrence.project(SemanticModelEncoding.model_validate(product.descriptor), policy)
    return product.model_copy(update={"descriptor": encoding.model_dump(mode="json")}), occurrence.provenance
