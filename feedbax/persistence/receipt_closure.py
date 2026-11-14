"""Resolve exact typed Store roots to the objects they require."""

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.effective_success import (
    BoundMember,
    Success,
    SuccessIndex,
    read_committed_success,
    read_effective_request,
    read_success_object,
    select_bound_product,
    select_success_product,
)
from feedbax.contracts.request_identity import DecodedRequest
from feedbax.contracts.request_refs import ContentRef, ProductRef
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.evaluation_subject import evaluation_subject_objects, verify_evaluation_success_evidence
from feedbax.persistence.receipt_content import ClosureVisitors, continuation_objects, success_content_objects, tree_objects


class ReceiptClosure:
    """Walk frozen receipt selections in one locked Store snapshot."""

    def __init__(
        self, store: ImmutableArtifactBlobProvider, index: SuccessIndex | None = None
    ) -> None:
        self.store = store
        self.index = index
        self._receipts: dict[str, tuple[ContentObjectRef, Success]] = {}
        self._requests: dict[str, DecodedRequest] = {}

    def receipt_at(self, ref: ContentObjectRef) -> Success:
        cached = self._receipts.get(ref.sha256)
        if cached is None:
            read = self.store.get_object
            receipt = (
                read_success_object(ref, read)
                if self.index is None
                else read_committed_success(self.index, ref, read)
            )
            self._receipts[ref.sha256] = ref, receipt
            return receipt
        if cached[0] != ref:
            raise ValueError(f"Conflicting receipt evidence: {ref.sha256}")
        return cached[1]

    def receipt_for_attempt(self, attempt: str) -> tuple[ContentObjectRef, Success]:
        if self.index is None or (ref := self.index.attempts.get(attempt)) is None:
            raise ValueError(f"No committed receipt for attempt {attempt!r}")
        receipt = self.receipt_at(ref)
        if receipt.attempt != attempt:
            raise ValueError(f"Receipt differs from the indexed attempt {attempt!r}")
        return ref, receipt

    def _request_at(self, ref: ContentObjectRef, receipt: Success) -> DecodedRequest:
        if ref.sha256 not in self._requests:
            self._requests[ref.sha256] = read_effective_request(receipt, self.store.get_object)
        return self._requests[ref.sha256]

    def owned_objects(self, ref: ContentObjectRef) -> tuple[ContentObjectRef, ...]:
        """Resolve evidence owned by the result, independent of previously read inputs."""
        receipt = self.receipt_at(ref)
        objects = [ref, receipt.request.object, *receipt.request.documents]
        if receipt.evaluation_manifest is not None:
            objects.append(receipt.evaluation_manifest)
        for group in receipt.products.values():
            for product in group:
                objects.extend(success_content_objects(self.store, product))
        if receipt.continuation is not None:
            objects.extend(continuation_objects(receipt.continuation))
        return tuple(objects)

    def walk(
        self, closure: StoreClosureRef, *, visitors: ClosureVisitors | None = None,
    ) -> tuple[ContentObjectRef, ...]:
        """Return the closure and optionally visit exact objects and array anchors."""
        objects: dict[str, ContentObjectRef] = {}
        pending: list[tuple[ReceiptRoot, tuple[DecodedRequest, BoundMember] | None]] = []
        visited: set[tuple[str, str | None, str | None, bytes | None]] = set()

        def add(ref: ContentObjectRef) -> None:
            if ref.sha256 not in objects and visitors is not None:
                visitors.object(ref)
            if objects.setdefault(ref.sha256, ref) != ref:
                raise ValueError(f"Conflicting object evidence: {ref.sha256}")

        for root in closure.roots:
            if isinstance(root, ReceiptRoot):
                pending.append((root, None))
            elif isinstance(root, ContentObjectRef):
                add(root)
            else:
                for ref in tree_objects(self.store, root, visitors=visitors):
                    add(ref)

        while pending:
            root, consumer = pending.pop()
            add(root.receipt)
            receipt = self.receipt_at(root.receipt)
            if root.product is None:
                key = root.receipt.sha256, None, None, None
                products = tuple(
                    product for group in receipt.products.values() for product in group
                )
            else:
                if consumer is None:
                    product = select_success_product(root.product, receipt)
                    projection = None
                else:
                    request, binding = consumer
                    product, occurrence = select_bound_product(request, root.product, receipt)
                    if (
                        binding.selection is None
                        or product.name != binding.selection.member
                        or occurrence != binding.selection.occurrence
                        or product.content != binding.content
                        or product.descriptor != binding.descriptor
                    ):
                        raise ValueError(f"Receipt input provenance differs for {binding.key!r}")
                    projection = product.model_dump_json().encode()
                key = root.receipt.sha256, root.product.role, product.name, projection
                products = (product,)
            if key in visited:
                continue
            visited.add(key)
            for ref in (receipt.request.object, *receipt.request.documents):
                add(ref)
            decoded = self._request_at(root.receipt, receipt)
            if receipt.evaluation_manifest is not None:
                add(receipt.evaluation_manifest)
                for ref in verify_evaluation_success_evidence(self.store, receipt, visitors=visitors):
                    add(ref)
            for product in products:
                for ref in success_content_objects(self.store, product, visitors=visitors):
                    add(ref)
            if root.product is None and receipt.continuation is not None:
                for ref in continuation_objects(receipt.continuation, visitors=visitors):
                    add(ref)

            for role, members in decoded.inputs.items():
                for member, bound in zip(members, receipt.inputs[role], strict=True):
                    reference = member.reference
                    selection = bound.selection
                    if bound.evaluation_subject is not None:
                        if closure.include_ancestry:
                            for ref in evaluation_subject_objects(
                                self.store, decoded, member, bound, visitors=visitors,
                            ):
                                add(ref)
                    elif isinstance(reference, ProductRef):
                        if selection is None:
                            raise ValueError(f"Missing producer selection for {role}.{bound.key}")
                        if closure.include_ancestry:
                            pending.append(
                                (ReceiptRoot(receipt=selection.receipt, product=reference), (decoded, bound))
                            )
                    elif not isinstance(reference, ContentRef) or selection is not None or bound.descriptor is not None:
                        raise ValueError(f"Unexpected producer evidence for {role}.{bound.key}")
                    elif closure.include_ancestry:
                        for ref in success_content_objects(self.store, reference):
                            add(ref)

        return tuple(objects[digest] for digest in sorted(objects))
