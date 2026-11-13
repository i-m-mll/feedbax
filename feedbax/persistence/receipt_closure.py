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
from feedbax.contracts.request_refs import ProductRef
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.receipt_content import continuation_objects, success_content_objects, tree_objects


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

    def _request_at(self, ref: ContentObjectRef, receipt: Success) -> DecodedRequest:
        if ref.sha256 not in self._requests:
            self._requests[ref.sha256] = read_effective_request(receipt, self.store.get_object)
        return self._requests[ref.sha256]

    def walk(self, closure: StoreClosureRef) -> tuple[ContentObjectRef, ...]:
        objects: dict[str, ContentObjectRef] = {}
        pending: list[tuple[ReceiptRoot, tuple[DecodedRequest, BoundMember] | None]] = []
        visited: set[tuple[str, str | None, str | None, bytes | None]] = set()

        def add(ref: ContentObjectRef) -> None:
            if objects.setdefault(ref.sha256, ref) != ref:
                raise ValueError(f"Conflicting object evidence: {ref.sha256}")

        for root in closure.roots:
            if isinstance(root, ReceiptRoot):
                pending.append((root, None))
            elif isinstance(root, ContentObjectRef):
                add(root)
            else:
                for ref in tree_objects(self.store, root):
                    add(ref)

        while pending:
            root, consumer = pending.pop()
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
            decoded = self._request_at(root.receipt, receipt)
            for ref in (root.receipt, receipt.request.object, *receipt.request.documents):
                add(ref)
            for product in products:
                for ref in success_content_objects(self.store, product):
                    add(ref)
            if root.product is None and receipt.continuation is not None:
                for ref in continuation_objects(receipt.continuation):
                    add(ref)

            for role, members in decoded.inputs.items():
                for member, bound in zip(members, receipt.inputs[role], strict=True):
                    reference = member.reference
                    selection = bound.selection
                    if isinstance(reference, ProductRef):
                        if selection is None:
                            raise ValueError(f"Missing producer selection for {role}.{bound.key}")
                        if closure.include_ancestry:
                            pending.append(
                                (ReceiptRoot(receipt=selection.receipt, product=reference), (decoded, bound))
                            )
                    elif selection is not None or bound.descriptor is not None:
                        raise ValueError(f"Unexpected producer evidence for {role}.{bound.key}")
                    elif closure.include_ancestry:
                        for ref in success_content_objects(self.store, reference):
                            add(ref)

        return tuple(objects[digest] for digest in sorted(objects))
