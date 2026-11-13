"""Verify a committed result closure or the declared Store in a read-only snapshot."""

from datetime import UTC, datetime

from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentObjectRef
from feedbax.contracts.effective_success import INDEX_REF, decode_success_index
from feedbax.contracts.store_check import StoreCheckFinding, StoreCheckResult
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.persistence.array_container import open_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.content_retention import _inventory_stats, _target_objects
from feedbax.persistence.reachability import store_roots
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.receipt_content import ClosureVisitors


class _Check:
    def __init__(self, store: ImmutableArtifactBlobProvider) -> None:
        self.store = store
        self.objects: dict[str, tuple[ContentObjectRef, bool]] = {}
        self.arrays: set[tuple[str, str]] = set()
        self.findings: list[StoreCheckFinding] = []
        self.visitors = ClosureVisitors(object=self.verify, array=self.verify_array)

    def object_problem(self, ref: ContentObjectRef, error: Exception) -> None:
        self.findings.append(StoreCheckFinding(
            code="damaged_object", address=ref.object_id, path=str(self.store.object_path(ref)),
            detail=str(error),
        ))

    def verify(self, ref: ContentObjectRef) -> None:
        if previous := self.objects.get(ref.sha256):
            if previous[0] != ref:
                self.object_problem(ref, ValueError(f"conflicting size for {ref.object_id}"))
            return
        try:
            self.store.verify_object(ref)
        except (ValueError, OSError) as error:
            self.object_problem(ref, error)
            self.objects[ref.sha256] = ref, False
            return
        self.objects[ref.sha256] = ref, True

    def verify_array(self, ref: ContentObjectRef, descriptor: str) -> None:
        binding = (ref.sha256, descriptor)
        self.verify(ref)
        if binding in self.arrays or self.objects[ref.sha256] != (ref, True):
            return
        self.arrays.add(binding)
        try:
            with (
                self.store.open_owned_object(ref) as path,
                open_array_container(path, expected_descriptor_sha256=descriptor) as reader,
            ):
                reader.verify_tensors()
        except (ValueError, OSError) as error:
            self.object_problem(ref, error)
            self.objects[ref.sha256] = ref, False

    def incomplete(self, address: str, detail: str) -> None:
        self.findings.append(StoreCheckFinding(
            code="check_incomplete", address=address, path=str(self.store.root), detail=detail,
        ))

    def receipt(self, attempt: str) -> ContentObjectRef:
        index = decode_success_index(self.store.read_ref(INDEX_REF))
        if (ref := index.attempts.get(attempt)) is None:
            raise ValueError(f"No committed receipt for attempt {attempt!r}")
        self.verify(ref)
        if not self.objects[ref.sha256][1]:
            return ref
        closure = ReceiptClosure(self.store, index)
        try:
            closure.receipt_for_attempt(attempt)
            selection = StoreClosureRef(roots=(ReceiptRoot(receipt=ref),), include_ancestry=True)
            closure.walk(selection, visitors=self.visitors)
        except (ValueError, OSError, KeyError) as error:
            self.incomplete(ref.object_id, str(error))
        return ref

    def inventory(self) -> None:
        inventory, errors = _inventory_stats(self.store)
        for digest, info in inventory.items():
            self.verify(ContentObjectRef(
                object_id=CONTENT_OBJECT_REF_PREFIX + digest, sha256=digest, size_bytes=info.st_size,
            ))
        # Retention blockers do not imply damage to known objects or references.
        for root in store_roots(
            self.store, datetime.now(UTC), errors, retention_gaps=[], include_history=True,
            visitors=self.visitors,
        ):
            for target in root.targets:
                try:
                    if target.tree is not None:
                        self.verify(target.tree.descriptor)
                    if target.manifest is not None:
                        self.verify(target.manifest.object)
                    for ref in _target_objects(self.store, target):
                        self.verify(ref)
                except (ValueError, OSError, KeyError) as error:
                    self.incomplete(root.name, str(error))
        for error in errors:
            self.incomplete(str(self.store.root), error)


def check_store(store: ImmutableArtifactBlobProvider, *, attempt: str | None = None) -> StoreCheckResult:
    """Check exact Store bytes without changing refs, receipts, or object contents."""
    check = _Check(store)
    with store._exclusive_lock(create=False):
        if attempt is None:
            check.inventory()
            receipt = None
        else:
            receipt = check.receipt(attempt)
    verified = [ref for ref, valid in check.objects.values() if valid]
    return StoreCheckResult(
        store=str(store.root), attempt=attempt, receipt=receipt,
        verified_objects=len(verified), verified_bytes=sum(ref.size_bytes for ref in verified),
        findings=tuple(check.findings),
    )
