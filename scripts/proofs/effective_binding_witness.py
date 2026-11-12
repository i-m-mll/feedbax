"""Read exact retained manifests and exercise frozen and rule Store lookups."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_refs import (
    ContentRef,
    InputMember,
    LastMember,
    NamedMember,
    ProductRef,
)
from feedbax.contracts.request_specs import ReportRequestSpec
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from scripts.proofs.effective_binding_store import (
    Product,
    PublishedSuccess,
    RequestObject,
    SuccessView,
    store_request,
)


@dataclass(frozen=True)
class Witness:
    producer: RequestObject
    rule: RequestObject
    frozen: RequestObject
    original: PublishedSuccess


def document_product(store: ImmutableArtifactBlobProvider, name: str, data: bytes) -> Product:
    manifest = json.loads(data)
    ref = store.store_rooted_object(data, namespace="proof-effective-manifest")
    return Product(
        name=name,
        content=ContentRef(sha256=ref.sha256, size=ref.size_bytes, media_type="application/json"),
        descriptor={"kind": manifest["kind"]},
    )


def summarize(view: SuccessView, request: RequestObject) -> Product:
    _, inputs = view.bind(request)
    store = view.store
    rows = []
    for member in inputs["documents"]:
        content = member.content
        data = store.get_object(
            f"content-object://sha256/{content.sha256}", size_bytes=content.size
        )
        manifest = json.loads(data)
        rows.append({"key": member.key, "kind": manifest["kind"], "id": manifest["id"]})
    ref = store.store_rooted_object(
        canonical_json_v2_bytes(rows), namespace="proof-effective-summary"
    )
    return Product(
        name="summary",
        content=ContentRef(sha256=ref.sha256, size=ref.size_bytes, media_type="application/json"),
    )


def prepare(store: ImmutableArtifactBlobProvider, supplier: bytes, analysis: bytes) -> Witness:
    retained = (
        document_product(store, "supplier", supplier),
        document_product(store, "analysis", analysis),
    )
    producer = store_request(
        store,
        encode_request(
            ReportRequestSpec(report_type="proof.retain-manifests"),
            {
                "imports": tuple(
                    InputMember(key=item.name, reference=item.content) for item in retained
                )
            },
        ),
    )
    SuccessView(store).publish(producer, attempt="retain-01", products={"manifests": retained})
    synopsis = ReportRequestSpec(report_type="proof.ordered-manifest-synopsis")

    def consumer(selected: LastMember | NamedMember) -> RequestObject:
        return store_request(
            store,
            encode_request(
                synopsis,
                {
                    "documents": (
                        InputMember(
                            key="rank-1",
                            reference=ProductRef(
                                request_id=producer.id,
                                role="manifests",
                                select=selected,
                            ),
                        ),
                        InputMember(
                            key="rank-2",
                            reference=ProductRef(
                                request_id=producer.id,
                                role="manifests",
                                select=NamedMember(member="supplier"),
                            ),
                        ),
                    )
                },
            ),
        )

    rule = consumer(LastMember())
    frozen = consumer(NamedMember(member="analysis"))
    view = SuccessView(store)
    product = summarize(view, rule)
    original = view.publish(rule, attempt="synopsis-01", products={"report": (product,)})
    return Witness(producer, rule, frozen, original)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    for name in ("supplier", "analysis"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    supplier, analysis = args.supplier.read_bytes(), args.analysis.read_bytes()
    if (
        sha256_bytes(supplier) != args.supplier_sha256
        or sha256_bytes(analysis) != args.analysis_sha256
    ):
        raise ValueError("Retained manifest differs from its pinned digest")
    store = ImmutableArtifactBlobProvider(args.store)
    witness = prepare(store, supplier, analysis)
    view = SuccessView(ImmutableArtifactBlobProvider(args.store))
    if (
        witness.rule.id == witness.frozen.id
        or view.authored(witness.frozen.id) is not None
        or view.lookup(witness.rule) != witness.original
        or view.lookup(witness.frozen) != witness.original
    ):
        raise AssertionError("Frozen lookup did not preserve authored IDs and the original success")
    print(
        json.dumps(
            {
                "store_id": store.store_id,
                "rule": witness.rule.model_dump(mode="json"),
                "frozen": witness.frozen.model_dump(mode="json"),
                "receipt_object": witness.original.ref.model_dump(mode="json"),
                "receipt": witness.original.receipt.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
