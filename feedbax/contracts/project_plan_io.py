"""Portable review files for the authored request portion of a Project Plan."""

from collections.abc import Iterable, Mapping
import json
from typing import Literal

from pydantic import Field, JsonValue

from feedbax.contracts._request_codec import same_typed_value
from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.project_plan import ElementKey, PlanRequest, PlanRequests, RunKey, SubsetSource
from feedbax.contracts.request_identity import DecodedRequest, decode_request, stored_request_descriptor
from feedbax.contracts.request_refs import BestRank, ProductRef, RequestId, RequestKind, Sha256
from feedbax.contracts.strict_json import strict_json_loads


_REVIEW_VERSION = "feedbax.project-plan.v2"
JsonObject = dict[str, JsonValue]


class _RequestGroup(StrictModel):
    storage_schema: str = Field(alias="schema")
    kind: RequestKind
    common: JsonObject
    requests: dict[RequestId, JsonObject]


class _SubsetSource(StrictModel):
    variant: str
    step: str
    role: str
    source: SubsetSource


class _PlanFile(StrictModel):
    schema_version: Literal["feedbax.project-plan.v2"]
    variants: tuple[str, ...] | None = None
    entries: list[PlanRequest]
    request_groups: list[_RequestGroup]
    documents: dict[Sha256, JsonObject | None]
    subset_sources: list[_SubsetSource] | None = None


def run_key(key: RunKey | ElementKey) -> RunKey:
    return key.run if isinstance(key, ElementKey) else key


def select_plan_requests(plan: PlanRequests, variants: Iterable[str] | None) -> PlanRequests:
    if variants is None:
        return plan
    names = frozenset(variants)
    return PlanRequests(
        {key: entry for key, entry in plan.entries.items() if run_key(key).variant in names},
        plan.requests,
        plan.documents,
        {key: source for key, source in plan.subset_sources.items() if key[0] in names},
    )


def _common_fields(left: JsonObject, right: JsonObject) -> JsonObject:
    common = {}
    for name, value in left.items():
        if name not in right:
            continue
        other = right[name]
        if same_typed_value(value, other):
            common[name] = value
        elif isinstance(value, dict) and isinstance(other, dict):
            shared = _common_fields(value, other)
            if shared:
                common[name] = shared
    return common


def _remaining_fields(obj: JsonObject, common: JsonObject) -> JsonObject:
    remaining = {}
    for name, value in obj.items():
        if name not in common:
            remaining[name] = value
        elif isinstance(value, dict) and isinstance(shared := common[name], dict):
            extra = _remaining_fields(value, shared)
            if extra:
                remaining[name] = extra
    return remaining


def _restore_fields(common: JsonObject, remaining: JsonObject) -> JsonObject:
    restored = common.copy()
    for name, value in remaining.items():
        if name not in common:
            restored[name] = value
        elif isinstance(value, dict) and isinstance(shared := common[name], dict):
            restored[name] = _restore_fields(shared, value)
        else:
            raise ValueError(f"Saved request repeats shared field {name!r}")
    return restored


def dump_plan_requests(plan: PlanRequests, variants: tuple[str, ...] | None) -> bytes:
    """Write authored addresses and canonical objects, factoring shared request fields."""
    scoped = select_plan_requests(plan, variants)
    grouped: dict[tuple[str, RequestKind], dict[RequestId, JsonObject]] = {}
    digests = set()
    for identity in sorted({entry.request_id for entry in scoped.entries.values()}):
        obj = json.loads(scoped.requests[identity])
        schema, kind = obj.pop("schema"), obj.pop("kind")
        grouped.setdefault((schema, kind), {})[identity] = obj
        descriptor = stored_request_descriptor(schema, kind)
        digests.update(obj["spec"][name] for name in descriptor.documents)

    groups = []
    for (schema, kind), objects in sorted(grouped.items()):
        rows = iter(objects.values())
        common = next(rows)
        for obj in rows:
            common = _common_fields(common, obj)
        groups.append(_RequestGroup(
            schema=schema,
            kind=kind,
            common=common,
            requests={identity: _remaining_fields(obj, common) for identity, obj in objects.items()},
        ))

    file = _PlanFile(
        schema_version=_REVIEW_VERSION,
        variants=variants,
        entries=list(scoped.entries.values()),
        request_groups=groups,
        documents={digest: json.loads(scoped.documents[digest]) for digest in sorted(digests)},
        subset_sources=[
            _SubsetSource(variant=variant, step=step, role=role, source=source)
            for (variant, step, role), source in sorted(scoped.subset_sources.items())
        ] or None,
    )
    return (file.model_dump_json(indent=1, exclude_none=True, by_alias=True) + "\n").encode()


def _restore_requests(groups: list[_RequestGroup]) -> dict[RequestId, bytes]:
    requests = {}
    seen = set()
    for group in groups:
        key = (group.storage_schema, group.kind)
        if key in seen or not group.requests:
            raise ValueError(f"Duplicate or empty saved request group {key!r}")
        seen.add(key)
        stored_request_descriptor(*key)
        for identity, remaining in group.requests.items():
            if identity in requests:
                raise ValueError(f"Duplicate saved request {identity}")
            restored = _restore_fields(group.common, remaining)
            if restored.keys() != {"spec", "inputs"}:
                raise ValueError(f"Invalid saved request fields for {identity}")
            requests[identity] = canonical_json_v2_bytes(
                {"schema": group.storage_schema, "kind": group.kind, **restored}
            )
    return requests


def load_plan_requests(data: bytes) -> tuple[PlanRequests, tuple[str, ...] | None]:
    """Read a review file and authenticate the identities before comparing them."""
    file = _PlanFile.model_validate(strict_json_loads(data))
    entries = {entry.key: entry for entry in file.entries}
    if len(entries) != len(file.entries):
        raise ValueError("Duplicate authored request address")
    if file.variants is not None and any(
        run_key(key).variant not in file.variants for key in entries
    ):
        raise ValueError("Request address is outside the saved variant selection")
    requests = _restore_requests(file.request_groups)
    if {entry.request_id for entry in entries.values()} != requests.keys():
        raise ValueError("Saved request objects differ from the authored request addresses")
    documents = {digest: canonical_json_v2_bytes(obj) for digest, obj in file.documents.items()}
    for digest, raw in documents.items():
        if sha256_bytes(raw) != digest:
            raise ValueError(f"Saved document does not match {digest}")
    used_documents = set()

    def read_document(digest: str) -> bytes:
        used_documents.add(digest)
        return documents[digest]

    decoded = {}
    for identity, raw in requests.items():
        try:
            request = decode_request(raw, read_document)
        except KeyError as error:
            raise ValueError(f"Missing document {error.args[0]} for {identity}") from error
        if request.id != identity:
            raise ValueError(f"Saved request object does not match {identity}")
        decoded[identity] = request
    if documents.keys() != used_documents:
        raise ValueError("Saved documents differ from the referenced request documents")
    for key, entry in entries.items():
        if isinstance(key, ElementKey) != (entry.kind in {"analyze", "render"}):
            raise ValueError(f"Invalid address for {entry.request_id}")
        for role, producer in entry.element_inputs.items():
            members = decoded[entry.request_id].inputs.get(role, ())
            if len(members) != 1 or not isinstance(members[0].reference, ProductRef):
                raise ValueError(f"Invalid element input {role!r} for {entry.request_id}")
            if (
                producer in entries
                and entries[producer].request_id != members[0].reference.request_id
            ):
                raise ValueError(
                    f"Element input {role!r} names the wrong producer for {entry.request_id}"
                )
    sources = {}
    for item in file.subset_sources or ():
        key = item.variant, item.step, item.role
        if key in sources:
            raise ValueError(f"Duplicate subset source {key!r}")
        sources[key] = item.source
    _validate_subset_sources(entries, decoded, sources, file.variants)
    return PlanRequests(entries, requests, documents, sources), file.variants


def _validate_subset_sources(
    entries: Mapping[RunKey | ElementKey, PlanRequest],
    decoded: Mapping[RequestId, DecodedRequest],
    sources: Mapping[tuple[str, str, str], SubsetSource],
    variants: tuple[str, ...] | None,
) -> None:
    if not sources:
        return
    by_step = {}
    for key, entry in entries.items():
        if isinstance(key, RunKey):
            by_step.setdefault((key.variant, key.step), []).append(entry)
    for (variant, step, role), source in sources.items():
        rows = {row.key: row for row in source.rows}
        if len(rows) != len(source.rows):
            raise ValueError(f"Duplicate subset member for {variant}/{step}/{role}")
        if role != "subject":
            raise ValueError(f"Invalid subset source role {role!r}")
        # Ordered static rows are in the source table. Request inputs keep rank keys,
        # but a frozen ProductRef alone does not identify its producer row.
        uses = by_step.get((variant, step), ())
        if not uses:
            raise ValueError(f"Subset source has no requests at {variant}/{step}")
        for entry in uses:
            members = decoded[entry.request_id].inputs.get(role, ())
            if rows:
                valid = all(
                    isinstance(member.reference, ProductRef)
                    and (row := rows.get(member.key)) is not None
                    and _matches_producer(entries, variants, source, row.variant, row.row, member.reference)
                    for member in members
                )
            else:
                valid = all(isinstance(member.reference, BestRank) and all(
                    _matches_producer(entries, variants, source, candidate.variant, candidate.row, candidate.reference)
                    for candidate in member.reference.candidates
                ) for member in members)
            if entry.kind != "evaluate" or not members or not valid:
                raise ValueError(f"Invalid subset source for {entry.request_id} input {role!r}")


def _matches_producer(
    entries: Mapping[RunKey | ElementKey, PlanRequest],
    variants: tuple[str, ...] | None, source: SubsetSource, variant: str, row: str, reference: ProductRef,
) -> bool:
    if variants is not None and variant not in variants:
        return True
    key = RunKey(variant, source.producer_step, row)
    producer = entries.get(key)
    return producer is not None and producer.request_id == reference.request_id
