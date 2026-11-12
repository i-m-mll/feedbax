"""Portable review files for the authored request portion of a Project Plan."""

from collections.abc import Iterable
import json
from typing import Literal

from pydantic import JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.project_plan import ElementKey, PlanRequest, PlanRequests, RunKey
from feedbax.contracts.request_identity import REQUEST_KINDS, decode_request
from feedbax.contracts.request_refs import ProductRef, RequestId, Sha256
from feedbax.contracts.strict_json import strict_json_loads


class _PlanFile(StrictModel):
    schema_version: Literal["feedbax.project-plan.v1"]
    variants: tuple[str, ...] | None = None
    entries: list[PlanRequest]
    requests: dict[RequestId, dict[str, JsonValue]]
    documents: dict[Sha256, dict[str, JsonValue]]


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
    )


def dump_plan_requests(plan: PlanRequests, variants: tuple[str, ...] | None) -> bytes:
    """Write a deterministic JSON review file with one copy of each stored object."""
    scoped = select_plan_requests(plan, variants)
    requests = {
        identity: json.loads(scoped.requests[identity])
        for identity in sorted({entry.request_id for entry in scoped.entries.values()})
    }
    digests = {
        request["spec"][name]
        for request in requests.values()
        for name in REQUEST_KINDS[request["kind"]].documents
    }
    file = _PlanFile(
        schema_version="feedbax.project-plan.v1",
        variants=variants,
        entries=list(scoped.entries.values()),
        requests=requests,
        documents={digest: json.loads(scoped.documents[digest]) for digest in sorted(digests)},
    )
    return (file.model_dump_json(indent=2, exclude_none=True) + "\n").encode()


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
    if {entry.request_id for entry in entries.values()} != file.requests.keys():
        raise ValueError("Saved request objects differ from the authored request addresses")
    requests = {identity: canonical_json_v2_bytes(obj) for identity, obj in file.requests.items()}
    documents = {digest: canonical_json_v2_bytes(obj) for digest, obj in file.documents.items()}
    decoded = {}
    for identity, raw in requests.items():
        try:
            request = decode_request(raw, documents.__getitem__)
        except KeyError as error:
            raise ValueError(f"Missing document {error.args[0]} for {identity}") from error
        if request.id != identity:
            raise ValueError(f"Saved request object does not match {identity}")
        decoded[identity] = request
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
    return PlanRequests(entries, requests, documents), file.variants
