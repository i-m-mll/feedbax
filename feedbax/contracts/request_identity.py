"""Canonical request identity and stored-object codec for M2."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import to_json

from feedbax.contracts._request_codec import decode, describe_models, encode
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.request_refs import (
    ContentRef,
    InputMember,
    RequestId,
    RequestInputs,
    RequestKind,
)
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec,
    EvaluationRequestSpec,
    RenderRequestSpec,
    ReportRequestSpec,
    RequestSpec,
    TrainingRequestSpec,
)

_IDENTITY = "feedbax.identity.request.v1"
_STORAGE_SCHEMA = "feedbax.request.v1"
_INPUTS = TypeAdapter(RequestInputs)


@dataclass(frozen=True)
class RequestDescriptor:
    spec_type: type[RequestSpec]
    documents: tuple[str, ...] = ()


REQUEST_KINDS: Mapping[RequestKind, RequestDescriptor] = {
    "train": RequestDescriptor(
        TrainingRequestSpec, ("graph", "task", "objective", "method_payload")
    ),
    "evaluate": RequestDescriptor(EvaluationRequestSpec),
    "analyze": RequestDescriptor(AnalysisRequestSpec),
    "render": RequestDescriptor(RenderRequestSpec),
    "report": RequestDescriptor(ReportRequestSpec),
}
@dataclass(frozen=True)
class EncodedRequest:
    """A request ID, its stored bytes, and the named bytes it references."""

    id: RequestId
    data: bytes
    documents: Mapping[str, bytes]


@dataclass(frozen=True)
class DecodedRequest:
    kind: RequestKind
    spec: RequestSpec
    inputs: RequestInputs
    id: RequestId


@cache
def _identity_contract() -> dict[str, Any]:
    data = files("feedbax.contracts").joinpath("request_identity_contract.json").read_bytes()
    return json.loads(data)


def identity_contract() -> dict[str, Any]:
    """Return a copy of the checked-in field and retirement table."""
    return deepcopy(_identity_contract())


def request_codec_authority() -> tuple[object, ...]:
    """Identify the loaded request descriptors, validators and identity pins."""
    return (
        encode_request,
        encode,
        tuple(
            (kind, descriptor, descriptor.spec_type.__pydantic_validator__)
            for kind, descriptor in REQUEST_KINDS.items()
        ),
        sha256_bytes(to_json(_identity_contract())),
    )


def generate_identity_fields() -> dict[str, Any]:
    roots = [InputMember]
    for descriptor in REQUEST_KINDS.values():
        for name in descriptor.documents:
            if not descriptor.spec_type.model_fields[name].is_required():
                raise ValueError(
                    f"Factored request document must be present: {descriptor.spec_type.__name__}.{name}"
                )
        roots.append(descriptor.spec_type)
    return describe_models(roots)


def _validated_inputs(inputs: RequestInputs) -> RequestInputs:
    groups = _INPUTS.validate_python(inputs)
    for role, members in groups.items():
        if not members:
            raise ValueError(f"Input role {role!r} must have at least one member")
        if len({member.key for member in members}) != len(members):
            raise ValueError(f"Duplicate input key in role {role!r}")
    return groups


def request_id(kind: RequestKind, spec: dict[str, Any], inputs: RequestInputs) -> RequestId:
    """Hash an already normalized spec projection and the resolved input references."""
    if kind not in REQUEST_KINDS:
        raise ValueError(f"Unknown request kind: {kind!r}")
    fields = _identity_contract()["fields"]
    references = {
        role: [
            {
                "key": member.key,
                "reference": (
                    (
                        {"sha256": member.reference.sha256, "tree": True}
                        if member.reference.tree
                        else {"sha256": member.reference.sha256}
                    )
                    if isinstance(member.reference, ContentRef)
                    else encode(member.reference, type(member.reference), fields)
                ),
            }
            for member in members
        ]
        for role, members in _validated_inputs(inputs).items()
    }
    preimage = {"identity": _IDENTITY, "kind": kind, "spec": spec, "inputs": references}
    return kind + ":" + sha256_bytes(canonical_json_v2_bytes(preimage))


def encode_request(spec: RequestSpec, inputs: RequestInputs) -> EncodedRequest:
    """Encode a typed, validated author request and compute its identity."""
    kind = next(
        (kind for kind, descriptor in REQUEST_KINDS.items() if descriptor.spec_type is type(spec)),
        None,
    )
    if kind is None:
        raise ValueError(f"Unknown request spec model: {type(spec).__name__}")
    descriptor = REQUEST_KINDS[kind]
    inputs = _validated_inputs(inputs)
    fields = _identity_contract()["fields"]
    projection = encode(spec, descriptor.spec_type, fields)
    documents = {}
    for name in descriptor.documents:
        data = canonical_json_v2_bytes(projection[name])
        digest = sha256_bytes(data)
        projection[name] = digest
        documents[digest] = data

    envelope = {
        "schema": _STORAGE_SCHEMA,
        "kind": kind,
        "spec": projection,
        "inputs": {
            role: [encode(member, InputMember, fields) for member in members]
            for role, members in inputs.items()
        },
    }
    return EncodedRequest(
        id=request_id(kind, projection, inputs),
        data=canonical_json_v2_bytes(envelope),
        documents=documents,
    )


def _read_object(data: bytes) -> dict:
    value = json.loads(data)
    if not isinstance(value, dict) or canonical_json_v2_bytes(value) != data:
        raise ValueError("Expected a canonical request object")
    return value


def decode_request(data: bytes, read_document: Callable[[str], bytes]) -> DecodedRequest:
    """Verify stored bytes and fill pinned values before validating the request."""
    payload = _read_object(data)
    if (
        payload.keys() != {"schema", "kind", "spec", "inputs"}
        or payload["schema"] != _STORAGE_SCHEMA
    ):
        raise ValueError("Unsupported stored request envelope")
    kind = payload["kind"]
    if (
        not isinstance(kind, str)
        or kind not in REQUEST_KINDS
        or not isinstance(payload["spec"], dict)
    ):
        raise ValueError("Unknown request kind or invalid stored request spec")
    descriptor = REQUEST_KINDS[kind]
    spec_payload = dict(payload["spec"])
    fields = _identity_contract()["fields"]
    input_groups = payload["inputs"]
    if not isinstance(input_groups, dict) or not all(
        isinstance(group, list) for group in input_groups.values()
    ):
        raise ValueError("Invalid stored request input groups")
    inputs = _validated_inputs(
        {
            role: tuple(decode(member, InputMember, fields) for member in group)
            for role, group in input_groups.items()
        }
    )
    for name in descriptor.documents:
        digest = spec_payload.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"Invalid request document digest for {name}")
        document = read_document(digest)
        if sha256_bytes(document) != digest:
            raise ValueError(f"Invalid request document bytes for {name}")
        spec_payload[name] = _read_object(document)
    restored = decode(spec_payload, descriptor.spec_type, fields)
    spec = descriptor.spec_type.model_validate_json(json.dumps(restored))
    encoded = encode_request(spec, inputs)
    if encoded.data != data:
        raise ValueError("Stored request object is not minimal or did not round-trip")
    return DecodedRequest(kind=kind, spec=spec, inputs=inputs, id=encoded.id)
