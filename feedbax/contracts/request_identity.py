"""Canonical request identity and stored-object codec for M2."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import cache
from importlib.resources import files
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_json

from feedbax.contracts._request_codec import decode, describe_models, encode
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.request_refs import (
    ContentRef,
    EffectiveBindingId,
    EffectiveInputs,
    InputMember,
    RequestId,
    RequestInputs,
    RequestKind,
)
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec,
    DocumentTrainingRequestSpec,
    EvaluationRequestSpec,
    RenderRequestSpec,
    ReportRequestSpec,
    RequestSpec,
    TrainingRequestSpec,
)

_IDENTITY = "feedbax.identity.request.v1"
_STORAGE_SCHEMA = "feedbax.request.v1"
_DOCUMENT_TRAIN_STORAGE_SCHEMA = "feedbax.request.train.documents.v1"
_EFFECTIVE_IDENTITY = "feedbax.identity.effective-binding.v1"
_INPUTS = TypeAdapter(RequestInputs)


@dataclass(frozen=True)
class RequestDescriptor:
    spec_type: type[RequestSpec]
    documents: tuple[str, ...] = ()
    storage_schema: str = _STORAGE_SCHEMA


REQUEST_KINDS: Mapping[RequestKind, RequestDescriptor] = {
    "train": RequestDescriptor(
        TrainingRequestSpec, ("graph", "task", "objective", "method_payload")
    ),
    "evaluate": RequestDescriptor(EvaluationRequestSpec),
    "analyze": RequestDescriptor(AnalysisRequestSpec),
    "render": RequestDescriptor(RenderRequestSpec),
    "report": RequestDescriptor(ReportRequestSpec),
}

DOCUMENT_TRAIN_REQUEST = RequestDescriptor(
    DocumentTrainingRequestSpec, ("method_payload",), _DOCUMENT_TRAIN_STORAGE_SCHEMA
)
_REQUEST_FORMATS = (*REQUEST_KINDS.items(), ("train", DOCUMENT_TRAIN_REQUEST))


def stored_request_descriptor(schema: str, kind: RequestKind) -> RequestDescriptor:
    for request_kind, descriptor in _REQUEST_FORMATS:
        if request_kind == kind and descriptor.storage_schema == schema:
            return descriptor
    raise ValueError("Unsupported stored request envelope")


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


def _same_document_value(left: Any, right: Any) -> bool:
    """Compare typed values before projection can erase distinctions such as bool/int."""
    if type(left) is not type(right):
        return False
    if left is None:
        return True
    if type(left) in (bool, int, float, str):
        return left == right
    if isinstance(left, Enum):
        return left is right
    if isinstance(left, BaseModel):
        return all(
            _same_document_value(getattr(left, name), getattr(right, name))
            for name in type(left).model_fields
        )
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            type(key) is str and _same_document_value(value, right[key])
            for key, value in left.items()
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            _same_document_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return False


@dataclass(frozen=True)
class _FactoredDocument:
    value: Any
    projection: Any
    digest: str
    data: bytes


class RequestDocumentMemo:
    """Retain the last verified document per field within one caller-verified codec authority."""

    def __init__(self) -> None:
        self._entries: dict[tuple[type[RequestSpec], str], _FactoredDocument] = {}

    def fork(self) -> RequestDocumentMemo:
        work = RequestDocumentMemo()
        work._entries.update(self._entries)
        return work

    def lookup(self, spec: RequestSpec, name: str) -> _FactoredDocument | None:
        entry = self._entries.get((type(spec), name))
        if entry is not None and _same_document_value(getattr(spec, name), entry.value):
            return entry
        return None

    def remember(
        self, spec: RequestSpec, name: str, projection: Any, digest: str, data: bytes
    ) -> None:
        self._entries[type(spec), name] = _FactoredDocument(
            deepcopy(getattr(spec, name)), projection, digest, data
        )


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
            for kind, descriptor in _REQUEST_FORMATS
        ),
        sha256_bytes(to_json(_identity_contract())),
    )


def generate_identity_fields() -> dict[str, Any]:
    roots = [InputMember]
    for _, descriptor in _REQUEST_FORMATS:
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


def encode_request(
    spec: RequestSpec, inputs: RequestInputs, *, document_memo: RequestDocumentMemo | None = None
) -> EncodedRequest:
    """Encode a typed, validated author request and compute its identity."""
    match = next(
        ((kind, descriptor) for kind, descriptor in _REQUEST_FORMATS
         if descriptor.spec_type is type(spec)),
        None,
    )
    if match is None:
        raise ValueError(f"Unknown request spec model: {type(spec).__name__}")
    kind, descriptor = match
    inputs = _validated_inputs(inputs)
    fields = _identity_contract()["fields"]
    reused = {}
    if document_memo is not None:
        reused = {
            name: entry
            for name in descriptor.documents
            if (entry := document_memo.lookup(spec, name)) is not None
        }
    projection = encode(
        spec, descriptor.spec_type, fields,
        projected_fields={name: entry.projection for name, entry in reused.items()},
    )
    documents = {}
    for name in descriptor.documents:
        if name in reused:
            digest, data = reused[name].digest, reused[name].data
        else:
            data = canonical_json_v2_bytes(projection[name])
            digest = sha256_bytes(data)
            if document_memo is not None:
                document_memo.remember(spec, name, projection[name], digest, data)
        projection[name] = digest
        documents[digest] = data

    envelope = {
        "schema": descriptor.storage_schema,
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


def _read_object(data: bytes, *, nullable: bool = False) -> dict | None:
    value = json.loads(data)
    if value is None and nullable and data == b"null":
        return None
    if not isinstance(value, dict) or canonical_json_v2_bytes(value) != data:
        raise ValueError("Expected a canonical request object")
    return value


def decode_request(data: bytes, read_document: Callable[[str], bytes]) -> DecodedRequest:
    """Verify stored bytes and fill pinned values before validating the request."""
    payload = _read_object(data)
    if payload.keys() != {"schema", "kind", "spec", "inputs"}:
        raise ValueError("Unsupported stored request envelope")
    kind = payload["kind"]
    if (
        not isinstance(kind, str)
        or kind not in REQUEST_KINDS
        or not isinstance(payload["spec"], dict)
    ):
        raise ValueError("Unknown request kind or invalid stored request spec")
    descriptor = stored_request_descriptor(payload["schema"], kind)
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
        spec_payload[name] = _read_object(document, nullable=kind == "train" and name != "method_payload")
    restored = decode(spec_payload, descriptor.spec_type, fields)
    spec = descriptor.spec_type.model_validate_json(json.dumps(restored))
    encoded = encode_request(spec, inputs)
    if encoded.data != data:
        raise ValueError("Stored request object is not minimal or did not round-trip")
    return DecodedRequest(kind=kind, spec=spec, inputs=inputs, id=encoded.id)


def effective_binding_id(request: DecodedRequest, inputs: EffectiveInputs) -> EffectiveBindingId:
    """Project verified bindings to a separately scoped successful-result lookup key."""
    if inputs.keys() != request.inputs.keys():
        raise ValueError("Effective inputs must cover the authored input roles")
    for role, members in inputs.items():
        authored = request.inputs[role]
        if tuple(member.key for member in members) != tuple(member.key for member in authored):
            raise ValueError(f"Effective inputs must preserve authored member order for {role!r}")
        for bound, source in zip(members, authored, strict=True):
            if isinstance(source.reference, ContentRef) and source.reference != bound.content:
                raise ValueError(f"Effective input differs from frozen content for {role}.{bound.key}")
    encoded = encode_request(request.spec, request.inputs)
    if encoded.id != request.id:
        raise ValueError("Effective binding requires an authenticated authored request")
    normalized = json.loads(encoded.data)
    preimage = {
        "identity": _EFFECTIVE_IDENTITY,
        "kind": normalized["kind"],
        "spec": normalized["spec"],
        "inputs": {
            role: [
                {
                    "key": member.key,
                    "content": member.content.model_dump(mode="json"),
                    "descriptor": member.descriptor,
                }
                for member in members
            ]
            for role, members in inputs.items()
        },
    }
    return f"effective:{sha256_bytes(canonical_json_v2_bytes(preimage))}"
