"""Versioned Studio draft hashes shared by manifests and saved workspaces."""

from __future__ import annotations

from typing import Any, Final, Literal, Mapping, TypeAlias, TypedDict, cast

from feedbax.contracts.canonical_json import canonical_json_v2_bytes


STUDIO_DRAFT_HASH_SCHEMA_ID: Final = "feedbax.studio.draft_hashes"
STUDIO_DRAFT_HASH_SCHEMA_VERSION: Final = "feedbax.studio.draft_hashes.v2"
STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION: Final = "feedbax.studio.draft_hashes.v1"
STUDIO_DRAFT_HASH_PIN: Final = "fnv1a32-canonical_json_v2"
STUDIO_DRAFT_HASH_LEGACY_PIN: Final = "fnv1a32-runtime_local_json_v1"

StudioDraftHashSchemaVersion: TypeAlias = Literal[
    "feedbax.studio.draft_hashes.v1",
    "feedbax.studio.draft_hashes.v2",
]
StudioDraftHashPin: TypeAlias = Literal[
    "fnv1a32-runtime_local_json_v1",
    "fnv1a32-canonical_json_v2",
]


class StudioDraftHashes(TypedDict):
    """Current cross-language draft-hash envelope."""

    schema_id: Literal["feedbax.studio.draft_hashes"]
    schema_version: Literal["feedbax.studio.draft_hashes.v2"]
    pin: Literal["fnv1a32-canonical_json_v2"]
    hashes: dict[str, str | None]


class LegacyStudioDraftHashes(TypedDict):
    """Admitted historical hashes that cannot prove current draft equality."""

    schema_id: Literal["feedbax.studio.draft_hashes"]
    schema_version: Literal["feedbax.studio.draft_hashes.v1"]
    pin: Literal["fnv1a32-runtime_local_json_v1"]
    rehash_required: Literal[True]
    hashes: dict[str, str | None]


AdmittedStudioDraftHashes: TypeAlias = StudioDraftHashes | LegacyStudioDraftHashes


def studio_draft_digest_v2(value: object) -> str:
    """Return FNV-1a-32 over the canonical v2 UTF-8 bytes for ``value``."""

    hash_value = 2166136261
    for byte in canonical_json_v2_bytes(value):
        hash_value ^= byte
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return f"{hash_value:08x}"


def studio_draft_hashes(payloads: Mapping[str, object]) -> StudioDraftHashes:
    """Hash named Studio draft payloads into the current durable envelope."""

    return {
        "schema_id": STUDIO_DRAFT_HASH_SCHEMA_ID,
        "schema_version": STUDIO_DRAFT_HASH_SCHEMA_VERSION,
        "pin": STUDIO_DRAFT_HASH_PIN,
        "hashes": {key: studio_draft_digest_v2(value) for key, value in payloads.items()},
    }


def admit_studio_draft_hashes(value: object) -> AdmittedStudioDraftHashes:
    """Admit current hashes or migrate an old raw map to rehash-required history."""

    if not isinstance(value, Mapping):
        raise ValueError("Studio draft hashes must be an object")
    mapping = cast(Mapping[object, object], value)
    if not any(key in mapping for key in ("schema_id", "schema_version", "pin", "hashes")):
        return {
            "schema_id": STUDIO_DRAFT_HASH_SCHEMA_ID,
            "schema_version": STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION,
            "pin": STUDIO_DRAFT_HASH_LEGACY_PIN,
            "rehash_required": True,
            "hashes": _nullable_string_hashes(mapping),
        }

    schema_id = mapping.get("schema_id")
    schema_version = mapping.get("schema_version")
    pin = mapping.get("pin")
    if schema_id != STUDIO_DRAFT_HASH_SCHEMA_ID:
        raise ValueError(f"unsupported Studio draft hash schema id: {schema_id!r}")
    if schema_version == STUDIO_DRAFT_HASH_SCHEMA_VERSION:
        if pin != STUDIO_DRAFT_HASH_PIN:
            raise ValueError(f"unsupported Studio draft hash pin: {pin!r}")
        hashes = _nullable_string_hashes(mapping.get("hashes"))
        if any(
            digest is not None
            and (len(digest) != 8 or any(c not in "0123456789abcdef" for c in digest))
            for digest in hashes.values()
        ):
            raise ValueError("current Studio draft hash digests must be eight lowercase hex digits")
        return {
            "schema_id": STUDIO_DRAFT_HASH_SCHEMA_ID,
            "schema_version": STUDIO_DRAFT_HASH_SCHEMA_VERSION,
            "pin": STUDIO_DRAFT_HASH_PIN,
            "hashes": hashes,
        }
    if schema_version == STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION:
        if pin != STUDIO_DRAFT_HASH_LEGACY_PIN:
            raise ValueError(f"unsupported Studio draft hash pin: {pin!r}")
        if mapping.get("rehash_required") is not True:
            raise ValueError("legacy Studio draft hashes must declare rehash_required=true")
        return {
            "schema_id": STUDIO_DRAFT_HASH_SCHEMA_ID,
            "schema_version": STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION,
            "pin": STUDIO_DRAFT_HASH_LEGACY_PIN,
            "rehash_required": True,
            "hashes": _nullable_string_hashes(mapping.get("hashes")),
        }
    raise ValueError(f"unsupported Studio draft hash schema version: {schema_version!r}")


def _nullable_string_hashes(value: object) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        raise ValueError("Studio draft hash values must be an object")
    hashes: dict[str, str | None] = {}
    for key, digest in cast(Mapping[Any, Any], value).items():
        if not isinstance(key, str) or (digest is not None and not isinstance(digest, str)):
            raise ValueError("Studio draft hash names must be strings and values strings or null")
        hashes[key] = digest
    return hashes


__all__ = [
    "AdmittedStudioDraftHashes",
    "LegacyStudioDraftHashes",
    "STUDIO_DRAFT_HASH_LEGACY_PIN",
    "STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION",
    "STUDIO_DRAFT_HASH_PIN",
    "STUDIO_DRAFT_HASH_SCHEMA_ID",
    "STUDIO_DRAFT_HASH_SCHEMA_VERSION",
    "StudioDraftHashes",
    "admit_studio_draft_hashes",
    "studio_draft_digest_v2",
    "studio_draft_hashes",
]
