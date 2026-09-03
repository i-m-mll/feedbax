"""The legacy v1 hash domain and on-disk form for authored documents.

Every content pin the engine writes — envelope hashes, lineage pins, compiled
document hashes, execution identity — remains on ``canonical_json_v1`` until
its owning durable schema explicitly migrates. Existing pins are permanent.

The projection is Feedbax's existing stable-JSON encoding
(:func:`~feedbax.contracts.manifest.canonical_json_bytes`: sorted keys, compact
separators, UTF-8), reused rather than re-implemented. The name
:data:`CANONICAL_PIN_ALGORITHM` is written into every pin record so a stored
hash always says which domain produced it. V2 exists beside this domain and
never redefines it.

The *emit* form is deliberately different from the *hash* form: hashing wants the
tightest projection, while a tracked compiler output wants to be readable and to
diff line by line. Both are fixed here so a regenerated document is byte-identical
to the tracked one, which is what makes the choke-point comparison meaningful.
"""

from __future__ import annotations

import json
from typing import Any

from feedbax.contracts.base import (
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.canonical_json import CANONICAL_JSON_V1

#: The name recorded beside every content hash the engine writes. A pin that does
#: not name its algorithm cannot be re-verified; a pin naming an algorithm the
#: reader does not know fails closed instead of being compared anyway.
CANONICAL_PIN_ALGORITHM = CANONICAL_JSON_V1


def canonical_bytes(value: Any) -> bytes:
    """Return the canonical serialization every content pin is computed over."""
    return canonical_json_bytes(value)


def canonical_sha256(value: Any) -> str:
    """Return the canonical sha256 content hash of ``value``."""
    return sha256_bytes(canonical_bytes(value))


def emit_text(value: Any) -> str:
    """Return the deterministic on-disk form of a compiler output document."""
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def content_pin(ref: str, value: Any) -> dict[str, str]:
    """Return one pin record: what was read, its hash, and the domain that made it."""
    return {
        "ref": ref,
        "content_hash": canonical_sha256(value),
        "pin_algorithm": CANONICAL_PIN_ALGORITHM,
    }


__all__ = [
    "CANONICAL_PIN_ALGORITHM",
    "canonical_bytes",
    "canonical_sha256",
    "content_pin",
    "emit_text",
]
