from __future__ import annotations

import hashlib

import pytest

from feedbax.contracts import (
    AuthenticatedManifestDigest,
    EvaluationManifestProvenanceEnvelope,
    verify_evaluation_manifest_provenance,
)
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    SpecPayload,
)


def _ref(kind: str, id_: str, role: str, digest: str, size: int = 123) -> ParentRef:
    return ParentRef(
        kind=kind,
        id=id_,
        role=role,
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": digest,
            "size_bytes": size,
        },
    )


def _case() -> tuple[EvaluationRunManifest, ParentRef, bytes, tuple[ParentRef, ...]]:
    training = _ref("TrainingRunManifest", "training", "training_run", "a" * 64)
    bank = _ref("EvaluationRunManifest", "bank", "evaluation_run", "b" * 64)
    spec = EvaluationRunSpec(
        evaluation_type="example.rollout",
        inputs=[training],
        params={
            "staged_prerequisites": {
                "bank": {"parent": bank.model_dump(mode="json")}
            }
        },
    )
    manifest = EvaluationRunManifest(
        id="evaluation",
        status="completed",
        evaluation_spec=SpecPayload(
            kind="EvaluationRunSpec",
            schema_id="feedbax.spec.evaluation_run",
            schema_version="feedbax.spec.evaluation_run.v1",
            inline=spec.model_dump(mode="json"),
        ),
        input_training_runs=[training],
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-evaluation-recipe", name="example.rollout"
            ),
            parents=[training, bank],
        ),
    )
    raw_bytes = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    authority = _ref(
        "EvaluationRunManifest",
        manifest.id,
        "evaluation_run",
        hashlib.sha256(raw_bytes).hexdigest(),
        len(raw_bytes),
    )
    return manifest, authority, raw_bytes, (training, bank)


def test_verifier_returns_authenticated_producer_source_digest_envelope() -> None:
    manifest, authority, raw_bytes, sources = _case()

    envelope = verify_evaluation_manifest_provenance(
        authority,
        raw_bytes,
        expected_producer_identity="example.rollout",
        expected_source_refs=sources,
    )

    assert isinstance(envelope, EvaluationManifestProvenanceEnvelope)
    assert envelope.producer_identity == "example.rollout"
    assert envelope.source_refs == sources
    assert envelope.digest_envelope == (
        AuthenticatedManifestDigest(
            "EvaluationRunManifest",
            "evaluation",
            "evaluation_run",
            hashlib.sha256(raw_bytes).hexdigest(),
            len(raw_bytes),
        ),
        AuthenticatedManifestDigest(
            "TrainingRunManifest", "training", "training_run", "a" * 64, 123
        ),
        AuthenticatedManifestDigest(
            "EvaluationRunManifest", "bank", "evaluation_run", "b" * 64, 123
        ),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("pending", "completed"),
        ("producer", "producer identity"),
        ("authority", "authority is not authenticated"),
        ("parents", "parents disagree"),
        ("expected", "expected sources"),
        ("source", "source 'training' is not authenticated"),
    ],
)
def test_verifier_fails_closed_on_envelope_drift(mutation: str, message: str) -> None:
    manifest, authority, raw_bytes, sources = _case()
    producer = "example.rollout"
    expected = sources
    if mutation == "pending":
        manifest.status = "running"
    elif mutation == "producer":
        producer = "other.rollout"
    elif mutation == "authority":
        authority.metadata = {}
    elif mutation == "parents":
        manifest.provenance.parents = list(sources[:1])
    elif mutation == "expected":
        expected = sources[:1]
    elif mutation == "source":
        manifest.evaluation_spec.inline["inputs"][0]["metadata"] = {}
        manifest.input_training_runs[0].metadata.clear()
        manifest.provenance.parents[0].metadata.clear()
        expected[0].metadata.clear()
    if mutation in {"pending", "parents", "source"}:
        raw_bytes = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
        authority = _ref(
            "EvaluationRunManifest",
            manifest.id,
            "evaluation_run",
            hashlib.sha256(raw_bytes).hexdigest(),
            len(raw_bytes),
        )

    with pytest.raises(ValueError, match=message):
        verify_evaluation_manifest_provenance(
            authority,
            raw_bytes,
            expected_producer_identity=producer,
            expected_source_refs=expected,
        )


def test_verifier_rejects_altered_manifest_bytes_with_preserved_identity() -> None:
    manifest, authority, raw_bytes, sources = _case()
    altered = raw_bytes.replace(b"example.rollout", b"tamperd.rollout", 1)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_evaluation_manifest_provenance(
            authority,
            altered,
            expected_producer_identity="example.rollout",
            expected_source_refs=sources,
        )
