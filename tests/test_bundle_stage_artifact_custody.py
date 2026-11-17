"""Artifact custody verifies declared digests against saved analysis bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from feedbax.analysis.fulfillment import admit_manifest_artifact_custody
from feedbax.contracts.base import (
    ArtifactRef,
    Provenance,
)
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    spec_payload,
)

pytestmark = [pytest.mark.feedbax_contract]


MANIFEST_ID = "feedbax-analysis:stage-custody"


def _artifact(sha256: str, size_bytes: int, relative: str) -> ArtifactRef:
    return ArtifactRef(
        role="analysis_data",
        logical_name="stage-product",
        media_type="application/octet-stream",
        sha256=sha256,
        size_bytes=size_bytes,
        metadata={"relative_path": relative},
    )


def _manifest(artifact: ArtifactRef) -> AnalysisRunManifest:
    spec = AnalysisRunSpec(analysis_type="feedbax.test.stage_custody", inputs=[], params={})
    return AnalysisRunManifest(
        id=MANIFEST_ID,
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec", spec.model_dump(mode="json", exclude_none=True)
        ),
        provenance=Provenance(parents=[]),
        artifacts=[artifact],
    )


def _write_product(root: Path, payload: bytes, relative: str = "products/stage.bin") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_artifact_custody_admits_bytes_that_match_what_the_manifest_recorded(
    tmp_path: Path,
) -> None:
    payload = b"stage product bytes"
    _write_product(tmp_path, payload)
    manifest = _manifest(
        _artifact(hashlib.sha256(payload).hexdigest(), len(payload), "products/stage.bin")
    )

    outcome = admit_manifest_artifact_custody(manifest, node_kind="analysis", root=tmp_path)

    assert outcome.admitted
    assert outcome.failures == []


@pytest.mark.parametrize(
    ("stored", "code"),
    [
        (b"replaced by something else entirely", "artifact_sha256_mismatch"),
        (None, "artifact_bytes_absent"),
    ],
)
def test_artifact_custody_refuses_bytes_that_are_not_what_was_recorded(
    tmp_path: Path, stored: bytes | None, code: str
) -> None:
    declared = b"stage product bytes"
    if stored is not None:
        _write_product(tmp_path, stored)
    manifest = _manifest(
        _artifact(hashlib.sha256(declared).hexdigest(), len(declared), "products/stage.bin")
    )

    outcome = admit_manifest_artifact_custody(manifest, node_kind="analysis", root=tmp_path)

    assert not outcome.admitted
    assert outcome.codes == (code,)


def test_artifact_custody_refuses_an_artifact_that_declares_no_digest(tmp_path: Path) -> None:
    _write_product(tmp_path, b"anything")
    artifact = ArtifactRef(
        role="analysis_data",
        logical_name="stage-product",
        media_type="application/octet-stream",
        metadata={"relative_path": "products/stage.bin"},
    )

    outcome = admit_manifest_artifact_custody(
        _manifest(artifact), node_kind="analysis", root=tmp_path
    )

    assert not outcome.admitted
    assert outcome.codes == ("artifact_digest_absent",)
