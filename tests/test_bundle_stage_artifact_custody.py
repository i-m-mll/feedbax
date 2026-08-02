"""A freshly written bundle stage product carries the custody a reused one does.

Reusing a cached stage product runs the full per-kind admission, and artifact
byte custody is part of it. Binding a *freshly written* product on its manifest
bytes alone would make a first run's custody weaker than the identical rerun's:
the same artifacts, admitted on the second pass and trusted on the first.

So the artifact half of that admission runs where the fresh receipt is built.
These tests state it at both ends — the reusable gate on its own, and the
bundle-stage receipt that now routes through it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from feedbax.analysis.fulfillment import (
    FulfillmentAdmissionError,
    admit_manifest_artifact_custody,
)
from feedbax.analysis.fulfillment_adapters import _bundle_stage_receipt
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    Provenance,
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


def _write_manifest_bytes(root: Path, manifest: AnalysisRunManifest) -> tuple[Path, bytes]:
    path = root / "manifests" / "analysis_runs" / "stage-custody.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return path, raw


# -- the reusable gate ------------------------------------------------------


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


# -- the bundle-stage receipt now routes through it -------------------------


def test_fresh_bundle_stage_receipt_refuses_artifacts_that_do_not_verify(
    tmp_path: Path,
) -> None:
    declared = b"stage product bytes"
    _write_product(tmp_path, b"a different product")
    manifest = _manifest(
        _artifact(hashlib.sha256(declared).hexdigest(), len(declared), "products/stage.bin")
    )
    path, raw = _write_manifest_bytes(tmp_path, manifest)

    with pytest.raises(FulfillmentAdmissionError) as excinfo:
        _bundle_stage_receipt(manifest, path, raw, root=tmp_path, node_key="analysis:stage")

    assert excinfo.value.outcome.codes == ("artifact_sha256_mismatch",)


def test_fresh_bundle_stage_receipt_binds_when_artifact_custody_holds(tmp_path: Path) -> None:
    """Control: the receipt is still bound when the product really is there."""
    payload = b"stage product bytes"
    _write_product(tmp_path, payload)
    manifest = _manifest(
        _artifact(hashlib.sha256(payload).hexdigest(), len(payload), "products/stage.bin")
    )
    path, raw = _write_manifest_bytes(tmp_path, manifest)

    receipt = _bundle_stage_receipt(
        manifest, path, raw, root=tmp_path, node_key="analysis:stage"
    )

    assert receipt.manifest.id == MANIFEST_ID
    assert receipt.node_kind == "analysis"


def test_fresh_bundle_stage_receipt_gate_also_runs_without_captured_bytes(
    tmp_path: Path,
) -> None:
    """The template path hands back no read, and is held to the same custody."""
    declared = b"stage product bytes"
    _write_product(tmp_path, b"a different product")
    manifest = _manifest(
        _artifact(hashlib.sha256(declared).hexdigest(), len(declared), "products/stage.bin")
    )
    path, _raw = _write_manifest_bytes(tmp_path, manifest)

    with pytest.raises(FulfillmentAdmissionError):
        _bundle_stage_receipt(manifest, path, None, root=tmp_path, node_key="analysis:stage")
