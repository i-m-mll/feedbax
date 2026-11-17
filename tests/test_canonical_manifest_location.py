"""Managed Train and Evaluate manifests have exact canonical locations."""

from pathlib import Path

import pytest

from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    TrainingRunManifest,
    canonical_manifest_path,
    load_manifest,
    spec_payload,
    write_manifest,
)


@pytest.mark.parametrize(
    ("manifest", "relative_path"),
    [
        (
            TrainingRunManifest(id="canonical-train", status="completed"),
            Path("manifests/training_runs/canonical-train.json"),
        ),
        (
            EvaluationRunManifest(
                id="canonical-evaluate",
                status="completed",
                evaluation_spec=spec_payload(
                    "EvaluationRunSpec",
                    EvaluationRunSpec(evaluation_type="feedbax.rollout").model_dump(mode="json"),
                ),
            ),
            Path("manifests/evaluation_runs/canonical-evaluate.json"),
        ),
    ],
    ids=["training", "evaluation"],
)
def test_written_manifests_reopen_at_their_canonical_location(tmp_path: Path, manifest, relative_path):
    written = write_manifest(manifest, root=tmp_path, index=False)
    assert written == tmp_path / relative_path
    assert canonical_manifest_path(manifest.kind, manifest.id, root=tmp_path) == written
    reopened = load_manifest(written)
    assert reopened.id == manifest.id
    assert reopened.kind == manifest.kind
