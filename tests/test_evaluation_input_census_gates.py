"""The evaluation-input uniqueness census decides every candidate it discovers.

Resolution asserts that exactly one manifest under the allowed root carries the
declared ID. A candidate the census cannot read or parse is undecidable, and an
undecidable candidate may well be the second copy — so it refuses the census
rather than dropping out of it and leaving the survivor looking unique.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.analysis.evaluation_inputs import (
    EvaluationInputAmbiguityError,
    resolve_evaluation_inputs,
)
from feedbax.contracts.base import ParentRef
from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    TrainingRunManifest,
)


CENSUS_MANIFEST_ID = "feedbax-training-run:census-input"
CENSUS_MANIFEST_URI = "manifests/training_runs/census-input.json"


def _census_manifest() -> TrainingRunManifest:
    return TrainingRunManifest(id=CENSUS_MANIFEST_ID, status="completed")


def _write_census_manifest(root: Path) -> None:
    path = root / CENSUS_MANIFEST_URI
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (_census_manifest().model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    )


def _census_spec(uri: str | None = None) -> EvaluationRunSpec:
    return EvaluationRunSpec(
        evaluation_type="tests.identity_gate_census",
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id=CENSUS_MANIFEST_ID,
                role="training_run",
                uri=uri,
            )
        ],
    )


def test_census_admits_a_root_whose_other_manifests_all_parse(tmp_path: Path) -> None:
    """Control: unrelated readable manifests are decided, not refused."""
    _write_census_manifest(tmp_path)
    other = tmp_path / "manifests" / "imported" / "other.json"
    other.parent.mkdir(parents=True)
    other.write_text(
        TrainingRunManifest(id="feedbax-training-run:other", status="completed").model_dump_json(),
        encoding="utf-8",
    )

    resolved = resolve_evaluation_inputs(_census_spec(), manifest_root=tmp_path)

    assert resolved[0].id == CENSUS_MANIFEST_ID


def test_malformed_candidate_refuses_the_uniqueness_census(tmp_path: Path) -> None:
    """A candidate that will not parse might be the duplicate, so it refuses."""
    _write_census_manifest(tmp_path)
    broken = tmp_path / "manifests" / "imported" / "broken.json"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"{ this is not json")

    with pytest.raises(EvaluationInputAmbiguityError, match="cannot be proved unique"):
        resolve_evaluation_inputs(_census_spec(), manifest_root=tmp_path)


def test_unreadable_candidate_refuses_the_census_even_behind_an_exact_uri(
    tmp_path: Path,
) -> None:
    """An exact URI still asserts uniqueness, so an undecidable sibling refuses."""
    _write_census_manifest(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-census-outside.json"
    outside.write_text(_census_manifest().model_dump_json(), encoding="utf-8")
    link = tmp_path / "manifests" / "imported" / "linked.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    with pytest.raises(EvaluationInputAmbiguityError, match="unreadable or malformed"):
        resolve_evaluation_inputs(_census_spec(uri=CENSUS_MANIFEST_URI), manifest_root=tmp_path)


def test_undecidable_candidate_names_the_reference_it_could_not_decide(
    tmp_path: Path,
) -> None:
    _write_census_manifest(tmp_path)
    broken = tmp_path / "manifests" / "imported" / "broken.json"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"\xff\xfe not utf-8")

    with pytest.raises(EvaluationInputAmbiguityError) as excinfo:
        resolve_evaluation_inputs(_census_spec(), manifest_root=tmp_path)

    assert "manifests/imported/broken.json" in str(excinfo.value)
