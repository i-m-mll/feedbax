from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from feedbax.analysis import (
    EvaluationRowCoverageError,
    EvaluationRowProjectionError,
    EvaluationRowProjectionErrorCategory,
    EvaluationRowProjector,
    ResolvedAnalysisInput,
    project_authenticated_evaluation_rows,
    require_exact_authored_cartesian_coverage,
)
from feedbax.analysis.manifest_inputs import ResolvedManifestInput
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    AnalysisEvaluationStateSource,
    ArtifactRef,
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    SpecPayload,
)


class _Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: str
    target: int


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    states_schema: str


def _authenticated_ref(
    kind: str,
    id_: str,
    role: str,
    raw_bytes: bytes,
) -> ParentRef:
    return ParentRef(
        kind=kind,
        id=id_,
        role=role,
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "size_bytes": len(raw_bytes),
        },
    )


def _input(*, arm: str = "trained", target: int = 0) -> ResolvedAnalysisInput:
    training_bytes = b"training"
    training = _authenticated_ref(
        "TrainingRunManifest",
        "training",
        "training_run",
        training_bytes,
    )
    run_spec = EvaluationRunSpec(
        evaluation_type="fixture.row_projection",
        inputs=[training],
        params={"arm": arm, "target": target},
    )
    states_bytes = b"states"
    states_digest = hashlib.sha256(states_bytes).hexdigest()
    artifact = ArtifactRef(
        role="evaluation_states",
        logical_name="states",
        artifact_id=f"artifact://sha256/{states_digest}",
        sha256=states_digest,
        size_bytes=len(states_bytes),
        storage_backend="fixture",
        uri=f"artifact://sha256/{states_digest}",
    )
    manifest = EvaluationRunManifest(
        id=f"evaluation:{arm}:{target}",
        status="completed",
        evaluation_spec=SpecPayload(
            kind="EvaluationRunSpec",
            schema_id="feedbax.spec.evaluation_run",
            schema_version="feedbax.spec.evaluation_run.v1",
            inline=run_spec.model_dump(mode="json"),
        ),
        input_training_runs=[training],
        artifacts=[artifact],
        metadata={"states_schema": "fixture.states.v1"},
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-evaluation-recipe",
                name=run_spec.evaluation_type,
            ),
            parents=[training],
        ),
    )
    raw_bytes = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    authority = _authenticated_ref(
        "EvaluationRunManifest",
        manifest.id,
        "evaluation_run",
        raw_bytes,
    )
    portable_authority = authority.model_copy(update={"uri": None})
    source = AnalysisEvaluationStateSource(
        source_kind="durable",
        requested_evaluation_manifest_id=manifest.id,
        evaluation_manifest_authority=portable_authority,
        supplying_evaluation_manifest_id=manifest.id,
        artifact_id=artifact.artifact_id,
        artifact_sha256=artifact.sha256,
        artifact_size_bytes=artifact.size_bytes,
        artifact_storage_backend=artifact.storage_backend,
        container_schema_id="feedbax.manifest.evaluation_states_container",
        container_schema_version="feedbax.manifest.evaluation_states_container.v3",
        container_storage_backend="npz.v3",
    )
    return ResolvedAnalysisInput(
        ref=authority,
        manifest=manifest,
        path=None,
        states={"samples": [target]},
        evaluation_state_source=source,
        manifest_input=ResolvedManifestInput(
            ref=authority,
            manifest=manifest,
            path=Path(f"/fixture/{manifest.id}.json"),
            raw_bytes=raw_bytes,
        ),
    )


def _projector() -> EvaluationRowProjector[
    tuple[int, ...],
    _Parameters,
    _Metadata,
    tuple[str, int],
]:
    return EvaluationRowProjector(
        state=lambda row: tuple(row.states["samples"]),
        parameters=lambda row: _Parameters.model_validate(row.parameters),
        metadata=lambda row: _Metadata.model_validate(row.metadata),
        row_key=lambda _row, _state, params, _metadata: (params.arm, params.target),
    )


def test_projects_typed_rows_after_authenticating_authority_and_provenance() -> None:
    rows = project_authenticated_evaluation_rows(
        [_input(target=0), _input(target=1)],
        projector=_projector(),
    )

    assert tuple(row.row_key for row in rows) == (("trained", 0), ("trained", 1))
    assert rows[1].state == (1,)
    assert rows[0].parameters == _Parameters(arm="trained", target=0)
    assert rows[0].metadata == _Metadata(states_schema="fixture.states.v1")
    assert rows[0].authority.provenance.producer_identity == "fixture.row_projection"
    assert rows[0].authority.provenance.source_refs == (
        rows[0].authority.run_spec.inputs[0],
    )


@pytest.mark.parametrize(
    ("mutate", "category"),
    [
        (
            lambda item: replace(item, manifest_input=None),
            EvaluationRowProjectionErrorCategory.INPUT_CONTRACT,
        ),
        (
            lambda item: replace(
                item,
                manifest_input=replace(item.manifest_input, raw_bytes=b"tampered"),
            ),
            EvaluationRowProjectionErrorCategory.PROVENANCE,
        ),
        (
            lambda item: replace(item, evaluation_state_source=None),
            EvaluationRowProjectionErrorCategory.STATE_AUTHORITY,
        ),
    ],
)
def test_authentication_failures_have_structured_categories(
    mutate: Any,
    category: EvaluationRowProjectionErrorCategory,
) -> None:
    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_authenticated_evaluation_rows(
            [mutate(_input())],
            projector=_projector(),
        )

    assert caught.value.category is category
    assert caught.value.row_index == 0


def test_downstream_projection_failure_is_categorized_without_message_parsing() -> None:
    projector = replace(
        _projector(),
        metadata=lambda _row: (_ for _ in ()).throw(RuntimeError("downstream detail")),
    )

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_authenticated_evaluation_rows([_input()], projector=projector)

    assert caught.value.category is EvaluationRowProjectionErrorCategory.PROJECTION
    assert caught.value.projection_field == "metadata"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_duplicate_projected_row_key_is_a_distinct_error_category() -> None:
    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_authenticated_evaluation_rows(
            [_input(), _input()],
            projector=_projector(),
        )

    assert caught.value.category is EvaluationRowProjectionErrorCategory.DUPLICATE_ROW_KEY


def test_exact_authored_cartesian_coverage_preserves_authored_order() -> None:
    axes = {"arm": ["analytical", "trained"], "target": [0, 1]}
    observed = [
        ("trained", 1),
        ("analytical", 0),
        ("trained", 0),
        ("analytical", 1),
    ]

    expected = require_exact_authored_cartesian_coverage(
        observed,
        axes=axes,
        row_key=lambda coordinate: (coordinate["arm"], coordinate["target"]),
    )

    assert expected == (
        ("analytical", 0),
        ("analytical", 1),
        ("trained", 0),
        ("trained", 1),
    )


def test_exact_authored_cartesian_coverage_reports_structured_delta() -> None:
    with pytest.raises(EvaluationRowCoverageError) as caught:
        require_exact_authored_cartesian_coverage(
            [("trained", 0), ("other", 1)],
            axes={"arm": ["trained"], "target": [0, 1]},
            row_key=lambda coordinate: (coordinate["arm"], coordinate["target"]),
        )

    assert caught.value.category is EvaluationRowProjectionErrorCategory.COVERAGE
    assert caught.value.missing == (("trained", 1),)
    assert caught.value.unexpected == (("other", 1),)


def test_exact_authored_cartesian_coverage_rejects_duplicates_and_key_collisions() -> None:
    with pytest.raises(EvaluationRowCoverageError) as duplicate:
        require_exact_authored_cartesian_coverage(
            [("trained", 0), ("trained", 0)],
            axes={"arm": ["trained"], "target": [0]},
            row_key=lambda coordinate: (coordinate["arm"], coordinate["target"]),
        )
    assert duplicate.value.duplicates == (("trained", 0),)

    with pytest.raises(EvaluationRowCoverageError) as collision:
        require_exact_authored_cartesian_coverage(
            ["trained"],
            axes={"arm": ["trained"], "target": [0, 1]},
            row_key=lambda coordinate: coordinate["arm"],
        )
    assert collision.value.duplicates == ("trained",)
