from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from feedbax.analysis import (
    EvaluationRowCoverageError,
    EvaluationRowProjection,
    EvaluationRowProjectionError,
    EvaluationRowProjectionErrorCategory,
    EvaluationRowProjectionErrorReason,
    EvaluationStateMaterializationReceipt,
    ResolvedAnalysisInput,
    ResolvedManifestInput,
    project_verified_evaluation_rows,
    require_exact_authored_cartesian_coverage,
    resolve_analysis_inputs,
)
from feedbax.analysis.evaluation import write_evaluation_states_cache
from feedbax.contracts.evaluation_states import store_evaluation_states_artifact
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    AnalysisRunSpec,
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    SpecPayload,
    evaluation_states_cache_path,
    write_manifest,
)


class _VelocityParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: str
    target: int
    conditioning: str
    gain: float


class _VelocityMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    states_schema: str
    controller: str
    partition: str


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


def _manifest_and_input(
    tmp_path: Path,
    *,
    target: int,
    durable: bool,
) -> tuple[EvaluationRunManifest, ResolvedManifestInput]:
    training_bytes = b"training"
    training = _authenticated_ref(
        "TrainingRunManifest",
        "training",
        "training_run",
        training_bytes,
    )
    manifest_id = f"evaluation:trained:{target}"
    run_spec = EvaluationRunSpec(
        evaluation_type="fixture.row_projection",
        inputs=[training],
        params={
            "arm": "trained",
            "target": target,
            "conditioning": "reach",
            "gain": 2.0,
        },
    )
    states = {"sample": target, "velocity": target + 0.5}
    artifacts = (
        [store_evaluation_states_artifact(states, root=tmp_path, manifest_id=manifest_id)]
        if durable
        else []
    )
    manifest = EvaluationRunManifest(
        id=manifest_id,
        status="completed",
        evaluation_spec=SpecPayload(
            kind="EvaluationRunSpec",
            schema_id="feedbax.spec.evaluation_run",
            schema_version="feedbax.spec.evaluation_run.v1",
            inline=run_spec.model_dump(mode="json"),
        ),
        input_training_runs=[training],
        artifacts=artifacts,
        metadata={
            "states_schema": "fixture.states.v1",
            "controller": "feedback",
            "partition": "holdout",
        },
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
    return manifest, ResolvedManifestInput(
        ref=authority,
        manifest=manifest,
        path=Path(f"/fixture/{manifest.id}.json"),
        raw_bytes=raw_bytes,
    )


def _resolved_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_kind: str = "evaluation_cache",
    target: int = 0,
) -> ResolvedAnalysisInput:
    durable = source_kind == "durable"
    manifest, manifest_input = _manifest_and_input(
        tmp_path,
        target=target,
        durable=durable,
    )
    states = {"sample": target, "velocity": target + 0.5}
    cache_path = evaluation_states_cache_path(manifest.id, root=tmp_path)
    if source_kind == "evaluation_cache":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_evaluation_states_cache(cache_path, manifest_id=manifest.id, states=states)
    elif source_kind == "analysis_time_recompute":

        def rederive(*_args: Any, **_kwargs: Any) -> tuple[EvaluationRunManifest, Path]:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            write_evaluation_states_cache(cache_path, manifest_id=manifest.id, states=states)
            return manifest, write_manifest(manifest, root=tmp_path, index=False)

        monkeypatch.setattr("feedbax.analysis.specs._rederive_evaluation_states", rederive)
    policy = "require_durable" if durable else "recompute"
    analysis_spec = AnalysisRunSpec(
        analysis_type="fixture.row_projection.analysis",
        inputs=[manifest_input.ref],
        evaluation_states_policy=policy,
    )
    return resolve_analysis_inputs(
        analysis_spec,
        root=tmp_path,
        authenticated_inputs={0: manifest_input},
    )[0]


def _velocity_projection(facts: Any) -> EvaluationRowProjection:
    params = _VelocityParameters.model_validate(facts.parameters)
    metadata = _VelocityMetadata.model_validate(facts.metadata)
    if metadata.states_schema != "fixture.states.v1":
        raise ValueError("unsupported state geometry")
    velocity = float(facts.states["velocity"])
    if velocity <= params.target:
        raise ValueError("state/parameter relationship is invalid")
    return EvaluationRowProjection(
        row_key=(params.arm, params.target, params.conditioning),
        state=velocity,
        parameters=params,
        metadata=metadata,
    )


def _controller_response_projection(facts: Any) -> EvaluationRowProjection:
    params = _VelocityParameters.model_validate(facts.parameters)
    metadata = _VelocityMetadata.model_validate(facts.metadata)
    if metadata.controller != "feedback" or metadata.partition != "holdout":
        raise ValueError("consumer-owned controller partition is invalid")
    response = float(facts.states["velocity"]) * params.gain
    return EvaluationRowProjection(
        row_key=(metadata.controller, params.target, metadata.partition),
        state={"peak_response": response},
        parameters=params,
        metadata=metadata,
    )


@pytest.mark.parametrize(
    ("source_kind", "proof_kind"),
    [
        ("durable", "authenticated_artifact"),
        ("evaluation_cache", "manifest_keyed_cache"),
        ("analysis_time_recompute", "authenticated_recompute"),
    ],
)
def test_projects_all_resolver_source_kinds_with_truthful_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    proof_kind: str,
) -> None:
    item = _resolved_input(tmp_path, monkeypatch, source_kind=source_kind)

    row = project_verified_evaluation_rows([item], project=_velocity_projection)[0]

    assert row.row_key == ("trained", 0, "reach")
    assert row.state == 0.5
    assert row.facts.state_source.source_kind == source_kind
    assert row.facts.state_receipt.proof_kind == proof_kind
    assert row.facts.provenance.producer_identity == "fixture.row_projection"


def test_one_projector_supports_two_materially_different_consumer_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = [
        _resolved_input(tmp_path, monkeypatch, target=0),
        _resolved_input(tmp_path, monkeypatch, target=1),
    ]

    velocity = project_verified_evaluation_rows(inputs, project=_velocity_projection)
    controller = project_verified_evaluation_rows(
        inputs,
        project=_controller_response_projection,
    )

    assert tuple(row.state for row in velocity) == (0.5, 1.5)
    assert tuple(row.row_key for row in controller) == (
        ("feedback", 0, "holdout"),
        ("feedback", 1, "holdout"),
    )
    assert controller[1].state == {"peak_response": 3.0}


def test_state_receipt_cannot_be_minted_through_the_public_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _resolved_input(tmp_path, monkeypatch)

    with pytest.raises(TypeError, match="issued by resolve_analysis_inputs"):
        EvaluationStateMaterializationReceipt(
            item.evaluation_state_source,
            "manifest_keyed_cache",
            item.states,
            _token=object(),
        )


@pytest.mark.parametrize(
    ("mutate", "category", "reason"),
    [
        (
            lambda item: replace(item, manifest_input=None),
            EvaluationRowProjectionErrorCategory.INPUT_CONTRACT,
            EvaluationRowProjectionErrorReason.INPUT_MANIFEST_MISSING,
        ),
        (
            lambda item: replace(
                item,
                manifest_input=replace(item.manifest_input, raw_bytes=b"tampered"),
            ),
            EvaluationRowProjectionErrorCategory.PROVENANCE,
            EvaluationRowProjectionErrorReason.MANIFEST_PROVENANCE_INVALID,
        ),
        (
            lambda item: replace(item, evaluation_state_receipt=None),
            EvaluationRowProjectionErrorCategory.STATE_MATERIALIZATION,
            EvaluationRowProjectionErrorReason.STATE_RECEIPT_MISSING,
        ),
        (
            lambda item: replace(item, states={"sample": 99, "velocity": 99.5}),
            EvaluationRowProjectionErrorCategory.STATE_MATERIALIZATION,
            EvaluationRowProjectionErrorReason.STATE_RECEIPT_MISMATCH,
        ),
    ],
)
def test_failures_have_stable_categories_and_reason_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    category: EvaluationRowProjectionErrorCategory,
    reason: EvaluationRowProjectionErrorReason,
) -> None:
    item = _resolved_input(tmp_path, monkeypatch, source_kind="durable")

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_verified_evaluation_rows([mutate(item)], project=_velocity_projection)

    assert caught.value.category is category
    assert caught.value.reason is reason
    assert caught.value.row_index == 0


@pytest.mark.parametrize(
    ("source_kind", "field", "value"),
    [
        ("durable", "container_schema_version", "wrong"),
        ("evaluation_cache", "cache_key", "wrong"),
        ("analysis_time_recompute", "resulting_evaluation_manifest_id", "wrong"),
    ],
)
def test_complete_typed_source_is_bound_at_the_receipt_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    field: str,
    value: str,
) -> None:
    item = _resolved_input(tmp_path, monkeypatch, source_kind=source_kind)
    tampered = replace(
        item,
        evaluation_state_source=item.evaluation_state_source.model_copy(update={field: value}),
    )

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_verified_evaluation_rows([tampered], project=_velocity_projection)

    assert caught.value.reason is EvaluationRowProjectionErrorReason.STATE_RECEIPT_MISMATCH


def test_downstream_projection_failure_is_reason_coded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_facts: Any) -> EvaluationRowProjection:
        raise RuntimeError("downstream detail")

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_verified_evaluation_rows(
            [_resolved_input(tmp_path, monkeypatch)],
            project=fail,
        )

    assert caught.value.category is EvaluationRowProjectionErrorCategory.PROJECTION
    assert caught.value.reason is EvaluationRowProjectionErrorReason.PROJECTOR_FAILED
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_duplicate_projected_row_key_carries_identity_and_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _resolved_input(tmp_path, monkeypatch)

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_verified_evaluation_rows([item, item], project=_velocity_projection)

    assert caught.value.reason is EvaluationRowProjectionErrorReason.PROJECTED_KEY_DUPLICATE
    assert caught.value.row_key == ("trained", 0, "reach")
    assert caught.value.first_index == 0
    assert caught.value.row_index == 1


def test_exact_authored_cartesian_coverage_preserves_authored_order() -> None:
    expected = require_exact_authored_cartesian_coverage(
        [
            ("trained", 1),
            ("analytical", 0),
            ("trained", 0),
            ("analytical", 1),
        ],
        axes={"arm": ["analytical", "trained"], "target": [0, 1]},
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

    assert caught.value.reason is EvaluationRowProjectionErrorReason.COVERAGE_KEY_SET_MISMATCH
    assert caught.value.missing == (("trained", 1),)
    assert caught.value.unexpected == (("other", 1),)


def test_exact_coverage_reports_duplicate_keys_and_indices() -> None:
    with pytest.raises(EvaluationRowCoverageError) as duplicate:
        require_exact_authored_cartesian_coverage(
            [("trained", 0), ("trained", 0)],
            axes={"arm": ["trained"], "target": [0]},
            row_key=lambda coordinate: (coordinate["arm"], coordinate["target"]),
        )

    assert duplicate.value.reason is (
        EvaluationRowProjectionErrorReason.COVERAGE_OBSERVED_KEY_DUPLICATE
    )
    assert duplicate.value.duplicates == (("trained", 0),)
    assert duplicate.value.duplicate_indices == ((0, 1),)


def test_exact_coverage_reports_expected_key_collision() -> None:
    with pytest.raises(EvaluationRowCoverageError) as collision:
        require_exact_authored_cartesian_coverage(
            ["trained"],
            axes={"arm": ["trained"], "target": [0, 1]},
            row_key=lambda coordinate: coordinate["arm"],
        )

    assert collision.value.reason is (
        EvaluationRowProjectionErrorReason.COVERAGE_EXPECTED_KEY_COLLISION
    )
    assert collision.value.duplicates == ("trained",)
    assert collision.value.duplicate_indices == ((0, 1),)
