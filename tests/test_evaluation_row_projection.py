from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any, get_type_hints

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict
import pytest

from feedbax.analysis import (
    EvaluationRowProjectionError,
    EvaluationRowProjectionErrorCode,
    ResolvedAnalysisInput,
    ResolvedEvaluationStateHandle,
    ResolvedManifestInput,
    project_evaluation_rows,
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


class _Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: int
    gain: float


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    controller: str


def _authenticated_ref(kind: str, id_: str, role: str, raw_bytes: bytes) -> ParentRef:
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


def _manifest_input(
    root: Path,
    *,
    target: int,
    durable: bool,
    states: Any,
) -> tuple[EvaluationRunManifest, ResolvedManifestInput]:
    training = _authenticated_ref("TrainingRunManifest", "training", "training_run", b"train")
    manifest_id = f"evaluation:{target}"
    spec = EvaluationRunSpec(
        evaluation_type="fixture.row_projection",
        inputs=[training],
        params={"target": target, "gain": 2.0},
    )
    artifacts = (
        [store_evaluation_states_artifact(states, root=root, manifest_id=manifest_id)]
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
            inline=spec.model_dump(mode="json"),
        ),
        input_training_runs=[training],
        artifacts=artifacts,
        metadata={"controller": "feedback"},
        provenance=Provenance(
            entrypoint=EntrypointRef(kind="feedbax-evaluation-recipe", name=spec.evaluation_type),
            parents=[training],
        ),
    )
    raw_bytes = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    ref = _authenticated_ref("EvaluationRunManifest", manifest.id, "evaluation_run", raw_bytes)
    return manifest, ResolvedManifestInput(
        ref=ref,
        manifest=manifest,
        path=Path(f"/fixture/{manifest.id}.json"),
        raw_bytes=raw_bytes,
    )


def _resolved_input(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_kind: str = "evaluation_cache",
    target: int = 0,
    states: Any | None = None,
) -> ResolvedAnalysisInput:
    states = {"velocity": np.asarray(target + 0.5)} if states is None else states
    manifest, manifest_input = _manifest_input(
        root,
        target=target,
        durable=source_kind == "durable",
        states=states,
    )
    cache_path = evaluation_states_cache_path(manifest.id, root=root)
    if source_kind == "evaluation_cache":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_evaluation_states_cache(cache_path, manifest_id=manifest.id, states=states)
    elif source_kind == "analysis_time_recompute":

        def rederive(*_args: Any, **_kwargs: Any) -> tuple[EvaluationRunManifest, Path]:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            write_evaluation_states_cache(cache_path, manifest_id=manifest.id, states=states)
            return manifest, write_manifest(manifest, root=root, index=False)

        monkeypatch.setattr("feedbax.analysis.specs._rederive_evaluation_states", rederive)
    policy = "require_durable" if source_kind == "durable" else "recompute"
    return resolve_analysis_inputs(
        AnalysisRunSpec(
            analysis_type="fixture.row_projection.analysis",
            inputs=[manifest_input.ref],
            evaluation_states_policy=policy,
        ),
        root=root,
        authenticated_inputs={0: manifest_input},
    )[0]


def _project(row: Any) -> tuple[int, float, str]:
    params = _Parameters.model_validate(row.parameters)
    metadata = _Metadata.model_validate(row.metadata)
    velocity = float(row.states["velocity"])
    if velocity <= params.target:
        raise ValueError("consumer-owned cross-field relationship failed")
    return params.target, velocity * params.gain, metadata.controller


@pytest.mark.parametrize(
    ("source_kind", "proof_kind"),
    [
        ("durable", "authenticated_artifact"),
        ("evaluation_cache", "manifest_keyed_cache"),
        ("analysis_time_recompute", "authenticated_recompute"),
    ],
)
def test_projects_all_resolver_source_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    proof_kind: str,
) -> None:
    item = _resolved_input(tmp_path, monkeypatch, source_kind=source_kind)

    assert project_evaluation_rows([item], project=_project) == ((0, 1.0, "feedback"),)
    assert item.evaluation_state_handle.proof_kind == proof_kind
    assert item.evaluation_state_handle.source is item.evaluation_state_source
    assert item.evaluation_state_handle.states is item.states


@pytest.mark.parametrize(
    ("signal", "source_kind", "dtype_name"),
    [
        (np.asarray([1 + 2j], dtype=np.complex64), "durable", "complex64"),
        (jnp.asarray([1.0], dtype=jnp.bfloat16), "evaluation_cache", "bfloat16"),
    ],
)
def test_handle_issuance_is_transparent_to_resolved_state_dtypes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal: Any,
    source_kind: str,
    dtype_name: str,
) -> None:
    item = _resolved_input(
        tmp_path,
        monkeypatch,
        source_kind=source_kind,
        states={"signal": signal},
    )

    projected = project_evaluation_rows([item], project=lambda row: row.states["signal"])[0]

    assert str(np.asarray(projected).dtype) == dtype_name


def test_manifest_facts_come_from_authenticated_raw_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _resolved_input(tmp_path, monkeypatch)
    item.manifest.evaluation_spec.inline["params"]["target"] = 99
    item.manifest.metadata["controller"] = "mutated"

    assert project_evaluation_rows([item], project=_project) == ((0, 1.0, "feedback"),)


def test_two_genuine_rows_cannot_splice_manifest_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _resolved_input(tmp_path, monkeypatch, target=0)
    second = _resolved_input(tmp_path, monkeypatch, target=1)
    spliced = replace(first, ref=second.ref, manifest_input=second.manifest_input)

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows([spliced], project=_project)

    assert caught.value.code is EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH
    assert caught.value.row_index == 0


def test_replaced_state_object_is_outside_the_resolver_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = replace(
        _resolved_input(tmp_path, monkeypatch),
        states={"velocity": np.asarray(99.5)},
    )

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows([item], project=_project)

    assert caught.value.code is EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH


def test_post_resolution_mutation_is_not_a_content_authentication_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _resolved_input(tmp_path, monkeypatch, source_kind="evaluation_cache")
    item.states["velocity"][...] = 3.5

    assert project_evaluation_rows([item], project=_project) == ((0, 7.0, "feedback"),)


def test_forged_exact_type_handle_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genuine = _resolved_input(tmp_path, monkeypatch)
    forged = object.__new__(ResolvedEvaluationStateHandle)
    object.__setattr__(forged, "source", genuine.evaluation_state_source)
    object.__setattr__(forged, "proof_kind", "manifest_keyed_cache")
    object.__setattr__(forged, "evaluation_manifest_authority", genuine.ref)
    object.__setattr__(forged, "states", genuine.states)
    item = replace(genuine, evaluation_state_handle=forged)

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows([item], project=_project)

    assert caught.value.code is EvaluationRowProjectionErrorCode.STATE_HANDLE_INVALID


def test_cross_row_source_retarget_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _resolved_input(tmp_path, monkeypatch, target=0)
    second = _resolved_input(tmp_path, monkeypatch, target=1)
    retargeted = replace(first, evaluation_state_source=second.evaluation_state_source)

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows([retargeted], project=_project)

    assert caught.value.code is EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH


def test_projector_failure_is_reason_coded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows(
            [_resolved_input(tmp_path, monkeypatch)],
            project=lambda _row: (_ for _ in ()).throw(RuntimeError("downstream")),
        )

    assert caught.value.code is EvaluationRowProjectionErrorCode.PROJECTOR_FAILED
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_public_projection_type_hints_resolve_at_runtime() -> None:
    assert set(get_type_hints(project_evaluation_rows)) == {"inputs", "project", "return"}
