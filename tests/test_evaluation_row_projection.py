from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any, Literal, get_type_hints

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict
import pytest

from feedbax.analysis import (
    EvaluationRowProjectionError,
    EvaluationRowProjectionErrorCode,
    ResolvedAnalysisInput,
    ResolvedEvaluationStateHandle,
    project_evaluation_rows,
    resolve_analysis_inputs,
)
from feedbax.analysis.evaluation import EvaluationRecipeResult, execute_evaluation_run_spec
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.contracts.base import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    ParentRef,
)
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunSpec,
    evaluation_states_cache_path,
)
from feedbax.plugins.application import new_application_registry_bundle


class _Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: int
    gain: float
    states_custody: Literal["cache", "durable"] = "cache"


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    controller: str
    cache: dict[str, Any]


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


def _resolved_input(
    root: Path,
    *,
    source_kind: str = "evaluation_cache",
    target: int = 0,
    states: Any | None = None,
) -> ResolvedAnalysisInput:
    registries = new_application_registry_bundle()
    states = {"velocity": np.asarray(target + 0.5)} if states is None else states
    durable = source_kind == "durable"
    training = _authenticated_ref("TrainingRunManifest", "training", "training_run", b"train")
    spec = EvaluationRunSpec(
        evaluation_type="fixture.row_projection",
        inputs=[training],
        params={"target": target, "gain": 2.0, **({"states_custody": "durable"} if durable else {})},
    )

    def recipe(_spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult(states=states, metadata={"controller": "feedback"})

    registries.evaluation_recipes.register(spec.evaluation_type, recipe)
    manifest, path = execute_evaluation_run_spec(
        spec,
        root=root,
        registry=registries.evaluation_recipes,
    )
    if source_kind == "analysis_time_recompute":
        evaluation_states_cache_path(manifest.id, root=root).unlink()
    return resolve_analysis_inputs(
        AnalysisRunSpec(
            analysis_type="fixture.row_projection.analysis",
            inputs=[authenticated_manifest_ref(manifest, path, "evaluation_run")],
            evaluation_states_policy="require_durable" if durable else "recompute",
        ),
        root=root,
        registry=registries.analysis_recipes,
        evaluation_registry=registries.evaluation_recipes,
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
    source_kind: str,
    proof_kind: str,
) -> None:
    item = _resolved_input(tmp_path, source_kind=source_kind)

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
    signal: Any,
    source_kind: str,
    dtype_name: str,
) -> None:
    item = _resolved_input(
        tmp_path,
        source_kind=source_kind,
        states={"signal": signal},
    )

    projected = project_evaluation_rows([item], project=lambda row: row.states["signal"])[0]

    assert str(np.asarray(projected).dtype) == dtype_name


def test_manifest_facts_come_from_authenticated_raw_bytes(tmp_path: Path) -> None:
    item = _resolved_input(tmp_path)
    item.manifest.evaluation_spec.inline["params"]["target"] = 99
    item.manifest.metadata["controller"] = "mutated"

    assert project_evaluation_rows([item], project=_project) == ((0, 1.0, "feedback"),)


def test_two_genuine_rows_cannot_splice_manifest_authority(tmp_path: Path) -> None:
    first = _resolved_input(tmp_path, target=0)
    second = _resolved_input(tmp_path, target=1)
    spliced = replace(first, ref=second.ref, manifest_input=second.manifest_input)

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows([spliced], project=_project)

    assert caught.value.code is EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH
    assert caught.value.row_index == 0


def test_replaced_state_object_is_outside_the_resolver_handle(tmp_path: Path) -> None:
    item = replace(
        _resolved_input(tmp_path),
        states={"velocity": np.asarray(99.5)},
    )

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows([item], project=_project)

    assert caught.value.code is EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH


def test_post_resolution_mutation_is_not_a_content_authentication_claim(tmp_path: Path) -> None:
    item = _resolved_input(tmp_path, source_kind="evaluation_cache")
    item.states["velocity"][...] = 3.5

    assert project_evaluation_rows([item], project=_project) == ((0, 7.0, "feedback"),)


def test_forged_exact_type_handle_is_rejected(tmp_path: Path) -> None:
    genuine = _resolved_input(tmp_path)
    forged = object.__new__(ResolvedEvaluationStateHandle)
    object.__setattr__(forged, "source", genuine.evaluation_state_source)
    object.__setattr__(forged, "proof_kind", "manifest_keyed_cache")
    object.__setattr__(forged, "evaluation_manifest_authority", genuine.ref)
    object.__setattr__(forged, "states", genuine.states)
    item = replace(genuine, evaluation_state_handle=forged)

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows([item], project=_project)

    assert caught.value.code is EvaluationRowProjectionErrorCode.STATE_HANDLE_INVALID


def test_cross_row_source_retarget_is_rejected(tmp_path: Path) -> None:
    first = _resolved_input(tmp_path, target=0)
    second = _resolved_input(tmp_path, target=1)
    retargeted = replace(first, evaluation_state_source=second.evaluation_state_source)

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows([retargeted], project=_project)

    assert caught.value.code is EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: setattr(source, "cache_key", "retargeted"),
        lambda source: source.evaluation_manifest_authority.metadata.__setitem__(
            "manifest_sha256", "f" * 64
        ),
    ],
    ids=["top-level-source-field", "nested-authority-field"],
)
def test_in_place_source_mutation_breaks_handle_snapshot(
    tmp_path: Path,
    mutate: Any,
) -> None:
    item = _resolved_input(tmp_path)
    mutate(item.evaluation_state_source)

    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows([item], project=_project)

    assert caught.value.code is EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH


def test_projector_failure_is_reason_coded(tmp_path: Path) -> None:
    with pytest.raises(EvaluationRowProjectionError) as caught:
        project_evaluation_rows(
            [_resolved_input(tmp_path)],
            project=lambda _row: (_ for _ in ()).throw(RuntimeError("downstream")),
        )

    assert caught.value.code is EvaluationRowProjectionErrorCode.PROJECTOR_FAILED
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_public_projection_type_hints_resolve_at_runtime() -> None:
    assert set(get_type_hints(project_evaluation_rows)) == {"inputs", "project", "return"}
