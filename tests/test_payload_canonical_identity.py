"""PREFLIGHT payload-identity round-trip: staged bytes must be the canonical projection.

These tests gate the incident where a run-spec staged with explicit ``null`` fields
(serialized without ``exclude_none=True``) passed PREFLIGHT and only failed the billable
SMOKE payload-sha check, where the training executor recomputes the canonical
validate->dump round-trip. The fix single-sources the canonical projection and makes
PREFLIGHT run it over the actual staged artifact that SMOKE will consume.
"""

from __future__ import annotations

import hashlib

import pytest

import feedbax.training.authoring as authoring
import feedbax.training.executor as executor
import feedbax.training.preparation as preparation
from feedbax.contracts.spec_storage import (
    canonical_training_run_spec_bytes,
    canonical_training_run_spec_projection,
    canonical_training_run_spec_sha256,
    training_spec_canonical_bytes,
)
from feedbax.orchestration.stages import (
    STAGE_PREFLIGHT,
    OrchestrationStageError,
    StageEngine,
    run_preflight_checks,
)
from feedbax.orchestration.state import RunSetState, StageState

from tests.test_orchestration_core import FakeDriver, _bundle, _compiled_row
from tests.test_training_run_executor import _run_spec


def _checks_by_name(bundle) -> dict:
    return {check.name: check for check in run_preflight_checks(bundle)}


def _contains_null(value) -> bool:
    """Return whether a JSON-projected value carries any explicit null anywhere."""
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_null(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_null(child) for child in value)
    return False


def test_single_source_projection_matches_executor_smoke_computation() -> None:
    """The single canonical source reproduces the executor's SMOKE-side payload sha."""
    spec = _run_spec()

    # The executor's _validate_execution_payload_binding recomputes the payload identity
    # as sha256(canonical_bytes(validate->dump exclude_none)). The single source must be
    # byte-for-byte identical so producer and verifier cannot diverge.
    expected_projection = spec.model_dump(mode="json", exclude_none=True)
    expected_bytes = training_spec_canonical_bytes(expected_projection)
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()

    assert canonical_training_run_spec_projection(spec) == expected_projection
    assert canonical_training_run_spec_bytes(spec) == expected_bytes
    assert canonical_training_run_spec_sha256(spec) == expected_sha

    # A mapping that validates as a TrainingRunSpec projects identically, so a staged
    # payload with explicit nulls is normalized to the same canonical identity.
    payload_with_nulls = spec.model_dump(mode="json")
    assert _contains_null(payload_with_nulls)
    assert canonical_training_run_spec_sha256(payload_with_nulls) == expected_sha


def test_single_source_is_shared_by_every_call_site() -> None:
    """Payload staging, PREFLIGHT, and the executor import the one canonical source."""
    assert (
        authoring.canonical_training_run_spec_projection
        is canonical_training_run_spec_projection
    )
    assert (
        executor.canonical_training_run_spec_projection
        is canonical_training_run_spec_projection
    )
    assert (
        preparation.canonical_training_run_spec_sha256 is canonical_training_run_spec_sha256
    )
    # stages.py consumes the bytes helper for the PREFLIGHT canonical-identity check.
    import feedbax.orchestration.stages as stages

    assert stages.canonical_training_run_spec_bytes is canonical_training_run_spec_bytes


def test_preflight_fails_when_staged_payload_has_explicit_nulls(tmp_path) -> None:
    """A run-spec staged with explicit nulls fails the named PREFLIGHT identity check."""
    spec = _run_spec()
    payload_with_nulls = spec.model_dump(mode="json")
    assert _contains_null(payload_with_nulls)

    bundle = _bundle(tmp_path, rows=[_compiled_row("row-a", run_spec=payload_with_nulls)])

    checks = _checks_by_name(bundle)
    identity = checks["payload-canonical-identity"]
    assert identity.status == "fail"
    assert "row-a" in (identity.detail or "")
    assert "explicit null fields present" in (identity.detail or "")

    # The gap this closes: normalization alone accepts the null-bearing payload, so the
    # divergence would otherwise only surface at the billable SMOKE stage.
    assert checks["manifest-payload-normalization"].status == "pass"

    # The recorded digest is the null-bearing bytes; the canonical projection differs.
    recorded = bundle.rows[0].execution.payload.sha256
    assert identity.observed["row-a"]["recorded_sha256"] == recorded
    assert identity.observed["row-a"]["canonical_sha256"] == canonical_training_run_spec_sha256(
        spec
    )
    assert recorded != canonical_training_run_spec_sha256(spec)


def test_preflight_passes_and_digest_equals_smoke_computation(tmp_path) -> None:
    """A canonically staged spec passes PREFLIGHT with the SMOKE-side digest."""
    spec = _run_spec()
    canonical_payload = canonical_training_run_spec_projection(spec)
    # A canonically staged payload is a fixed point of the projection: re-projecting it
    # (validate->dump exclude_none) does not change its bytes. Data nulls that live inside
    # free-form dict field values are legitimate and must survive unchanged.
    assert canonical_training_run_spec_bytes(canonical_payload) == training_spec_canonical_bytes(
        canonical_payload
    )

    bundle = _bundle(tmp_path, rows=[_compiled_row("row-a", run_spec=canonical_payload)])

    identity = _checks_by_name(bundle)["payload-canonical-identity"]
    assert identity.status == "pass"

    ref = bundle.rows[0].execution.payload
    # The digest PREFLIGHT validated is exactly what the executor recomputes at SMOKE.
    assert ref.sha256 == canonical_training_run_spec_sha256(spec)
    assert identity.observed["row-a"]["recorded_sha256"] == ref.sha256
    assert identity.observed["row-a"]["canonical_sha256"] == ref.sha256


def _preflight_completed_state(bundle, digests: dict[str, str]) -> RunSetState:
    return RunSetState(
        run_set_id=bundle.run_set_id,
        stages={
            STAGE_PREFLIGHT: StageState(
                status="completed",
                outputs={"payload_canonical_digests": digests},
            )
        },
    )


def test_smoke_rejects_payload_restaged_after_preflight(tmp_path) -> None:
    """SMOKE fails closed if a row payload digest changed since PREFLIGHT validated it."""
    spec = _run_spec()
    bundle = _bundle(tmp_path, rows=[_compiled_row("row-a", run_spec=canonical_training_run_spec_projection(spec))])
    engine = StageEngine(bundle=bundle, driver=FakeDriver())
    recorded = bundle.rows[0].execution.payload.sha256

    # Matching digest carried forward is accepted.
    engine._assert_preflight_payload_handoff(_preflight_completed_state(bundle, {"row-a": recorded}))

    # A digest that no longer matches the staged payload means it was restaged.
    with pytest.raises(OrchestrationStageError) as raised:
        engine._assert_preflight_payload_handoff(
            _preflight_completed_state(bundle, {"row-a": "0" * 64})
        )
    assert "diverged from the PREFLIGHT-validated digest" in str(raised.value)
    assert "row-a" in str(raised.value)

    # Legacy state without recorded digests does not spuriously fail.
    engine._assert_preflight_payload_handoff(
        RunSetState(run_set_id=bundle.run_set_id)
    )
