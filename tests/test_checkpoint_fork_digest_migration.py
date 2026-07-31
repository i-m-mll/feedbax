"""Acceptance suite for sanctioned checkpoint fork derived-digest migration.

These tests cover the versioned-verification-and-migration contract from
``docs/design/derived_digest_migration.md`` (Mandible feedbax/8cab07a). They are
xdist-safe: every filesystem write lands under ``tmp_path`` and no process-global
state is mutated without ``monkeypatch`` restoration.
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any

import pytest

import feedbax.__main__ as feedbax_main
import feedbax.training.checkpoint_custody as custody_module
from feedbax.contracts.checkpoints import (
    CHECKPOINT_FORK_PLAN_SCHEMA_VERSION,
    CheckpointForkCompatibilityProjection,
    CheckpointForkPlan,
    CheckpointForkSourcePreparation,
    CheckpointForkTarget,
    RunContractBinding,
)
from feedbax.contracts.training import (
    TRAINING_RUN_SPEC_SCHEMA_ID,
    TRAINING_RUN_SPEC_SCHEMA_VERSION_V3,
    TrainingRunSpec,
)
from feedbax.training import (
    CheckpointDigestMigrationRequired,
    CheckpointForkPlanBindings,
    RunContractProjectionEvidence,
    classify_checkpoint_fork_plan_derived_digests,
    derive_checkpoint_fork_compatibility_projection,
    fork_checkpoint_plan,
    migrate_checkpoint_fork_plan_derived_digests,
    relock_checkpoint_fork_plan_file,
    run_contract_binding,
)
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    _RUN_CONTRACT_BINDING_ALGORITHM_V2,
    _RUN_CONTRACT_BINDING_ALGORITHM_V3,
    _RUN_CONTRACT_BINDING_ALGORITHM_V4,
    _RUN_CONTRACT_HASH_DOMAIN,
    _RUN_CONTRACT_MIGRATION_EDGES,
    _compatible_stored_canonical_projection,
)
from tests.test_checkpoint_custody import _minimax_slots, _run_spec
from tests.checkpoint_minimax_plugin import PLUGIN_MODULE as CHECKPOINT_MINIMAX_PLUGIN

pytestmark = pytest.mark.migration_contract


def _phase_program(run_spec: TrainingRunSpec) -> Any:
    return run_spec.worker_execution.method_contract.phase_program


def _drifted_run_spec(run_spec: TrainingRunSpec, *, learning_rate_delta: float) -> TrainingRunSpec:
    """Return a run spec whose science (learning rate) differs, ABI unchanged."""
    training_config = run_spec.training_config.model_copy(
        update={"learning_rate": run_spec.training_config.learning_rate + learning_rate_delta}
    )
    return run_spec.model_copy(update={"training_config": training_config})


def _historical_v3_evidence(
    run_spec: TrainingRunSpec,
) -> tuple[dict[str, Any], str, RunContractProjectionEvidence]:
    """Build an authenticated historical v3 run-contract projection for ``run_spec``."""
    projection = copy.deepcopy(
        run_contract_binding(run_spec, _phase_program(run_spec)).canonical_projection
    )
    assert projection is not None
    # Historical projection envelopes are stamped v2; the binding-level version
    # distinguishes v2 from v3. v3 records carry a v3 TrainingRunSpec with launch
    # policy that the forward migration strips.
    projection["algorithm_version"] = _RUN_CONTRACT_BINDING_ALGORITHM_V2
    projection["training_run_spec"]["schema_version"] = TRAINING_RUN_SPEC_SCHEMA_VERSION_V3
    projection["training_run_spec"]["execution"] = {}
    recorded_sha256 = custody_module._run_contract_hash(projection)
    evidence = RunContractProjectionEvidence(
        canonical_projection=projection,
        recorded_sha256=recorded_sha256,
        recorded_algorithm_version=_RUN_CONTRACT_BINDING_ALGORITHM_V3,
        training_run_spec_schema_id=TRAINING_RUN_SPEC_SCHEMA_ID,
        training_run_spec_schema_version=TRAINING_RUN_SPEC_SCHEMA_VERSION_V3,
        evidence_refs=("custody://transaction/source",),
    )
    return projection, recorded_sha256, evidence


def _stored_v3_projection(
    current_run_spec: TrainingRunSpec,
    slots: dict[str, object],
    recorded_sha256: str,
    *,
    update: dict[str, Any] | None = None,
) -> CheckpointForkCompatibilityProjection:
    """Build a stored fork lock projection pinned to a v3 run-contract digest."""
    candidate = derive_checkpoint_fork_compatibility_projection(
        current_run_spec, _phase_program(current_run_spec), slots
    )
    pinned = {
        "run_contract_algorithm_version": _RUN_CONTRACT_BINDING_ALGORITHM_V3,
        "run_contract_projection_sha256": recorded_sha256,
    }
    if update:
        pinned.update(update)
    return candidate.model_copy(update=pinned)


def _fork_plan(
    stored_projections: dict[str, CheckpointForkCompatibilityProjection],
) -> CheckpointForkPlan:
    targets = [
        CheckpointForkTarget(
            target_id=target_id,
            row_id=f"row-{target_id}",
            checkpoint_root_ref=f"root-{target_id}",
            run_spec_ref="run",
            slot_template_ref="slots",
            compatibility=projection,
        )
        for target_id, projection in stored_projections.items()
    ]
    return CheckpointForkPlan(
        source=CheckpointForkSourcePreparation(checkpoint_root_ref="source"),
        targets=targets,
    )


def _bindings(
    current_run_spec: TrainingRunSpec,
    slots: dict[str, object],
    plan: CheckpointForkPlan,
    tmp_path: Path,
) -> CheckpointForkPlanBindings:
    roots: dict[str, Path] = {"source": tmp_path / "source"}
    for target in plan.targets:
        roots[target.checkpoint_root_ref] = tmp_path / target.checkpoint_root_ref
    return CheckpointForkPlanBindings(
        checkpoint_roots=roots,
        run_specs={"run": current_run_spec},
        slot_templates={"slots": slots},
    )


def test_verification_signals_migration_and_authenticated_migration_succeeds(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    _, recorded_sha256, evidence = _historical_v3_evidence(run_spec)
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    # Verification (the fork gate, without evidence) advertises the migration.
    with pytest.raises(CheckpointDigestMigrationRequired) as excinfo:
        fork_checkpoint_plan(plan, bindings)
    message = str(excinfo.value)
    assert _RUN_CONTRACT_BINDING_ALGORITHM_V3 in message
    assert _RUN_CONTRACT_BINDING_ALGORITHM_V4 in message
    assert "checkpoint relock" in message
    # The evidence-free gate must not assert the drift is algorithm-only.
    assert "algorithm-only" not in message
    assert "input drift cannot be ruled out" in message

    classification = classify_checkpoint_fork_plan_derived_digests(plan, bindings)
    assert classification.classifications[0].is_migratable_algorithm_drift

    result = migrate_checkpoint_fork_plan_derived_digests(
        plan,
        bindings,
        requalification_requirements=["re-run mapped_fork_proof rehearsal"],
        historical_evidence={"target-a": evidence},
    )
    assert result.record.proof_mode == "authenticated-historical-evidence"
    assert result.record.source_algorithm_version == _RUN_CONTRACT_BINDING_ALGORITHM_V3
    assert result.record.target_algorithm_version == _RUN_CONTRACT_BINDING_ALGORITHM_V4
    migrated_projection = result.plan.targets[0].compatibility
    assert migrated_projection.run_contract_algorithm_version == _RUN_CONTRACT_BINDING_ALGORITHM_V4
    candidate = derive_checkpoint_fork_compatibility_projection(
        run_spec, _phase_program(run_spec), slots
    )
    assert migrated_projection.run_contract_projection_sha256 == (
        candidate.run_contract_projection_sha256
    )

    # Re-verification passes strict equality under v4.
    reverify = classify_checkpoint_fork_plan_derived_digests(result.plan, bindings)
    assert reverify.classifications[0].is_unchanged


def test_invisible_run_spec_drift_refuses_migration(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    # Evidence built from a science-drifted spec: it authenticates but forward
    # migrates to a projection that does not match the current one.
    drifted = _drifted_run_spec(run_spec, learning_rate_delta=0.001)
    _, recorded_sha256, evidence = _historical_v3_evidence(drifted)
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    with pytest.raises(CheckpointCompatibilityError, match="input drift"):
        migrate_checkpoint_fork_plan_derived_digests(
            plan,
            bindings,
            requalification_requirements=["re-run rehearsal"],
            historical_evidence={"target-a": evidence},
        )


def test_migration_refuses_authored_field_mutation(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    _, recorded_sha256, evidence = _historical_v3_evidence(run_spec)
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    # A successful migration preserves every authored plan field (only the
    # derived digests and the provenance migration_history change).
    result = migrate_checkpoint_fork_plan_derived_digests(
        plan,
        bindings,
        requalification_requirements=["re-run rehearsal"],
        historical_evidence={"target-a": evidence},
    )

    def _authored_without_history(fork_plan: CheckpointForkPlan) -> dict[str, Any]:
        projection = custody_module._checkpoint_fork_authored_projection(fork_plan)
        projection.pop("migration_history", None)
        return projection

    assert _authored_without_history(result.plan) == _authored_without_history(plan)

    # The invariance guard the migration routes through refuses an authored change.
    tampered_target = plan.targets[0].model_copy(
        update={"compatibility": result.plan.targets[0].compatibility, "row_id": "tampered-row"}
    )
    with pytest.raises(CheckpointCompatibilityError, match="authored-field change"):
        custody_module._replace_checkpoint_fork_compatibility_locks(plan, [tampered_target])


def test_algorithm_bump_with_real_input_drift_refuses(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    drifted = _drifted_run_spec(run_spec, learning_rate_delta=0.002)
    _, recorded_sha256, evidence = _historical_v3_evidence(drifted)
    # Sentinels stay identical (slot ABI/population are run-spec independent);
    # only the run-contract digest carries the older algorithm version.
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    # The evidence-free fork gate cannot detect the hidden science drift: it
    # raises the migration-required signal on the clean-sentinel version drift
    # without asserting the drift is algorithm-only (spec §4.3, §3.1).
    with pytest.raises(CheckpointDigestMigrationRequired) as excinfo:
        fork_checkpoint_plan(plan, bindings)
    gate_message = str(excinfo.value)
    assert "algorithm-only" not in gate_message
    assert "input drift cannot be ruled out" in gate_message

    # The evidence-bearing classify/migrate layer adjudicates and refuses.
    classification = classify_checkpoint_fork_plan_derived_digests(
        plan, bindings, historical_evidence={"target-a": evidence}
    )
    assert classification.classifications[0].run_contract_class == "input_drift"
    assert classification.classifications[0].has_input_drift

    with pytest.raises(CheckpointCompatibilityError, match="input drift"):
        migrate_checkpoint_fork_plan_derived_digests(
            plan,
            bindings,
            requalification_requirements=["re-run rehearsal"],
            historical_evidence={"target-a": evidence},
        )


def test_mixed_sentinel_and_algorithm_drift_refuses(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    _, recorded_sha256, evidence = _historical_v3_evidence(run_spec)
    candidate = derive_checkpoint_fork_compatibility_projection(
        run_spec, _phase_program(run_spec), slots
    )
    tampered_slots = dict(candidate.slot_structural_abi_sha256)
    (first_slot,) = list(tampered_slots)[:1]
    tampered_slots[first_slot] = "f" * 64
    stored = _stored_v3_projection(
        run_spec,
        slots,
        recorded_sha256,
        update={"slot_structural_abi_sha256": tampered_slots},
    )
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    # The fork gate raises the compatibility error, never the migration signal.
    with pytest.raises(CheckpointCompatibilityError) as excinfo:
        fork_checkpoint_plan(plan, bindings)
    assert not isinstance(excinfo.value, CheckpointDigestMigrationRequired)

    with pytest.raises(CheckpointCompatibilityError, match="input drift"):
        migrate_checkpoint_fork_plan_derived_digests(
            plan,
            bindings,
            requalification_requirements=["re-run rehearsal"],
            historical_evidence={"target-a": evidence},
        )


def test_current_version_run_contract_mismatch_is_input_drift(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    candidate = derive_checkpoint_fork_compatibility_projection(
        run_spec, _phase_program(run_spec), slots
    )
    # Same (current) algorithm version but a different projection digest.
    stored = candidate.model_copy(update={"run_contract_projection_sha256": "a" * 64})
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    classification = classify_checkpoint_fork_plan_derived_digests(plan, bindings)
    assert classification.classifications[0].run_contract_class == "input_drift"

    with pytest.raises(CheckpointCompatibilityError) as excinfo:
        fork_checkpoint_plan(plan, bindings)
    assert not isinstance(excinfo.value, CheckpointDigestMigrationRequired)


@pytest.mark.parametrize(
    "update",
    [
        {"run_contract_algorithm_version": "feedbax.training_checkpoint.run_contract_binding.v99"},
        {"run_contract_hash_domain": "some-other-hash-domain"},
    ],
)
def test_unsupported_edge_fails_closed(update: dict[str, Any], tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    candidate = derive_checkpoint_fork_compatibility_projection(
        run_spec, _phase_program(run_spec), slots
    )
    stored = candidate.model_copy(
        update={"run_contract_projection_sha256": "b" * 64, **update}
    )
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    classification = classify_checkpoint_fork_plan_derived_digests(plan, bindings)
    assert classification.classifications[0].run_contract_class == "unsupported"

    with pytest.raises(CheckpointCompatibilityError, match="unsupported") as excinfo:
        fork_checkpoint_plan(plan, bindings)
    assert not isinstance(excinfo.value, CheckpointDigestMigrationRequired)

    with pytest.raises(CheckpointCompatibilityError, match="unsupported"):
        migrate_checkpoint_fork_plan_derived_digests(
            plan,
            bindings,
            requalification_requirements=["re-run rehearsal"],
        )


def test_missing_and_invalid_evidence_refused_and_attestation_opt_in(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    _, recorded_sha256, _ = _historical_v3_evidence(run_spec)
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    # Missing evidence: authenticated migration refuses.
    with pytest.raises(CheckpointCompatibilityError, match="authenticated historical evidence"):
        migrate_checkpoint_fork_plan_derived_digests(
            plan,
            bindings,
            requalification_requirements=["re-run rehearsal"],
        )

    # Invalid evidence (wrong recorded digest): authenticated migration refuses.
    invalid_evidence = RunContractProjectionEvidence(
        canonical_projection={"schema_id": "wrong"},
        recorded_sha256="c" * 64,
        recorded_algorithm_version=_RUN_CONTRACT_BINDING_ALGORITHM_V3,
        training_run_spec_schema_id=TRAINING_RUN_SPEC_SCHEMA_ID,
        training_run_spec_schema_version=TRAINING_RUN_SPEC_SCHEMA_VERSION_V3,
    )
    with pytest.raises(CheckpointCompatibilityError, match="authenticated historical evidence"):
        migrate_checkpoint_fork_plan_derived_digests(
            plan,
            bindings,
            requalification_requirements=["re-run rehearsal"],
            historical_evidence={"target-a": invalid_evidence},
        )

    # Owner-attestation opt-in (no evidence) succeeds and records mode and duties.
    result = migrate_checkpoint_fork_plan_derived_digests(
        plan,
        bindings,
        requalification_requirements=["re-run fork-proof and rehearsal tests"],
        owner_attestation=True,
    )
    assert result.record.proof_mode == "owner-attestation"
    assert result.record.requalification_requirements == (
        "re-run fork-proof and rehearsal tests",
    )

    # Requalification duties are mandatory in every mode.
    with pytest.raises(CheckpointCompatibilityError, match="re-qualification"):
        migrate_checkpoint_fork_plan_derived_digests(
            plan,
            bindings,
            requalification_requirements=[],
            owner_attestation=True,
        )


def test_supplied_evidence_always_adjudicates_over_attestation_flag(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    _, recorded_sha256, valid_evidence = _historical_v3_evidence(run_spec)
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    # Evidence + flag + valid evidence: the authenticated proof mode is recorded,
    # the attestation flag does not downgrade it.
    result = migrate_checkpoint_fork_plan_derived_digests(
        plan,
        bindings,
        requalification_requirements=["re-run rehearsal"],
        historical_evidence={"target-a": valid_evidence},
        owner_attestation=True,
    )
    assert result.record.proof_mode == "authenticated-historical-evidence"

    # Evidence + flag + evidence that proves input drift: refuse even with the flag.
    drifted = _drifted_run_spec(run_spec, learning_rate_delta=0.004)
    _, drift_sha256, drift_evidence = _historical_v3_evidence(drifted)
    drift_stored = _stored_v3_projection(run_spec, slots, drift_sha256)
    drift_plan = _fork_plan({"target-a": drift_stored})
    drift_bindings = _bindings(run_spec, slots, drift_plan, tmp_path)
    with pytest.raises(CheckpointCompatibilityError, match="input drift"):
        migrate_checkpoint_fork_plan_derived_digests(
            drift_plan,
            drift_bindings,
            requalification_requirements=["re-run rehearsal"],
            historical_evidence={"target-a": drift_evidence},
            owner_attestation=True,
        )

    # Invalid evidence + flag: refuse (attestation applies only with no evidence).
    invalid_evidence = RunContractProjectionEvidence(
        canonical_projection={"schema_id": "wrong"},
        recorded_sha256="c" * 64,
        recorded_algorithm_version=_RUN_CONTRACT_BINDING_ALGORITHM_V3,
        training_run_spec_schema_id=TRAINING_RUN_SPEC_SCHEMA_ID,
        training_run_spec_schema_version=TRAINING_RUN_SPEC_SCHEMA_VERSION_V3,
    )
    with pytest.raises(CheckpointCompatibilityError, match="did not authenticate"):
        migrate_checkpoint_fork_plan_derived_digests(
            plan,
            bindings,
            requalification_requirements=["re-run rehearsal"],
            historical_evidence={"target-a": invalid_evidence},
            owner_attestation=True,
        )


def test_multi_target_all_or_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    _, recorded_sha256, evidence = _historical_v3_evidence(run_spec)
    stored_a = _stored_v3_projection(run_spec, slots, recorded_sha256)
    stored_b = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored_a, "target-b": stored_b})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    plan_path = tmp_path / "plan.json"
    original_bytes = json.dumps(
        plan.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True
    ).encode("utf-8")
    plan_path.write_bytes(original_bytes)

    def _fail_replace(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("forced mid-write failure")

    monkeypatch.setattr(custody_module.os, "replace", _fail_replace)
    with pytest.raises(RuntimeError, match="forced mid-write failure"):
        relock_checkpoint_fork_plan_file(
            plan_path,
            bindings,
            requalification_requirements=["re-run rehearsal"],
            historical_evidence={"target-a": evidence, "target-b": evidence},
        )

    # Neither the migrated digests nor a record persisted; the file is untouched.
    assert plan_path.read_bytes() == original_bytes
    reloaded = CheckpointForkPlan.model_validate(json.loads(plan_path.read_text()))
    assert reloaded.migration_history == []
    assert reloaded.targets[0].compatibility.run_contract_algorithm_version == (
        _RUN_CONTRACT_BINDING_ALGORITHM_V3
    )
    # No stray temp files remain in the plan directory.
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["plan.json"]


def test_record_binds_plan_hashes_and_survives_reload(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    _, recorded_sha256, evidence = _historical_v3_evidence(run_spec)
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    result = migrate_checkpoint_fork_plan_derived_digests(
        plan,
        bindings,
        requalification_requirements=["re-run rehearsal"],
        historical_evidence={"target-a": evidence},
    )
    record = result.record
    assert record.source_plan_canonical_sha256 == custody_module.checkpoint_fork_plan_sha256(plan)
    assert record.source_plan_canonical_sha256 != record.target_plan_canonical_sha256
    assert result.plan.schema_version == CHECKPOINT_FORK_PLAN_SCHEMA_VERSION

    reloaded = CheckpointForkPlan.model_validate(
        json.loads(json.dumps(result.plan.model_dump(mode="json", exclude_none=True)))
    )
    assert reloaded.migration_history == [record]
    assert reloaded.migration_history[0].source_plan_canonical_sha256 == (
        record.source_plan_canonical_sha256
    )

    # Hash self-reference convention: the target hash is the persisted plan's
    # canonical hash with its final migration-history entry stripped.
    stripped = reloaded.model_copy(
        update={"migration_history": reloaded.migration_history[:-1]}
    )
    assert record.target_plan_canonical_sha256 == custody_module.checkpoint_fork_plan_sha256(
        stripped
    )
    assert record.target_plan_canonical_sha256 != custody_module.checkpoint_fork_plan_sha256(
        reloaded
    )


def test_heterogeneous_stored_versions_refused(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    candidate = derive_checkpoint_fork_compatibility_projection(
        run_spec, _phase_program(run_spec), slots
    )
    # One target stored at v2, one at v3: both are supported migratable edges with
    # clean sentinels, so the plan-level op reaches the single-edge requirement.
    stored_v2 = candidate.model_copy(
        update={
            "run_contract_algorithm_version": _RUN_CONTRACT_BINDING_ALGORITHM_V2,
            "run_contract_projection_sha256": "a" * 64,
        }
    )
    stored_v3 = candidate.model_copy(
        update={
            "run_contract_algorithm_version": _RUN_CONTRACT_BINDING_ALGORITHM_V3,
            "run_contract_projection_sha256": "b" * 64,
        }
    )
    plan = _fork_plan({"target-a": stored_v2, "target-b": stored_v3})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    with pytest.raises(CheckpointCompatibilityError, match="one shared migration edge"):
        migrate_checkpoint_fork_plan_derived_digests(
            plan,
            bindings,
            requalification_requirements=["re-run rehearsal"],
            owner_attestation=True,
        )


def test_stale_concurrent_write_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    _, recorded_sha256, evidence = _historical_v3_evidence(run_spec)
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings = _bindings(run_spec, slots, plan, tmp_path)

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.model_dump(mode="json", exclude_none=True)))

    concurrent_plan = plan.model_copy(
        update={"targets": [plan.targets[0].model_copy(update={"row_id": "concurrently-changed"})]}
    )
    real_migrate = custody_module.migrate_checkpoint_fork_plan_derived_digests

    def _racing_migrate(*args: Any, **kwargs: Any):
        plan_path.write_text(json.dumps(concurrent_plan.model_dump(mode="json", exclude_none=True)))
        return real_migrate(*args, **kwargs)

    monkeypatch.setattr(
        custody_module, "migrate_checkpoint_fork_plan_derived_digests", _racing_migrate
    )
    with pytest.raises(CheckpointCompatibilityError, match="changed under the relock"):
        relock_checkpoint_fork_plan_file(
            plan_path,
            bindings,
            requalification_requirements=["re-run rehearsal"],
            historical_evidence={"target-a": evidence},
        )


def test_fork_and_resume_share_migration_edge_registry() -> None:
    edges = {
        (edge.source_algorithm_version, edge.target_algorithm_version, edge.hash_domain)
        for edge in _RUN_CONTRACT_MIGRATION_EDGES.edges()
    }
    assert edges == {
        (
            _RUN_CONTRACT_BINDING_ALGORITHM_V2,
            _RUN_CONTRACT_BINDING_ALGORITHM_V4,
            _RUN_CONTRACT_HASH_DOMAIN,
        ),
        (
            _RUN_CONTRACT_BINDING_ALGORITHM_V3,
            _RUN_CONTRACT_BINDING_ALGORITHM_V4,
            _RUN_CONTRACT_HASH_DOMAIN,
        ),
    }

    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    projection, recorded_sha256, evidence = _historical_v3_evidence(run_spec)
    current_binding = run_contract_binding(run_spec, _phase_program(run_spec))

    # The resume verifier accepts the same v3 -> v4 edge the fork classifier uses.
    recorded_binding = RunContractBinding(
        algorithm_version=_RUN_CONTRACT_BINDING_ALGORITHM_V3,
        training_run_spec_schema_id=TRAINING_RUN_SPEC_SCHEMA_ID,
        training_run_spec_schema_version=TRAINING_RUN_SPEC_SCHEMA_VERSION_V3,
        training_run_spec_sha256="d" * 64,
        method_payload_schema_id=current_binding.method_payload_schema_id,
        method_payload_schema_version=current_binding.method_payload_schema_version,
        method_payload_sha256="e" * 64,
        phase_program_sha256="f" * 64,
        canonical_projection=projection,
        canonical_projection_sha256=recorded_sha256,
    )
    assert _compatible_stored_canonical_projection(recorded_binding, current_binding)

    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    candidate = derive_checkpoint_fork_compatibility_projection(
        run_spec, _phase_program(run_spec), slots
    )
    classification = custody_module.classify_fork_compatibility_projection(
        stored,
        candidate,
        historical_evidence=evidence,
        current_canonical_projection=current_binding.canonical_projection,
    )
    assert classification.run_contract_class == "migratable_algorithm_drift"
    assert classification.migration_edge is not None
    assert classification.migration_edge.source_algorithm_version == (
        _RUN_CONTRACT_BINDING_ALGORITHM_V3
    )


def _write_cli_bindings(
    tmp_path: Path,
    current_run_spec: TrainingRunSpec,
    slots: dict[str, object],
    plan: CheckpointForkPlan,
) -> Path:
    run_spec_path = tmp_path / "run_spec.json"
    run_spec_path.write_text(json.dumps(current_run_spec.model_dump(mode="json", exclude_none=True)))
    slots_path = tmp_path / "slots.pkl"
    with slots_path.open("wb") as stream:
        pickle.dump(slots, stream)
    checkpoint_roots = {"source": str(tmp_path / "source")}
    for target in plan.targets:
        checkpoint_roots[target.checkpoint_root_ref] = str(tmp_path / target.checkpoint_root_ref)
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(
        json.dumps(
            {
                "schema_id": "feedbax.runtime.checkpoint_fork_plan_bindings",
                "schema_version": "feedbax.runtime.checkpoint_fork_plan_bindings.v1",
                "checkpoint_roots": checkpoint_roots,
                "run_specs": {"run": str(run_spec_path)},
                "slot_templates": {"slots": str(slots_path)},
            }
        )
    )
    return bindings_path


def _write_cli_evidence(tmp_path: Path, evidence: RunContractProjectionEvidence) -> Path:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "target-a": {
                    "canonical_projection": evidence.canonical_projection,
                    "recorded_sha256": evidence.recorded_sha256,
                    "recorded_algorithm_version": evidence.recorded_algorithm_version,
                    "training_run_spec_schema_id": evidence.training_run_spec_schema_id,
                    "training_run_spec_schema_version": evidence.training_run_spec_schema_version,
                    "evidence_refs": list(evidence.evidence_refs),
                }
            }
        )
    )
    return evidence_path


def test_cli_check_and_write_behaviors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    _, recorded_sha256, evidence = _historical_v3_evidence(run_spec)
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings_path = _write_cli_bindings(tmp_path, run_spec, slots, plan)
    evidence_path = _write_cli_evidence(tmp_path, evidence)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.model_dump(mode="json", exclude_none=True)))

    # --check reports drift and exits nonzero.
    exit_code = feedbax_main.main(
        ["checkpoint", "relock", "--plan", str(plan_path), "--bindings", str(bindings_path),
         "--plugin", CHECKPOINT_MINIMAX_PLUGIN,
         "--check"]
    )
    assert exit_code == 1
    check_report = json.loads(capsys.readouterr().out)
    assert check_report["status"] == "drift"
    assert check_report["targets"]["target-a"]["run_contract_class"] == "migratable_algorithm_drift"

    # --write migrates the plan file in place.
    exit_code = feedbax_main.main(
        ["checkpoint", "relock", "--plan", str(plan_path), "--bindings", str(bindings_path),
         "--plugin", CHECKPOINT_MINIMAX_PLUGIN,
         "--write", "--historical-evidence", str(evidence_path),
         "--requalification", "re-run rehearsal"]
    )
    assert exit_code == 0
    write_report = json.loads(capsys.readouterr().out)
    assert write_report["status"] == "migrated"
    assert write_report["record"]["proof_mode"] == "authenticated-historical-evidence"
    migrated_plan = CheckpointForkPlan.model_validate(json.loads(plan_path.read_text()))
    assert migrated_plan.targets[0].compatibility.run_contract_algorithm_version == (
        _RUN_CONTRACT_BINDING_ALGORITHM_V4
    )
    assert len(migrated_plan.migration_history) == 1

    # A clean re-check exits zero.
    exit_code = feedbax_main.main(
        ["checkpoint", "relock", "--plan", str(plan_path), "--bindings", str(bindings_path),
         "--plugin", CHECKPOINT_MINIMAX_PLUGIN,
         "--check"]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "clean"

    # A --write on the already-current plan refuses the no-op.
    with pytest.raises(CheckpointCompatibilityError, match="nothing to migrate"):
        feedbax_main.main(
            ["checkpoint", "relock", "--plan", str(plan_path), "--bindings", str(bindings_path),
             "--plugin", CHECKPOINT_MINIMAX_PLUGIN,
             "--write", "--requalification", "re-run rehearsal"]
        )


def test_cli_requires_exactly_one_mode(tmp_path: Path) -> None:
    # argparse fails before the handler when neither or both modes are given.
    with pytest.raises(SystemExit):
        feedbax_main.main(
            ["checkpoint", "relock", "--plan", "plan.json", "--bindings", "bindings.json"]
        )
    with pytest.raises(SystemExit):
        feedbax_main.main(
            ["checkpoint", "relock", "--plan", "plan.json", "--bindings", "bindings.json",
             "--check", "--write"]
        )


def test_cli_write_refuses_input_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    drifted = _drifted_run_spec(run_spec, learning_rate_delta=0.003)
    _, recorded_sha256, evidence = _historical_v3_evidence(drifted)
    stored = _stored_v3_projection(run_spec, slots, recorded_sha256)
    plan = _fork_plan({"target-a": stored})
    bindings_path = _write_cli_bindings(tmp_path, run_spec, slots, plan)
    evidence_path = _write_cli_evidence(tmp_path, evidence)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.model_dump(mode="json", exclude_none=True)))

    with pytest.raises(CheckpointCompatibilityError, match="input drift"):
        feedbax_main.main(
            ["checkpoint", "relock", "--plan", str(plan_path), "--bindings", str(bindings_path),
             "--plugin", CHECKPOINT_MINIMAX_PLUGIN,
             "--write", "--historical-evidence", str(evidence_path),
             "--requalification", "re-run rehearsal"]
        )
    # The plan file is untouched by the refused write.
    unchanged = CheckpointForkPlan.model_validate(json.loads(plan_path.read_text()))
    assert unchanged.migration_history == []
