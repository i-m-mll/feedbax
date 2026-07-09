from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from feedbax.contracts.manifest import TrainingRunManifest, spec_payload
from feedbax.contracts.migrations import default_spec_registry, migrate_structured_spec_payload
from feedbax.contracts.training import LrScheduleSpec, OptimizerSpec
from feedbax.orchestration.conformance import (
    RUN_CONFORMANCE_SCHEMA_ID,
    RUN_CONFORMANCE_SCHEMA_VERSION,
    CheckEntry,
    CheckRegistry,
    ConformanceRowArtifacts,
    RunConformanceCertificate,
    assert_certificate_allows_completed_registration,
    build_core_check_registry,
    check_completed_batches,
    check_events_terminal,
    check_lr_trace,
    check_manifest_valid,
    missing_input_check,
    pass_check,
    run_conformance_checks,
    write_conformance_certificate,
)
from feedbax.plugins.discovery import load_conformance_check_plugins


pytestmark = pytest.mark.feedbax_contract


GENERATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _row(**updates: object) -> ConformanceRowArtifacts:
    base = {
        "row_id": "row-a",
        "bundle_row_spec": {
            "n_batches": 10,
            "checkpoint_interval": 5,
            "seeds": {"controller": 123},
        },
        "training_diagnostics": {
            "completed_batches": 10,
            "checkpoint_coordinates": [5, 10],
            "seeds": {"controller": 123},
        },
        "manifest_payload": {
            "kind": "TrainingRunManifest",
            "metadata": {
                "environment_fingerprint": {"python": "3.12", "jax": "0.test"},
                "seeds": {"controller": 123},
            },
            "training_spec": {"inline": {"method_payload": {"payload": {}}}},
            "summary_metrics": {"completed_batches": 10},
        },
        "recorded_environment_fingerprint": {"python": "3.12", "jax": "0.test"},
        "preflight_normalized_payload": {"method_payload": {"payload": {}}},
    }
    base.update(updates)
    return ConformanceRowArtifacts(**base)


def test_certificate_assembly_rules_and_deterministic_write(tmp_path: Path) -> None:
    registry = CheckRegistry(
        {
            "z_pass": lambda row: pass_check("z_pass"),
            "a_missing": lambda row: missing_input_check("a_missing", "required"),
            "m_skip": lambda row: CheckEntry(
                check_id="m_skip",
                status="skipped",
                detail="not applicable for fixture",
            ),
        }
    )

    certificate = write_conformance_certificate(
        run_set_dir=tmp_path,
        run_set_id="run-set-a",
        rows=[_row(row_id="row-b"), _row(row_id="row-a")],
        registry=registry,
        generated_at=GENERATED_AT,
    )

    assert certificate.overall == "fail"
    assert list(certificate.rows) == ["row-a", "row-b"]
    assert [check.check_id for check in certificate.rows["row-a"].checks] == [
        "a_missing",
        "m_skip",
        "z_pass",
    ]
    written = (tmp_path / "conformance.json").read_text(encoding="utf-8")
    assert written == json.dumps(certificate.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    with pytest.raises(ValueError, match="skipped conformance checks require a detail"):
        CheckEntry(check_id="bad_skip", status="skipped")


def test_schema_round_trip_and_registry_identity() -> None:
    payload = {
        "schema_id": RUN_CONFORMANCE_SCHEMA_ID,
        "schema_version": RUN_CONFORMANCE_SCHEMA_VERSION,
        "run_set_id": "run-set-a",
        "generated_at": GENERATED_AT.isoformat(),
        "overall": "pass",
        "rows": {"row-a": {"checks": [{"check_id": "ok", "status": "pass"}]}},
    }

    certificate = RunConformanceCertificate.model_validate(payload)
    assert certificate.model_dump(mode="json")["schema_version"] == RUN_CONFORMANCE_SCHEMA_VERSION

    family = default_spec_registry.resolve("RunConformanceCertificate")
    assert family.identity == RUN_CONFORMANCE_SCHEMA_ID
    assert family.current_version == RUN_CONFORMANCE_SCHEMA_VERSION
    migrated = migrate_structured_spec_payload("RunConformanceCertificate", payload)
    assert migrated.payload == payload


def test_core_checks_pass_fail_and_missing_inputs() -> None:
    assert check_completed_batches(_row()).status == "pass"
    assert check_completed_batches(
        _row(training_diagnostics={"completed_batches": 9})
    ).status == "fail"
    missing = check_completed_batches(_row(bundle_row_spec={}))
    assert missing.status == "fail"
    assert "missing required input" in str(missing.detail)

    registry = build_core_check_registry()
    certificate = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[_row(event_log=None)],
        registry=registry,
        generated_at=GENERATED_AT,
    )

    checks = {check.check_id: check for check in certificate.rows["row-a"].checks}
    assert checks["completed_batches"].status == "pass"
    assert checks["checkpoint_cadence"].status == "pass"
    assert checks["environment_fingerprint"].status == "pass"
    assert checks["seeds"].status == "pass"
    assert checks["events_terminal"].status == "skipped"
    assert checks["events_terminal"].detail


def test_manifest_valid_loads_manifest_and_compares_preflight_payload(tmp_path: Path) -> None:
    training_payload = {"method_payload": {"payload": {"optimizer": {"type": "adamw"}}}}
    manifest = TrainingRunManifest(
        id="feedbax-training-run:row-a",
        training_spec=spec_payload("TrainingRunSpec", training_payload),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json", exclude_none=True), sort_keys=True),
        encoding="utf-8",
    )
    registry = CheckRegistry({"manifest_valid": check_manifest_valid})

    certificate = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[
            _row(
                manifest_path=manifest_path,
                manifest_payload=None,
                preflight_normalized_payload=training_payload,
            )
        ],
        registry=registry,
        generated_at=GENERATED_AT,
    )

    assert certificate.rows["row-a"].checks[0].status == "pass"


@pytest.mark.parametrize("expected_status", ["complete", "completed", "done"])
def test_events_terminal_complete_event_matches_success_aliases(
    expected_status: str,
) -> None:
    result = check_events_terminal(
        _row(
            event_log=[
                {"type": "started"},
                {"type": "complete"},
            ],
            bundle_row_spec={"expected_terminal_status": expected_status},
        )
    )

    assert result.status == "pass"


def test_events_terminal_legacy_skip_and_terminal_validation() -> None:
    assert check_events_terminal(_row(event_log=None)).status == "skipped"
    passed = check_events_terminal(
        _row(
            event_log=[
                {"type": "started"},
                {"type": "completed"},
            ],
            bundle_row_spec={"expected_terminal_status": "completed"},
        )
    )
    failed = check_events_terminal(
        _row(
            event_log=[
                {"type": "completed"},
                {"type": "failed"},
            ],
        )
    )

    assert passed.status == "pass"
    assert failed.status == "fail"


@pytest.mark.parametrize(
    ("event_status", "expected_status"),
    [
        ("complete", "failed"),
        ("failed", "complete"),
    ],
)
def test_events_terminal_fails_success_failure_disagreement(
    event_status: str,
    expected_status: str,
) -> None:
    result = check_events_terminal(
        _row(
            event_log=[{"type": event_status}],
            bundle_row_spec={"expected_terminal_status": expected_status},
        )
    )

    assert result.status == "fail"


def test_lr_trace_uses_optimizer_builder_resume_context_and_rejects_flat_terminal_lr() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=10,
        constant_lr_iterations=4,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.2,
    )
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=schedule,
    ).model_dump(mode="json")

    passing_row = _row(
        bundle_row_spec={
            "optimizer": optimizer,
            "resume_context": {
                "schedule_origin_step": 100,
                "current_step": 100,
                "optimizer_count_at_current_step": 12_000,
            },
        },
        training_diagnostics={
            "lr_trace": {
                100: 0.01,
                104: 0.1,
                110: 0.02,
            }
        },
    )
    failing_row = _row(
        bundle_row_spec=passing_row.bundle_row_spec,
        training_diagnostics={
            "lr_trace": {
                100: 0.02,
                104: 0.02,
                110: 0.02,
            }
        },
    )

    assert check_lr_trace(passing_row).status == "pass"
    failed = check_lr_trace(failing_row)
    assert failed.status == "fail"
    assert failed.observed[100] == pytest.approx(0.02)


def test_plugin_check_discovery_and_failure_propagation() -> None:
    def plugin_check(row: ConformanceRowArtifacts) -> CheckEntry:
        return CheckEntry(
            check_id="project_check",
            status="fail",
            expected="project invariant",
            observed="broken",
        )

    class FakeEntryPoint:
        name = "fixture"

        def load(self) -> object:
            return SimpleNamespace(
                feedbax_conformance_checks=lambda: [("project_check", plugin_check)]
            )

    registry = CheckRegistry()
    load_conformance_check_plugins(registry=registry, entry_points=[FakeEntryPoint()])
    certificate = run_conformance_checks(
        run_set_id="run-set-a",
        rows=[_row()],
        registry=registry,
        generated_at=GENERATED_AT,
    )

    assert certificate.overall == "fail"
    assert certificate.rows["row-a"].checks[0].check_id == "project_check"
    assert certificate.rows["row-a"].checks[0].status == "fail"


def test_conformance_plugin_discovery_ignores_non_conformance_registrars() -> None:
    def experiment_registrar(registry: object) -> None:
        raise AssertionError("non-conformance registrar should not be called")

    class FakeEntryPoint:
        name = "experiment"

        def load(self) -> object:
            return experiment_registrar

    registry = CheckRegistry()
    load_conformance_check_plugins(registry=registry, entry_points=[FakeEntryPoint()])

    assert len(registry) == 0


def test_register_coupling_rejects_failing_certificate() -> None:
    certificate = RunConformanceCertificate(
        run_set_id="run-set-a",
        generated_at=GENERATED_AT,
        overall="fail",
        rows={"row-a": {"checks": [{"check_id": "bad", "status": "fail"}]}},
    )

    with pytest.raises(ValueError, match="phase=completed"):
        assert_certificate_allows_completed_registration(certificate)
