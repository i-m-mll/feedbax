from __future__ import annotations

import json
from pathlib import Path

from feedbax.bin import orchestrate
from feedbax.contracts.studio_training import StudioTrainingAssemblySpec
from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.worker import (
    AxisSpec,
    MethodTrainingDiagnosticsSpec,
    PhaseSpec,
    SlotAxisBindingSpec,
    StateSlotSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
)
from feedbax.orchestration.payload_report import (
    MeasurementBindingExpectation,
    assert_measurement_binding,
    build_payload_report,
    format_measurement_assertion_mismatches,
)
from feedbax.orchestration import CompiledExecutionRow, RowLaunchSpec
from tests.test_orchestration_cli import _standard_training_run_payload, _write_bundle
from tests.test_orchestration_core import _bundle, _compiled_row


TRACE_SCHEMA_ID = "feedbax.tests.training_batch_measurement"
TRACE_SCHEMA_VERSION = "feedbax.tests.training_batch_measurement.v1"


def _payload(
    *,
    trace_schema_id: str = TRACE_SCHEMA_ID,
    trace_schema_version: str = TRACE_SCHEMA_VERSION,
) -> dict[str, object]:
    payload = _standard_training_run_payload()
    run_spec = TrainingRunSpec.model_validate(payload)
    contract = run_spec.worker_execution.method_contract
    measurement_step = UpdateStepSpec(
        name="training_batch_measurement",
        kind="measurement",
        kernel=UpdateKernelSpec(kernel_ref="tests.measure_training_batch"),
        reads=["model"],
        writes=["measurement_metric"],
    )
    evaluation_phase = PhaseSpec(
        name="heldout_evaluation",
        kind="evaluation",
        reads=["measurement_metric"],
        metadata={
            "evaluation_contract": {
                "schema_id": "feedbax.tests.evaluation_contract",
                "schema_version": "feedbax.tests.evaluation_contract.v1",
            }
        },
    )
    phase_program = contract.phase_program.model_copy(
        update={
            "phases": [*contract.phase_program.phases, evaluation_phase],
            "update_steps": [*contract.phase_program.update_steps, measurement_step],
        }
    )
    contract = contract.model_copy(
        update={
            "axes": [*contract.axes, AxisSpec(name="replica", role="replicate", size=2)],
            "state_slots": [
                *contract.state_slots,
                StateSlotSpec(
                    name="measurement_metric",
                    role="metric",
                    axis_bindings=[
                        SlotAxisBindingSpec(axis="replica", mode="mapped", array_axis=0)
                    ],
                ),
            ],
            "phase_program": phase_program,
            "training_diagnostics": MethodTrainingDiagnosticsSpec(
                trace_schema_id=trace_schema_id,
                trace_schema_version=trace_schema_version,
                measurement_basis="training_batch",
                metric_payload_slot="measurement_metric",
                replica_axis="replica",
            ),
        }
    )
    graph = run_spec.graph.model_copy(
        update={
            "schema_id": "feedbax.tests.model_contract",
            "schema_version": "feedbax.tests.model_contract.v1",
        }
    )
    updated = run_spec.model_copy(
        update={
            "graph": graph,
            "worker_execution": run_spec.worker_execution.model_copy(
                update={"method_contract": contract}
            ),
        }
    )
    return updated.model_dump(mode="json", exclude_none=True)


def _report_bundle(tmp_path: Path, *rows: tuple[str, dict[str, object]]):
    compiled_rows = [
        _compiled_row(
            row_id,
            run_spec={
                key: value
                for key, value in payload.items()
                if key not in {"schema_id", "schema_version"}
            },
        )
        for row_id, payload in rows
    ]
    return _bundle(tmp_path, rows=compiled_rows)


def test_payload_report_extracts_binding_and_is_deterministic(tmp_path: Path) -> None:
    bundle = _report_bundle(tmp_path, ("row-b", _payload()), ("row-a", _payload()))

    first = build_payload_report(bundle)
    second = build_payload_report(bundle)

    assert first.model_dump_json(exclude_none=True) == second.model_dump_json(exclude_none=True)
    assert [row.row_id for row in first.rows] == ["row-a", "row-b"]
    row = first.rows[0]
    assert row.model_contract is not None
    assert row.model_contract.model_dump() == {
        "schema_id": "feedbax.tests.model_contract",
        "schema_version": "feedbax.tests.model_contract.v1",
    }
    assert row.measurement_binding is not None
    assert row.measurement_binding.model_dump() == {
        "trace_schema_id": TRACE_SCHEMA_ID,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "measurement_basis": "training_batch",
    }
    assert [(item.name, item.kind) for item in row.measurement_steps] == [
        ("training_batch_measurement", "measurement")
    ]
    assert [(item.name, item.kind) for item in row.evaluation_phases] == [
        ("heldout_evaluation", "evaluation")
    ]
    assert [identity.model_dump() for identity in row.evaluation_contract_identities] == [
        {
            "schema_id": "feedbax.tests.evaluation_contract",
            "schema_version": "feedbax.tests.evaluation_contract.v1",
        }
    ]


def test_payload_report_preserves_root_and_nested_contract_identities(tmp_path: Path) -> None:
    training_spec = _payload()
    training_spec["data_contract"] = {
        "schema_id": "feedbax.tests.data_contract",
        "schema_version": "feedbax.tests.data_contract.v1",
    }
    payload = StudioTrainingAssemblySpec(
        total_batches=1,
        training_spec=training_spec,
        task_spec={
            "schema_id": "feedbax.tests.task_contract",
            "schema_version": "feedbax.tests.task_contract.v1",
        },
        graph_spec={
            "schema_id": "feedbax.tests.model_contract",
            "schema_version": "feedbax.tests.model_contract.v1",
        },
    ).model_dump(mode="json", exclude_none=True)
    row = CompiledExecutionRow(
        row_id="studio-row",
        payload=payload,
        resolved_semantics=payload,
        launch=RowLaunchSpec(command=["python", "-c", "pass"]),
    )

    report = build_payload_report(_bundle(tmp_path, rows=[row]))

    projected = report.rows[0]
    assert projected.task_contract is not None
    assert projected.task_contract.schema_id == "feedbax.tests.task_contract"
    assert projected.model_contract is not None
    assert projected.model_contract.schema_id == "feedbax.tests.model_contract"
    assert projected.data_contract is not None
    assert projected.data_contract.schema_id == "feedbax.tests.data_contract"


def test_measurement_assertion_passes_and_shows_both_sides_on_mismatch(
    tmp_path: Path,
) -> None:
    report = build_payload_report(_report_bundle(tmp_path, ("row-a", _payload())))

    passing = assert_measurement_binding(
        report,
        MeasurementBindingExpectation(
            trace_schema_id=TRACE_SCHEMA_ID,
            trace_schema_version=TRACE_SCHEMA_VERSION,
            measurement_basis="training_batch",
        ),
    )
    assert passing.matches is True

    failing = assert_measurement_binding(
        report,
        MeasurementBindingExpectation(
            trace_schema_id="feedbax.tests.locked_measurement",
            trace_schema_version="feedbax.tests.locked_measurement.v2",
        ),
    )
    assert failing.matches is False
    message = format_measurement_assertion_mismatches(failing)
    assert "feedbax.tests.locked_measurement@feedbax.tests.locked_measurement.v2" in message
    assert f"{TRACE_SCHEMA_ID}@{TRACE_SCHEMA_VERSION}" in message


def test_measurement_assertion_reports_only_the_mismatching_row(tmp_path: Path) -> None:
    bundle = _report_bundle(
        tmp_path,
        ("matching", _payload()),
        (
            "mismatching",
            _payload(
                trace_schema_id="feedbax.tests.heldout_measurement",
                trace_schema_version="feedbax.tests.heldout_measurement.v1",
            ),
        ),
    )

    result = assert_measurement_binding(
        build_payload_report(bundle),
        {
            row_id: MeasurementBindingExpectation(
                trace_schema_id=TRACE_SCHEMA_ID,
                trace_schema_version=TRACE_SCHEMA_VERSION,
            )
            for row_id in ("matching", "mismatching")
        },
    )

    assert result.matches is False
    assert [row.row_id for row in result.rows if not row.matches] == ["mismatching"]


def test_payload_report_rejects_unmaterialized_payload(tmp_path: Path) -> None:
    bundle = _report_bundle(tmp_path, ("row-a", _payload()))
    row = bundle.rows[0]
    unmaterialized = row.model_copy(
        update={
            "execution": row.execution.model_copy(
                update={
                    "payload": row.execution.payload.model_copy(update={"uri": None})
                }
            )
        }
    )

    try:
        build_payload_report(bundle.model_copy(update={"rows": [unmaterialized]}))
    except ValueError as exc:
        assert "row-a" in str(exc)
        assert "not locally materialized" in str(exc)
    else:
        raise AssertionError("unmaterialized payload was accepted")


def test_describe_cli_text_json_and_failing_assertion(
    tmp_path: Path,
    capsys,
) -> None:
    bundle = _report_bundle(tmp_path, ("row-a", _payload()))
    bundle_path = _write_bundle(bundle, tmp_path / "bundle.json")

    assert orchestrate.main(["describe", "--bundle", str(bundle_path)]) == 0
    text_output = capsys.readouterr().out
    assert "row: row-a" in text_output
    assert (
        f"measurement binding: {TRACE_SCHEMA_ID}@{TRACE_SCHEMA_VERSION}" in text_output
    )

    assert orchestrate.main(["describe", "--bundle", str(bundle_path), "--json"]) == 0
    json_output = json.loads(capsys.readouterr().out)
    assert json_output["rows"][0]["row_id"] == "row-a"

    assert (
        orchestrate.main(
            [
                "describe",
                "--bundle",
                str(bundle_path),
                "--assert-measurement",
                TRACE_SCHEMA_ID,
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = orchestrate.main(
        [
            "describe",
            "--bundle",
            str(bundle_path),
            "--assert-measurement",
            "feedbax.tests.wrong@feedbax.tests.wrong.v1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == orchestrate.EXIT_PREFLIGHT
    assert "expected=" in captured.err
    assert "actual=" in captured.err
