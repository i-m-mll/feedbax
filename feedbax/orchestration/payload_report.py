"""Ephemeral, read-only projections of compiled scientific run payloads."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from feedbax.contracts.worker import MethodContractSpec
from feedbax.orchestration.assembly import AssemblyContext, load_schema_artifact
from feedbax.orchestration.bundle import RunBundle

if TYPE_CHECKING:
    from feedbax.contracts.migrations import SpecSchemaRegistry


class _EphemeralModel(BaseModel):
    """Strict base for projections that are not durable contract families."""

    model_config = ConfigDict(extra="forbid")


class ContractIdentity(_EphemeralModel):
    """Declared schema identity for one scientific contract."""

    schema_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)


class NamedKind(_EphemeralModel):
    """Name and worker-contract kind of one phase-program entry."""

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)


class MeasurementBinding(_EphemeralModel):
    """Actual measurement contract bound by a lowered execution payload."""

    trace_schema_id: str = Field(min_length=1)
    trace_schema_version: str = Field(min_length=1)
    measurement_basis: str = Field(min_length=1)


class MeasurementBindingExpectation(_EphemeralModel):
    """Required measurement identity, with optional version and basis constraints."""

    trace_schema_id: str = Field(min_length=1)
    trace_schema_version: str | None = None
    measurement_basis: str | None = None


class RunPayloadRowReport(_EphemeralModel):
    """Scientific payload projection for one compiled bundle row."""

    row_id: str = Field(min_length=1)
    task_contract: ContractIdentity | None = None
    model_contract: ContractIdentity | None = None
    data_contract: ContractIdentity | None = None
    measurement_binding: MeasurementBinding | None = None
    measurement_steps: list[NamedKind] = Field(default_factory=list)
    evaluation_phases: list[NamedKind] = Field(default_factory=list)
    evaluation_contract_identities: list[ContractIdentity] = Field(default_factory=list)


class RunPayloadReport(_EphemeralModel):
    """Deterministic, ephemeral projection of a compiled run bundle."""

    rows: list[RunPayloadRowReport]


class MeasurementBindingComparison(_EphemeralModel):
    """Expected and actual measurement bindings for one bundle row."""

    row_id: str = Field(min_length=1)
    matches: bool
    expected: MeasurementBindingExpectation | None
    actual: MeasurementBinding | None


class MeasurementBindingAssertionResult(_EphemeralModel):
    """Aggregate result of comparing measurement bindings across rows."""

    matches: bool
    rows: list[MeasurementBindingComparison]


def _project_payload(row_id: str, payload: Mapping[str, Any]) -> RunPayloadRowReport:
    """Project all report fields from one payload in one extraction boundary."""

    def mapping(value: Any) -> Mapping[str, Any] | None:
        return value if isinstance(value, Mapping) else None

    def identity(value: Any, *, path: str) -> ContractIdentity | None:
        candidate = mapping(value)
        if candidate is None:
            return None
        schema_id = candidate.get("schema_id")
        schema_version = candidate.get("schema_version")
        if schema_id is None and schema_version is None:
            return None
        if not isinstance(schema_id, str) or not isinstance(schema_version, str):
            raise ValueError(
                f"row {row_id!r} declares a partial contract identity at {path!r}; "
                "schema_id and schema_version must both be strings"
            )
        return ContractIdentity(schema_id=schema_id, schema_version=schema_version)

    nested_training = mapping(payload.get("training_spec"))
    execution_payload = (
        nested_training
        if nested_training is not None and "worker_execution" in nested_training
        else payload
    )

    task_candidates = [
        ("task_spec", payload.get("task_spec")),
        ("task_contract", payload.get("task_contract")),
        ("task_contract", execution_payload.get("task_contract")),
        ("task", execution_payload.get("task")),
    ]
    model_candidates = [
        ("graph_spec", payload.get("graph_spec")),
        ("model_contract", payload.get("model_contract")),
        ("graph", execution_payload.get("graph")),
        ("model_contract", execution_payload.get("model_contract")),
        ("model", execution_payload.get("model")),
    ]
    data_candidates = [
        ("data_contract", payload.get("data_contract")),
        ("data", payload.get("data")),
        ("data_contract", execution_payload.get("data_contract")),
        ("data", execution_payload.get("data")),
    ]

    def first_identity(candidates: list[tuple[str, Any]]) -> ContractIdentity | None:
        declared = [
            projected
            for path, value in candidates
            if (projected := identity(value, path=path)) is not None
        ]
        distinct = {(item.schema_id, item.schema_version) for item in declared}
        if len(distinct) > 1:
            raise ValueError(
                f"row {row_id!r} declares conflicting contract identities: "
                f"{sorted(distinct)!r}"
            )
        return declared[0] if declared else None

    worker_execution = mapping(execution_payload.get("worker_execution"))
    method_contract_payload = (
        None if worker_execution is None else worker_execution.get("method_contract")
    )
    method_contract: MethodContractSpec | None = None
    if method_contract_payload is not None:
        if not isinstance(method_contract_payload, Mapping):
            raise ValueError(
                f"row {row_id!r} worker_execution.method_contract must be an object"
            )
        method_contract = MethodContractSpec.model_validate(method_contract_payload)

    measurement_binding = None
    measurement_steps: list[NamedKind] = []
    evaluation_phases: list[NamedKind] = []
    if method_contract is not None:
        diagnostics = method_contract.training_diagnostics
        if diagnostics is not None:
            measurement_binding = MeasurementBinding(
                trace_schema_id=diagnostics.trace_schema_id,
                trace_schema_version=diagnostics.trace_schema_version,
                measurement_basis=diagnostics.measurement_basis,
            )
        measurement_steps = sorted(
            (
                NamedKind(name=step.name, kind=step.kind)
                for step in method_contract.phase_program.update_steps
                if step.kind == "measurement"
            ),
            key=lambda item: (item.name, item.kind),
        )
        evaluation_phases = sorted(
            (
                NamedKind(name=phase.name, kind=phase.kind)
                for phase in method_contract.phase_program.phases
                if phase.kind == "evaluation"
            ),
            key=lambda item: (item.name, item.kind),
        )

    evaluation_candidates: list[tuple[str, Any]] = [
        ("evaluation_contract", execution_payload.get("evaluation_contract")),
        ("evaluation_spec", execution_payload.get("evaluation_spec")),
    ]
    raw_evaluation_contracts = execution_payload.get("evaluation_contracts")
    if raw_evaluation_contracts is not None:
        if not isinstance(raw_evaluation_contracts, list):
            raise ValueError(f"row {row_id!r} evaluation_contracts must be a list")
        evaluation_candidates.extend(
            (f"evaluation_contracts[{index}]", value)
            for index, value in enumerate(raw_evaluation_contracts)
        )
    if method_contract is not None:
        for phase in method_contract.phase_program.phases:
            if phase.kind != "evaluation":
                continue
            evaluation_candidates.extend(
                (
                    (
                        f"worker_execution.method_contract.phase_program."
                        f"phases[{phase.name!r}].metadata.{key}",
                        phase.metadata.get(key),
                    )
                    for key in ("evaluation_contract", "evaluation_spec")
                )
            )

    evaluation_identities = {
        (projected.schema_id, projected.schema_version): projected
        for path, value in evaluation_candidates
        if (projected := identity(value, path=path)) is not None
    }

    return RunPayloadRowReport(
        row_id=row_id,
        task_contract=first_identity(task_candidates),
        model_contract=first_identity(model_candidates),
        data_contract=first_identity(data_candidates),
        measurement_binding=measurement_binding,
        measurement_steps=measurement_steps,
        evaluation_phases=evaluation_phases,
        evaluation_contract_identities=[
            evaluation_identities[key] for key in sorted(evaluation_identities)
        ],
    )


def build_payload_report(
    bundle: RunBundle,
    *,
    schema_registry: "SpecSchemaRegistry | None" = None,
) -> RunPayloadReport:
    """Build a report using only payload bytes already materialized locally."""

    rows: list[RunPayloadRowReport] = []
    context = AssemblyContext(custody_root=Path.cwd(), schema_registry=schema_registry)
    for row in sorted(bundle.rows, key=lambda item: item.row_id):
        ref = row.execution.payload
        if ref.uri is None:
            raise ValueError(
                f"row {row.row_id!r} execution payload {ref.artifact_id!r} is not "
                "locally materialized: no local URI"
            )
        local_path = Path(ref.uri).expanduser()
        if not local_path.is_file():
            raise ValueError(
                f"row {row.row_id!r} execution payload {ref.artifact_id!r} is not "
                f"locally materialized at {local_path}"
            )
        local_ref = ref.model_copy(update={"uri": str(local_path)})
        try:
            migrated = load_schema_artifact(local_ref, context=context)
        except OSError as exc:
            raise ValueError(
                f"row {row.row_id!r} execution payload {ref.artifact_id!r} is not "
                f"locally materialized at {local_path}: {exc}"
            ) from exc
        rows.append(_project_payload(row.row_id, migrated.payload))
    return RunPayloadReport(rows=rows)


def assert_measurement_binding(
    report: RunPayloadReport,
    expected: (
        MeasurementBindingExpectation
        | MeasurementBinding
        | Mapping[str, Any]
    ),
) -> MeasurementBindingAssertionResult:
    """Compare actual row bindings with one uniform or per-row expectation."""

    def expectation(value: Any) -> MeasurementBindingExpectation:
        if isinstance(value, MeasurementBindingExpectation):
            return value
        if isinstance(value, MeasurementBinding):
            return MeasurementBindingExpectation.model_validate(value.model_dump())
        return MeasurementBindingExpectation.model_validate(value)

    uniform = isinstance(expected, (MeasurementBindingExpectation, MeasurementBinding)) or (
        isinstance(expected, Mapping) and "trace_schema_id" in expected
    )
    expected_by_row: Mapping[str, MeasurementBindingExpectation]
    if uniform:
        uniform_expected = expectation(expected)
        expected_by_row = {row.row_id: uniform_expected for row in report.rows}
    else:
        if not isinstance(expected, Mapping):
            raise TypeError("measurement expectation must be uniform or mapped by row id")
        expected_by_row = {
            str(row_id): expectation(row_expected)
            for row_id, row_expected in expected.items()
        }
        known_rows = {row.row_id for row in report.rows}
        unexpected_rows = sorted(set(expected_by_row) - known_rows)
        if unexpected_rows:
            raise ValueError(
                f"measurement expectations name unknown rows: {unexpected_rows!r}"
            )

    comparisons: list[MeasurementBindingComparison] = []
    for row in sorted(report.rows, key=lambda item: item.row_id):
        row_expected = expected_by_row.get(row.row_id)
        actual = row.measurement_binding
        matches = row_expected is not None and actual is not None
        if matches:
            assert row_expected is not None and actual is not None
            matches = row_expected.trace_schema_id == actual.trace_schema_id
            if row_expected.trace_schema_version is not None:
                matches = matches and (
                    row_expected.trace_schema_version == actual.trace_schema_version
                )
            if row_expected.measurement_basis is not None:
                matches = matches and row_expected.measurement_basis == actual.measurement_basis
        comparisons.append(
            MeasurementBindingComparison(
                row_id=row.row_id,
                matches=matches,
                expected=row_expected,
                actual=actual,
            )
        )
    return MeasurementBindingAssertionResult(
        matches=all(item.matches for item in comparisons),
        rows=comparisons,
    )


def _format_contract_identity(identity: ContractIdentity | None) -> str:
    if identity is None:
        return "<not declared>"
    return f"{identity.schema_id}@{identity.schema_version}"


def _format_named_kinds(items: list[NamedKind]) -> str:
    return ", ".join(f"{item.name} ({item.kind})" for item in items) or "<none>"


def format_payload_report(report: RunPayloadReport) -> str:
    """Render a deterministic human-readable payload report."""

    lines: list[str] = []
    for row in sorted(report.rows, key=lambda item: item.row_id):
        lines.extend(
            [
                f"row: {row.row_id}",
                f"  task contract: {_format_contract_identity(row.task_contract)}",
                f"  model contract: {_format_contract_identity(row.model_contract)}",
                f"  data contract: {_format_contract_identity(row.data_contract)}",
            ]
        )
        if row.measurement_binding is None:
            lines.append("  measurement binding: <not declared>")
        else:
            binding = row.measurement_binding
            lines.append(
                "  measurement binding: "
                f"{binding.trace_schema_id}@{binding.trace_schema_version} "
                f"(basis={binding.measurement_basis})"
            )
        lines.append(f"  measurement steps: {_format_named_kinds(row.measurement_steps)}")
        lines.append(f"  evaluation phases: {_format_named_kinds(row.evaluation_phases)}")
        evaluation = ", ".join(
            _format_contract_identity(identity)
            for identity in row.evaluation_contract_identities
        )
        lines.append(f"  evaluation contracts: {evaluation or '<none>'}")
    return "\n".join(lines)


def format_measurement_assertion_mismatches(
    result: MeasurementBindingAssertionResult,
) -> str:
    """Render only measurement mismatches with expected and actual side by side."""

    lines: list[str] = []
    for comparison in sorted(result.rows, key=lambda item: item.row_id):
        if comparison.matches:
            continue
        expected = comparison.expected
        if expected is None:
            expected_text = "<missing per-row expectation>"
        else:
            version = (
                "*" if expected.trace_schema_version is None else expected.trace_schema_version
            )
            basis = "*" if expected.measurement_basis is None else expected.measurement_basis
            expected_text = f"{expected.trace_schema_id}@{version} (basis={basis})"
        actual = comparison.actual
        actual_text = (
            "<not declared>"
            if actual is None
            else (
                f"{actual.trace_schema_id}@{actual.trace_schema_version} "
                f"(basis={actual.measurement_basis})"
            )
        )
        lines.append(
            f"row {comparison.row_id}: expected={expected_text}; actual={actual_text}"
        )
    return "\n".join(lines)


__all__ = [
    "ContractIdentity",
    "MeasurementBinding",
    "MeasurementBindingAssertionResult",
    "MeasurementBindingComparison",
    "MeasurementBindingExpectation",
    "NamedKind",
    "RunPayloadReport",
    "RunPayloadRowReport",
    "assert_measurement_binding",
    "build_payload_report",
    "format_measurement_assertion_mismatches",
    "format_payload_report",
]
