"""CLI for Feedbax deterministic run-set orchestration.

Status lines are a consumer contract. Text status output is one line per row:

``row=<id> status=<s> batch=<i>/<n> last_loss=<x|-> last_event_age_s=<t|-> seq=<n>``

followed by ``stages=<stage>:<status>,...`` in orchestration stage order.

Exit codes: 0 success; 2 preflight or conformance failure; 3 row failure;
4 budget exceeded; 5 lock conflict; 1 all other errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.shadow_launch import ShadowLaunchEvidence, ShadowLaunchRowEvidence
from feedbax.contracts.evaluation_lifecycle import (
    EvaluationLifecycleEvidence,
    EvaluationShadowLaunchEvidence,
    EvaluationWorkerTopologyEvidence,
)
from feedbax.contracts.staged_execution import validate_staged_binding_name
from feedbax.contracts.training import TrainingProgramRegistry
from feedbax.orchestration import (
    STAGE_ORDER,
    MatrixAuthorityError,
    AssemblyContext,
    EmergencyRunSetRecord,
    RunAssemblyRequest,
    RunBundle,
    RunEvent,
    RunEventReader,
    RunSetState,
    RunSetStateStore,
    StateLockError,
    assemble_run_bundle,
    build_default_assembly_registry,
    build_training_run_matrix_authority,
    run_authority_preflight_checks,
)
from feedbax.orchestration.bundle import (
    canonical_run_bundle_sha256,
    default_orchestration_root,
    mint_run_set_id,
)
from feedbax.orchestration.revision import (
    FeedbaxRevisionError,
    assert_feedbax_source_residence,
)
from feedbax.orchestration.collection_recovery import CollectionRecoveryBinding
from feedbax.orchestration.staged_root_custody import StagedRootSnapshotBinding
from feedbax.orchestration.conformance import CheckRegistry
from feedbax.orchestration.drivers.capabilities import (
    DriverAuthority,
    DriverConstructionContext,
    DriverHook,
    DriverRegistry,
)
from feedbax.orchestration.drivers.runpod import load_runpod_api_key
from feedbax.orchestration.input_materialization import InputProviderRootBinding
from feedbax.orchestration.executor_family import evaluation_matrix_ordered_union
from feedbax.orchestration.payload_report import (
    MeasurementBindingExpectation,
    assert_measurement_binding,
    build_payload_report,
    format_measurement_assertion_mismatches,
    format_payload_report,
)
from feedbax.orchestration.stages import (
    BudgetExceeded,
    PreflightFailed,
    StageEngine,
)
from feedbax.plugins.composition import compose_application
from feedbax.training.interruption import CancellationDecision, RunInterruptionController


EXIT_SUCCESS = 0
EXIT_OTHER = 1
EXIT_PREFLIGHT = 2
EXIT_ROW_FAILURE = 3
EXIT_BUDGET = 4
EXIT_LOCK = 5

TERMINAL_ROW_STATUSES = frozenset({"completed", "failed", "stopped"})


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.bootstrap_state = asyncio.run(compose_application())
        return int(args.func(args))
    except StateLockError as exc:
        _print_error(exc)
        return EXIT_LOCK
    except PreflightFailed as exc:
        _print_error(exc)
        return EXIT_PREFLIGHT
    except BudgetExceeded as exc:
        _print_error(exc)
        return EXIT_BUDGET
    except FeedbaxRevisionError as exc:
        # The imported package is not the one the invocation or the request says
        # it is. That is a gate failure before any work, not an internal error.
        _print_error(exc)
        return EXIT_PREFLIGHT
    except Exception as exc:
        _print_error(exc)
        return EXIT_OTHER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive Feedbax run-set orchestration.",
        epilog=(
            "Exit codes: 0 success; 2 preflight/conformance failure; "
            "3 row failure; 4 budget exceeded; 5 lock conflict; 1 other."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Run ASSEMBLE and PREFLIGHT only")
    preflight_input = preflight.add_mutually_exclusive_group(required=True)
    preflight_input.add_argument("--assembly-request", help="RunAssemblyRequest JSON path")
    preflight_input.add_argument(
        "--bundle",
        help="Content-pinned, already assembled RunBundle JSON path (authority-only)",
    )
    preflight.add_argument(
        "--bundle-sha256",
        help="Expected canonical SHA-256 for --bundle authority input",
    )
    preflight.add_argument(
        "--authority-only",
        action="store_true",
        help="Emit matrix authority without provider readiness checks",
    )
    preflight.add_argument(
        "--run-set-id",
        help="Bind authority and later provider preflight to one run-set identity",
    )
    preflight.set_defaults(func=cmd_preflight)

    launch = subparsers.add_parser("launch", help="Launch or resume a run bundle")
    launch.add_argument("--assembly-request", required=True, help="RunAssemblyRequest JSON path")
    launch.add_argument("--driver", help="Driver identity (validated after application bootstrap)")
    launch.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and bind RunPod launch rows without contacting RunPod",
    )
    launch.add_argument(
        "--deadman",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the bundle dead-man switch policy",
    )
    launch.add_argument(
        "--deadman-silence-seconds",
        type=int,
        help="Override the bundle dead-man silence threshold",
    )
    launch.add_argument("--resume-run-set", help="Resume an existing run-set id")
    for sub in (preflight, launch):
        sub.add_argument(
            "--input-provider", action="append", default=[], metavar="NAME=ABSOLUTE_PATH"
        )
        sub.add_argument(
            "--staged-root",
            action="append",
            default=[],
            metavar="KIND:NAME=ABSOLUTE_PATH",
        )
        sub.add_argument(
            "--feedbax-checkout",
            metavar="PATH",
            help=(
                "Assert that the imported feedbax package is supplied by this checkout. "
                "Runtime-only operator assertion; never recorded in any durable artifact."
            ),
        )
    launch.set_defaults(func=cmd_launch)

    shadow_launch = subparsers.add_parser(
        "shadow-launch",
        help="Exercise one provider-free local continuation update through COLLECT",
    )
    shadow_launch.add_argument(
        "--assembly-request", required=True, help="RunAssemblyRequest JSON path"
    )
    shadow_launch.add_argument(
        "--input-provider", action="append", default=[], metavar="NAME=ABSOLUTE_PATH"
    )
    shadow_launch.add_argument(
        "--staged-root",
        action="append",
        default=[],
        metavar="KIND:NAME=ABSOLUTE_PATH",
    )
    shadow_launch.set_defaults(func=cmd_shadow_launch)

    describe = subparsers.add_parser(
        "describe",
        help="Describe the scientific payload in an assembled run bundle",
    )
    describe.add_argument("--bundle", required=True, help="RunBundle JSON path")
    describe.add_argument("--json", action="store_true", help="Emit report JSON")
    describe.add_argument(
        "--assert-measurement",
        metavar="TRACE_SCHEMA_ID[@VERSION]",
        help="Fail when a row does not match the expected measurement binding",
    )
    describe.set_defaults(func=cmd_describe)

    status = subparsers.add_parser("status", help="Print current run-set status")
    status.add_argument("--run-set", required=True, help="Run-set id")
    status.add_argument("--json", action="store_true", help="Emit state JSON")
    status.set_defaults(func=cmd_status)

    watch = subparsers.add_parser("watch", help="Follow run events until all rows are terminal")
    watch.add_argument("--run-set", required=True, help="Run-set id")
    watch.add_argument("--poll-interval", type=float, default=0.5, help=argparse.SUPPRESS)
    watch.set_defaults(func=cmd_watch)

    for name in ("collect", "certify", "resume"):
        sub = subparsers.add_parser(name, help=f"Run the {name} orchestration action")
        sub.add_argument("--run-set", required=True, help="Run-set id")
        if name in {"collect", "resume"}:
            sub.add_argument(
                "--recover-collected-root",
                action="append",
                default=[],
                metavar="ROW=ABSOLUTE_PATH",
                help="Recover a preserved run-owned row collection after verified teardown",
            )
        if name in {"certify", "resume"}:
            sub.add_argument("--input-provider", action="append", default=[])
        sub.set_defaults(func=globals()[f"cmd_{name}"])

    teardown = subparsers.add_parser("teardown", help="Tear down run-set resources")
    teardown.add_argument("--run-set", required=True, help="Run-set id")
    teardown.add_argument("--force", action="store_true", help="Break a stale state lock")
    teardown.set_defaults(func=cmd_teardown)

    return parser


def _assert_declared_feedbax_checkout(args: argparse.Namespace) -> None:
    """Honour the runtime-only ``--feedbax-checkout`` residence assertion."""
    declared = getattr(args, "feedbax_checkout", None)
    if declared:
        assert_feedbax_source_residence(declared)


def cmd_preflight(args: argparse.Namespace) -> int:
    _assert_declared_feedbax_checkout(args)
    if not args.authority_only:
        if args.bundle:
            raise ValueError("--bundle is supported only with --authority-only")
        if args.bundle_sha256:
            raise ValueError("--bundle-sha256 is supported only with --authority-only")
    if args.authority_only:
        if not args.run_set_id:
            raise ValueError("--authority-only requires --run-set-id")
        if args.bundle:
            if not args.bundle_sha256:
                raise ValueError("--bundle authority input requires --bundle-sha256")
            bundle = _load_bundle(args.bundle)
        else:
            if args.bundle_sha256:
                raise ValueError("--bundle-sha256 requires --bundle")
            request_path = Path(args.assembly_request)
            request = _load_assembly_request(request_path)
            root = Path(request.orchestration_root or request_path.parent).expanduser()
            bundle = assemble_run_bundle(
                request,
                run_set_id=args.run_set_id,
                context=AssemblyContext(custody_root=root / "custody", repo_root=Path.cwd()),
                registry=_assembly_registry(args.bootstrap_state.bundle),
            )
        checks = run_authority_preflight_checks(bundle)
        if any(check.status == "fail" for check in checks):
            return EXIT_PREFLIGHT
        metadata = bundle.environment.metadata
        try:
            authority = build_training_run_matrix_authority(
                bundle,
                local_repos=metadata.get("runpod_local_repos", {}),
                protected_refs=metadata.get("runpod_protected_refs", {}),
                expected_bundle_sha256=args.bundle_sha256,
                expected_run_set_id=args.run_set_id,
            )
        except MatrixAuthorityError as exc:
            _print_error(exc)
            return EXIT_PREFLIGHT
        _write_json(authority)
        return EXIT_SUCCESS
    request_path = Path(args.assembly_request)
    request = _load_assembly_request(request_path)
    engine = _request_engine(
        request,
        request_path=request_path,
        run_set_id=args.run_set_id,
        input_provider_bindings=_input_provider_bindings(args.input_provider),
        staged_root_bindings=_staged_root_bindings(args.staged_root),
        conformance_registry=args.bootstrap_state.bundle.conformance_checks,
        plugin_provenance=args.bootstrap_state.provenance,
        registry_bundle=args.bootstrap_state.bundle,
    )
    try:
        state = engine.run(stop_after_stage="PREFLIGHT")
    except PreflightFailed:
        state = engine.store.load()
    checks = state.stage("PREFLIGHT").checks
    for check in checks:
        detail = f" detail={check.detail}" if check.detail else ""
        print(f"{check.name} {check.status}{detail}")
    return EXIT_PREFLIGHT if any(check.status == "fail" for check in checks) else EXIT_SUCCESS


def cmd_launch(args: argparse.Namespace) -> int:
    _assert_declared_feedbax_checkout(args)
    request_path = Path(args.assembly_request)
    request = _load_assembly_request(request_path)
    if args.driver:
        if args.driver != request.deployment_policy.driver:
            raise ValueError(
                "--driver conflicts with deployment_policy.driver; edit and re-authorize "
                "the versioned RunAssemblyRequest instead of overriding launch policy"
            )
    overrides: dict[str, Any] = {}
    if args.deadman is not None:
        overrides["deadman_enabled"] = args.deadman
    if args.deadman_silence_seconds is not None:
        overrides["deadman_silence_seconds"] = args.deadman_silence_seconds
    if overrides:
        request = RunAssemblyRequest.model_validate(
            {**request.model_dump(mode="json"), **overrides}
        )
    input_provider_bindings = _input_provider_bindings(args.input_provider)
    staged_root_bindings = _staged_root_bindings(args.staged_root)
    if args.dry_run:
        if args.resume_run_set:
            raise ValueError("--dry-run does not support --resume-run-set")
        root = (
            Path(request.orchestration_root).expanduser()
            if request.orchestration_root
            else request_path.parent
        )
        bundle = assemble_run_bundle(
            request,
            run_set_id=mint_run_set_id(),
            context=AssemblyContext(custody_root=root / "custody", repo_root=Path.cwd()),
            registry=_assembly_registry(args.bootstrap_state.bundle),
        )
        driver = _construct_driver(
            bundle,
            driver_registry=args.bootstrap_state.bundle.drivers,
            training_method_registry=args.bootstrap_state.bundle.training_programs,
            input_provider_bindings=input_provider_bindings,
            staged_root_bindings=staged_root_bindings,
            load_credentials=False,
        )
        if not driver.realized_capabilities.facts.supports(DriverHook.DRY_RUN_LAUNCH):
            raise ValueError(
                f"driver {bundle.deployment_policy.driver!r} capability variant "
                f"{driver.realized_capabilities.variant_id!r} does not support dry-run launch"
            )
        commands = driver.dry_run_launch(bundle)
        for row, _command in zip(bundle.rows, commands, strict=True):
            print(f"row={row.row_id} dry-run=accepted")
        return EXIT_SUCCESS
    with RunInterruptionController() as interruption:
        engine = _request_engine(
            request,
            request_path=request_path,
            run_set_id=args.resume_run_set,
            interruption_probe=interruption.poll,
            input_provider_bindings=input_provider_bindings,
            staged_root_bindings=staged_root_bindings,
            conformance_registry=args.bootstrap_state.bundle.conformance_checks,
            plugin_provenance=args.bootstrap_state.provenance,
            registry_bundle=args.bootstrap_state.bundle,
        )
        state = engine.run()
    return _state_exit_code(state)


def cmd_shadow_launch(args: argparse.Namespace) -> int:
    """Run the one local governed scenario without entering provider readiness stages."""
    request_path = Path(args.assembly_request)
    request = _load_assembly_request(request_path)
    _require_provider_free_shadow_request(request)
    engine = _request_engine(
        request,
        request_path=request_path,
        input_provider_bindings=_input_provider_bindings(args.input_provider),
        staged_root_bindings=_staged_root_bindings(args.staged_root),
        native_update_budget=1,
        conformance_registry=args.bootstrap_state.bundle.conformance_checks,
        plugin_provenance=args.bootstrap_state.provenance,
        registry_bundle=args.bootstrap_state.bundle,
    )
    state = engine.run(stop_after_stage="COLLECT")
    if engine.bundle is None:
        raise RuntimeError("shadow launch did not persist an assembled bundle")
    if engine.bundle.execution_family == "evaluation-matrix":
        state = engine.run(stop_after_stage="TEARDOWN")
    bundle_path = engine.bundle.run_set_dir / "bundle.json"
    persisted_bundle = _load_bundle(bundle_path)
    evidence = _shadow_launch_evidence(persisted_bundle, state)
    _write_json(evidence)
    return _state_exit_code(state)


def _require_provider_free_shadow_request(request: RunAssemblyRequest) -> None:
    policy = request.deployment_policy
    if policy.venue != "local" or policy.cloud_authorized or policy.review_authorized:
        raise ValueError(
            "shadow-launch requires an unauthorized local DeploymentPolicy; "
            "provider-capable requests are not eligible"
        )


def _shadow_launch_evidence(
    bundle: RunBundle, state: RunSetState
) -> ShadowLaunchEvidence | EvaluationShadowLaunchEvidence:
    """Validate the bounded local result and emit non-readiness evidence only."""
    policy = bundle.deployment_policy
    if policy.venue != "local" or policy.cloud_authorized:
        raise ValueError("shadow-launch cannot emit evidence for a provider-capable bundle")
    if state.stage("COLLECT").status != "completed":
        raise ValueError("shadow-launch requires a completed COLLECT stage")
    execution_family = getattr(bundle, "execution_family", "native-training")
    if execution_family == "native-training":
        if len(bundle.rows) != 1:
            raise ValueError("native shadow-launch requires exactly one assembled row")
        for stage in ("CERTIFY", "TEARDOWN", "REGISTER"):
            if state.stage(stage).status != "pending":
                raise ValueError(f"shadow-launch must stop before {stage}")
    elif (
        state.stage("CERTIFY").status != "completed"
        or state.stage("TEARDOWN").status != "completed"
        or state.stage("REGISTER").status != "pending"
    ):
        raise ValueError(
            "evaluation shadow-launch requires completed CERTIFY and TEARDOWN "
            "and must stop before REGISTER"
        )

    for row in bundle.rows:
        if state.rows.get(row.row_id) is None or state.rows[row.row_id].status != "completed":
            raise ValueError(f"shadow-launch row {row.row_id!r} did not complete")
    if execution_family == "evaluation-matrix":
        lifecycles = []
        for row in bundle.rows:
            lifecycle_path = state.rows[row.row_id].collected_outputs.get(
                "evaluation-matrix-result.json"
            )
            if not isinstance(lifecycle_path, str):
                raise ValueError("evaluation shadow launch lacks collected lifecycle evidence")
            lifecycles.append(
                EvaluationLifecycleEvidence.model_validate_json(
                    Path(lifecycle_path).read_text(encoding="utf-8")
                )
            )
        return EvaluationShadowLaunchEvidence(
            run_set_id=bundle.run_set_id,
            bundle_sha256=canonical_run_bundle_sha256(bundle),
            lifecycles=tuple(lifecycles),
            ordered_union=evaluation_matrix_ordered_union(bundle, lifecycles),
            worker_topology=EvaluationWorkerTopologyEvidence.model_validate_json(
                Path(
                    state.rows[bundle.rows[0].row_id].collected_outputs[
                        "evaluation-worker-topology.json"
                    ]
                ).read_text(encoding="utf-8")
            ),
        )
    row = bundle.rows[0]
    if "--resume" not in row.launch.command:
        raise ValueError("shadow-launch requires a native continuation row")
    if row.execution.row_provenance is None:
        raise ValueError("shadow-launch row lacks planned-run provenance")

    row_dir = bundle.run_set_dir / "rows" / row.row_id
    diagnostics = json.loads((row_dir / "training-diagnostics.json").read_text(encoding="utf-8"))
    if diagnostics.get("segment_completed_batches") != 1:
        raise ValueError("shadow-launch requires exactly one completed continuation batch")
    try:
        native_result = json.loads((row_dir / "stdout.log").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "shadow-launch requires one native executor JSON result; "
            "multiple stdout log lines are not accepted"
        ) from exc
    if not isinstance(native_result, dict):
        raise ValueError("shadow-launch native executor result must be a JSON object")
    if native_result.get("payload_binding_status") != "verified":
        raise ValueError("shadow-launch native payload binding was not verified")

    return ShadowLaunchEvidence(
        run_set_id=bundle.run_set_id,
        bundle_sha256=canonical_run_bundle_sha256(bundle),
        rows=(
            ShadowLaunchRowEvidence(
                row_id=row.row_id,
                planned_run_id=row.execution.row_provenance.planned_run_id,
            ),
        ),
    )


def cmd_describe(args: argparse.Namespace) -> int:
    """Render and optionally assert the scientific payload of a run bundle."""
    report = build_payload_report(_load_bundle(args.bundle))
    if args.json:
        _write_json(report)
    else:
        print(format_payload_report(report))
    if args.assert_measurement is None:
        return EXIT_SUCCESS
    result = assert_measurement_binding(
        report,
        _parse_measurement_expectation(args.assert_measurement),
    )
    if result.matches:
        return EXIT_SUCCESS
    print(format_measurement_assertion_mismatches(result), file=sys.stderr)
    return EXIT_PREFLIGHT


def cmd_status(args: argparse.Namespace) -> int:
    state = _load_status_state(args.run_set)
    if isinstance(state, EmergencyRunSetRecord):
        if args.json:
            _write_json(state)
        else:
            print(_format_emergency_status(state))
        return EXIT_OTHER
    if args.json:
        _write_json(state)
    else:
        run_set_dir = _run_set_dir(args.run_set)
        for row_id in sorted(state.rows):
            print(format_status_line(state, row_id, run_set_dir=run_set_dir))
    return _state_exit_code(state)


def cmd_watch(args: argparse.Namespace) -> int:
    run_set_dir = _run_set_dir(args.run_set)
    yielded_through: dict[str, int] = {}
    while True:
        state = _load_state(args.run_set)
        for row_id in sorted(state.rows):
            path = run_set_dir / "events" / f"{row_id}.events.jsonl"
            from_seq = yielded_through.get(row_id, -1) + 1
            events = RunEventReader(path).read_all(from_seq=from_seq)
            for event in events:
                yielded_through[row_id] = event.seq
                print(format_event_line(event))
        if _all_rows_terminal(state):
            return _state_exit_code(state)
        time.sleep(args.poll_interval)


def cmd_collect(args: argparse.Namespace) -> int:
    state = _run_existing(
        args.run_set,
        stop_after_stage="COLLECT",
        collection_recovery_bindings=_collection_recovery_bindings(args.recover_collected_root),
        conformance_registry=args.bootstrap_state.bundle.conformance_checks,
        training_method_registry=args.bootstrap_state.bundle.training_programs,
        driver_registry=args.bootstrap_state.bundle.drivers,
        plugin_provenance=args.bootstrap_state.provenance,
    )
    return _state_exit_code(state)


def cmd_certify(args: argparse.Namespace) -> int:
    run_options: dict[str, Any] = {
        "stop_after_stage": "CERTIFY",
        "retry_failed_certification": True,
    }
    input_bindings = _input_provider_bindings(args.input_provider)
    if input_bindings:
        run_options["input_provider_bindings"] = input_bindings
    state = _run_existing(
        args.run_set,
        conformance_registry=args.bootstrap_state.bundle.conformance_checks,
        training_method_registry=args.bootstrap_state.bundle.training_programs,
        driver_registry=args.bootstrap_state.bundle.drivers,
        plugin_provenance=args.bootstrap_state.provenance,
        **run_options,
    )
    if state.stage("CERTIFY").outputs.get("overall") == "fail":
        return EXIT_PREFLIGHT
    return _state_exit_code(state)


def cmd_teardown(args: argparse.Namespace) -> int:
    state = _run_existing(
        args.run_set,
        stop_after_stage="TEARDOWN",
        break_stale_lock=args.force,
        conformance_registry=args.bootstrap_state.bundle.conformance_checks,
        training_method_registry=args.bootstrap_state.bundle.training_programs,
        driver_registry=args.bootstrap_state.bundle.drivers,
        plugin_provenance=args.bootstrap_state.provenance,
    )
    return _state_exit_code(state)


def cmd_resume(args: argparse.Namespace) -> int:
    with RunInterruptionController() as interruption:
        state = _run_existing(
            args.run_set,
            interruption_probe=interruption.poll,
            input_provider_bindings=_input_provider_bindings(args.input_provider),
            collection_recovery_bindings=_collection_recovery_bindings(args.recover_collected_root),
            conformance_registry=args.bootstrap_state.bundle.conformance_checks,
            training_method_registry=args.bootstrap_state.bundle.training_programs,
            driver_registry=args.bootstrap_state.bundle.drivers,
            plugin_provenance=args.bootstrap_state.provenance,
        )
    return _state_exit_code(state)


def format_status_line(
    state: RunSetState,
    row_id: str,
    *,
    run_set_dir: Path,
    now_ms: int | None = None,
) -> str:
    """Render one stable machine-readable status line for a row."""
    row = state.rows[row_id]
    events = RunEventReader(run_set_dir / "events" / f"{row_id}.events.jsonl").read_all()
    latest = events[-1] if events else None
    progress = next((event for event in reversed(events) if event.type == "progress"), None)
    batch = _format_batch(progress)
    last_loss = _format_loss(progress)
    event_age = _format_event_age(latest, now_ms=now_ms)
    seq = latest.seq if latest is not None else row.event_seq_high_water_mark
    stages = ",".join(f"{stage}:{state.stage(stage).status}" for stage in STAGE_ORDER)
    return (
        f"row={row_id} status={row.status} batch={batch} last_loss={last_loss} "
        f"last_event_age_s={event_age} seq={seq} stages={stages}"
    )


def format_event_line(event: RunEvent) -> str:
    """Render one event line for ``watch``."""
    payload = json.dumps(event.payload, sort_keys=True, separators=(",", ":"))
    return f"row={event.row_id} seq={event.seq} type={event.type} payload={payload}"


def _run_engine(
    bundle: RunBundle,
    *,
    stop_after_stage: str | None = None,
    break_stale_lock: bool = False,
    retry_failed_certification: bool = False,
    interruption_probe: Callable[[], CancellationDecision | None] | None = None,
    input_provider_bindings: tuple[InputProviderRootBinding, ...] = (),
    collection_recovery_bindings: tuple[CollectionRecoveryBinding, ...] = (),
    staged_root_bindings: tuple[StagedRootSnapshotBinding, ...] = (),
    conformance_registry: CheckRegistry,
    training_method_registry: TrainingProgramRegistry,
    driver_registry: DriverRegistry,
    plugin_provenance: Sequence[Any],
) -> RunSetState:
    driver = _construct_driver(
        bundle,
        driver_registry=driver_registry,
        training_method_registry=training_method_registry,
        input_provider_bindings=input_provider_bindings,
        collection_recovery_bindings=collection_recovery_bindings,
        staged_root_bindings=staged_root_bindings,
    )
    state = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=conformance_registry,
        plugin_provenance=plugin_provenance,
        interruption_probe=interruption_probe,
    ).run(
        break_stale_lock=break_stale_lock,
        stop_after_stage=stop_after_stage,
        retry_failed_certification=retry_failed_certification,
    )
    if state.abort_reason == "budget-exceeded":
        raise BudgetExceeded("budget exceeded")
    return state


def _run_existing(
    run_set_id: str,
    *,
    stop_after_stage: str | None = None,
    break_stale_lock: bool = False,
    retry_failed_certification: bool = False,
    interruption_probe: Callable[[], CancellationDecision | None] | None = None,
    conformance_registry: CheckRegistry,
    training_method_registry: TrainingProgramRegistry,
    driver_registry: DriverRegistry,
    plugin_provenance: Sequence[Any],
    input_provider_bindings: tuple[InputProviderRootBinding, ...] = (),
    collection_recovery_bindings: tuple[CollectionRecoveryBinding, ...] = (),
) -> RunSetState:
    bundle = _load_existing_bundle(run_set_id)
    return _run_engine(
        bundle,
        stop_after_stage=stop_after_stage,
        break_stale_lock=break_stale_lock,
        retry_failed_certification=retry_failed_certification,
        interruption_probe=interruption_probe,
        conformance_registry=conformance_registry,
        training_method_registry=training_method_registry,
        driver_registry=driver_registry,
        plugin_provenance=plugin_provenance,
        input_provider_bindings=input_provider_bindings,
        collection_recovery_bindings=collection_recovery_bindings,
    )


def _construct_driver(
    bundle: RunBundle,
    *,
    driver_registry: DriverRegistry,
    training_method_registry: TrainingProgramRegistry,
    input_provider_bindings: tuple[InputProviderRootBinding, ...] = (),
    collection_recovery_bindings: tuple[CollectionRecoveryBinding, ...] = (),
    native_update_budget: int | None = None,
    staged_root_bindings: tuple[StagedRootSnapshotBinding, ...] = (),
    load_credentials: bool = True,
):
    return driver_registry.construct(
        bundle.deployment_policy.driver,
        _driver_construction_context(
            bundle,
            input_provider_bindings=input_provider_bindings,
            collection_recovery_bindings=collection_recovery_bindings,
            native_update_budget=native_update_budget,
            staged_root_bindings=staged_root_bindings,
            training_method_registry=training_method_registry,
            load_credentials=load_credentials,
        ),
    )


def _driver_construction_context(
    bundle: RunBundle,
    *,
    input_provider_bindings: tuple[InputProviderRootBinding, ...] = (),
    collection_recovery_bindings: tuple[CollectionRecoveryBinding, ...] = (),
    native_update_budget: int | None = None,
    staged_root_bindings: tuple[StagedRootSnapshotBinding, ...] = (),
    training_method_registry: TrainingProgramRegistry,
    load_credentials: bool = True,
) -> DriverConstructionContext:
    api_key = load_runpod_api_key() if load_credentials else None
    return DriverConstructionContext(
        configuration={
            "bundle": bundle,
            "preserve_owned_resources": bundle.keep_alive,
        },
        runtime_bindings={
            "input_provider_bindings": input_provider_bindings,
            "collection_recovery_bindings": collection_recovery_bindings,
            "native_update_budget": native_update_budget,
            "staged_root_bindings": staged_root_bindings,
            "training_method_registry": training_method_registry,
        },
        credentials=({"runpod_api_key": api_key} if api_key is not None else {}),
        authority=DriverAuthority(
            cloud_authorized=bundle.deployment_policy.cloud_authorized,
            spend_authorized=bundle.deployment_policy.cloud_authorized,
            credential_names=frozenset({"runpod_api_key"} if api_key is not None else ()),
        ),
        recovery_inputs=(
            {"collection_recovery_bindings": collection_recovery_bindings}
            if collection_recovery_bindings
            else {}
        ),
    )


def _load_bundle(path: str | Path) -> RunBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    migrated = default_spec_registry.migrate("RunBundle", payload)
    return RunBundle.model_validate(migrated.payload)


def _load_assembly_request(path: str | Path) -> RunAssemblyRequest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    migrated = default_spec_registry.migrate("RunAssemblyRequest", payload)
    return RunAssemblyRequest.model_validate(migrated.payload)


def _request_engine(
    request: RunAssemblyRequest,
    *,
    request_path: Path,
    run_set_id: str | None = None,
    interruption_probe: Callable[[], CancellationDecision | None] | None = None,
    input_provider_bindings: tuple[InputProviderRootBinding, ...] = (),
    native_update_budget: int | None = None,
    staged_root_bindings: tuple[StagedRootSnapshotBinding, ...] = (),
    conformance_registry: CheckRegistry,
    plugin_provenance: Sequence[Any],
    registry_bundle: Any,
) -> StageEngine:
    root = (
        Path(request.orchestration_root).expanduser()
        if request.orchestration_root
        else request_path.parent
    )
    context = AssemblyContext(
        custody_root=root / "custody",
        repo_root=Path.cwd(),
    )
    return StageEngine.from_request(
        request,
        context=context,
        registry=_assembly_registry(registry_bundle),
        driver_registry=registry_bundle.drivers,
        driver_context=lambda bundle: _driver_construction_context(
            bundle,
            input_provider_bindings=input_provider_bindings,
            native_update_budget=native_update_budget,
            staged_root_bindings=staged_root_bindings,
            training_method_registry=registry_bundle.training_programs,
        ),
        run_set_id=run_set_id,
        conformance_registry=conformance_registry,
        plugin_provenance=plugin_provenance,
        interruption_probe=interruption_probe,
    )


def _assembly_registry(bundle: Any) -> Any:
    return build_default_assembly_registry(
        method_registry=bundle.training_programs,
        row_lowerer_registry=bundle.row_lowerers,
        evaluation_registry=bundle.evaluation_recipes,
    )


def _input_provider_bindings(values: list[str]) -> tuple[InputProviderRootBinding, ...]:
    bindings = []
    for value in values:
        name, separator, raw_root = value.partition("=")
        root = Path(raw_root)
        if not separator or not name or not root.is_absolute():
            raise ValueError("--input-provider requires NAME=ABSOLUTE_PATH")
        validate_staged_binding_name(name)
        bindings.append(InputProviderRootBinding(name, root))
    return tuple(bindings)


def _staged_root_bindings(values: list[str]) -> tuple[StagedRootSnapshotBinding, ...]:
    bindings = []
    for value in values:
        identity, separator, raw_root = value.partition("=")
        kind, kind_separator, name = identity.partition(":")
        root = Path(raw_root)
        if (
            not separator
            or not kind_separator
            or kind not in {"manifest-store", "artifact-provider", "checkpoint-custody"}
            or not name
            or not root.is_absolute()
        ):
            raise ValueError("--staged-root requires KIND:NAME=ABSOLUTE_PATH with a supported KIND")
        validate_staged_binding_name(name)
        info = root.stat()
        bindings.append(
            StagedRootSnapshotBinding(
                name=name,
                kind=kind,
                root=root,
                expected_root_identity=(info.st_dev, info.st_ino),
            )
        )
    return tuple(bindings)


def _parse_measurement_expectation(value: str) -> MeasurementBindingExpectation:
    schema_id, separator, schema_version = value.rpartition("@")
    if not separator:
        schema_id = value
        schema_version = ""
    if not schema_id or (separator and not schema_version):
        raise ValueError("--assert-measurement requires TRACE_SCHEMA_ID or TRACE_SCHEMA_ID@VERSION")
    return MeasurementBindingExpectation(
        trace_schema_id=schema_id,
        trace_schema_version=schema_version or None,
    )


def _collection_recovery_bindings(
    values: list[str],
) -> tuple[CollectionRecoveryBinding, ...]:
    bindings = []
    for value in values:
        row_id, separator, raw_root = value.partition("=")
        root = Path(raw_root)
        if not separator or not row_id or not root.is_absolute():
            raise ValueError("--recover-collected-root requires ROW=ABSOLUTE_PATH")
        bindings.append(CollectionRecoveryBinding(row_id=row_id, root=root))
    return tuple(bindings)


def _load_launch_bundle(path: str | Path, resume_run_set: str | None) -> RunBundle:
    if resume_run_set:
        existing = _run_set_dir(resume_run_set) / "bundle.json"
        if existing.exists():
            return _load_bundle(existing)
        return _load_bundle(path).model_copy(update={"run_set_id": resume_run_set})
    return _load_bundle(path)


def _load_existing_bundle(run_set_id: str) -> RunBundle:
    bundle_path = _run_set_dir(run_set_id) / "bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Run-set bundle not found: {bundle_path}")
    return _load_bundle(bundle_path)


def _load_state(run_set_id: str) -> RunSetState:
    return RunSetStateStore(_run_set_dir(run_set_id) / "state.json").load()


def _load_status_state(run_set_id: str) -> RunSetState | EmergencyRunSetRecord:
    store = RunSetStateStore(_run_set_dir(run_set_id) / "state.json")
    emergency = store.load_emergency() if store.emergency_path.exists() else None
    if emergency is not None and emergency.preservation_state != "release-authorized":
        return emergency
    try:
        return store.load()
    except (OSError, ValueError):
        if emergency is not None:
            return emergency
        raise


def _format_emergency_status(record: EmergencyRunSetRecord) -> str:
    return " ".join(
        (
            f"run_set={record.run_set_id}",
            f"recovery={record.preservation_state}",
            f"provider={record.provider_identity.provider}",
            f"resource_id={record.provider_identity.resource_id}",
            f"custody_complete={str(record.custody_complete).lower()}",
            "next_recovery_action=" + json.dumps(record.next_recovery_action),
        )
    )


def _run_set_dir(run_set_id: str) -> Path:
    return default_orchestration_root(run_set_id)


def _write_json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json", exclude_none=True)
    else:
        payload = value
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _state_exit_code(state: RunSetState) -> int:
    if state.abort_reason == "budget-exceeded":
        return EXIT_BUDGET
    if any(row.status == "failed" for row in state.rows.values()):
        return EXIT_ROW_FAILURE
    return EXIT_SUCCESS


def _all_rows_terminal(state: RunSetState) -> bool:
    return all(row.status in TERMINAL_ROW_STATUSES for row in state.rows.values())


def _format_batch(event: RunEvent | None) -> str:
    if event is None:
        return "-/-"
    batch = event.payload.get("batch")
    total = event.payload.get("total_batches")
    if batch is None or total is None:
        return "-/-"
    return f"{batch}/{total}"


def _format_loss(event: RunEvent | None) -> str:
    if event is None or event.payload.get("loss") is None:
        return "-"
    return f"{float(event.payload['loss']):.6g}"


def _format_event_age(event: RunEvent | None, *, now_ms: int | None = None) -> str:
    if event is None:
        return "-"
    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    return str(max(0, int((current_ms - event.emitted_at_ms) / 1000)))


def _print_error(exc: BaseException) -> None:
    print(f"feedbax-orchestrate: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
