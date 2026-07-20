"""CLI for Feedbax deterministic run-set orchestration.

Status lines are a consumer contract. Text status output is one line per row:

``row=<id> status=<s> batch=<i>/<n> last_loss=<x|-> last_event_age_s=<t|-> seq=<n>``

followed by ``stages=<stage>:<status>,...`` in orchestration stage order.

Exit codes: 0 success; 2 preflight or conformance failure; 3 row failure;
4 budget exceeded; 5 lock conflict; 1 all other errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.staged_execution import validate_staged_binding_name
from feedbax.orchestration import (
    STAGE_ORDER,
    LocalOrchestrationDriver,
    MatrixAuthorityError,
    AssemblyContext,
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
    run_preflight_checks,
)
from feedbax.orchestration.bundle import default_orchestration_root
from feedbax.orchestration.collection_recovery import CollectionRecoveryBinding
from feedbax.orchestration.conformance import build_default_check_registry
from feedbax.orchestration.drivers.runpod import (
    RunPodDriverConfig,
    RunPodOrchestrationDriver,
    load_runpod_api_key,
)
from feedbax.orchestration.input_materialization import InputProviderRootBinding
from feedbax.orchestration.stages import (
    BudgetExceeded,
    PreflightFailed,
    StageEngine,
)
from feedbax.plugins import load_training_method_plugins
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
    preflight_input.add_argument(
        "--assembly-request", help="RunAssemblyRequest JSON path"
    )
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
    launch.add_argument(
        "--assembly-request", required=True, help="RunAssemblyRequest JSON path"
    )
    launch.add_argument("--driver", choices=["local", "runpod"], help="Driver override")
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
    launch.set_defaults(func=cmd_launch)

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


def cmd_preflight(args: argparse.Namespace) -> int:
    if not args.authority_only:
        if args.bundle:
            raise ValueError("--bundle is supported only with --authority-only")
        if args.bundle_sha256:
            raise ValueError("--bundle-sha256 is supported only with --authority-only")
        load_training_method_plugins(fail_on_load_error=True)
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
                registry=build_default_assembly_registry(),
            )
        checks = (
            run_authority_preflight_checks(bundle)
            if args.bundle
            else run_preflight_checks(bundle)
        )
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
    load_training_method_plugins(fail_on_load_error=True)
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
    with RunInterruptionController() as interruption:
        engine = _request_engine(
            request,
            request_path=request_path,
            run_set_id=args.resume_run_set,
            interruption_probe=interruption.poll,
            input_provider_bindings=_input_provider_bindings(args.input_provider),
        )
        state = engine.run()
    return _state_exit_code(state)


def cmd_status(args: argparse.Namespace) -> int:
    state = _load_state(args.run_set)
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
        collection_recovery_bindings=_collection_recovery_bindings(
            args.recover_collected_root
        ),
    )
    return _state_exit_code(state)


def cmd_certify(args: argparse.Namespace) -> int:
    load_training_method_plugins(fail_on_load_error=True)
    run_options: dict[str, Any] = {
        "stop_after_stage": "CERTIFY",
        "retry_failed_certification": True,
    }
    input_bindings = _input_provider_bindings(args.input_provider)
    if input_bindings:
        run_options["input_provider_bindings"] = input_bindings
    state = _run_existing(args.run_set, **run_options)
    if state.stage("CERTIFY").outputs.get("overall") == "fail":
        return EXIT_PREFLIGHT
    return _state_exit_code(state)


def cmd_teardown(args: argparse.Namespace) -> int:
    state = _run_existing(args.run_set, stop_after_stage="TEARDOWN", break_stale_lock=args.force)
    return _state_exit_code(state)


def cmd_resume(args: argparse.Namespace) -> int:
    load_training_method_plugins(fail_on_load_error=True)
    with RunInterruptionController() as interruption:
        state = _run_existing(
            args.run_set,
            interruption_probe=interruption.poll,
            input_provider_bindings=_input_provider_bindings(args.input_provider),
            collection_recovery_bindings=_collection_recovery_bindings(
                args.recover_collected_root
            ),
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
) -> RunSetState:
    if collection_recovery_bindings:
        driver = _driver_for_bundle(
            bundle,
            input_provider_bindings,
            collection_recovery_bindings=collection_recovery_bindings,
        )
    else:
        driver = _driver_for_bundle(bundle, input_provider_bindings)
    state = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=build_default_check_registry(),
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
        input_provider_bindings=input_provider_bindings,
        collection_recovery_bindings=collection_recovery_bindings,
    )


def _driver_for_bundle(
    bundle: RunBundle,
    bindings: tuple[InputProviderRootBinding, ...] = (),
    collection_recovery_bindings: tuple[CollectionRecoveryBinding, ...] = (),
) -> LocalOrchestrationDriver | RunPodOrchestrationDriver:
    driver_name = bundle.deployment_policy.driver
    if driver_name == "local":
        if collection_recovery_bindings:
            raise ValueError("collection recovery is only supported for a torn-down RunPod run")
        return LocalOrchestrationDriver(input_provider_bindings=bindings)
    if driver_name == "runpod":
        return RunPodOrchestrationDriver(
            config=_runpod_config_for_bundle(bundle),
            input_provider_bindings=bindings,
            collection_recovery_bindings=collection_recovery_bindings,
        )
    raise RuntimeError(f"Unsupported orchestration driver: {driver_name!r}")


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
) -> StageEngine:
    root = Path(request.orchestration_root).expanduser() if request.orchestration_root else request_path.parent
    context = AssemblyContext(
        custody_root=root / "custody",
        repo_root=Path.cwd(),
    )
    return StageEngine.from_request(
        request,
        context=context,
        registry=build_default_assembly_registry(),
        driver_factory=lambda bundle: _driver_for_bundle(bundle, input_provider_bindings),
        run_set_id=run_set_id,
        conformance_registry=build_default_check_registry(),
        interruption_probe=interruption_probe,
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


def _runpod_config_for_bundle(bundle: RunBundle) -> RunPodDriverConfig:
    metadata = bundle.environment.metadata
    resources = bundle.deployment_policy.resources
    raw_patches = metadata.get("runpod_path_patches", ())
    path_patches = tuple(
        (str(item["remote_file"]), str(item["from"]), str(item["to"])) for item in raw_patches
    )
    return RunPodDriverConfig(
        pod_id=_optional_string(metadata.get("runpod_pod_id")),
        ssh_host=_optional_string(metadata.get("runpod_ssh_host")),
        ssh_port=(int(metadata["runpod_ssh_port"]) if metadata.get("runpod_ssh_port") else None),
        gpu_id=resources.gpu_id,
        datacenters=tuple(resources.regions),
        api_key=load_runpod_api_key(),
        min_balance_usd=float(metadata.get("runpod_min_balance_usd", 5.0)),
        image=bundle.environment.image_id or "runpod/pytorch:latest",
        local_repos={
            str(name): str(path) for name, path in metadata.get("runpod_local_repos", {}).items()
        },
        remote_repos={
            str(name): str(path) for name, path in metadata.get("runpod_remote_repos", {}).items()
        },
        protected_refs={
            str(name): str(ref) for name, ref in metadata.get("runpod_protected_refs", {}).items()
        },
        path_patches=path_patches,
    )


def _optional_string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


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
