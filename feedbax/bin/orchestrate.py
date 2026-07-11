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
from feedbax.orchestration import (
    STAGE_ORDER,
    LocalOrchestrationDriver,
    RunBundle,
    RunEvent,
    RunEventReader,
    RunSetState,
    RunSetStateStore,
    StateLockError,
)
from feedbax.orchestration.bundle import default_orchestration_root
from feedbax.orchestration.conformance import build_default_check_registry
from feedbax.orchestration.drivers.runpod import RunPodDriverConfig, RunPodOrchestrationDriver
from feedbax.orchestration.stages import (
    BudgetExceeded,
    PreflightFailed,
    StageEngine,
    run_preflight_checks,
)
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
    preflight.add_argument("--bundle", required=True, help="RunBundle JSON path")
    preflight.set_defaults(func=cmd_preflight)

    launch = subparsers.add_parser("launch", help="Launch or resume a run bundle")
    launch.add_argument("--bundle", required=True, help="RunBundle JSON path")
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
        sub.set_defaults(func=globals()[f"cmd_{name}"])

    teardown = subparsers.add_parser("teardown", help="Tear down run-set resources")
    teardown.add_argument("--run-set", required=True, help="Run-set id")
    teardown.add_argument("--force", action="store_true", help="Break a stale state lock")
    teardown.set_defaults(func=cmd_teardown)

    return parser


def cmd_preflight(args: argparse.Namespace) -> int:
    bundle = _load_bundle(args.bundle)
    try:
        state = StageEngine(
            bundle=bundle,
            driver=_driver_for_bundle(bundle),
            conformance_registry=build_default_check_registry(),
        ).run(stop_after_stage="PREFLIGHT")
    except PreflightFailed:
        state = RunSetStateStore(bundle.run_set_dir / "state.json").load()
    checks = state.stage("PREFLIGHT").checks or run_preflight_checks(bundle)
    for check in checks:
        detail = f" detail={check.detail}" if check.detail else ""
        print(f"{check.name} {check.status}{detail}")
    return EXIT_PREFLIGHT if any(check.status == "fail" for check in checks) else EXIT_SUCCESS


def cmd_launch(args: argparse.Namespace) -> int:
    bundle = _load_launch_bundle(args.bundle, args.resume_run_set)
    if args.driver:
        bundle = bundle.model_copy(update={"driver": args.driver})
    overrides: dict[str, Any] = {}
    if args.deadman is not None:
        overrides["deadman_enabled"] = args.deadman
    if args.deadman_silence_seconds is not None:
        overrides["deadman_silence_seconds"] = args.deadman_silence_seconds
    if overrides:
        bundle = RunBundle.model_validate({**bundle.model_dump(mode="json"), **overrides})
    with RunInterruptionController() as interruption:
        state = _run_engine(bundle, interruption_probe=interruption.poll)
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
    state = _run_existing(args.run_set, stop_after_stage="COLLECT")
    return _state_exit_code(state)


def cmd_certify(args: argparse.Namespace) -> int:
    state = _run_existing(args.run_set, stop_after_stage="CERTIFY")
    if state.stage("CERTIFY").outputs.get("overall") == "fail":
        return EXIT_PREFLIGHT
    return _state_exit_code(state)


def cmd_teardown(args: argparse.Namespace) -> int:
    state = _run_existing(args.run_set, stop_after_stage="TEARDOWN", break_stale_lock=args.force)
    return _state_exit_code(state)


def cmd_resume(args: argparse.Namespace) -> int:
    with RunInterruptionController() as interruption:
        state = _run_existing(args.run_set, interruption_probe=interruption.poll)
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
    interruption_probe: Callable[[], CancellationDecision | None] | None = None,
) -> RunSetState:
    driver = _driver_for_bundle(bundle)
    state = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=build_default_check_registry(),
        interruption_probe=interruption_probe,
    ).run(
        break_stale_lock=break_stale_lock,
        stop_after_stage=stop_after_stage,
    )
    if state.abort_reason == "budget-exceeded":
        raise BudgetExceeded("budget exceeded")
    return state


def _run_existing(
    run_set_id: str,
    *,
    stop_after_stage: str | None = None,
    break_stale_lock: bool = False,
    interruption_probe: Callable[[], CancellationDecision | None] | None = None,
) -> RunSetState:
    bundle = _load_existing_bundle(run_set_id)
    return _run_engine(
        bundle,
        stop_after_stage=stop_after_stage,
        break_stale_lock=break_stale_lock,
        interruption_probe=interruption_probe,
    )


def _driver_for_bundle(bundle: RunBundle) -> LocalOrchestrationDriver | RunPodOrchestrationDriver:
    if bundle.driver == "local":
        return LocalOrchestrationDriver()
    if bundle.driver == "runpod":
        return RunPodOrchestrationDriver(config=_runpod_config_for_bundle(bundle))
    raise RuntimeError(f"Unsupported orchestration driver: {bundle.driver!r}")


def _load_bundle(path: str | Path) -> RunBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    migrated = default_spec_registry.migrate("RunBundle", payload)
    return RunBundle.model_validate(migrated.payload)


def _runpod_config_for_bundle(bundle: RunBundle) -> RunPodDriverConfig:
    metadata = bundle.environment.metadata
    raw_patches = metadata.get("runpod_path_patches", ())
    path_patches = tuple(
        (str(item["remote_file"]), str(item["from"]), str(item["to"])) for item in raw_patches
    )
    return RunPodDriverConfig(
        pod_id=_optional_string(metadata.get("runpod_pod_id")),
        ssh_host=_optional_string(metadata.get("runpod_ssh_host")),
        ssh_port=(int(metadata["runpod_ssh_port"]) if metadata.get("runpod_ssh_port") else None),
        gpu_id=_optional_string(metadata.get("runpod_gpu_id")),
        datacenters=tuple(str(item) for item in metadata.get("runpod_datacenters", ())),
        min_balance_usd=float(metadata.get("runpod_min_balance_usd", 5.0)),
        image=bundle.environment.image_id or "runpod/pytorch:latest",
        local_repos={
            str(name): str(path) for name, path in metadata.get("runpod_local_repos", {}).items()
        },
        remote_repos={
            str(name): str(path) for name, path in metadata.get("runpod_remote_repos", {}).items()
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
