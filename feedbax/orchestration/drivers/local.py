"""Local subprocess orchestration driver."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from feedbax.contracts.evaluation_lifecycle import (
    EVALUATION_COLLECTION_OUTPUTS,
    EvaluationMatrixOrderedUnionEvidence,
)
from feedbax.orchestration.bundle import (
    ResolvedAssemblyInput,
    RunBundle,
    RunRowSpec,
    environment_declaration_identity_projection,
)
from feedbax.orchestration.conformance import (
    RunConformanceCertificate,
    assert_certificate_allows_completed_registration,
)
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.drivers.capabilities import (
    AcquisitionSemantics,
    AuthorizationSemantics,
    CustodySemantics,
    DriverCapabilityEnvelope,
    DriverCapabilityFacts,
    DriverConstructionContext,
    DriverHook,
    DriverRegistration,
    DriverVenue,
    EnvironmentSemantics,
    MonitoringSemantics,
    RecoverySemantics,
    RealizedDriverCapabilities,
    ResourceSemantics,
    RetrySemantics,
    SpendSemantics,
    TeardownSemantics,
)
from feedbax.orchestration.drivers.native_execution import (
    native_resume_checkpoint_source,
    seed_authenticated_checkpoint,
)
from feedbax.orchestration.executor_family import executor_family_adapter
from feedbax.orchestration.input_materialization import (
    InputMaterializationError,
    InputProviderRootBinding,
    materialize_bundle_inputs,
    preflight_bundle_input_bindings,
    reclaim_materialized_staged_roots,
)
from feedbax.orchestration.staged_root_custody import StagedRootSnapshotBinding
from feedbax.orchestration.state import PreflightCheckEntry, RunSetState
from feedbax.training import publish_directory_no_replace


class LocalDriverError(RuntimeError):
    """Raised when the local driver cannot complete a requested action."""


_DEPENDENCY_INVENTORY_SCRIPT = """
import importlib.metadata
import json
import platform
import sys

inventory = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"]
    version = distribution.version
    if not name or not version:
        raise RuntimeError("installed distribution lacks required name or version metadata")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text is not None else None
    if direct_url is not None and not isinstance(direct_url, dict):
        raise RuntimeError(f"distribution {name!r} has invalid direct_url.json")
    inventory.append({"direct_url": direct_url, "name": name, "version": version})
payload = {
    "schema_version": "feedbax.local_dependency_inventory.v1",
    "interpreter": {
        "cache_tag": sys.implementation.cache_tag,
        "executable": sys.executable,
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
    },
    "distributions": inventory,
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""


class LocalOrchestrationDriver:
    """Run orchestration rows as local subprocesses under the run-set directory."""

    poll_interval_seconds = 0.05

    capability_envelope = DriverCapabilityEnvelope(
        driver_name="local",
        variants={
            "local-stop": DriverCapabilityFacts(
            variant_id="local-stop",
            venue=DriverVenue.LOCAL_PROCESS,
            resources=ResourceSemantics.LOCAL_PROCESS,
            spend=SpendSemantics.NONE,
            authorization=AuthorizationSemantics.NONE,
            environment=EnvironmentSemantics.LOCAL_INVENTORY,
            monitoring=MonitoringSemantics.ROW_POLL,
            recovery=RecoverySemantics.PROCESS_LOCAL,
            retry=RetrySemantics.NONE,
            acquisition=AcquisitionSemantics.NONE,
            teardown=TeardownSemantics.LOCAL_PROCESS_STOP,
            custody=CustodySemantics.LOCAL_RUN_SET,
            optional_hooks=frozenset(
                {
                    DriverHook.PREFLIGHT_CHECKS,
                    DriverHook.CHECKPOINT_STOP,
                }
            ),
            ),
            "local-preserved": DriverCapabilityFacts(
                variant_id="local-preserved",
                venue=DriverVenue.LOCAL_PROCESS,
                resources=ResourceSemantics.LOCAL_PROCESS,
                spend=SpendSemantics.NONE,
                authorization=AuthorizationSemantics.NONE,
                environment=EnvironmentSemantics.LOCAL_INVENTORY,
                monitoring=MonitoringSemantics.ROW_POLL,
                recovery=RecoverySemantics.PROCESS_LOCAL,
                retry=RetrySemantics.NONE,
                acquisition=AcquisitionSemantics.NONE,
                teardown=TeardownSemantics.RESOURCES_PRESERVED,
                custody=CustodySemantics.LOCAL_RUN_SET,
                optional_hooks=frozenset({DriverHook.PREFLIGHT_CHECKS}),
            ),
        },
    )
    realized_capabilities = capability_envelope.realize("local-stop")

    def __init__(
        self,
        *,
        cwd: Path | str | None = None,
        python_executable: str | None = None,
        freeze_lines: Sequence[str] | None = None,
        input_provider_bindings: Sequence[InputProviderRootBinding] = (),
        staged_root_bindings: Sequence[StagedRootSnapshotBinding] = (),
        update_budget: int | None = None,
        realized_capabilities: RealizedDriverCapabilities | None = None,
    ) -> None:
        self.realized_capabilities = realized_capabilities or type(self).realized_capabilities
        self.cwd = Path(cwd or Path.cwd())
        self.python_executable = python_executable or sys.executable
        self.freeze_lines = tuple(freeze_lines) if freeze_lines is not None else None
        self.input_provider_bindings = tuple(input_provider_bindings)
        self.staged_root_bindings = tuple(staged_root_bindings)
        self.update_budget = update_budget
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def preflight_checks(self, bundle: RunBundle) -> list[PreflightCheckEntry]:
        """Authenticate every local input binding before PROVISION."""
        if not bundle.staged_roots:
            return []
        failures, observed = preflight_bundle_input_bindings(
            bundle,
            provider_bindings=self.input_provider_bindings,
            staged_root_bindings=self.staged_root_bindings,
        )
        return [
            PreflightCheckEntry(
                name="local-input-bindings",
                status="fail" if failures else "pass",
                detail="; ".join(failures) if failures else None,
                observed=observed or "no-resolved-inputs",
            )
        ]
    def provision(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        run_set_dir = bundle.run_set_dir
        for dirname in ("events", "sentinels", "rows", "collected"):
            (run_set_dir / dirname).mkdir(parents=True, exist_ok=True)
        return {"driver": "local", "run_set_dir": str(run_set_dir)}

    def realize_env(self, bundle: RunBundle, state: RunSetState) -> str:
        return compute_environment_fingerprint(
            bundle,
            cwd=self.cwd,
            python_executable=self.python_executable,
            freeze_lines=self.freeze_lines,
        )

    def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        inputs_dir = bundle.run_set_dir / "inputs"
        if os.path.lexists(inputs_dir):
            raise LocalDriverError(f"input destination already exists: {inputs_dir}")
        attempt_root = (
            bundle.run_set_dir
            / ".stage-attempts"
            / (f"stage-inputs-{state.stage('STAGE_INPUTS').attempts}")
        )
        try:
            staged_inputs = materialize_bundle_inputs(
                bundle,
                destination_root=attempt_root,
                provider_bindings=self.input_provider_bindings,
                staged_root_bindings=self.staged_root_bindings,
            )
        except InputMaterializationError as exc:
            raise LocalDriverError(str(exc)) from exc
        payloads: list[dict[str, str]] = []
        if bundle.execution_family == "evaluation-matrix":
            bundle_target = attempt_root / "inputs" / "run-bundle.json"
            bundle_target.write_text(
                bundle.model_dump_json(exclude_none=True),
                encoding="utf-8",
            )
        for row in bundle.rows:
            if row.launch.payload_routing.get("kind") != "registered-execution-payload":
                continue
            source = Path(row.execution.payload.uri or "")
            if not source.is_file():
                raise LocalDriverError(
                    f"registered execution payload is not materialized for row {row.row_id!r}"
                )
            if _sha256_file(source) != row.execution.payload.sha256:
                raise LocalDriverError(
                    f"registered execution payload digest mismatch for row {row.row_id!r}"
                )
            target = attempt_root / "inputs" / f"{row.row_id}.json"
            shutil.copy2(source, target)
            payloads.append(
                {
                    "row_id": row.row_id,
                    "source": str(source),
                    "target": str(inputs_dir / target.name),
                }
            )
        parent_descriptor = os.open(
            bundle.run_set_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            source_name = (attempt_root / "inputs").relative_to(bundle.run_set_dir).as_posix()
            publish_directory_no_replace(
                parent_descriptor,
                source_name,
                "inputs",
                expected_identity=os.stat(
                    source_name, dir_fd=parent_descriptor, follow_symlinks=False
                ),
            )
        finally:
            os.close(parent_descriptor)
        return {
            "input_count": len(staged_inputs),
            "inputs": [
                {
                    "target_role": staged.target_role,
                    "destination": str(inputs_dir / staged.target_role),
                    "files": [asdict(item) for item in staged.files],
                }
                for staged in staged_inputs
            ],
            "inputs_dir": str(inputs_dir),
            "payload_count": len(payloads),
            "payloads": payloads,
        }

    def launch_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        paths = _row_paths(bundle, row.row_id)
        if paths["done"].exists():
            return {"row_id": row.row_id, "status": "completed"}
        if paths["failed"].exists():
            return {"row_id": row.row_id, "status": "failed"}
        if row.row_id in self._processes and self._processes[row.row_id].poll() is None:
            return {"row_id": row.row_id, "pid": self._processes[row.row_id].pid}
        if paths["started"].exists():
            pid = _read_pid(paths["pid"])
            if pid and _pid_alive(pid):
                self._processes.pop(row.row_id, None)
                return {
                    "row_id": row.row_id,
                    "pid": pid,
                    "status": "launched",
                    "adopted": True,
                }
            discrepancy = {
                "code": "orphaned_launch",
                "message": (
                    "orphaned launch: started sentinel present, process dead, no terminal sentinel"
                ),
                "pid": pid,
            }
            paths["sentinels"].mkdir(parents=True, exist_ok=True)
            paths["failed"].write_text(discrepancy["message"] + "\n", encoding="utf-8")
            return {
                "row_id": row.row_id,
                "pid": pid,
                "status": "failed",
                "detail": discrepancy["message"],
                "event_discrepancies": [discrepancy],
            }

        paths["row_dir"].mkdir(parents=True, exist_ok=True)
        paths["sentinels"].mkdir(parents=True, exist_ok=True)
        paths["events"].mkdir(parents=True, exist_ok=True)
        checkpoint_source = native_resume_checkpoint_source(bundle, row)
        if checkpoint_source is not None:
            _seed_native_resume_checkpoint_root(
                bundle.run_set_dir / "inputs" / checkpoint_source.custody.target_role,
                paths["row_dir"],
                checkpoint_source,
            )
        paths["started"].write_text(str(time.time()), encoding="utf-8")

        env = os.environ.copy()
        env.update(
            {
                "FEEDBAX_RUN_SET_ID": bundle.run_set_id,
                "FEEDBAX_ROW_ID": row.row_id,
                "FEEDBAX_RUN_EVENTS_DIR": str(paths["events"]),
                "FEEDBAX_ENV_FINGERPRINT": state.environment_fingerprint or "",
                "FEEDBAX_ROW_DIR": str(paths["row_dir"]),
            }
        )
        env["PYTHONPATH"] = _prepend_feedbax_source_root(env.get("PYTHONPATH"))
        command, bound_row = executor_family_adapter(row.execution_family).bind_command(
            _row_command(row, self.python_executable),
            bundle=bundle,
            row=row,
            payload_path=bundle.run_set_dir / "inputs" / f"{row.row_id}.json",
            collection_root=paths["row_dir"],
            inputs_root=bundle.run_set_dir / "inputs",
            repo_root=self.cwd,
            environment_fingerprint=state.environment_fingerprint or "",
            update_budget=(
                self.update_budget if row.execution_family == "native-training" else None
            ),
        )
        stdout = (paths["row_dir"] / "stdout.log").open("ab")
        stderr = (paths["row_dir"] / "stderr.log").open("ab")
        process = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        stdout.close()
        stderr.close()
        self._processes[row.row_id] = process
        paths["pid"].write_text(f"{process.pid}\n", encoding="utf-8")
        return {"row_id": row.row_id, "pid": process.pid, "command": command}

    def probe(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> DriverRowProbe:
        paths = _row_paths(bundle, row.row_id)
        process = self._processes.get(row.row_id)
        if process is not None:
            returncode = process.poll()
            if returncode is None:
                return DriverRowProbe(status="running", pid=process.pid)
            if returncode == 0:
                paths["done"].write_text("0\n", encoding="utf-8")
                return DriverRowProbe(status="completed", pid=process.pid)
            paths["failed"].write_text(f"{returncode}\n", encoding="utf-8")
            return DriverRowProbe(status="failed", pid=process.pid, detail=f"exit={returncode}")
        if paths["done"].exists():
            return DriverRowProbe(status="completed", pid=_read_pid(paths["pid"]))
        if paths["failed"].exists():
            return DriverRowProbe(status="failed", pid=_read_pid(paths["pid"]))
        if paths["pid"].exists():
            pid = _read_pid(paths["pid"])
            if pid and _pid_alive(pid):
                return DriverRowProbe(status="running", pid=pid)
            return DriverRowProbe(status="failed", pid=pid, detail="pid exited without sentinel")
        return DriverRowProbe(status="pending")

    def stop_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        paths = _row_paths(bundle, row.row_id)
        process = self._processes.get(row.row_id)
        pid = process.pid if process is not None else _read_pid(paths["pid"])
        if pid:
            _terminate_process_group(pid, process=process)
        paths["failed"].write_text("stopped\n", encoding="utf-8")
        return {"row_id": row.row_id, "pid": pid, "status": "stopped"}

    def request_stop_at_checkpoint(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        """Ask a local row to stop itself at its next durable checkpoint."""
        del state
        paths = _row_paths(bundle, row.row_id)
        process = self._processes.get(row.row_id)
        pid = process.pid if process is not None else _read_pid(paths["pid"])
        if pid and _pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            except OSError:
                os.kill(pid, signal.SIGINT)
        return {"row_id": row.row_id, "pid": pid, "status": "stop_requested"}

    def collect(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, str]:
        paths = _row_paths(bundle, row.row_id)
        dest_dir = bundle.run_set_dir / "collected" / row.row_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        collected: dict[str, str] = {}
        sources = row.launch.collect or [str(paths["event_log"])]
        failed_row = state.rows[row.row_id].status in {"failed", "stopped"}
        for source in sources:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = paths["row_dir"] / source_path
            if bundle.execution_family == "evaluation-matrix" and os.path.abspath(
                source_path
            ) == os.path.abspath(paths["row_dir"] / "evaluation"):
                # Compact products supersede this raw working store. Older
                # bundles may still declare it, but collection must not create
                # a second terminal copy.
                continue
            if not source_path.exists():
                continue
            # A failed executor may leave a large, actively-written raw/cache tree.
            # Small declared files remain useful failure evidence, but recursive
            # directory collection is terminal-output publication and is therefore
            # forbidden until the executor has completed successfully.
            if failed_row and source_path.is_dir():
                continue
            dest = dest_dir / source_path.name
            if source_path.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(source_path, dest)
            else:
                shutil.copy2(source_path, dest)
            collected[source_path.name] = str(dest)
        payload_sha256 = row.launch.metadata.get("payload_sha256")
        if payload_sha256 and collected:
            first = Path(next(iter(collected.values())))
            if first.is_file() and _sha256_file(first) != str(payload_sha256):
                raise LocalDriverError(f"payload sha256 mismatch for row {row.row_id!r}")
        return collected

    def teardown(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        if self.realized_capabilities.facts.teardown is TeardownSemantics.RESOURCES_PRESERVED:
            return {
                "driver": "local",
                "teardown": "skipped",
                "skip_reason": "realized-capability-preserves-resources",
                "stopped_rows": [],
            }
        stopped: list[str] = []
        for row in bundle.rows:
            probe = self.probe(bundle, row, state)
            paths = _row_paths(bundle, row.row_id)
            pid = probe.pid or _read_pid(paths["pid"])
            if probe.status == "running" or (pid and _process_group_alive(pid)):
                self.stop_row(bundle, row, state)
                stopped.append(row.row_id)
        failed_row_reclamation = [
            _reclaim_failed_row_tree(bundle, row)
            for row in bundle.rows
            if state.rows[row.row_id].status in {"failed", "stopped"}
            and state.stage("COLLECT").status in {"completed", "failed"}
        ]
        successful_evaluation_reclamation = _reclaim_successful_evaluation_stores(
            bundle,
            state,
        )
        reclamation: dict[str, Any] = {
            "status": "retained",
            "custody_refs": [custody.custody_ref for custody in bundle.staged_roots],
            "reclaimed_bytes": 0,
            "reason": "terminal-consumer-barrier-not-complete",
        }
        if _terminal_staged_root_reclamation_allowed(bundle, state):
            result = reclaim_materialized_staged_roots(
                bundle,
                inputs_root=bundle.run_set_dir / "inputs",
            )
            reclamation = {
                "status": result.status,
                "custody_refs": list(result.custody_refs),
                "reclaimed_bytes": result.reclaimed_bytes,
            }
        return {
            "driver": "local",
            "stopped_rows": stopped,
            "failed_row_reclamation": failed_row_reclamation,
            "successful_evaluation_reclamation": successful_evaluation_reclamation,
            "staged_root_reclamation": reclamation,
        }


def compute_environment_fingerprint(
    bundle: RunBundle,
    *,
    cwd: Path | str,
    python_executable: str | None = None,
    freeze_lines: Sequence[str] | None = None,
) -> str:
    """Compute a deterministic local environment fingerprint."""
    root = Path(cwd)
    interpreter, discovered_packages = _probe_dependency_inventory(python_executable)
    package_lines = list(freeze_lines) if freeze_lines is not None else discovered_packages
    payload = environment_declaration_identity_projection(bundle.environment)
    payload["declared_python_version"] = payload.pop("python_version")
    payload["repo_revisions"] = []
    payload.update(
        {
            "dependency_inventory_source": (
                "provided" if freeze_lines is not None else "importlib.metadata"
            ),
            "interpreter": interpreter,
            "packages": sorted(package_lines),
        }
    )
    for revision in bundle.environment.repo_revisions:
        repo_path = root / revision.path
        dirty = _git_dirty(repo_path)
        if dirty and not revision.dirty_allowed:
            raise LocalDriverError(f"dirty repo not allowed for fingerprint: {repo_path}")
        payload["repo_revisions"].append(
            {
                "path": revision.path,
                "revision": revision.revision,
                "dirty_allowed": revision.dirty_allowed,
                "head": _git_output(repo_path, "rev-parse", "HEAD") or None,
                "patch_sha256": _git_patch_sha256(repo_path) if dirty else None,
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _terminal_staged_root_reclamation_allowed(
    bundle: RunBundle,
    state: RunSetState,
) -> bool:
    if not bundle.staged_roots:
        return False
    if any(state.rows.get(row.row_id) is None for row in bundle.rows):
        return False
    row_statuses = {state.rows[row.row_id].status for row in bundle.rows}
    if not row_statuses <= {"completed", "failed", "stopped"}:
        return False
    if state.stage("STAGE_INPUTS").status != "completed":
        return False
    if row_statuses == {"completed"}:
        if any(state.stage(stage_id).status != "completed" for stage_id in ("COLLECT", "CERTIFY")):
            return False
    elif state.stage("COLLECT").status not in {"completed", "failed"}:
        return False
    if bundle.execution_family == "evaluation-matrix":
        ordered_union = state.stage("COLLECT").outputs.get("evaluation_matrix_ordered_union")
        if row_statuses == {"completed"} and not isinstance(ordered_union, Mapping):
            return False
    return True


def _reclaim_successful_evaluation_stores(
    bundle: RunBundle,
    state: RunSetState,
) -> list[Mapping[str, Any]]:
    """Reclaim raw evaluation stores after their compact terminal custody is certified."""
    if bundle.execution_family != "evaluation-matrix":
        return []
    completed_rows = [row for row in bundle.rows if state.rows.get(row.row_id, None) is not None]
    if (
        len(completed_rows) != len(bundle.rows)
        or any(state.rows[row.row_id].status != "completed" for row in completed_rows)
        or state.stage("COLLECT").status != "completed"
        or state.stage("CERTIFY").status != "completed"
    ):
        return [
            {
                "row_id": row.row_id,
                "status": "retained",
                "reclaimed_bytes": 0,
                "reason": "terminal-consumer-barrier-not-complete",
            }
            for row in completed_rows
        ]
    _verify_successful_evaluation_terminal_custody(bundle, state)
    return [_reclaim_successful_evaluation_store(bundle, row) for row in completed_rows]


def _verify_successful_evaluation_terminal_custody(
    bundle: RunBundle,
    state: RunSetState,
) -> None:
    collect = state.stage("COLLECT")
    collected_rows = collect.outputs.get("rows")
    if not isinstance(collected_rows, Mapping):
        raise LocalDriverError("evaluation reclamation requires durable collected-row outputs")
    ordered_union = collect.outputs.get("evaluation_matrix_ordered_union")
    if not isinstance(ordered_union, Mapping):
        raise LocalDriverError("evaluation reclamation requires durable ordered-union evidence")
    union_payload = dict(ordered_union)
    union_path = union_payload.pop("path", None)
    expected_union_path = bundle.run_set_dir / "evaluation-matrix-ordered-union.json"
    if not isinstance(union_path, str) or Path(union_path) != expected_union_path:
        raise LocalDriverError("evaluation reclamation ordered-union path is not run-owned")
    _require_run_owned_path(
        expected_union_path,
        root=bundle.run_set_dir,
        directory=False,
    )
    try:
        recorded_union = EvaluationMatrixOrderedUnionEvidence.model_validate(union_payload)
        durable_union = EvaluationMatrixOrderedUnionEvidence.model_validate_json(
            expected_union_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise LocalDriverError("evaluation reclamation ordered-union evidence is invalid") from exc
    if recorded_union != durable_union:
        raise LocalDriverError("evaluation reclamation ordered-union evidence drifted")
    for row in bundle.rows:
        outputs = state.rows[row.row_id].collected_outputs
        recorded_outputs = collected_rows.get(row.row_id)
        if not isinstance(recorded_outputs, Mapping) or dict(recorded_outputs) != outputs:
            raise LocalDriverError(
                f"evaluation reclamation collected outputs drifted for row {row.row_id!r}"
            )
        event_name = f"{row.row_id}.events.jsonl"
        for name, path in outputs.items():
            if not isinstance(name, str) or not isinstance(path, str):
                raise LocalDriverError(
                    f"evaluation reclamation collected output is invalid for row {row.row_id!r}"
                )
            expected_path = (
                bundle.run_set_dir / "events" / event_name
                if name == event_name
                else bundle.run_set_dir / "collected" / row.row_id / name
            )
            if Path(path) != expected_path:
                raise LocalDriverError(
                    f"evaluation reclamation collected output path is not run-owned for "
                    f"row {row.row_id!r}: {name!r}"
                )
            _require_run_owned_path(
                expected_path,
                root=bundle.run_set_dir,
                directory=name == "evaluation-batch-compaction",
            )
        missing_declared = set(EVALUATION_COLLECTION_OUTPUTS) - outputs.keys()
        if missing_declared:
            raise LocalDriverError(
                f"evaluation reclamation compact outputs are incomplete for row "
                f"{row.row_id!r}: {sorted(missing_declared)!r}"
            )
        missing = executor_family_adapter(row.execution_family).missing_collection_outputs(
            bundle,
            row,
            outputs,
        )
        if missing:
            raise LocalDriverError(
                f"evaluation reclamation compact outputs are incomplete for row "
                f"{row.row_id!r}: {missing!r}"
            )
    certify = state.stage("CERTIFY")
    certificate_ref = certify.outputs.get("certificate_ref")
    certificate_sha256 = certify.outputs.get("certificate_sha256")
    expected_certificate = bundle.run_set_dir / "conformance.json"
    if (
        certify.outputs.get("overall") != "pass"
        or state.certificate_ref != str(expected_certificate)
        or certificate_ref != str(expected_certificate)
        or not isinstance(certificate_sha256, str)
    ):
        raise LocalDriverError("evaluation reclamation requires a durable passing certificate")
    _require_run_owned_path(
        expected_certificate,
        root=bundle.run_set_dir,
        directory=False,
    )
    if _sha256_file(expected_certificate) != certificate_sha256:
        raise LocalDriverError("evaluation reclamation requires a durable passing certificate")
    try:
        certificate = RunConformanceCertificate.model_validate_json(
            expected_certificate.read_text(encoding="utf-8")
        )
        assert_certificate_allows_completed_registration(certificate)
    except (OSError, ValueError) as exc:
        raise LocalDriverError(
            "evaluation reclamation requires a valid passing certificate"
        ) from exc
    if certificate.run_set_id != bundle.run_set_id or set(certificate.rows) != {
        row.row_id for row in bundle.rows
    }:
        raise LocalDriverError("evaluation reclamation certificate authority drifted")


def _require_run_owned_path(path: Path, *, root: Path, directory: bool) -> None:
    """Require an existing non-symlink file or directory at an exact run-owned path."""
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise LocalDriverError(
            f"evaluation reclamation durable path is outside the run set: {path}"
        ) from exc
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise LocalDriverError(f"evaluation reclamation durable path is unsafe: {path}")
    if not os.path.lexists(path):
        raise LocalDriverError(f"evaluation reclamation durable path is unsafe: {path}")
    if directory:
        valid = path.is_dir()
    else:
        valid = path.is_file()
    if not valid:
        raise LocalDriverError(f"evaluation reclamation durable path has wrong type: {path}")


def _reclaim_successful_evaluation_store(
    bundle: RunBundle,
    row: RunRowSpec,
) -> Mapping[str, Any]:
    """Atomically isolate and remove one run-owned raw evaluation store."""
    row_root = bundle.run_set_dir / "rows" / row.row_id
    _require_run_owned_path(
        row_root,
        root=bundle.run_set_dir,
        directory=True,
    )
    raw_root = row_root / "evaluation"
    isolated = row_root / ".evaluation.success-reclaiming"
    record_path = row_root / ".evaluation-store-reclamation.json"
    if os.path.lexists(raw_root) and os.path.lexists(isolated):
        raise LocalDriverError(
            f"raw evaluation store and reclamation isolate both exist for {row.row_id!r}"
        )
    record: dict[str, Any] | None = None
    if record_path.is_symlink():
        raise LocalDriverError(f"raw evaluation reclamation record is unsafe for {row.row_id!r}")
    if record_path.is_file():
        try:
            loaded = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalDriverError(
                f"raw evaluation reclamation record is invalid for {row.row_id!r}"
            ) from exc
        if not isinstance(loaded, dict):
            raise LocalDriverError(
                f"raw evaluation reclamation record is invalid for {row.row_id!r}"
            )
        record = loaded
        if (
            record.get("schema_version")
            != "feedbax.orchestration.local_evaluation_store_reclamation.v1"
            or record.get("row_id") != row.row_id
            or record.get("source") != str(raw_root)
            or record.get("status") not in {"deleting", "completed"}
            or not isinstance(record.get("reclaimed_bytes"), int)
            or record["reclaimed_bytes"] < 0
        ):
            raise LocalDriverError(f"raw evaluation reclamation record drifted for {row.row_id!r}")
        if record["status"] == "completed":
            if os.path.lexists(raw_root) or os.path.lexists(isolated):
                raise LocalDriverError(
                    f"reclaimed raw evaluation store reappeared for {row.row_id!r}"
                )
            return {
                "row_id": row.row_id,
                "status": "already-reclaimed",
                "reclaimed_bytes": 0,
            }
    elif not os.path.lexists(raw_root):
        if os.path.lexists(isolated):
            raise LocalDriverError(
                f"raw evaluation reclamation isolate lacks intent for {row.row_id!r}"
            )
        raise LocalDriverError(
            f"raw evaluation store disappeared without reclamation authority for {row.row_id!r}"
        )

    target = isolated if os.path.lexists(isolated) else raw_root
    if record is None:
        if target.is_symlink() or not target.is_dir():
            raise LocalDriverError(f"raw evaluation store is not a run-owned directory: {target}")
        reclaimed_bytes = sum(
            entry.stat(follow_symlinks=False).st_blocks * 512
            for entry in (target, *target.rglob("*"))
            if not entry.is_symlink()
        )
        record = {
            "schema_version": "feedbax.orchestration.local_evaluation_store_reclamation.v1",
            "row_id": row.row_id,
            "source": str(raw_root),
            "reclaimed_bytes": reclaimed_bytes,
            "status": "deleting",
        }
        _atomic_write_local_json(record_path, record)
    else:
        reclaimed_bytes = record["reclaimed_bytes"]

    if os.path.lexists(target):
        if target.is_symlink() or not target.is_dir():
            raise LocalDriverError(f"raw evaluation store is not a run-owned directory: {target}")
        if target == raw_root:
            os.replace(raw_root, isolated)
        shutil.rmtree(isolated)
    record["status"] = "completed"
    _atomic_write_local_json(record_path, record)
    return {
        "row_id": row.row_id,
        "status": "reclaimed",
        "reclaimed_bytes": reclaimed_bytes,
    }


def _atomic_write_local_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one run-local JSON record without exposing partial bytes."""
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _terminate_process_group(
    pid: int,
    *,
    process: subprocess.Popen[bytes] | None,
    timeout_seconds: float = 2.0,
) -> None:
    """Terminate a row's whole process group even when its leader already exited."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    deadline = time.monotonic() + timeout_seconds
    while _process_group_alive(pid) and time.monotonic() < deadline:
        if process is not None:
            process.poll()
        time.sleep(0.01)
    if _process_group_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process is not None:
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise LocalDriverError(f"local row process group {pid} did not terminate") from exc
    deadline = time.monotonic() + timeout_seconds
    while _process_group_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _process_group_alive(pid):
        raise LocalDriverError(f"local row process group {pid} remained live after SIGKILL")


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reclaim_failed_row_tree(
    bundle: RunBundle,
    row: RunRowSpec,
) -> Mapping[str, Any]:
    """Atomically isolate and remove one failed run-owned row tree."""
    rows_root = bundle.run_set_dir / "rows"
    row_root = rows_root / row.row_id
    isolated = rows_root / f".{row.row_id}.failed-reclaiming"
    if os.path.lexists(row_root) and os.path.lexists(isolated):
        raise LocalDriverError(
            f"failed row tree and reclamation isolate both exist for {row.row_id!r}"
        )
    target = isolated if os.path.lexists(isolated) else row_root
    if not os.path.lexists(target):
        return {"row_id": row.row_id, "status": "already-reclaimed", "reclaimed_bytes": 0}
    if target.is_symlink() or not target.is_dir():
        raise LocalDriverError(f"failed row tree is not a run-owned directory: {target}")
    reclaimed_bytes = sum(
        entry.stat(follow_symlinks=False).st_blocks * 512
        for entry in target.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )
    if target == row_root:
        os.replace(row_root, isolated)
    shutil.rmtree(isolated)
    return {
        "row_id": row.row_id,
        "status": "reclaimed",
        "reclaimed_bytes": reclaimed_bytes,
    }


def _row_command(row: RunRowSpec, python_executable: str) -> list[str]:
    if row.launch.command:
        return [str(part) for part in row.launch.command]
    assert row.launch.entry is not None
    return [python_executable, row.launch.entry]


def _prepend_feedbax_source_root(existing: str | None) -> str:
    """Make this checkout importable to rows launched from an arbitrary cwd."""
    source_root = str(Path(__file__).resolve().parents[3])
    return source_root if not existing else os.pathsep.join((source_root, existing))


def _row_paths(bundle: RunBundle, row_id: str) -> dict[str, Path]:
    run_set_dir = bundle.run_set_dir
    sentinels = run_set_dir / "sentinels"
    events = run_set_dir / "events"
    row_dir = run_set_dir / "rows" / row_id
    return {
        "sentinels": sentinels,
        "events": events,
        "row_dir": row_dir,
        "started": sentinels / f"{row_id}.started",
        "pid": sentinels / f"{row_id}.pid",
        "done": sentinels / f"{row_id}.done",
        "failed": sentinels / f"{row_id}.failed",
        "event_log": events / f"{row_id}.events.jsonl",
    }


def _probe_dependency_inventory(
    python_executable: str | None,
) -> tuple[dict[str, str | None], list[str]]:
    """Read interpreter identity and distribution provenance from the selected Python."""
    executable = python_executable or sys.executable
    try:
        result = subprocess.run(
            [executable, "-c", _DEPENDENCY_INVENTORY_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", None)
        detail = str(stderr).strip() if stderr else str(exc)
        raise LocalDriverError(
            f"dependency inventory failed for interpreter {executable!r}: {detail}"
        ) from exc

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} was not valid JSON"
        ) from exc
    if not isinstance(probe, dict) or set(probe) != {
        "distributions",
        "interpreter",
        "schema_version",
    }:
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} had invalid structure"
        )
    if probe["schema_version"] != "feedbax.local_dependency_inventory.v1":
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} had unknown schema"
        )
    interpreter = probe["interpreter"]
    if (
        not isinstance(interpreter, dict)
        or set(interpreter) != {"cache_tag", "executable", "implementation", "version"}
        or not isinstance(interpreter["executable"], str)
        or not interpreter["executable"]
        or not isinstance(interpreter["implementation"], str)
        or not interpreter["implementation"]
        or not isinstance(interpreter["version"], str)
        or not interpreter["version"]
        or (interpreter["cache_tag"] is not None and not isinstance(interpreter["cache_tag"], str))
    ):
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} had invalid interpreter identity"
        )
    inventory = probe["distributions"]
    return dict(interpreter), _canonicalize_dependency_inventory(inventory, executable=executable)


def _canonicalize_dependency_inventory(inventory: Any, *, executable: str) -> list[str]:
    """Validate and canonicalize importlib distribution records."""
    if not isinstance(inventory, list) or any(
        not isinstance(entry, dict)
        or set(entry) != {"direct_url", "name", "version"}
        or not isinstance(entry["name"], str)
        or not entry["name"]
        or not isinstance(entry["version"], str)
        or not entry["version"]
        or (entry["direct_url"] is not None and not isinstance(entry["direct_url"], dict))
        for entry in inventory
    ):
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} had invalid structure"
        )
    by_name: dict[str, dict[str, Any]] = {}
    for raw_entry in inventory:
        entry = dict(raw_entry)
        entry["name"] = re.sub(r"[-_.]+", "-", entry["name"]).lower()
        previous = by_name.get(entry["name"])
        if previous is not None and previous != entry:
            raise LocalDriverError(
                "dependency inventory contains conflicting distributions for normalized name "
                f"{entry['name']!r}"
            )
        by_name[entry["name"]] = entry
    return [
        json.dumps(by_name[name], sort_keys=True, separators=(",", ":")) for name in sorted(by_name)
    ]


def _git_output(cwd: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _git_dirty(cwd: Path) -> bool:
    return bool(_git_output(cwd, "status", "--porcelain"))


def _git_patch_sha256(cwd: Path) -> str:
    diff = _git_output(cwd, "diff", "--binary")
    untracked = _git_output(cwd, "ls-files", "--others", "--exclude-standard")
    h = hashlib.sha256()
    h.update(diff.encode("utf-8"))
    h.update(untracked.encode("utf-8"))
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_native_resume_checkpoint_root(
    source: Path, row_dir: Path, resolved: ResolvedAssemblyInput
) -> None:
    """Clone one staged checkpoint tree and publish it without replacement."""

    if not source.is_dir() or source.is_symlink():
        raise LocalDriverError(f"native resume checkpoint source is not a directory: {source}")
    target = row_dir / "checkpoints"
    attempt = row_dir / ".checkpoint-seed-attempt"
    if os.path.lexists(target):
        raise LocalDriverError(f"native resume checkpoint target already exists: {target}")
    if os.path.lexists(attempt):
        raise LocalDriverError(f"native resume checkpoint attempt already exists: {attempt}")
    try:
        seed_authenticated_checkpoint(source, attempt, target, resolved)
    except LocalDriverError:
        raise
    except Exception as exc:
        raise LocalDriverError(f"native resume checkpoint seeding failed: {exc}") from exc


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def local_driver_registration() -> DriverRegistration:
    """Return the built-in context-aware local driver registration."""

    def resolve(context: DriverConstructionContext):
        preserve = context.configuration.get("preserve_owned_resources", False)
        if not isinstance(preserve, bool):
            raise TypeError("preserve_owned_resources must be a bool")
        variant = (
            "local-preserved"
            if preserve
            else "local-stop"
        )
        return LocalOrchestrationDriver.capability_envelope.realize(variant)

    def factory(context: DriverConstructionContext, realized):
        runtime = context.runtime_bindings
        if runtime.get("collection_recovery_bindings"):
            raise ValueError(
                "local capability variant does not support durable remote collection recovery"
            )
        driver = LocalOrchestrationDriver(
            cwd=runtime.get("cwd"),
            python_executable=_optional_runtime_string(runtime, "python_executable"),
            freeze_lines=runtime.get("freeze_lines"),
            input_provider_bindings=runtime.get("input_provider_bindings", ()),
            staged_root_bindings=runtime.get("staged_root_bindings", ()),
            update_budget=_optional_runtime_int(runtime, "native_update_budget"),
            realized_capabilities=realized,
        )
        if driver.realized_capabilities != realized:
            raise ValueError("local driver factory received inconsistent realized capabilities")
        return driver

    return DriverRegistration(
        name="local",
        supported_capabilities=LocalOrchestrationDriver.capability_envelope,
        resolve_capabilities=resolve,
        factory=factory,
    )


def _optional_runtime_string(runtime: Mapping[str, object], key: str) -> str | None:
    value = runtime.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"local driver runtime binding {key!r} must be a string")
    return value


def _optional_runtime_int(runtime: Mapping[str, object], key: str) -> int | None:
    value = runtime.get(key)
    if value is not None and not isinstance(value, int):
        raise TypeError(f"local driver runtime binding {key!r} must be an integer")
    return value
