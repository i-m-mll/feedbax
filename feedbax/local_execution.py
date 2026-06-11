"""Local execution runner for Feedbax execution specs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from feedbax.execution_models import ExecutionSpec, LocalExecutionResult
from feedbax.execution_plan import prepare_execution_plan
from feedbax.manifest import (
    EntrypointRef,
    ManifestStatus,
    Provenance,
    default_manifest_root,
    utc_now,
    write_training_run_manifest,
)


def run_local_execution(
    spec: ExecutionSpec,
    *,
    root: Path | str | None = None,
    timeout: Optional[float] = None,
) -> LocalExecutionResult:
    """Run a local execution and emit a durable training manifest."""
    if spec.backend != "local":
        raise ValueError("run_local_execution only accepts backend='local'")
    job_id = spec.resolved_job_id()
    root_path = Path(root) if root is not None else default_manifest_root()
    run_dir = root_path / "executions" / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    plan = prepare_execution_plan(ExecutionSpec(**{**spec.model_dump(), "job_id": job_id}))
    (run_dir / "execution-plan.json").write_text(
        plan.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )

    cwd = Path(spec.local.cwd).expanduser() if spec.local.cwd else None
    env = {**os.environ, **spec.env}
    proc = subprocess.run(
        [spec.local.shell, "-lc", spec.command],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    status: ManifestStatus = "completed" if proc.returncode == 0 else "failed"
    studio_metadata = (
        spec.metadata.get("studio", {}) if isinstance(spec.metadata.get("studio"), dict) else {}
    )
    training_spec = (
        studio_metadata.get("training_spec")
        if isinstance(studio_metadata.get("training_spec"), dict)
        else None
    )
    task_spec = (
        studio_metadata.get("task_spec")
        if isinstance(studio_metadata.get("task_spec"), dict)
        else None
    )
    task_binding_spec = (
        studio_metadata.get("task_binding_spec")
        if isinstance(studio_metadata.get("task_binding_spec"), dict)
        else None
    )
    graph_spec = (
        studio_metadata.get("graph_spec")
        if isinstance(studio_metadata.get("graph_spec"), dict)
        else None
    )
    total_batches = 0
    if training_spec is not None:
        try:
            total_batches = int(training_spec.get("n_batches") or 0)
        except (TypeError, ValueError):
            total_batches = 0
    final_loss: Optional[float] = None
    training_summary: dict[str, Any] | None = None
    summary_path = cwd / "artifacts" / "training-summary.json" if cwd is not None else None
    if summary_path is not None and summary_path.exists():
        try:
            training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_loss = training_summary.get("final_loss")
            if isinstance(summary_loss, (int, float)):
                final_loss = float(summary_loss)
        except (OSError, json.JSONDecodeError):
            training_summary = None
    history_events: list[dict[str, Any]] = [
        {
            "type": "execution_result",
            "backend": "local",
            "return_code": proc.returncode,
            "completed_at": utc_now().isoformat(),
        }
    ]
    if training_summary is not None:
        for event in training_summary.get("history", []):
            if isinstance(event, dict):
                history_events.append({"type": "training_progress", **event})
    provenance = Provenance(
        entrypoint=EntrypointRef(
            kind="feedbax-execution",
            command=spec.command,
            metadata={"backend": "local", "return_code": proc.returncode},
        ),
        issues=list(spec.issues),
        metadata={
            "execution_plan": str(run_dir / "execution-plan.json"),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "execution_metadata": spec.metadata,
        },
    )
    manifest, manifest_path = write_training_run_manifest(
        job_id=job_id,
        total_batches=total_batches,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=task_binding_spec,
        graph_spec=graph_spec,
        status=status,
        history_events=history_events,
        final_loss=final_loss,
        root=root_path,
        provenance=provenance,
    )
    return LocalExecutionResult(
        job_id=job_id,
        status=status,
        return_code=proc.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        manifest_path=str(manifest_path),
        manifest_payload=manifest.model_dump(mode="json", exclude_none=True),
        plan=plan,
    )
