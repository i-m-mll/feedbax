# Feedbax Studio: Cloud Training Dispatch — Implementation Spec

## 1. Executive Summary

This spec covers the work needed to close six gaps between what the Feedbax Studio UI configures and what the training worker actually executes. The infrastructure — GCP orchestration, SSE streaming, WebSocket relay, worker process lifecycle — is substantially complete. The missing pieces are all in the data-forwarding layer: specs configured in the UI are silently dropped before reaching the worker, trained weights are never persisted, and the startup script installs a stale PyPI package that may be missing critical classes.

### What exists (no changes needed)
- GCP instance lifecycle via `OrchestrationManager` and `gcp.py`
- Worker SSE stream with seq-based reconnect replay
- WebSocket relay (`ws/training.py`) bridging worker to browser
- Frontend orchestration panel, worker-config panel, loss builder, optimizer selector
- `TrainingService` remote/local mode switching

### What this spec adds
- Forward `training_spec` (optimizer type/params) and `task_spec` to worker
- Worker reads optimizer from spec instead of hardcoding AdamW
- Worker reads task params (n_reach_steps, effort_weight) from spec
- Worker serializes trained model weights with `equinox.tree_serialise_leaves`
- Weight download endpoint (`GET /checkpoint/download`) proxied through Studio backend
- Startup script installs feedbax from git at a pinned ref, not PyPI
- `LaunchRequest` and `InstanceConfig` grow a `feedbax_install_cmd` field
- Optional: SSE endpoint at `GET /api/orchestration/events` replacing 5s polling
- Frontend UX: auth token persistence, "Start Training" disabled when worker not ready, cleaner state machine display

Phase 1 covers Gaps 1–4 (critical, no breaking API changes). Phase 2 covers Gaps 5–6 (enhancements).

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Browser                                                                      │
│                                                                               │
│  TrainingPanel.tsx                                                            │
│    ├── useOrchestration   → POST /api/orchestration/launch                    │
│    │                        GET  /api/orchestration/status  (poll 5s)         │
│    │                        DEL  /api/orchestration/instance                  │
│    ├── useTraining        → POST /api/training              (start job)        │
│    │                        WS   /ws/training/{job_id}      (stream)           │
│    └── useWorkerConfig    → POST /api/training/worker/connect                 │
│                             GET  /api/training/worker/status                  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ HTTP / WebSocket
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  Studio FastAPI Backend  (feedbax/web/)                                       │
│                                                                               │
│  api/orchestration.py                                                         │
│    └── OrchestrationManager.launch()                                          │
│          1. gcloud create instance  (startup_script installs worker)          │
│          2. poll GCP until RUNNING                                            │
│          3. wait_for_health(worker_url)                                       │
│          4. training_service.connect_remote(worker_url, token)                │
│                                                                               │
│  api/training.py  POST /api/training                                         │
│    └── TrainingService.start_training(n_batches, training_config,            │
│                                        training_spec, task_spec)  ◄── GAP 1/2│
│                                                                               │
│  ws/training.py  WS /ws/training/{job_id}                                    │
│    └── relay: worker SSE → browser WebSocket                                 │
│                                                                               │
│  api/training.py  GET /api/training/{job_id}/checkpoint/download             │
│    └── proxy:  worker GET /checkpoint/download → browser     ◄── GAP 3       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ HTTP (local subprocess or remote TCP)
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  Training Worker  (feedbax/web/worker/app.py)                                 │
│                                                                               │
│  POST /start  ← body: {total_batches, training_config?,                      │
│                         training_spec?, task_spec?}         ◄── GAP 1/2      │
│  GET  /stream               (SSE — seq-buffered, reconnectable)              │
│  GET  /status                                                                 │
│  POST /stop                                                                   │
│  GET  /checkpoint           (metadata: batch, loss, weights_available)       │
│  GET  /checkpoint/download  (binary eqx weights file)       ◄── GAP 3        │
│                                                                               │
│  _run_training_real(job, cfg)                                                 │
│    ├── build optimizer from training_spec.optimizer         ◄── GAP 1        │
│    ├── override task params from task_spec                  ◄── GAP 2        │
│    └── eqx.tree_serialise_leaves after completion           ◄── GAP 3        │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ (remote mode)
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  GCP Compute Instance                                                         │
│  startup_script.py (generated, not a constant):                              │
│    pip install 'git+https://github.com/…/feedbax.git@develop'  ◄── GAP 4    │
│    python -m feedbax.web.worker --host 0.0.0.0 --port $WORKER_PORT           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Gap Analysis

| # | Gap | Severity | Impact if unfixed |
|---|-----|----------|-------------------|
| 1 | TrainingSpec (optimizer, loss weights) not forwarded to worker | Critical | UI optimizer selection silently ignored; worker always hardcodes AdamW |
| 2 | TaskSpec not forwarded to worker | Critical | n_reach_steps, effort_weight hardcoded regardless of UI |
| 3 | Checkpoint weights not persisted or downloadable | Critical | Training produces no artifacts; trained model is lost |
| 4 | Startup script installs PyPI feedbax | Critical | `AnalyticalMusculoskeletalPlant` etc. missing from release; worker fails on import |
| 5 | Orchestration status uses polling, no SSE | Nice-to-have | 5s latency, extra requests |
| 6 | TrainingPanel cloud UX improvements | Nice-to-have | Auth token re-entry friction; Start button active when worker not ready |

---

## 4. Detailed Implementation Plan

### Gap 1: Forward TrainingSpec to Worker

**Root cause.** `api/training.py::start_training` calls `training_service.start_training(n_batches, training_config=...)`. `training_spec` and `task_spec` arrive in `TrainingRequest` but are never forwarded.

**Propagation chain.** `api/training.py` → `services/training_service.py` → `worker/client.py` → `worker/app.py`

New keyword argument `training_spec: Optional[dict] = None` added at each layer.

**Optimizer extraction (`worker/app.py`, new function):**

```python
def _build_optimizer_from_spec(
    training_spec: Optional[dict],
    cfg: _TrainingCfg,
) -> optax.GradientTransformation:
    clip = optax.clip_by_global_norm(cfg.grad_clip)
    if training_spec is None:
        return optax.chain(clip, optax.adamw(cfg.learning_rate, weight_decay=1e-6))

    opt_spec = training_spec.get("optimizer", {})
    opt_type = str(opt_spec.get("type", "adamw")).lower()
    params = opt_spec.get("params", {})

    def _p(key, default):
        v = params.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    lr = _p("learning_rate", cfg.learning_rate)

    if opt_type == "adam":
        inner = optax.adam(lr, b1=_p("b1", 0.9), b2=_p("b2", 0.999))
    elif opt_type == "sgd":
        inner = optax.sgd(lr, momentum=_p("momentum", 0.0))
    elif opt_type == "rmsprop":
        inner = optax.rmsprop(lr, decay=_p("decay", 0.9))
    else:  # adamw default
        inner = optax.adamw(lr, b1=_p("b1", 0.9), b2=_p("b2", 0.999),
                             weight_decay=_p("weight_decay", 1e-6))

    return optax.chain(clip, inner)
```

**Loss weight extraction.** Full dynamic JAX loss construction from `LossTermSpec` is deferred (requires building closures at runtime which is non-trivial). Phase 1 scopes to: extract `effort_weight` from `training_spec.loss.children.effort.weight` if present.

```python
def _extract_effort_weight_from_spec(training_spec: Optional[dict], default: float) -> float:
    if training_spec is None:
        return default
    loss = training_spec.get("loss", {})
    children = loss.get("children", {})
    effort_term = children.get("effort", {})
    w = effort_term.get("weight")
    try:
        return float(w) if w is not None else default
    except (TypeError, ValueError):
        return default
```

Call site in `_run_training_real`:
```python
cfg.effort_weight = _extract_effort_weight_from_spec(job.training_spec, cfg.effort_weight)
optimizer = _build_optimizer_from_spec(job.training_spec, cfg)
```

### Gap 2: Forward TaskSpec to Worker

Same propagation chain as Gap 1. New keyword argument `task_spec: Optional[dict] = None`.

**`_extract_training_cfg` extension:**

```python
def _extract_training_cfg(
    training_config: Optional[Dict[str, Any]],
    task_spec: Optional[Dict[str, Any]] = None,
) -> _TrainingCfg:
    cfg = _TrainingCfg()
    # ... existing training_config extraction unchanged ...

    if task_spec is not None:
        task_params = task_spec.get("params", {})
        for key, cast in [("n_reach_steps", int), ("effort_weight", float)]:
            if key in task_params:
                try:
                    setattr(cfg, key, cast(task_params[key]))
                except (TypeError, ValueError):
                    pass
    return cfg
```

### Gap 3: Checkpoint Weights Persistence and Download

**`_Job` additions:**
```python
checkpoint_path: Optional[str] = None
```

**`_run_training_real` — serialize after training:**
```python
# After job.status = WorkerStatus.COMPLETED, before terminal _emit:
try:
    import tempfile as _tmpfile, os as _os
    _ckpt_dir = _tmpfile.mkdtemp(prefix="feedbax_ckpt_")
    _ckpt_path = _os.path.join(_ckpt_dir, f"{job.job_id}.eqx")
    eqx.tree_serialise_leaves(_ckpt_path, controller)
    job.checkpoint_path = _ckpt_path
    _emit(job, {"type": "training_log", "job_id": job.job_id,
                "batch": job.total_batches, "level": "info",
                "message": f"Checkpoint saved ({_ckpt_path})"})
except Exception as _exc:
    _emit(job, {"type": "training_log", "job_id": job.job_id,
                "batch": job.total_batches, "level": "warning",
                "message": f"Failed to save checkpoint: {_exc}"})
```

**`/checkpoint` fix:**
```python
"weights_available": job.checkpoint_path is not None,  # was hardcoded False
```

**New worker endpoint:**
```python
from fastapi.responses import FileResponse

@app.get("/checkpoint/download", dependencies=[_auth_dep])
def checkpoint_download():
    job = _state.get("current")
    if job is None or job.checkpoint_path is None:
        raise HTTPException(status_code=404, detail="No checkpoint available")
    import os
    if not os.path.exists(job.checkpoint_path):
        raise HTTPException(status_code=410, detail="Checkpoint file gone")
    return FileResponse(
        job.checkpoint_path,
        media_type="application/octet-stream",
        filename=f"feedbax_checkpoint_{job.job_id}.eqx",
    )
```

**`worker/client.py` — new function:**
```python
async def download_checkpoint(
    base_url: str,
    dest_path: str,
    auth_token: Optional[str] = None,
) -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "GET", f"{base_url}/checkpoint/download",
            headers=_auth_headers(auth_token),
        ) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)
```

**`services/training_service.py` — new method:**
```python
async def download_checkpoint(self, job_id: str, dest_path: str) -> None:
    if self._current_job_id != job_id:
        raise ValueError(f"Unknown job {job_id!r}")
    if self._base_url is None:
        raise RuntimeError("No worker configured")
    await worker_client.download_checkpoint(self._base_url, dest_path, self._auth_token)
```

**`api/training.py` — new endpoint:**
```python
@router.get('/{job_id}/checkpoint/download')
async def download_checkpoint(job_id: str):
    import tempfile, os
    from starlette.background import BackgroundTask
    dest = tempfile.mktemp(suffix=".eqx")
    try:
        await training_service.download_checkpoint(job_id, dest)
    except ValueError:
        raise HTTPException(status_code=404, detail='Job not found')
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return FileResponse(
        dest,
        media_type="application/octet-stream",
        filename=f"feedbax_checkpoint_{job_id}.eqx",
        background=BackgroundTask(os.unlink, dest),
    )
```

**`api/client.ts` — new functions:**
```typescript
export async function fetchCheckpoint(jobId: string) {
  return request<{ batch: number; loss: number; weights_available: boolean }>(
    `/api/training/${jobId}/checkpoint`
  );
}

export async function downloadCheckpoint(jobId: string): Promise<void> {
  const response = await fetch(`/api/training/${jobId}/checkpoint/download`);
  if (!response.ok) throw new Error(`Download failed: ${response.status}`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `feedbax_checkpoint_${jobId}.eqx`;
  a.click();
  URL.revokeObjectURL(url);
}
```

**`TrainingPanel.tsx` — download button** (in completed state section):
```tsx
{status === 'completed' && jobId && (
  <button onClick={async () => {
    const meta = await fetchCheckpoint(jobId);
    if (meta.weights_available) await downloadCheckpoint(jobId);
  }}>
    <Download className="w-3.5 h-3.5" /> Download weights
  </button>
)}
```

### Gap 4: Startup Script — Git Install

**`orchestration/startup_script.py`** — replace module constant with generator:
```python
def make_startup_script(feedbax_install_cmd: str = "pip install feedbax") -> str:
    return f"""#!/bin/bash
set -euo pipefail
{feedbax_install_cmd}
pip install uvicorn httpx fastapi 2>&1 | tail -5
if [[ -n "${{TS_AUTH_KEY:-}}" ]]; then
  curl -fsSL https://tailscale.com/install.sh | sh
  tailscale up --authkey="${{TS_AUTH_KEY}}" --hostname="feedbax-worker-$(hostname)" 2>&1
fi
nohup python -m feedbax.web.worker \\
  --host 0.0.0.0 --port ${{WORKER_PORT:-8765}} \\
  ${{AUTH_TOKEN:+--auth-token "$AUTH_TOKEN"}} \\
  >> /var/log/feedbax-worker.log 2>&1 &
echo "Worker started on port ${{WORKER_PORT:-8765}}"
"""

# Backwards compat: keep STARTUP_SCRIPT for any direct import
STARTUP_SCRIPT = make_startup_script()
```

**`orchestration/gcp.py`** — `InstanceConfig` gets new field:
```python
feedbax_install_cmd: str = (
    "pip install 'git+https://github.com/mlll-io/feedbax.git@develop'"
)
```

`create_instance` calls `make_startup_script(config.feedbax_install_cmd)` instead of `STARTUP_SCRIPT`.

**`api/orchestration.py`** — `LaunchRequest` gets:
```python
feedbax_install_cmd: Optional[str] = None
```

`InstanceConfig` construction uses:
```python
feedbax_install_cmd=(
    payload.feedbax_install_cmd
    or "pip install 'git+https://github.com/mlll-io/feedbax.git@develop'"
),
```

**`api/client.ts`** — `LaunchInstanceRequest` gets:
```typescript
feedbax_install_cmd?: string;
```

**`TrainingPanel.tsx`** — add collapsed "Advanced" section in cloud panel with a text input for `feedbax_install_cmd`, pre-populated with the git-install default.

### Gap 5 (Nice-to-Have): Orchestration Status SSE

**`api/orchestration.py`** — new endpoint:
```python
@router.get("/events")
async def orchestration_events():
    async def _generate():
        last_status = None
        while True:
            state = orchestration_manager.state
            if state.status != last_status:
                last_status = state.status
                payload = {
                    "status": state.status,
                    "instance_name": state.instance.name if state.instance else None,
                    "worker_url": state.worker_url,
                    "error": state.error,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                if state.status in ("running", "error", "preempted"):
                    break
            await asyncio.sleep(2.0)
    return StreamingResponse(_generate(), media_type="text/event-stream")
```

**`useOrchestration.ts`** — replace `setTimeout` polling with `EventSource`. Keep the polling loop as fallback when `EventSource` is not supported or the endpoint returns 404.

### Gap 6 (Nice-to-Have): TrainingPanel Cloud UX

**Auth token localStorage persistence:**
```typescript
const [cloudAuthToken, setCloudAuthToken] = useState(
  () => localStorage.getItem('feedbax.cloud.authToken') ?? ''
);
// in onChange:
localStorage.setItem('feedbax.cloud.authToken', e.target.value);
```
Similarly for `cloudProject` (key `feedbax.cloud.project`) and `cloudZone` (key `feedbax.cloud.zone`). Do NOT persist `cloudTsAuthKey` — it's a one-time-use secret.

**"Start Training" disabled logic:**
```tsx
disabled={
  status === 'running' ||
  (workerMode === 'remote' && !workerConnected) ||
  orchestrationStatus === 'creating' ||
  orchestrationStatus === 'connecting'
}
```

**State machine chip** — icon+text pairs:
| Status | Icon | Label |
|--------|------|-------|
| `idle` | grey dot | "Idle" |
| `creating` | spinner | "Creating instance…" |
| `connecting` | spinner | "Connecting worker…" |
| `running` | green dot | "Running — {instanceName}" |
| `preempted` | amber ⚠ | "Preempted" |
| `error` | red ✕ | error message (40 chars max) |

---

## 5. End-to-End Flow: Launch Cloud → Train → Download Weights

1. User clicks "Launch instance" in TrainingPanel cloud section.
2. `useOrchestration.launch(params)` → `POST /api/orchestration/launch` → returns `{status: "creating", instance_name: "feedbax-worker-abc123"}` immediately.
3. Frontend starts polling. CloudStatusChip shows "Creating instance…".
4. Background task: `gcloud compute instances create` with generated startup script (git-install feedbax). GCP provisions VM (~30–90s).
5. Startup script runs: installs feedbax from git, installs uvicorn/httpx/fastapi, starts worker on port 8765.
6. OrchestrationManager polls until `status == RUNNING`, then polls `wait_for_health(worker_url)`. Sets status → "connecting".
7. Frontend poll returns "connecting". "Start Training" button disabled.
8. Health check passes. Manager calls `training_service.connect_remote(worker_url, token)`. Sets status → "running".
9. Frontend poll returns "running". `setWorkerConfig('remote', url, true)`. "Start Training" enabled. CloudStatusChip: green "Running — feedbax-worker-abc123".
10. User clicks "Start Training". `useTraining.start()` → `POST /api/training` with `{training_spec, task_spec, training_config, graph_spec}`.
11. Backend forwards all four to `training_service.start_training(...)` → `POST {worker_url}/start` with `{total_batches, training_config, training_spec, task_spec}`.
12. Worker builds cfg (with task_spec overrides), builds optimizer from training_spec, starts `_run_training_real` thread. Returns `{job_id}`.
13. Frontend opens `WebSocket /ws/training/{job_id}`. Studio relays worker SSE events to browser. Charts update in real time.
14. Training completes. Worker serializes `controller` with `eqx.tree_serialise_leaves`. Sets `checkpoint_path`. Emits `training_complete`.
15. Frontend receives `training_complete`. Status = "completed". "Download weights" button appears.
16. User clicks "Download weights". `GET /api/training/{job_id}/checkpoint/download` → Studio proxies from worker → browser downloads `feedbax_checkpoint_{job_id}.eqx`.
17. User clicks "Terminate". `DELETE /api/orchestration/instance` → gcloud delete. TrainingService disconnects. CloudStatusChip returns to "Idle".

---

## 6. GRU Experiment Reference

The current hardcoded training (`_run_training_real`) trains a 2-link musculoskeletal arm + GRU controller on a random-reach task.

### Currently hardcoded vs configurable

| Parameter | Currently | After Phase 1 |
|-----------|-----------|---------------|
| Optimizer type | AdamW (hardcoded) | from `training_spec.optimizer.type` |
| Learning rate | from `TrainingConfig` | from `training_spec.optimizer.params.learning_rate` |
| Weight decay | 1e-6 (hardcoded) | from `training_spec.optimizer.params.weight_decay` |
| `n_reach_steps` | from `TrainingConfig` (default 80) | also from `task_spec.params.n_reach_steps` |
| `effort_weight` | from `TrainingConfig` (default 2.5) | also from `training_spec.loss.children.effort.weight` |
| `hidden_dim` | from `TrainingConfig` (default 128) | unchanged |
| `network_type` | from `TrainingConfig` (default "gru") | unchanged |
| Body preset | midpoint of default_2link_bounds | hardcoded (future) |
| Loss structure | hardcoded (tracking L1 + effort + smoothness) | partially spec-driven (future) |

### Recommended parameters for first-run validation

Short run (verify E2E wiring works, ~2–5 min on n1-standard-4):
```json
{
  "n_batches": 100, "batch_size": 32, "learning_rate": 0.001,
  "hidden_dim": 64, "n_reach_steps": 40, "effort_weight": 1.0,
  "snapshot_interval": 20
}
```

Full experiment (meaningful training, ~30–40 min on n1-standard-4):
```json
{
  "n_batches": 2000, "batch_size": 128, "learning_rate": 0.001,
  "hidden_dim": 128, "n_reach_steps": 80, "effort_weight": 2.5,
  "snapshot_interval": 100
}
```

---

## 7. Implementation Phases

### Phase 1 — Critical (all additive, no breaking changes)

| Item | Files | Size |
|------|-------|------|
| Gap 1: training_spec propagation chain | `api/training.py`, `services/training_service.py`, `worker/client.py`, `worker/app.py` | Small |
| Gap 1: `_build_optimizer_from_spec` | `worker/app.py` | Small |
| Gap 1: `_extract_effort_weight_from_spec` | `worker/app.py` | Trivial |
| Gap 2: task_spec propagation + `_extract_training_cfg` extension | same files | Small |
| Gap 3: checkpoint serialization in `_run_training_real` | `worker/app.py` | Small |
| Gap 3: `/checkpoint/download` worker endpoint | `worker/app.py` | Small |
| Gap 3: `download_checkpoint` in client/service/api | `worker/client.py`, `services/training_service.py`, `api/training.py` | Small |
| Gap 3: frontend download button + `downloadCheckpoint` | `api/client.ts`, `TrainingPanel.tsx` | Small |
| Gap 4: startup_script generator | `orchestration/startup_script.py` | Small |
| Gap 4: `feedbax_install_cmd` field | `orchestration/gcp.py`, `api/orchestration.py`, `api/client.ts`, `TrainingPanel.tsx` | Small |

### Phase 2 — Enhancements

| Item | Files |
|------|-------|
| Gap 5: orchestration SSE endpoint | `api/orchestration.py`, `hooks/useOrchestration.ts` |
| Gap 6: localStorage auth persistence | `TrainingPanel.tsx` |
| Gap 6: Start button disabled logic | `TrainingPanel.tsx` |
| Gap 6: State machine chip | `TrainingPanel.tsx` |
| Gap 6: `feedbax_install_cmd` UI field | `TrainingPanel.tsx`, `api/client.ts` |
| Future: CDE dispatch | `worker/app.py` |
| Future: dynamic loss from spec | `worker/app.py` (significant) |
