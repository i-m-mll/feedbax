import { useCallback, useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { startTraining, stopTraining } from '@/api/client';
import { useTrainingStore } from '@/stores/trainingStore';
import { useGraphStore } from '@/stores/graphStore';
import { getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import { actionErrorMessage, withStoreActionFeedback } from '@/stores/storeActions';
import { ensureTaskBindingSpec } from '@/features/scenario/taskBindings';
import { normalizeTrainingTrajectoryPayload } from '@/features/scenario/liveTraining';
import { parseContract } from '@/generated/studioContracts';
import type { DomainDiagnostic, TrainingWebSocketEvent } from '@/generated/studioContracts';
import type { TaskSpec, TrainingConfig, TrainingProgress } from '@/types/training';
import type { TrainingStatus } from '@/stores/trainingStore';

export const TRAINING_WS_MAX_RECONNECT_ATTEMPTS = 4;
export const TRAINING_WS_BASE_RECONNECT_DELAY_MS = 500;
export const TRAINING_WS_MAX_RECONNECT_DELAY_MS = 4_000;
export const TRAINING_PROGRESS_BATCH_INTERVAL_MS = 50;

export function createTrainingProgressBatcher(
  applyProgress: (progress: TrainingProgress) => void,
  applyRunningStatus: () => void,
  delayMs = TRAINING_PROGRESS_BATCH_INTERVAL_MS
) {
  let pendingProgress: TrainingProgress | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const flushPending = () => {
    timer = null;
    const nextProgress = pendingProgress;
    pendingProgress = null;
    if (!nextProgress) return;
    applyProgress(nextProgress);
    applyRunningStatus();
  };

  return {
    enqueue(progress: TrainingProgress) {
      pendingProgress = progress;
      if (timer !== null) return;
      timer = globalThis.setTimeout(flushPending, delayMs);
    },
    flush() {
      if (timer !== null) {
        globalThis.clearTimeout(timer);
      }
      flushPending();
    },
    cancel() {
      if (timer !== null) {
        globalThis.clearTimeout(timer);
      }
      timer = null;
      pendingProgress = null;
    },
  };
}

export function trainingWebSocketReconnectDelayMs(attempt: number): number {
  return Math.min(
    TRAINING_WS_BASE_RECONNECT_DELAY_MS * 2 ** Math.max(0, attempt),
    TRAINING_WS_MAX_RECONNECT_DELAY_MS
  );
}

export function shouldReconnectTrainingWebSocket({
  attempt,
  intentionalClose,
  status,
}: {
  attempt: number;
  intentionalClose: boolean;
  status: TrainingStatus;
}): boolean {
  return !intentionalClose && status === 'running' && attempt < TRAINING_WS_MAX_RECONNECT_ATTEMPTS;
}

export function formatTrainingDiagnostic(diagnostic: DomainDiagnostic): string {
  const nodeSuffix =
    diagnostic.node_ids.length > 0 ? ` [${diagnostic.node_ids.join(', ')}]` : '';
  return (
    `${diagnostic.severity.toUpperCase()} ${diagnostic.code}${nodeSuffix}: ` +
    diagnostic.message
  );
}

/**
 * Build runtime worker controls from task/training specs. Graph topology and
 * model leaves are carried by GraphSpec, not inferred here.
 */
function buildTrainingConfig(
  task: TaskSpec,
  n_batches: number,
  batch_size: number,
  learning_rate: number
): TrainingConfig {
  const n_reach_steps =
    typeof task.params?.n_steps === 'number'
      ? task.params.n_steps
      : 80;

  return {
    n_batches,
    batch_size,
    learning_rate,
    grad_clip: 1.0,
    n_reach_steps,
  };
}

export function useTraining() {
  const trainingStore = useTrainingStore();
  const trainingScenario = useWorkspaceStore((state) => getTrainingScenario(state.workspace));
  const trainingSpec = trainingScenario?.training_spec ?? trainingStore.trainingSpec;
  const taskSpec = trainingScenario?.task_spec ?? trainingStore.taskSpec;
  const {
    status,
    jobId,
    setStatus,
    setJobId,
    setProgress,
    appendLog,
    clearHistory,
    setTrainingDiagnostics,
    setTrainingStreamError,
    setLatestTrajectory,
  } = trainingStore;
  const graphId = useGraphStore((state) => state.graphId);
  const graph = useGraphStore((state) => state.graph);
  const wsRef = useRef<WebSocket | null>(null);
  const wsJobIdRef = useRef<string | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);
  const intentionalCloseRef = useRef(false);
  const lastEventSeqRef = useRef(-1);
  const progressBatcherRef = useRef<ReturnType<typeof createTrainingProgressBatcher> | null>(null);
  if (progressBatcherRef.current === null) {
    progressBatcherRef.current = createTrainingProgressBatcher(
      setProgress,
      () => setStatus('running'),
    );
  }

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const connect = useCallback(
    (nextJobId: string) => {
      const existing = wsRef.current;
      if (
        existing &&
        wsJobIdRef.current === nextJobId &&
        (existing.readyState === WebSocket.CONNECTING || existing.readyState === WebSocket.OPEN)
      ) {
        return;
      }

      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${protocol}://${window.location.host}/ws/training/${nextJobId}`);
      wsRef.current = ws;
      wsJobIdRef.current = nextJobId;

      ws.onmessage = (event) => {
        reconnectAttemptRef.current = 0;
        let payload: TrainingWebSocketEvent;
        try {
          payload = parseContract('TrainingWebSocketEvent', JSON.parse(event.data) as unknown);
        } catch (error) {
          const message =
            error instanceof Error ? error.message : 'Invalid training WebSocket payload';
          setTrainingStreamError(message);
          setTrainingDiagnostics([]);
          appendLog({
            batch: 0,
            level: 'error',
            message,
            timestamp: Date.now(),
          });
          setStatus('error');
          intentionalCloseRef.current = true;
          ws.close();
          return;
        }
        if (payload.job_id !== nextJobId || payload.seq <= lastEventSeqRef.current) {
          return;
        }
        lastEventSeqRef.current = payload.seq;
        setTrainingStreamError(null);
        if (payload.type === 'training_progress') {
          progressBatcherRef.current?.enqueue({
            batch: payload.batch,
            total_batches: payload.total_batches,
            loss: payload.loss,
            loss_terms: payload.loss_terms ?? {},
            grad_norm: payload.grad_norm ?? 0,
            step_time_ms: payload.step_time_ms ?? 0,
            metrics: payload.metrics ?? {},
            status: payload.status ?? 'running',
          });
        }
        if (payload.type === 'training_log') {
          appendLog({
            batch: payload.batch,
            level: payload.level ?? 'info',
            message: payload.message,
            timestamp: Date.now(),
          });
        }
        if (payload.type === 'training_trajectory') {
          const traj = payload.trajectory;
          if (traj) {
            setLatestTrajectory(normalizeTrainingTrajectoryPayload(traj, payload.batch));
          }
        }
        if (payload.type === 'training_resync') {
          setTrainingStreamError(payload.reason === 'gap' ? payload.message : null);
          appendLog({
            batch: useTrainingStore.getState().progress?.batch ?? 0,
            level: payload.reason === 'gap' ? 'warning' : 'info',
            message: payload.message,
            timestamp: Date.now(),
          });
        }
        if (payload.type === 'training_complete') {
          progressBatcherRef.current?.flush();
          setStatus('completed');
          intentionalCloseRef.current = true;
          ws.close();
        }
        if (payload.type === 'training_error') {
          progressBatcherRef.current?.flush();
          const diagnostics = payload.diagnostics ?? [];
          setTrainingDiagnostics(diagnostics);
          setTrainingStreamError(payload.error);
          appendLog({
            batch: payload.batch ?? 0,
            level: 'error',
            message: payload.error,
            timestamp: Date.now(),
          });
          setStatus('error');
          intentionalCloseRef.current = true;
          ws.close();
        }
      };

      ws.onclose = () => {
        if (wsRef.current !== ws) return;
        wsRef.current = null;
        const currentStatus = useTrainingStore.getState().status;
        if (
          !shouldReconnectTrainingWebSocket({
            attempt: reconnectAttemptRef.current,
            intentionalClose: intentionalCloseRef.current,
            status: currentStatus,
          })
        ) {
          if (!intentionalCloseRef.current && currentStatus === 'running') {
            const message = 'Training stream disconnected after multiple reconnect attempts.';
            setTrainingStreamError(message);
            appendLog({
              batch: 0,
              level: 'error',
              message,
              timestamp: Date.now(),
            });
            setStatus('error');
          }
          return;
        }

        const attempt = reconnectAttemptRef.current;
        reconnectAttemptRef.current += 1;
        const delayMs = trainingWebSocketReconnectDelayMs(attempt);
        const message = `Training stream disconnected. Reconnecting in ${Math.round(
          delayMs / 1000
        )}s (attempt ${attempt + 1}/${TRAINING_WS_MAX_RECONNECT_ATTEMPTS}).`;
        setTrainingStreamError(message);
        appendLog({
          batch: 0,
          level: 'warning',
          message,
          timestamp: Date.now(),
        });
        clearReconnectTimer();
        reconnectTimerRef.current = window.setTimeout(() => {
          reconnectTimerRef.current = null;
          connect(nextJobId);
        }, delayMs);
      };
    },
    [
      setProgress,
      setStatus,
      appendLog,
      setTrainingStreamError,
      setTrainingDiagnostics,
      setLatestTrajectory,
      clearReconnectTimer,
    ]
  );

  useEffect(
    () => () => {
      intentionalCloseRef.current = true;
      clearReconnectTimer();
      progressBatcherRef.current?.cancel();
      wsRef.current?.close();
      wsRef.current = null;
    },
    [clearReconnectTimer]
  );

  const start = useCallback(async () => {
    if (!graphId) {
      setStatus('error');
      toast.error('Save the project before starting training.', { id: 'training-start-error' });
      return;
    }
    try {
      intentionalCloseRef.current = false;
      reconnectAttemptRef.current = 0;
      lastEventSeqRef.current = -1;
      progressBatcherRef.current?.cancel();
      clearReconnectTimer();
      setTrainingStreamError(null);
      setTrainingDiagnostics([]);
      clearHistory();
      const learningRate =
        typeof trainingSpec.optimizer.params.learning_rate === 'number'
          ? trainingSpec.optimizer.params.learning_rate
          : 0.001;
      const trainingConfig = buildTrainingConfig(
        taskSpec,
        trainingSpec.n_batches,
        trainingSpec.batch_size,
        learningRate
      );
      const response = await withStoreActionFeedback(
        () => startTraining(
          graphId,
          trainingSpec,
          taskSpec,
          graph,
          trainingConfig,
          ensureTaskBindingSpec(trainingScenario?.task_binding_spec, graph, taskSpec)
        ),
        {
          errorToast: (error) => actionErrorMessage(error, 'Failed to start training.'),
          toastId: 'training-start-error',
          onError: (error) => {
            setTrainingStreamError(actionErrorMessage(error, 'Failed to start training.'));
            setTrainingDiagnostics([]);
            setStatus('error');
          },
        },
      );
      if (!response) return;
      setJobId(response.job_id);
      setStatus('running');
      toast.success('Training started.', { id: 'training-start-success' });
      connect(response.job_id);
    } catch (error) {
      const message = actionErrorMessage(error, 'Failed to start training.');
      setTrainingStreamError(message);
      setTrainingDiagnostics([]);
      setStatus('error');
      toast.error(message, { id: 'training-start-error' });
    }
  }, [
    graphId,
    graph,
    trainingSpec,
    taskSpec,
    trainingScenario?.task_binding_spec,
    setJobId,
    setStatus,
    connect,
    clearHistory,
    clearReconnectTimer,
    setTrainingStreamError,
    setTrainingDiagnostics,
  ]);

  const stop = useCallback(async () => {
    if (!jobId) return;
    intentionalCloseRef.current = true;
    clearReconnectTimer();
    progressBatcherRef.current?.flush();
    const stopped = await withStoreActionFeedback(
      () => stopTraining(jobId),
      {
        errorToast: (error) =>
          `${actionErrorMessage(error, 'Failed to stop training.')} Marked idle locally.`,
        toastId: 'training-stop-error',
      },
    );
    wsRef.current?.close();
    wsJobIdRef.current = null;
    setStatus('idle');
    setJobId(null);
    setTrainingStreamError(null);
    setTrainingDiagnostics([]);
    if (stopped) {
      toast.success('Training stopped.', { id: 'training-stop-success' });
    }
  }, [
    jobId,
    setJobId,
    setStatus,
    clearReconnectTimer,
    setTrainingStreamError,
    setTrainingDiagnostics,
  ]);

  return {
    status,
    jobId,
    start,
    stop,
  };
}

// Re-export for consumers that want to display a config summary without
// triggering a full training start.
export { buildTrainingConfig };
