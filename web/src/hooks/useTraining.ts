import { useCallback, useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { startTraining, stopTraining } from '@/api/client';
import { useTrainingStore } from '@/stores/trainingStore';
import { useGraphStore } from '@/stores/graphStore';
import { getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import { actionErrorMessage, withStoreActionFeedback } from '@/stores/storeActions';
import { ensureTaskBindingSpec } from '@/features/scenario/taskBindings';
import { parseContract } from '@/generated/studioContracts';
import type { TrainingWebSocketEvent } from '@/generated/studioContracts';
import type { TaskSpec, TrainingConfig } from '@/types/training';
import type { TrainingStatus } from '@/stores/trainingStore';

export const TRAINING_WS_MAX_RECONNECT_ATTEMPTS = 4;
export const TRAINING_WS_BASE_RECONNECT_DELAY_MS = 500;
export const TRAINING_WS_MAX_RECONNECT_DELAY_MS = 4_000;

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
        setTrainingStreamError(null);
        let payload: TrainingWebSocketEvent;
        try {
          payload = parseContract('TrainingWebSocketEvent', JSON.parse(event.data) as unknown);
        } catch (error) {
          const message =
            error instanceof Error ? error.message : 'Invalid training WebSocket payload';
          setTrainingStreamError(message);
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
        if (payload.type === 'training_progress') {
          setProgress({
            batch: payload.batch,
            total_batches: payload.total_batches,
            loss: payload.loss,
            loss_terms: payload.loss_terms ?? {},
            grad_norm: payload.grad_norm ?? 0,
            step_time_ms: payload.step_time_ms ?? 0,
            metrics: payload.metrics ?? {},
            status: payload.status ?? 'running',
          });
          setStatus('running');
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
            setLatestTrajectory({
              batch: payload.batch,
              effector: Array.isArray(traj.effector)
                ? (traj.effector as [number, number][])
                : [],
              target: Array.isArray(traj.target)
                ? (traj.target as [number, number][] | [number, number])
                : null,
              t: Array.isArray(traj.t) ? (traj.t as number[]) : [],
              observables: traj.observables ?? {},
              outputs: traj.outputs ?? {},
            });
          }
        }
        if (payload.type === 'training_complete') {
          setStatus('completed');
          intentionalCloseRef.current = true;
          ws.close();
        }
        if (payload.type === 'training_error') {
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
      setLatestTrajectory,
      clearReconnectTimer,
    ]
  );

  useEffect(
    () => () => {
      intentionalCloseRef.current = true;
      clearReconnectTimer();
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
      clearReconnectTimer();
      setTrainingStreamError(null);
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
  ]);

  const stop = useCallback(async () => {
    if (!jobId) return;
    intentionalCloseRef.current = true;
    clearReconnectTimer();
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
    if (stopped) {
      toast.success('Training stopped.', { id: 'training-stop-success' });
    }
  }, [jobId, setJobId, setStatus, clearReconnectTimer, setTrainingStreamError]);

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
