import { useCallback, useRef } from 'react';
import { startTraining, stopTraining } from '@/api/client';
import { useTrainingStore } from '@/stores/trainingStore';
import { useGraphStore } from '@/stores/graphStore';
import { getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import { ensureTaskBindingSpec } from '@/features/scenario/taskBindings';
import { parseContract } from '@/generated/studioContracts';
import type { TrainingWebSocketEvent } from '@/generated/studioContracts';
import type { TaskSpec, TrainingConfig } from '@/types/training';

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
    setLatestTrajectory,
  } = trainingStore;
  const graphId = useGraphStore((state) => state.graphId);
  const graph = useGraphStore((state) => state.graph);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(
    (nextJobId: string) => {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${protocol}://${window.location.host}/ws/training/${nextJobId}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        let payload: TrainingWebSocketEvent;
        try {
          payload = parseContract('TrainingWebSocketEvent', JSON.parse(event.data) as unknown);
        } catch (error) {
          appendLog({
            batch: 0,
            level: 'error',
            message: error instanceof Error ? error.message : 'Invalid training WebSocket payload',
            timestamp: Date.now(),
          });
          setStatus('error');
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
          ws.close();
        }
        if (payload.type === 'training_error') {
          appendLog({
            batch: payload.batch ?? 0,
            level: 'error',
            message: payload.error,
            timestamp: Date.now(),
          });
          setStatus('error');
          ws.close();
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    },
    [setProgress, setStatus, appendLog, setLatestTrajectory]
  );

  const start = useCallback(async () => {
    if (!graphId) {
      setStatus('error');
      return;
    }
    try {
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
      const response = await startTraining(
        graphId,
        trainingSpec,
        taskSpec,
        graph,
        trainingConfig,
        ensureTaskBindingSpec(trainingScenario?.task_binding_spec, graph, taskSpec)
      );
      setJobId(response.job_id);
      setStatus('running');
      connect(response.job_id);
    } catch {
      setStatus('error');
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
  ]);

  const stop = useCallback(async () => {
    if (!jobId) return;
    await stopTraining(jobId);
    wsRef.current?.close();
    setStatus('idle');
    setJobId(null);
  }, [jobId, setJobId, setStatus]);

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
