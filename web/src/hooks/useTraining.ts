import { useCallback, useRef } from 'react';
import { startTraining, stopTraining } from '@/api/client';
import { useTrainingStore } from '@/stores/trainingStore';
import { useGraphStore } from '@/stores/graphStore';
import type { GraphSpec } from '@/types/graph';
import type { TrainingConfig } from '@/types/training';

/**
 * Map from the Network node's hidden_type param to the canonical network_type
 * string expected by the backend training script.
 */
const HIDDEN_TYPE_TO_NETWORK_TYPE: Record<string, string> = {
  GRUCell: 'gru',
  LSTMCell: 'lstm',
  SimpleRNNCell: 'rnn',
  CDECell: 'cde',
  LeakyRNNCell: 'leaky_rnn',
};

/**
 * Extract network hyperparameters from a graph spec by finding the first node
 * whose component type is "Network".
 */
function extractNetworkParams(
  graph: GraphSpec
): { hidden_dim: number; network_type: string } {
  const networkNode = Object.values(graph.nodes).find(
    (node) => node.type === 'Network'
  );

  if (!networkNode) {
    return { hidden_dim: 64, network_type: 'gru' };
  }

  const hiddenDim =
    typeof networkNode.params.hidden_size === 'number'
      ? networkNode.params.hidden_size
      : typeof networkNode.params.hidden_dim === 'number'
      ? networkNode.params.hidden_dim
      : 64;

  const hiddenType =
    typeof networkNode.params.hidden_type === 'string'
      ? networkNode.params.hidden_type
      : typeof networkNode.params.cell_type === 'string'
      ? networkNode.params.cell_type
      : 'GRUCell';

  const networkType = HIDDEN_TYPE_TO_NETWORK_TYPE[hiddenType] ?? hiddenType.toLowerCase();

  return { hidden_dim: hiddenDim, network_type: networkType };
}

/**
 * Build a TrainingConfig from the current graph and training spec.
 * Hardcoded fields (grad_clip, n_reach_steps, effort_weight) will be
 * configurable via UI in a future phase.
 */
function buildTrainingConfig(
  graph: GraphSpec,
  n_batches: number,
  batch_size: number,
  learning_rate: number
): TrainingConfig {
  const { hidden_dim, network_type } = extractNetworkParams(graph);

  // Read n_reach_steps from the task node (SimpleReaches n_steps param).
  // Bug: dc1adbc — read from graph instead of hardcoding
  const taskNode = Object.values(graph.nodes).find(
    (node) => node.type === 'SimpleReaches'
  );
  const n_reach_steps =
    typeof taskNode?.params?.n_steps === 'number'
      ? taskNode.params.n_steps
      : 80;

  return {
    n_batches,
    batch_size,
    learning_rate,
    grad_clip: 1.0,
    hidden_dim,
    network_type,
    n_reach_steps,
    effort_weight: 2.5,
  };
}

export function useTraining() {
  const {
    trainingSpec,
    taskSpec,
    status,
    jobId,
    setStatus,
    setJobId,
    setProgress,
    appendLog,
    clearHistory,
    setLatestTrajectory,
  } = useTrainingStore();
  const graphId = useGraphStore((state) => state.graphId);
  const graph = useGraphStore((state) => state.graph);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(
    (nextJobId: string) => {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${protocol}://${window.location.host}/ws/training/${nextJobId}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
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
              effector: traj.effector,
              target: traj.target,
              t: traj.t,
            });
          }
        }
        if (payload.type === 'training_complete') {
          setStatus('completed');
          ws.close();
        }
        if (payload.type === 'training_error') {
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
        graph,
        trainingSpec.n_batches,
        trainingSpec.batch_size,
        learningRate
      );
      const response = await startTraining(
        graphId,
        trainingSpec,
        taskSpec,
        graph,
        trainingConfig
      );
      setJobId(response.job_id);
      setStatus('running');
      connect(response.job_id);
    } catch {
      setStatus('error');
    }
  }, [graphId, graph, trainingSpec, taskSpec, setJobId, setStatus, connect, clearHistory]);

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
export { extractNetworkParams, buildTrainingConfig };
