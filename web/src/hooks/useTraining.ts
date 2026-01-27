import { useCallback, useRef } from 'react';
import { startTraining, stopTraining } from '@/api/client';
import { useTrainingStore } from '@/stores/trainingStore';
import { useGraphStore } from '@/stores/graphStore';

export function useTraining() {
  const { trainingSpec, taskSpec, status, jobId, setStatus, setJobId, setProgress } =
    useTrainingStore();
  const graphId = useGraphStore((state) => state.graphId);
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
            metrics: payload.metrics ?? {},
          });
          setStatus('running');
        }
        if (payload.type === 'training_complete') {
          setStatus('completed');
          ws.close();
        }
        if (payload.type === 'training_error') {
          setStatus('error');
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    },
    [setProgress, setStatus]
  );

  const start = useCallback(async () => {
    if (!graphId) {
      setStatus('error');
      return;
    }
    try {
      const response = await startTraining(graphId, trainingSpec, taskSpec);
      setJobId(response.job_id);
      setStatus('running');
      connect(response.job_id);
    } catch {
      setStatus('error');
    }
  }, [graphId, trainingSpec, taskSpec, setJobId, setStatus, connect]);

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
