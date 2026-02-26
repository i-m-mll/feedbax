import { useCallback, useEffect, useRef, useState } from 'react';
import {
  launchInstance,
  fetchOrchestrationStatus,
  terminateInstance,
  type LaunchInstanceRequest,
} from '@/api/client';
import { useTrainingStore } from '@/stores/trainingStore';

const POLL_INTERVAL_MS = 5000;

/** Terminal statuses that stop polling. */
const TERMINAL_STATUSES = new Set(['idle', 'running', 'preempted', 'error']);

export function useOrchestration() {
  const setOrchestrationState = useTrainingStore((s) => s.setOrchestrationState);
  const setWorkerConfig = useTrainingStore((s) => s.setWorkerConfig);
  const orchestrationStatus = useTrainingStore((s) => s.orchestrationStatus);
  const orchestrationInstanceName = useTrainingStore((s) => s.orchestrationInstanceName);
  const orchestrationWorkerUrl = useTrainingStore((s) => s.orchestrationWorkerUrl);

  const [launching, setLaunching] = useState(false);
  const [terminating, setTerminating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollingRef = useRef(false);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    pollingRef.current = false;
  }, []);

  const pollStatus = useCallback(async () => {
    if (!pollingRef.current) return;
    try {
      const res = await fetchOrchestrationStatus();
      setOrchestrationState(res.status, res.instance_name, res.worker_url);
      if (res.error) {
        setError(res.error);
      }
      // When the worker is running, also sync the TrainingService worker config.
      if (res.status === 'running' && res.worker_url) {
        setWorkerConfig('remote', res.worker_url, true);
      }
      // Stop polling once we reach a terminal state.
      if (TERMINAL_STATUSES.has(res.status)) {
        stopPolling();
        return;
      }
    } catch {
      // Ignore transient fetch errors during polling.
    }
    if (pollingRef.current) {
      pollTimerRef.current = setTimeout(pollStatus, POLL_INTERVAL_MS);
    }
  }, [setOrchestrationState, setWorkerConfig, stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    pollingRef.current = true;
    pollTimerRef.current = setTimeout(pollStatus, POLL_INTERVAL_MS);
  }, [pollStatus, stopPolling]);

  // Clean up on unmount.
  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const launch = useCallback(
    async (params: LaunchInstanceRequest) => {
      setLaunching(true);
      setError(null);
      try {
        const res = await launchInstance(params);
        setOrchestrationState(res.status, res.instance_name, res.worker_url);
        // Begin polling for status updates.
        startPolling();
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Failed to launch instance';
        setError(msg);
        setOrchestrationState('error', null, null);
      } finally {
        setLaunching(false);
      }
    },
    [setOrchestrationState, startPolling]
  );

  const terminate = useCallback(async () => {
    setTerminating(true);
    setError(null);
    stopPolling();
    try {
      await terminateInstance();
      setOrchestrationState('idle', null, null);
      // Disconnect the worker config.
      setWorkerConfig('local', null, false);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to terminate instance';
      setError(msg);
    } finally {
      setTerminating(false);
    }
  }, [setOrchestrationState, setWorkerConfig, stopPolling]);

  return {
    status: orchestrationStatus,
    instanceName: orchestrationInstanceName,
    workerUrl: orchestrationWorkerUrl,
    launching,
    terminating,
    error,
    launch,
    terminate,
  };
}
