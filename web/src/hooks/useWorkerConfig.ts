import { useCallback, useEffect, useState } from 'react';
import { connectWorker, fetchWorkerStatus } from '@/api/client';
import { useTrainingStore } from '@/stores/trainingStore';

export function useWorkerConfig() {
  const { workerMode, workerUrl, workerConnected, setWorkerConfig } = useTrainingStore();
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch current worker status from backend on mount.
  useEffect(() => {
    fetchWorkerStatus()
      .then((res) => setWorkerConfig(res.mode, res.url, res.connected))
      .catch(() => {
        // Backend may not be up yet; ignore.
      });
  }, [setWorkerConfig]);

  const connect = useCallback(
    async (url: string, authToken: string | null) => {
      setConnecting(true);
      setError(null);
      try {
        await connectWorker(url, authToken);
        // Re-fetch status to confirm connection.
        const status = await fetchWorkerStatus();
        setWorkerConfig(status.mode, status.url, status.connected);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to connect');
        setWorkerConfig('local', null, false);
      } finally {
        setConnecting(false);
      }
    },
    [setWorkerConfig]
  );

  return {
    workerMode,
    workerUrl,
    workerConnected,
    connecting,
    error,
    connect,
  };
}
