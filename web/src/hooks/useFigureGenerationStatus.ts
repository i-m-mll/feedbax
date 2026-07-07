import { useEffect, useRef } from 'react';
import { getFigureStatus } from '@/api/figureAPI';
import { useDemandStore } from '@/stores/demandStore';
import type { FigureRequestStatus } from '@/types/analysis';

const FIGURE_STATUS_POLL_MS = 2000;

export function useFigureGenerationStatus(
  nodeId: string | null | undefined,
  status: FigureRequestStatus
) {
  const setResult = useDemandStore((state) => state.setResult);
  const setError = useDemandStore((state) => state.setError);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (status !== 'running' || !nodeId) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return undefined;
    }

    const requestId = useDemandStore.getState().requests[nodeId]?.figureHash;
    if (!requestId) return undefined;

    pollRef.current = setInterval(async () => {
      try {
        const result = await getFigureStatus(requestId);
        if (result.status === 'complete' && result.figure_hashes?.length) {
          setResult(nodeId, result.figure_hashes[0]);
        } else if (result.status === 'error') {
          setError(nodeId, result.error ?? 'Generation failed');
        }
      } catch {
        // Keep polling on transient errors.
      }
    }, FIGURE_STATUS_POLL_MS);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [nodeId, setError, setResult, status]);
}
