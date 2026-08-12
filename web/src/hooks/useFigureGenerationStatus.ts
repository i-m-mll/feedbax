import { useEffect } from 'react';
import { toast } from 'sonner';
import { getFigureStatus } from '@/api/figureAPI';
import { useDemandStore } from '@/stores/demandStore';
import type { FigureRequestStatus } from '@/types/analysis';

const FIGURE_STATUS_POLL_MS = 2000;

export function useFigureGenerationStatus(
  nodeId: string | null | undefined,
  status: FigureRequestStatus
) {
  const requestId = useDemandStore((state) =>
    nodeId ? state.requests[nodeId]?.figureHash : undefined
  );
  const setResultForRequest = useDemandStore((state) => state.setResultForRequest);
  const setErrorForRequest = useDemandStore((state) => state.setErrorForRequest);

  useEffect(() => {
    if (status !== 'running' || !nodeId || !requestId) return undefined;

    let active = true;
    let requestInFlight = false;
    const intervalId = setInterval(async () => {
      if (requestInFlight) return;
      requestInFlight = true;
      try {
        const result = await getFigureStatus(requestId);
        if (!active) return;
        if (result.status === 'complete' && result.figure_hashes?.length) {
          clearInterval(intervalId);
          if (setResultForRequest(nodeId, requestId, result.figure_hashes[0])) {
            toast.success('Figure generated.', { id: `figure-generated-${nodeId}` });
          }
        } else if (result.status === 'error') {
          clearInterval(intervalId);
          const message = result.error ?? 'Generation failed';
          if (setErrorForRequest(nodeId, requestId, message)) {
            toast.error(message, { id: `figure-generation-error-${nodeId}` });
          }
        }
      } catch {
        // Keep polling on transient errors.
      } finally {
        requestInFlight = false;
      }
    }, FIGURE_STATUS_POLL_MS);

    return () => {
      active = false;
      clearInterval(intervalId);
    };
  }, [nodeId, requestId, setErrorForRequest, setResultForRequest, status]);
}
