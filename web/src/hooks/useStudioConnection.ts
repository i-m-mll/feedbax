import { useStudioJobActions, useStudioJobConnection, useStudioJobs } from '@/hooks/useStudioJobs';

export function useStudioConnection() {
  const state = useStudioJobConnection();
  const { refresh } = useStudioJobActions();
  const { error } = useStudioJobs();
  return {
    isError: state === 'disconnected',
    isSuccess: state === 'connected',
    error,
    refetch: refresh,
  };
}
