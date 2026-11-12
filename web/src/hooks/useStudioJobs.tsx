import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PropsWithChildren,
} from 'react';
import { skipToken, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  acknowledgeStudioJob,
  fetchStudioJobHistory,
  fetchStudioJobs,
  resumeStudioJob,
  stopStudioJobs,
} from '@/api/jobs';
import type { StudioJobEvent, StudioJobSnapshot } from '@/generated/studioContracts';
import {
  STUDIO_JOBS_WORKER_NAME,
  type StudioJobConnection,
  type StudioJobWorkerCommand,
  type StudioJobWorkerMessage,
} from '@/jobs/studioJobConnection';
import JobsWorker from '@/jobs/studioJobWorker?sharedworker';
import { useGraphStore } from '@/stores/graphStore';

const jobsKey = ['studio-jobs'] as const;
const historyKey = (projectId: string | null) => ['studio-job-history', projectId] as const;

const StudioJobsContext = createContext<{
  connection: StudioJobConnection;
  refresh: () => void;
  acknowledge: (jobId: string) => Promise<void>;
} | null>(null);

function useStudioJobsContext() {
  const context = useContext(StudioJobsContext);
  if (!context) throw new Error('Studio jobs require a connection provider');
  return context;
}

export function StudioJobsProvider({ children, folderProjectId }: PropsWithChildren<{ folderProjectId: string | null }>) {
  const cache = useQueryClient();
  const graphId = useGraphStore((state) => state.graphId);
  const projectId = folderProjectId || graphId;
  // The window owns replay. Closing its last inspector must not cancel it.
  useQuery<StudioJobEvent[]>({
    queryKey: historyKey(projectId),
    queryFn: skipToken,
    notifyOnChangeProps: [],
  });
  const [connection, setConnection] = useState<StudioJobConnection>('connecting');
  const retry = useRef<(() => void) | null>(null);
  const activePort = useRef<MessagePort | null>(null);

  useEffect(() => {
    let port: MessagePort | null = null;
    let generation = 0;
    let buffered: StudioJobEvent[] | null = null;
    let historyReady = false;
    let snapshotReady = false;
    const history = historyKey(projectId);

    function apply(event: StudioJobEvent) {
      const { activity } = event;
      if (activity) {
        cache.setQueryData<StudioJobSnapshot>(jobsKey, (snapshot) => {
          if (!snapshot || event.cursor <= snapshot.cursor) return snapshot;
          const known = snapshot.jobs.some((job) => job.job_id === activity.job_id);
          const jobs = known
            ? snapshot.jobs.map((job) => job.job_id === activity.job_id ? activity : job)
            : [...snapshot.jobs, activity];
          return { ...snapshot, cursor: event.cursor, jobs };
        });
      }
      if (projectId === event.project_id && historyReady) {
        cache.setQueryData<StudioJobEvent[]>(history, (events = []) => {
          if ((events[events.length - 1]?.cursor ?? 0) >= event.cursor) return events;
          return [...events, event];
        });
      }
    }

    async function hydrate() {
      const current = ++generation;
      buffered = [];
      historyReady = false;
      snapshotReady = false;
      await Promise.all([
        cache.cancelQueries({ queryKey: jobsKey, exact: true }),
        cache.cancelQueries({ queryKey: history, exact: true }),
      ]);
      if (current !== generation) return;
      let through = 0;
      try {
        const snapshot = await cache.fetchQuery({
          queryKey: jobsKey,
          queryFn: ({ signal }) => fetchStudioJobs(signal),
          retry: false,
          staleTime: 0,
        });
        if (current !== generation) return;
        through = snapshot.cursor;
        snapshotReady = true;
        if (projectId) {
          await cache.fetchQuery({
            queryKey: history,
            queryFn: ({ signal }) =>
              fetchStudioJobHistory(projectId, snapshot.oldest_cursor, snapshot.cursor, signal),
            retry: false,
            staleTime: 0,
          });
          if (current !== generation) return;
          historyReady = true;
        }
      } catch {
        // The failed query carries the error and remains available for retry.
      } finally {
        if (current === generation) {
          const arrived = buffered ?? [];
          buffered = null;
          for (const event of arrived) {
            if (event.cursor > through) apply(event);
          }
        }
      }
    }

    function open() {
      if (port) return;
      try {
        port = new JobsWorker({ name: STUDIO_JOBS_WORKER_NAME }).port;
      } catch {
        setConnection('disconnected');
        return;
      }
      activePort.current = port;
      port.onmessage = ({ data }: MessageEvent<StudioJobWorkerMessage>) => {
        switch (data.type) {
          case 'attached':
          case 'reset':
            void hydrate();
            break;
          case 'state':
            setConnection(data.state);
            if (data.state === 'connected' && !snapshotReady && buffered === null) void hydrate();
            break;
          case 'event':
            if (buffered) buffered.push(data.event);
            else apply(data.event);
            break;
          case 'seen':
            setSeen(data.jobId);
            break;
        }
      };
      port.postMessage({ type: 'subscribe', projectId } satisfies StudioJobWorkerCommand);
    }

    function close() {
      generation += 1;
      void cache.cancelQueries({ queryKey: jobsKey, exact: true });
      void cache.cancelQueries({ queryKey: history, exact: true });
      if (port) {
        port.onmessage = null;
        port.postMessage({ type: 'release' } satisfies StudioJobWorkerCommand);
        port.close();
      }
      activePort.current = null;
      port = null;
    }

    function setSeen(jobId: string) {
      cache.setQueryData<StudioJobSnapshot>(jobsKey, (snapshot) => snapshot && ({
        ...snapshot,
        jobs: snapshot.jobs.map((job) => job.job_id === jobId ? { ...job, seen: true } : job),
      }));
    }

    retry.current = () => {
      if (!port) open();
      else {
        port.postMessage({ type: 'retry' } satisfies StudioJobWorkerCommand);
        void hydrate();
      }
    };
    open();
    window.addEventListener('pagehide', close);
    window.addEventListener('pageshow', open);
    return () => {
      window.removeEventListener('pagehide', close);
      window.removeEventListener('pageshow', open);
      retry.current = null;
      close();
    };
  }, [cache, projectId]);

  const context = useMemo(() => ({
    connection,
    refresh: () => retry.current?.(),
    acknowledge: async (jobId: string) => {
      const result = await acknowledgeStudioJob(jobId);
      activePort.current?.postMessage({ type: 'seen', jobId: result.job_id } satisfies StudioJobWorkerCommand);
    },
  }), [connection]);

  return <StudioJobsContext.Provider value={context}>{children}</StudioJobsContext.Provider>;
}

export function useStudioJobs() {
  return useQuery<StudioJobSnapshot>({ queryKey: jobsKey, queryFn: skipToken });
}

export function useStudioJobHistory(projectId: string | null) {
  return useQuery<StudioJobEvent[]>({ queryKey: historyKey(projectId), queryFn: skipToken });
}

export function useStudioJobConnection() {
  return useStudioJobsContext().connection;
}

export function useStudioJobActions() {
  const { refresh, acknowledge } = useStudioJobsContext();
  const resumeTokens = useRef(new Map<string, string>());
  return useMemo(() => ({
    refresh,
    stop: stopStudioJobs,
    seen: acknowledge,
    async resume(jobId: string) {
      const token = resumeTokens.current.get(jobId) ?? crypto.randomUUID();
      resumeTokens.current.set(jobId, token);
      const result = await resumeStudioJob(jobId, token);
      resumeTokens.current.delete(jobId);
      return result;
    },
  }), [refresh, acknowledge]);
}
