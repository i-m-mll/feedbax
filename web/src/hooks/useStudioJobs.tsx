import {
  createContext,
  useCallback,
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
  warmStudioWorkers,
} from '@/api/jobs';
import type { StudioActivityJob, StudioJobEvent, StudioJobSnapshot } from '@/generated/studioContracts';
import { studioFormat, studioName, studioRunStatus, studioStep } from '@/components/activity/studioVocabulary';
import {
  STUDIO_JOBS_WORKER_NAME,
  type StudioJobWorkerCommand,
  type StudioJobWorkerMessage,
} from '@/jobs/studioJobConnection';
import { serviceDidNotRespond, serviceResponded, subscribeConnection, getConnection } from '@/api/connection';
import JobsWorker from '@/jobs/studioJobWorker?sharedworker';

const jobsKey = ['studio-jobs'] as const;
const historyKey = (projectId: string | null) => ['studio-job-history', projectId] as const;
const eventSource = (venue: StudioJobEvent['venue']) => venue
  ? JSON.stringify([venue.session_id, venue.stream_id]) : 'local';

const StudioJobsContext = createContext<{
  refresh: () => void;
  acknowledge: (jobId: string) => Promise<void>;
} | null>(null);

function useStudioJobsContext() {
  const context = useContext(StudioJobsContext);
  if (!context) throw new Error('Studio jobs require a connection provider');
  return context;
}

export function StudioJobsProvider({ children, projectId }: PropsWithChildren<{ projectId: string | null }>) {
  const cache = useQueryClient();
  const parentRefresh = useContext(StudioJobsContext)?.refresh;
  const ownsSnapshot = parentRefresh === undefined;
  // The window owns replay. Closing its last inspector must not cancel it.
  useQuery<StudioJobEvent[]>({
    queryKey: historyKey(projectId),
    queryFn: skipToken,
    notifyOnChangeProps: [],
  });
  const [announcements, setAnnouncements] = useState<{ id: number; text: string }[]>([]);
  const announcementId = useRef(0);
  const retry = useRef<(() => void) | null>(null);
  const activePort = useRef<MessagePort | null>(null);

  useEffect(() => {
    let port: MessagePort | null = null;
    let generation = 0;
    let historyBoundary: AbortController | undefined;
    let buffered: StudioJobEvent[] | null = null;
    let historyReady = false;
    let snapshotReady = false;
    let machineUpdatePending = false;
    let streamDown = false;
    let historyCursors = new Map<string, number>();
    let announcedThrough = 0;
    const statuses = new Map<string, StudioActivityJob['status']>();
    let initialSnapshot = true;
    const history = historyKey(projectId);

    function announce(jobs: readonly StudioActivityJob[], silent = false) {
      if (!ownsSnapshot) return;
      const updates = jobs.flatMap((job) => {
        if (projectId !== null && projectId !== job.project_id) return [];
        const previous = statuses.get(job.job_id);
        statuses.set(job.job_id, job.status);
        if (silent || previous === job.status) return [];
        const run = [
          projectId === null && job.project_id,
          studioStep(job.kind).name,
          job.stage_id,
          job.label !== job.stage_id && job.label,
          job.scenario_id !== 'default' && job.scenario_id,
        ].filter(Boolean).join(' · ');
        return [{ id: ++announcementId.current,
          text: studioFormat('runAnnouncement', { run, status: studioRunStatus(job.status) }) }];
      });
      if (updates.length) setAnnouncements((items) => [...items, ...updates].slice(-20));
    }

    function apply(event: StudioJobEvent) {
      const { activity } = event;
      if (activity && ownsSnapshot && !event.venue) {
        cache.setQueryData<StudioJobSnapshot>(jobsKey, (snapshot) => {
          if (!snapshot || event.cursor <= snapshot.cursor) return snapshot;
          const known = snapshot.jobs.some((job) => job.job_id === activity.job_id);
          const jobs = known
            ? snapshot.jobs.map((job) => job.job_id === activity.job_id ? activity : job)
            : [...snapshot.jobs, activity];
          return { ...snapshot, cursor: event.cursor, jobs };
        });
        if (snapshotReady && event.cursor > announcedThrough) {
          announcedThrough = event.cursor;
          announce([activity]);
        }
      }
      if (projectId === event.project_id && historyReady) {
        const source = eventSource(event.venue);
        if ((historyCursors.get(source) ?? 0) >= event.cursor) return;
        historyCursors.set(source, event.cursor);
        cache.setQueryData<StudioJobEvent[]>(history, (events = []) => [...events, event]);
      }
    }

    async function hydrate(rebuildHistory = false) {
      const current = ++generation;
      historyBoundary?.abort();
      const boundary = new AbortController();
      historyBoundary = boundary;
      buffered = [];
      historyReady = false;
      snapshotReady = false;
      await Promise.all([
        ownsSnapshot && cache.cancelQueries({ queryKey: jobsKey, exact: true }),
        cache.cancelQueries({ queryKey: history, exact: true }),
      ]);
      if (current !== generation) return;
      let through = 0;
      try {
        const snapshot = ownsSnapshot ? await cache.fetchQuery({
          queryKey: jobsKey,
          queryFn: () => fetchStudioJobs(boundary.signal),
          staleTime: 0,
        }) : await fetchStudioJobs(boundary.signal);
        if (current !== generation) return;
        through = snapshot.cursor;
        announcedThrough = through;
        announce(snapshot.jobs, initialSnapshot);
        initialSnapshot = false;
        snapshotReady = true;
        if (projectId !== null) {
          const venues = snapshot.venue_cursors ?? [];
          const retained = rebuildHistory ? undefined : cache.getQueryData<StudioJobEvent[]>(history);
          const before = (source: string, oldest: number, through: number) => {
            const cursor = retained && historyCursors.get(source);
            return cursor !== undefined && cursor >= oldest && cursor <= through ? cursor : oldest;
          };
          const localStart = before('local', snapshot.oldest_cursor, snapshot.cursor);
          const venueStarts = new Map(venues.map((venue) => [eventSource(venue), before(eventSource(venue), 0, venue.cursor)]));
          const retainedEvents = (retained ?? []).filter((event) => event.venue
            ? event.cursor <= (venueStarts.get(eventSource(event.venue)) ?? 0)
            : event.cursor > snapshot.oldest_cursor && event.cursor <= localStart);
          await cache.fetchQuery({
            queryKey: history,
            queryFn: async () => [
              ...retainedEvents,
              ...(await Promise.all([
                fetchStudioJobHistory(projectId, localStart, snapshot.cursor, boundary.signal),
                ...venues.map((venue) =>
                  fetchStudioJobHistory(projectId, venueStarts.get(eventSource(venue)) ?? 0, venue.cursor, boundary.signal, venue)),
              ])).flat(),
            ],
            staleTime: 0,
          });
          if (current !== generation) return;
          historyCursors = new Map([
            ['local', snapshot.cursor],
            ...venues.map((venue) => [eventSource(venue), venue.cursor] as const),
          ]);
        }
        if (current !== generation) return;
        historyReady = true;
      } catch {
        // The failed query carries the error and remains available for retry.
        if (current === generation && ownsSnapshot) snapshotReady = !initialSnapshot;
      } finally {
        if (current === generation) {
          const arrived = buffered ?? [];
          buffered = null;
          for (const event of arrived) {
            if (event.venue || event.cursor > through) apply(event);
          }
          if (machineUpdatePending) {
            machineUpdatePending = false;
            void hydrate();
          }
        }
      }
    }

    function warm() {
      if (ownsSnapshot && projectId !== null) void warmStudioWorkers().catch(() => undefined);
    }

    function open() {
      if (port) return;
      warm();
      port = new JobsWorker({ name: STUDIO_JOBS_WORKER_NAME }).port;
      activePort.current = port;
      port.onmessage = ({ data }: MessageEvent<StudioJobWorkerMessage>) => {
        switch (data.type) {
          case 'attached':
            void hydrate();
            break;
          case 'reset':
            void hydrate(true);
            break;
          case 'state':
            streamDown = data.state === 'disconnected';
            if (streamDown) {
              snapshotReady = false;
              serviceDidNotRespond();
            }
            if (data.state === 'connected') serviceResponded();
            if (data.state === 'connected' && !snapshotReady && buffered === null) void hydrate();
            break;
          case 'event':
            if (buffered) buffered.push(data.event);
            else apply(data.event);
            break;
          case 'machines-changed':
            if (ownsSnapshot) {
              if (buffered) machineUpdatePending = true;
              else void hydrate();
            }
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
      historyBoundary?.abort();
      if (ownsSnapshot) void cache.cancelQueries({ queryKey: jobsKey, exact: true });
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

    function retryConnection() {
      parentRefresh?.();
      warm();
      if (!port) open();
      else {
        if (ownsSnapshot) port.postMessage({ type: 'retry' } satisfies StudioJobWorkerCommand);
        void hydrate();
      }
    }
    retry.current = retryConnection;
    const stopReconnecting = ownsSnapshot ? subscribeConnection(() => {
      if (streamDown && getConnection().kind === 'connected') retryConnection();
    }) : undefined;
    open();
    window.addEventListener('pagehide', close);
    window.addEventListener('pageshow', open);
    if (ownsSnapshot) window.addEventListener('studio-native-reconnected', retryConnection);
    return () => {
      window.removeEventListener('pagehide', close);
      window.removeEventListener('pageshow', open);
      window.removeEventListener('studio-native-reconnected', retryConnection);
      retry.current = null;
      stopReconnecting?.();
      close();
    };
  }, [cache, ownsSnapshot, parentRefresh, projectId]);

  const refresh = useCallback(() => retry.current?.(), []);
  const context = useMemo(() => ({
    refresh,
    acknowledge: async (jobId: string) => {
      const result = await acknowledgeStudioJob(jobId);
      activePort.current?.postMessage({ type: 'seen', jobId: result.job_id } satisfies StudioJobWorkerCommand);
    },
  }), [refresh]);

  return <StudioJobsContext.Provider value={context}>
    {children}
    {ownsSnapshot && <div role="log" aria-label={studioName('status')} aria-live="polite" aria-relevant="additions" className="sr-only">
      {announcements.map(({ id, text }) => <div key={id}>{text}</div>)}
    </div>}
  </StudioJobsContext.Provider>;
}

export function useStudioJobs() {
  return useQuery<StudioJobSnapshot>({ queryKey: jobsKey, queryFn: skipToken });
}

export function useStudioJobHistory(projectId: string | null) {
  return useQuery<StudioJobEvent[]>({ queryKey: historyKey(projectId), queryFn: skipToken });
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
