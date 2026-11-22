import { useCallback, useMemo, useRef, useSyncExternalStore } from 'react';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { fetchSavedTrainRuns } from '@/api/jobs';
import { useFollowingQuery } from '@/hooks/useFollowingQuery';
import type { Project, StudioSavedTrainRuns } from '@/generated/studioContracts';
import type { ProjectEditController } from './projectEditController';

const planKey = (project: string, revision: string) => ['saved-train-runs', project, revision] as const;

/**
 * The plan of a project window. What is drawn reads `shown`, the last plan read, with `project`, the project it was
 * compiled from: while a newer revision's plan is read both stay as they were and switch together when it arrives.
 * What a gesture does reads `read()`, the plan of the latest saved revision once it has been read.
 */
export interface SavedPlan {
  readonly shown: StudioSavedTrainRuns | undefined;
  readonly project: Project;
  /** Whether `shown` is the plan of the shown revision. */
  readonly current: boolean;
  /** Whether the shown revision's plan is still to be read, so a gesture waits: until it is current or has failed. */
  readonly waiting: boolean;
  readonly error: Error | null;
  read(): Promise<StudioSavedTrainRuns>;
  refetch(): Promise<unknown>;
}

export function useSavedPlan(editor: ProjectEditController, version: string): SavedPlan {
  const { saved } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const queries = useQueryClient();
  const query = useFollowingQuery({
    queryKey: planKey(saved.project_id, saved.revision),
    queryFn: ({ signal }) => fetchSavedTrainRuns(saved.project_id, saved.revision, signal),
    placeholderData: keepPreviousData,
    gcTime: 0,
  }, version);
  const compiled = useRef<{ revision: string; project: Project }>();
  if (query.data?.revision === saved.revision) compiled.current = { revision: saved.revision, project: saved.project };
  const read = query.data;
  const earlier = read && read.revision !== saved.revision
    ? compiled.current?.revision === read.revision ? compiled.current.project : undefined
    : saved.project;
  const pair = useMemo(() => earlier ? { project: earlier, shown: read } : { project: saved.project, shown: undefined },
    [earlier, saved.project, read]);
  const readCurrent = useCallback(() => {
    const { project_id: project, revision } = editor.getSnapshot().saved;
    return queries.fetchQuery({
      queryKey: planKey(project, revision),
      queryFn: ({ signal }) => fetchSavedTrainRuns(project, revision, signal),
      staleTime: Infinity,
    });
  }, [editor, queries]);
  const current = pair.shown?.revision === saved.revision;
  const { refetch } = query;
  return {
    ...pair, current, error: query.error, read: readCurrent, refetch,
    waiting: !current && (!!read || query.isPending),
  };
}
