import { useMemo } from 'react';
import { useQueries } from '@tanstack/react-query';
import { fetchSavedEvaluationObjective } from '@/api/jobs';
import type {
  StudioJobEvent, StudioProjectWindowMapRow, StudioSavedEvaluationRow, StudioSavedTrainAttempt, StudioSavedTrainKey, StudioSavedTrainRow,
} from '@/generated/studioContracts';
import { useStudioJobActions, useStudioJobConnection, useStudioJobHistory } from '@/hooks/useStudioJobs';
import { heldOutTracks, type ObjectiveStage } from './heldOutObjective';
import { StudioRegion, StudioRegionBoundary, studioRegionIssue } from './StudioRegion';
import { LossPlot, studioRunHistory, type LossStage } from './StudioTrainingLoss';
import { studioName, studioReason } from './studioVocabulary';

type Stage = {
  row: StudioSavedTrainRow;
  attempt: StudioSavedTrainAttempt;
  through: number | null;
};

const emptyEvents: readonly StudioJobEvent[] = [];

function checkpointLineage(
  rows: readonly StudioSavedTrainRow[], row: StudioSavedTrainRow, attempt: StudioSavedTrainAttempt,
) {
  const byJob = new Map(rows.flatMap((candidate) =>
    candidate.attempts.map((job) => [job.job_id, { row: candidate, attempt: job }] as const),
  ));
  const stages: Stage[] = [{ row, attempt, through: null }];
  const visited = new Set([attempt.job_id]);
  let source = attempt.checkpoint_source;
  while (source) {
    const predecessor = byJob.get(source.job_id);
    if (visited.has(source.job_id) || predecessor?.row.request_id !== source.request_id) return null;
    visited.add(source.job_id);
    stages.push({ ...predecessor, through: source.completed_updates });
    source = predecessor.attempt.checkpoint_source;
  }
  return stages.reverse();
}

function sameRow(first: StudioSavedTrainKey, second: StudioSavedTrainKey) {
  return first.variant === second.variant && first.step === second.step && first.row === second.row;
}

function TrainLoss({ project, revision, evaluations, mapRows, rows, row, attempt }: {
  project: string;
  revision: string;
  evaluations: readonly StudioSavedEvaluationRow[];
  mapRows: readonly StudioProjectWindowMapRow[];
  rows: readonly StudioSavedTrainRow[];
  row: StudioSavedTrainRow;
  attempt: StudioSavedTrainAttempt;
}) {
  const history = useStudioJobHistory(project);
  const events = history.data ?? emptyEvents;
  const connection = useStudioJobConnection();
  const actions = useStudioJobActions();
  const stages = useMemo(() => {
    const lineage = checkpointLineage(rows, row, attempt);
    if (!lineage) return null;
    const projected: (LossStage & ObjectiveStage & { key: StudioSavedTrainKey })[] = [];
    let offset = 0;
    for (const stage of lineage) {
      const through = stage.through ?? stage.attempt.progress?.completed_updates ?? 0;
      const { loss, stageStart, attemptStarts, attemptCutoffs } = studioRunHistory(stage.attempt, stage.row.attempts, events);
      const source = stage.attempt.checkpoint_source;
      // A stateful worker reports the source's update clock. Resume keeps the
      // original stage origin, even when a later attempt starts at another checkpoint.
      if (source && through > 0 && stageStart !== 0 && stageStart !== source.completed_updates) return null;
      const origin = stageStart ?? 0;
      const completed = Math.max(0, through - origin);
      const points = loss.filter(({ end }) => end > origin && (stage.through === null || end <= through)).map((point) => ({
        ...point,
        end: point.end - origin,
        interval: point.interval && { ...point.interval, start: point.interval.start - origin },
      }));
      projected.push({
        label: stage.row.key.step, offset, completed, points,
        key: stage.row.key, origin, through: stage.through, request: stage.row.request_id,
        starts: attemptStarts, cutoffs: attemptCutoffs,
      });
      offset += completed;
    }
    return projected;
  }, [rows, row, attempt, events]);
  const reads = evaluations.flatMap(({ key, checkpoint_series, result }) => {
    const stage = checkpoint_series && stages?.find((candidate) => sameRow(candidate.key, checkpoint_series.subject));
    return stage && result ? [{ key, result, stage }] : [];
  });
  const objectives = useQueries({ queries: reads.map(({ key, result }) => ({
    queryKey: ['saved-evaluation-objective', project, revision, key, result.job_id, result.receipt],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      fetchSavedEvaluationObjective(project, { revision, key, job_id: result.job_id }, signal),
    staleTime: Infinity,
    retry: false,
  })) });
  const tracks = reads.flatMap(({ key, stage }, index) => {
    const result = objectives[index].data;
    if (!result) return [];
    const planRow = mapRows.find((candidate) => sameRow(candidate.key, key));
    const label = planRow && Object.keys(planRow.coordinates).length ? `${key.step} · ${planRow.label}` : key.step;
    return heldOutTracks(label, result, stage).map((track) => ({
      ...track, id: JSON.stringify([key, track.id]),
    }));
  });
  const objectiveError = objectives.find((query) => query.isError)?.error;
  const retry = () => { actions.refresh(); };
  const state = history.isError || connection === 'disconnected'
    ? { kind: 'failed' as const, issue: studioRegionIssue('results', history.error, connection === 'disconnected'), retry }
    : history.isPending
      ? { kind: 'loading' as const }
      : !stages
        ? { kind: 'failed' as const, issue: { reason: studioReason('unavailable', studioName('checkpoint')) }, retry }
        : { kind: 'result' as const, missing: [] };
  return (
    <section aria-label={studioName('results')} className="mt-3 rounded border border-shell-divider p-3">
      <StudioRegion state={state} skeleton={<div className="h-28 motion-safe:animate-pulse rounded bg-shell-divider" />} empty={null}>
        {stages && <LossPlot stages={stages} objectives={tracks} />}
        {objectiveError && <StudioRegion state={{
          kind: 'failed', issue: studioRegionIssue('objective', objectiveError),
          retry: () => { for (const query of objectives) if (query.isError) void query.refetch(); },
        }} skeleton={null} empty={null}>{null}</StudioRegion>}
      </StudioRegion>
    </section>
  );
}

export function StudioTrainResults(props: Parameters<typeof TrainLoss>[0]) {
  return (
    <StudioRegionBoundary name="results">
      <TrainLoss {...props} />
    </StudioRegionBoundary>
  );
}
