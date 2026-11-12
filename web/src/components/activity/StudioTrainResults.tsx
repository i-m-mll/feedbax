import { useMemo } from 'react';
import type { StudioJobEvent, StudioSavedTrainAttempt, StudioSavedTrainRow } from '@/generated/studioContracts';
import { useStudioJobActions, useStudioJobConnection, useStudioJobHistory } from '@/hooks/useStudioJobs';
import { StudioRegion, StudioRegionBoundary, studioRegionIssue } from './StudioRegion';
import { LossPlot, studioRunHistory } from './StudioTrainingLoss';
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

function TrainLoss({ project, rows, row, attempt }: {
  project: string;
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
    let offset = 0;
    return lineage.map((stage) => {
      const completed = stage.through ?? stage.attempt.progress?.completed_updates ?? 0;
      const { loss } = studioRunHistory(stage.attempt, stage.row.attempts, events);
      const points = stage.through === null ? loss : loss.filter(({ end }) => end <= completed);
      const projected = { label: stage.row.key.step, offset, completed, points };
      offset += completed;
      return projected;
    });
  }, [rows, row, attempt, events]);
  const retry = () => { actions.refresh(); };
  const state = !stages
    ? { kind: 'failed' as const, issue: { reason: studioReason('unavailable', studioName('checkpoint')) }, retry }
    : history.isError || connection === 'disconnected'
      ? { kind: 'failed' as const, issue: studioRegionIssue('results', history.error, connection === 'disconnected'), retry }
      : history.isPending
        ? { kind: 'loading' as const }
        : { kind: 'result' as const, missing: [] };
  return (
    <section aria-label={studioName('results')} className="mt-3 rounded border border-slate-200 p-3">
      <StudioRegion state={state} skeleton={<div className="h-28 animate-pulse rounded bg-slate-200" />} empty={null}>
        {stages && <LossPlot stages={stages} />}
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
