import { useMemo } from 'react';
import { useQueries } from '@tanstack/react-query';
import { fetchSavedEvaluationObjective } from '@/api/jobs';
import type {
  StudioJobEvent, StudioProjectWindowMapRow, StudioSavedEvaluationRow, StudioSavedTrainAttempt, StudioSavedTrainKey,
  StudioSavedTrainResult, StudioSavedTrainRow,
} from '@/generated/studioContracts';
import { useStudioJobActions, useStudioJobConnection, useStudioJobHistory } from '@/hooks/useStudioJobs';
import { heldOutTracks, type ObjectiveStage } from './heldOutObjective';
import { StudioRegion, StudioRegionBoundary, studioRegionIssue } from './StudioRegion';
import { LossPlot, studioRunHistory, type LossResultName, type LossSeries, type LossStage } from './StudioTrainingLoss';
import { studioName, studioReason } from './studioVocabulary';

type Stage = {
  row: Pick<StudioSavedTrainRow, 'key' | 'request_id' | 'attempts'>;
  attempt: StudioSavedTrainAttempt;
  through: number | null;
};

const emptyEvents: readonly StudioJobEvent[] = [];

function historicalRequest(row: StudioSavedTrainRow): Stage['row'] | undefined {
  return row.previous && row.previous.request_id !== row.request_id && row.previous_attempts?.length
    ? { key: row.key, request_id: row.previous.request_id, attempts: row.previous_attempts } : undefined;
}

function checkpointLineage(
  rows: readonly StudioSavedTrainRow[], row: Stage['row'], attempt: StudioSavedTrainAttempt,
  through: number | null = null,
) {
  const history = rows.flatMap((candidate): Stage['row'][] => {
    const historical = historicalRequest(candidate);
    return historical ? [historical] : [];
  });
  const byJob = new Map<string, Omit<Stage, 'through'>>([...history, ...rows].flatMap((candidate) =>
    candidate.attempts.map((job) => [job.job_id, { row: candidate, attempt: job }] as const),
  ));
  const stages: Stage[] = [{ row, attempt, through }];
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

type ProjectLossStage = LossStage & ObjectiveStage & { key: StudioSavedTrainKey };

function projectLoss(lineage: Stage[], events: readonly StudioJobEvent[]): ProjectLossStage[] | null {
  const stages: ProjectLossStage[] = [];
  let offset = 0;
  for (const stage of lineage) {
    const through = stage.through ?? stage.attempt.progress?.completed_updates ?? 0;
    const exactEvents = events.filter(({ request_id }) => request_id === stage.row.request_id);
    const { loss, stageStart, attemptStarts, attemptCutoffs } = studioRunHistory(stage.attempt, stage.row.attempts, exactEvents);
    const source = stage.attempt.checkpoint_source;
    // Stateful workers keep the source update clock across resumed attempts.
    if (source && through > 0 && stageStart !== 0 && stageStart !== source.completed_updates) return null;
    const origin = stageStart ?? 0;
    const completed = Math.max(0, through - origin);
    const points = loss.filter(({ end }) => end > origin && (stage.through === null || end <= through)).map((point) => ({
      ...point,
      end: point.end - origin,
      interval: point.interval && { ...point.interval, start: point.interval.start - origin },
    }));
    stages.push({
      label: stage.row.key.step, offset, completed, points,
      key: stage.row.key, origin, through: stage.through, request: stage.row.request_id,
      starts: attemptStarts, cutoffs: attemptCutoffs,
    });
    offset += completed;
  }
  return stages;
}

export type TrainLossResult = { name: LossResultName; result: StudioSavedTrainResult };

function resultLoss(
  { name, result }: TrainLossResult, row: StudioSavedTrainRow, rows: readonly StudioSavedTrainRow[],
  events: readonly StudioJobEvent[],
): LossSeries {
  const exactRow = result.request_id === row.request_id ? row : historicalRequest(row);
  const attempt = exactRow?.attempts.find(({ job_id, successful_receipt }) => job_id === result.job_id &&
    successful_receipt?.manifest_id === result.receipt.manifest_id &&
    successful_receipt.manifest_sha256 === result.receipt.manifest_sha256);
  const lineage = exactRow && attempt && result.progress &&
    checkpointLineage(rows, exactRow, attempt, result.progress.completed_updates);
  return { result: { name, jobId: result.job_id }, stages: (lineage && projectLoss(lineage, events)) || [] };
}

function TrainLoss({ project, revision, evaluations, mapRows, rows, row, selection }: {
  project: string;
  revision: string;
  evaluations: readonly StudioSavedEvaluationRow[];
  mapRows: readonly StudioProjectWindowMapRow[];
  rows: readonly StudioSavedTrainRow[];
  row: StudioSavedTrainRow;
  selection: { kind: 'attempt'; attempt: StudioSavedTrainAttempt } | { kind: 'results'; results: TrainLossResult[] };
}) {
  const history = useStudioJobHistory(project);
  const events = history.data ?? emptyEvents;
  const connection = useStudioJobConnection();
  const actions = useStudioJobActions();
  const stages = useMemo(() => {
    if (selection.kind !== 'attempt') return undefined;
    const lineage = checkpointLineage(rows, row, selection.attempt);
    return lineage && projectLoss(lineage, events);
  }, [rows, row, selection, events]);
  const series = selection.kind === 'results' ? selection.results.map((result) => resultLoss(result, row, rows, events))
    : stages ? [{ stages }] : [];
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
      : stages === null
        ? { kind: 'failed' as const, issue: { reason: studioReason('unavailable', studioName('checkpoint')) }, retry }
        : { kind: 'result' as const, missing: [] };
  return (
    <section aria-label={studioName('results')} className="mt-3 rounded border border-shell-divider p-3">
      <StudioRegion state={state} skeleton={<div className="h-28 motion-safe:animate-pulse rounded bg-shell-divider" />} empty={null}>
        <LossPlot series={series} objectives={tracks} />
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
