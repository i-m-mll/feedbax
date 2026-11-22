import { useEffect, useMemo, useRef } from 'react';
import { useQueries, useQueryClient } from '@tanstack/react-query';
import { fetchSavedEvaluationObjective } from '@/api/jobs';
import type {
  StudioEvaluationObjectiveSeries, StudioJobEvent, StudioProjectWindowMapRow,
  StudioSavedEvaluationRow, StudioSavedTrainAttempt, StudioSavedTrainKey,
  StudioSavedTrainResult, StudioSavedTrainRow,
} from '@/generated/studioContracts';
import { useStudioJobActions, useStudioJobHistory } from '@/hooks/useStudioJobs';
import { heldOutTracks, type ObjectiveStage } from './heldOutObjective';
import { StudioRegion, StudioRegionBoundary, studioRegionIssue } from './StudioRegion';
import { MetricPlots, studioRunHistory, type MetricResultName, type MetricSeries, type MetricStage } from './StudioTrainMetrics';
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

type ProjectMetricStage = MetricStage & ObjectiveStage & { key: StudioSavedTrainKey };

/** One Evaluate row's held-out objective to read: its finished result, or the live points of a running attempt. */
interface ObjectiveRead {
  key: StudioSavedTrainKey;
  stage: ProjectMetricStage;
  jobId: string;
  receipt: NonNullable<StudioSavedEvaluationRow['result']>['receipt'] | null;
  points: NonNullable<NonNullable<StudioSavedEvaluationRow['checkpoint_attempt']>['points']> | null;
}

function projectMetrics(lineage: Stage[], events: readonly StudioJobEvent[]): ProjectMetricStage[] | null {
  const stages: ProjectMetricStage[] = [];
  let offset = 0;
  for (const stage of lineage) {
    const through = stage.through ?? stage.attempt.progress?.completed_updates ?? 0;
    const exactEvents = events.filter(({ request_id }) => request_id === stage.row.request_id);
    const { metrics, stageStart, attemptStarts, attemptCutoffs } = studioRunHistory(stage.attempt, stage.row.attempts, exactEvents);
    const source = stage.attempt.checkpoint_source;
    // Stateful workers keep the source update clock across resumed attempts.
    if (source && through > 0 && stageStart !== 0 && stageStart !== source.completed_updates) return null;
    const origin = stageStart ?? 0;
    const completed = Math.max(0, through - origin);
    const shown = metrics.map((metric) => ({
      ...metric,
      points: metric.points.filter(({ end }) => end > origin && (stage.through === null || end <= through)).map((point) => ({
        ...point, end: point.end - origin, start: point.start === undefined ? undefined : point.start - origin,
      })),
    }));
    stages.push({
      label: stage.row.key.step, offset, completed, metrics: shown,
      key: stage.row.key, origin, through: stage.through, request: stage.row.request_id,
      starts: attemptStarts, cutoffs: attemptCutoffs,
    });
    offset += completed;
  }
  return stages;
}

export type TrainMetricResult = { name: MetricResultName; result: StudioSavedTrainResult };

function resultMetrics(
  { name, result }: TrainMetricResult, row: StudioSavedTrainRow, rows: readonly StudioSavedTrainRow[],
  events: readonly StudioJobEvent[],
): MetricSeries {
  const exactRow = result.request_id === row.request_id ? row : historicalRequest(row);
  const attempt = exactRow?.attempts.find(({ job_id, successful_receipt }) => job_id === result.job_id &&
    successful_receipt?.manifest_id === result.receipt.manifest_id &&
    successful_receipt.manifest_sha256 === result.receipt.manifest_sha256);
  const lineage = exactRow && attempt && result.progress &&
    checkpointLineage(rows, exactRow, attempt, result.progress.completed_updates);
  return { result: { name, jobId: result.job_id }, stages: (lineage && projectMetrics(lineage, events)) || [] };
}

function TrainMetrics({ project, revision, evaluations, mapRows, rows, row, selection }: {
  project: string;
  revision: string;
  evaluations: readonly StudioSavedEvaluationRow[];
  mapRows: readonly StudioProjectWindowMapRow[];
  rows: readonly StudioSavedTrainRow[];
  row: StudioSavedTrainRow;
  selection: { kind: 'attempt'; attempt: StudioSavedTrainAttempt } | { kind: 'results'; results: TrainMetricResult[] };
}) {
  const history = useStudioJobHistory(project);
  const events = history.data ?? emptyEvents;
  const actions = useStudioJobActions();
  const queryClient = useQueryClient();
  const stages = useMemo(() => {
    if (selection.kind !== 'attempt') return undefined;
    const lineage = checkpointLineage(rows, row, selection.attempt);
    return lineage && projectMetrics(lineage, events);
  }, [rows, row, selection, events]);
  const series = selection.kind === 'results' ? selection.results.map((result) => resultMetrics(result, row, rows, events))
    : stages ? [{ stages }] : [];
  const reads = evaluations.flatMap(({ key, checkpoint_series, checkpoint_attempt, result }): ObjectiveRead[] => {
    const stage = checkpoint_series && stages?.find((candidate) => sameRow(candidate.key, checkpoint_series.subject));
    if (!stage) return [];
    const live = checkpoint_attempt?.placement.location === 'live' && checkpoint_attempt.job_id !== result?.job_id
      ? checkpoint_attempt : null;
    if (live?.points?.some(({ status }) => status === 'scored')) {
      return [{ key, stage, jobId: live.job_id, receipt: null, points: live.points }];
    }
    return result ? [{ key, stage, jobId: result.job_id, receipt: result.receipt, points: null }] : [];
  });
  const queries = reads.map(({ key, jobId, receipt }) => ({
    queryKey: ['saved-evaluation-objective', project, revision, key, jobId] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      fetchSavedEvaluationObjective(project, { revision, key, job_id: jobId }, signal),
    staleTime: (query: { state: { data?: StudioEvaluationObjectiveSeries } }) =>
      receipt && query.state.data?.receipt?.manifest_sha256 === receipt.manifest_sha256 ? Infinity : 0,
  }));
  const objectives = useQueries({ queries });
  const versions = reads.map(({ receipt, points }, index) => {
    const queryKey = queries[index].queryKey;
    const version = JSON.stringify(receipt ?? points?.map(({ publication_cursor, transaction_root_sha256 }) =>
      [publication_cursor, transaction_root_sha256]));
    return { id: JSON.stringify(queryKey), queryKey, version };
  });
  const lastVersions = useRef(new Map<string, string>());
  useEffect(() => {
    const previous = lastVersions.current;
    lastVersions.current = new Map(versions.map(({ id, version }) => [id, version]));
    for (const { id, queryKey, version } of versions) {
      if (previous.has(id) && version !== previous.get(id)) {
        void queryClient.invalidateQueries({ queryKey, exact: true });
      }
    }
  }, [queryClient, versions]);
  const tracks = reads.flatMap(({ key, stage, points, receipt }, index) => {
    const result = objectives[index].data;
    if (!result || objectives[index].isError) return [];
    const selected = points ? new Map(points.map((point) => [point.publication_cursor, point])) : null;
    const current: StudioEvaluationObjectiveSeries = selected ? {
      ...result,
      points: result.points.filter(({ checkpoint }) => {
        const point = selected.get(checkpoint.publication_cursor);
        return point?.status === 'scored' && point.job_id === checkpoint.job_id
          && point.transaction_root_sha256 === checkpoint.transaction_root_sha256;
      }),
    } : result;
    if (receipt && result.receipt && result.receipt.manifest_sha256 !== receipt.manifest_sha256) return [];
    const planRow = mapRows.find((candidate) => sameRow(candidate.key, key));
    const several = mapRows.filter((candidate) => candidate.key.variant === key.variant && candidate.key.step === key.step).length > 1;
    const label = planRow && several ? `${key.step} · ${planRow.label}` : key.step;
    return heldOutTracks(label, current, stage).map((track) => ({
      ...track, id: JSON.stringify([key, track.id]),
    }));
  });
  const objectiveError = objectives.find((query) => query.isError)?.error;
  const retry = () => { actions.refresh(); };
  const state = history.isError
    ? { kind: 'failed' as const, issue: studioRegionIssue('results', history.error), retry }
    : history.isPending
      ? { kind: 'loading' as const }
      : stages === null
        ? { kind: 'failed' as const, issue: { reason: studioReason('unavailable', studioName('checkpoint')) }, retry }
        : { kind: 'result' as const, missing: [] };
  return (
    <section aria-label={studioName('results')} className="mt-3 rounded border border-shell-divider p-3">
      <StudioRegion state={state} skeleton={<div className="h-28 motion-safe:animate-pulse rounded bg-shell-divider" />} empty={null}>
        <MetricPlots series={series} objectives={tracks} />
        {objectiveError && <StudioRegion state={{
          kind: 'failed', issue: studioRegionIssue('objective', objectiveError),
          retry: () => { for (const query of objectives) if (query.isError) void query.refetch(); },
        }} skeleton={null} empty={null}>{null}</StudioRegion>}
      </StudioRegion>
    </section>
  );
}

export function StudioTrainResults(props: Parameters<typeof TrainMetrics>[0]) {
  return (
    <StudioRegionBoundary name="results">
      <TrainMetrics {...props} />
    </StudioRegionBoundary>
  );
}
