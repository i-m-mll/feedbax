import { useMemo } from 'react';
import {
  TrainingTrajectoryPayloadSchema,
  type StudioActivityJob,
  type StudioJobEvent,
} from '@/generated/studioContracts';
import {
  liveTrainingEffectorTrack,
  liveTrainingTargetTrack,
  normalizeTrainingTrajectoryPayload,
} from '@/features/scenario/liveTraining';
import { workspaceReplayPolyline } from '@/features/scenario/workspaceReplay';
import { useStudioJobActions, useStudioJobHistory, useStudioJobs } from '@/hooks/useStudioJobs';
import { StudioRegion, StudioRegionBoundary, studioRegionIssue, type StudioRegionState } from './StudioRegion';
import { StudioCollectionProgress, StudioRunProgress, StudioRunReason, StudioRunStatus } from './StudioRunStatus';
import { studioName, studioReason } from './studioVocabulary';
import { MetricPlots, studioRunHistory } from './StudioTrainMetrics';
import { PlotTrack, plotHeight, plotWidth, projectSeries } from './studioPlot';
import { ScrollFade } from '@/components/ui/ScrollFade';

const emptyJobs: readonly StudioActivityJob[] = [];
const emptyEvents: readonly StudioJobEvent[] = [];

function RunScene({ event, retry }: { event?: StudioJobEvent; retry: () => void }) {
  const parsed = useMemo(() => event ? TrainingTrajectoryPayloadSchema.safeParse(event.data.trajectory) : null, [event]);
  const tracks = useMemo(() => {
    if (!parsed?.success) return [];
    const frame = normalizeTrainingTrajectoryPayload(parsed.data);
    return projectSeries([
      workspaceReplayPolyline(liveTrainingEffectorTrack(frame)),
      workspaceReplayPolyline(liveTrainingTargetTrack(frame)),
    ], true);
  }, [parsed]);
  const [effector = [], goal = []] = tracks;
  return (
    <section aria-label={studioName('scene')} className="min-w-0 space-y-1">
      <h4 className="text-xs font-medium">{studioName('scene')}</h4>
      {parsed?.success === false ? (
        <StudioRegion
          state={{
            kind: 'failed',
            issue: { reason: studioReason('unavailable', studioName('scene')), detail: parsed.error.message },
            retry,
          }}
          skeleton={null}
          empty={null}
        >{null}</StudioRegion>
      ) : (
        <>
          <svg
            viewBox={`0 0 ${plotWidth} ${plotHeight}`}
            role="img"
            aria-label={studioName('scene')}
            className="h-28 w-full rounded border border-shell-divider bg-shell-surface"
          >
            <PlotTrack points={goal} dashed />
            <PlotTrack points={effector} />
          </svg>
          <div className="flex flex-wrap gap-2 text-xs">
            {effector.length > 0 && <span className="text-shell-foreground">{studioName('effector')}</span>}
            {goal.length > 0 && <span className="decoration-dashed underline">{studioName('goal')}</span>}
          </div>
        </>
      )}
    </section>
  );
}

function RunInspector({ job }: { job: StudioActivityJob }) {
  const jobs = useStudioJobs().data?.jobs ?? emptyJobs;
  const history = useStudioJobHistory(job.project_id);
  const events = history.data ?? emptyEvents;
  const actions = useStudioJobActions();
  const data = useMemo(() => studioRunHistory(job, jobs, events), [job, jobs, events]);
  const retry = () => { void actions.refresh(); };
  const state: StudioRegionState = history.isError
    ? { kind: 'failed', issue: studioRegionIssue('inspector', history.error), retry }
    : history.isPending
      ? { kind: 'loading' }
      : events.length > 0
        ? { kind: 'result', missing: [] }
        : { kind: 'empty' };
  const fields = (
    <>
      <div className="grid gap-3 sm:grid-cols-2">
        <MetricPlots series={[{ stages: [{ label: job.label, offset: 0, completed: 0, metrics: data.metrics }] }]} />
        <RunScene event={data.trajectory} retry={retry} />
      </div>
      <section aria-label={studioName('log')} className="min-w-0 space-y-1">
        <h4 className="text-xs font-medium">{studioName('log')}</h4>
        <ScrollFade className="min-h-9 max-h-32 rounded border border-shell-divider bg-shell-surface p-2">
          <pre className="select-text whitespace-pre-wrap break-words text-xs text-shell-muted">{job.log_tail?.join('\n') ?? data.log}</pre>
        </ScrollFade>
      </section>
    </>
  );
  return (
    <section aria-label={studioName('inspector')} className="space-y-3 rounded-md border border-shell-divider bg-shell-subtle p-3">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="break-words font-medium">{job.label}</h3>
          <StudioRunReason job={job} />
        </div>
        <StudioRunStatus status={job.status} />
      </header>
      {job.status === 'running' && job.progress && <StudioRunProgress progress={job.progress} etaSeconds={job.eta_seconds} />}
      {job.status === 'collecting' && <StudioCollectionProgress progress={job.collection_progress} />}
      <StudioRegion
        state={state}
        skeleton={<div className="h-36 motion-safe:animate-pulse rounded bg-shell-divider" />}
        empty={fields}
      >
        {fields}
      </StudioRegion>
    </section>
  );
}

export function StudioRunInspector({ job }: { job: StudioActivityJob }) {
  return (
    <StudioRegionBoundary name="inspector">
      <RunInspector job={job} />
    </StudioRegionBoundary>
  );
}
