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
import { useStudioJobActions, useStudioJobConnection, useStudioJobHistory, useStudioJobs } from '@/hooks/useStudioJobs';
import { StudioRegion, StudioRegionBoundary, studioRegionIssue, type StudioRegionState } from './StudioRegion';
import { StudioRunProgress, StudioRunReason, StudioRunStatus } from './StudioRunStatus';
import { studioName, studioReason } from './studioVocabulary';
import { LossPlot, studioRunHistory } from './StudioTrainingLoss';
import { PlotTrack, plotHeight, plotWidth, projectSeries } from './studioPlot';

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
            className="h-28 w-full rounded border border-slate-200 bg-white"
          >
            <PlotTrack points={goal} stroke="text-violet-700" />
            <PlotTrack points={effector} stroke="text-blue-700" />
          </svg>
          <div className="flex flex-wrap gap-2 text-xs">
            {effector.length > 0 && <span className="text-blue-700">{studioName('effector')}</span>}
            {goal.length > 0 && <span className="text-violet-700">{studioName('goal')}</span>}
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
  const connection = useStudioJobConnection();
  const actions = useStudioJobActions();
  const data = useMemo(() => studioRunHistory(job, jobs, events), [job, jobs, events]);
  const retry = () => { void actions.refresh(); };
  const state: StudioRegionState = connection === 'disconnected' || history.isError
    ? { kind: 'failed', issue: studioRegionIssue('inspector', history.error, connection === 'disconnected'), retry }
    : history.isPending
      ? { kind: 'loading' }
      : events.length > 0
        ? { kind: 'result', missing: [] }
        : { kind: 'empty' };
  const fields = (
    <>
      <div className="grid gap-3 sm:grid-cols-2">
        <LossPlot stages={[{ label: job.label, offset: 0, completed: 0, points: data.loss }]} />
        <RunScene event={data.trajectory} retry={retry} />
      </div>
      <section aria-label={studioName('log')} className="min-w-0 space-y-1">
        <h4 className="text-xs font-medium">{studioName('log')}</h4>
        <pre className="min-h-9 max-h-32 select-text overflow-auto whitespace-pre-wrap break-words rounded border border-slate-200 bg-white p-2 text-xs text-slate-700">
          {data.log}
        </pre>
      </section>
    </>
  );
  return (
    <section aria-label={studioName('inspector')} className="space-y-3 rounded-md border border-slate-200 bg-slate-50 p-3">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="break-words font-medium">{job.label}</h3>
          <StudioRunReason job={job} />
        </div>
        <StudioRunStatus status={job.status} />
      </header>
      {job.progress && <StudioRunProgress progress={job.progress} etaSeconds={job.eta_seconds} />}
      <StudioRegion
        state={state}
        skeleton={<div className="h-36 animate-pulse rounded bg-slate-200" />}
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
