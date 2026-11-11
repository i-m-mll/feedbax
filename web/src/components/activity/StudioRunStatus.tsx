import type { StudioActivityJob } from '@/generated/studioContracts';
import {
  studioName,
  studioReason,
  studioRunPhase,
  studioRunStatus,
  studioUnit,
  type StudioRunStatusId,
} from './studioVocabulary';

const statusColors: Record<StudioRunStatusId, string> = {
  not_run: 'bg-slate-400',
  cannot_run: 'bg-red-600',
  stale: 'bg-amber-500',
  queued: 'bg-slate-500',
  awaiting_authorization: 'bg-amber-500',
  blocked: 'bg-red-600',
  starting: 'bg-blue-600',
  running: 'bg-blue-600',
  stopping: 'bg-amber-500',
  collecting: 'bg-blue-600',
  succeeded: 'bg-green-600',
  failed: 'bg-red-600',
  stopped: 'bg-slate-500',
  interrupted: 'bg-amber-500',
  corrupt: 'bg-red-600',
};

const integer = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

export function StudioRunStatus({ status }: { status: StudioRunStatusId }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-sm text-slate-700" aria-live="polite">
      <span aria-hidden="true" className={`size-2 rounded-full ${statusColors[status]}`} />
      {studioRunStatus(status)}
    </span>
  );
}

export function StudioRunReason({ job }: { job: StudioActivityJob }) {
  const reason = job.reason ?? (job.status === 'starting' ? studioRunPhase(job.phase) : null);
  return (
    <>
      {reason && <p className="text-xs text-slate-700">{studioReason('observed', reason)}</p>}
      {job.can_resume && <p className="text-xs text-slate-700">{studioReason('canResume', studioName('run'))}</p>}
    </>
  );
}

export function StudioRunProgress({
  progress,
  etaSeconds,
}: {
  progress: NonNullable<StudioActivityJob['progress']>;
  etaSeconds?: number | null;
}) {
  const { completed_updates: completed, total_updates: total } = progress;
  return (
    <div className="min-w-24 space-y-1">
      <progress
        aria-label={studioName('progress')}
        className="block h-1.5 w-full accent-blue-600"
        value={total > 0 ? completed : undefined}
        max={total > 0 ? total : undefined}
      />
      <div className="flex flex-wrap gap-x-2 text-xs tabular-nums text-slate-700">
        <span>{integer.format(completed)} / {integer.format(total)}</span>
        {etaSeconds != null && Number.isFinite(etaSeconds) && (
          <span aria-label={studioName('remaining')}>
            {etaSeconds < 60
              ? `${integer.format(Math.ceil(etaSeconds))} ${studioUnit('second')}`
              : `${integer.format(Math.ceil(etaSeconds / 60))} ${studioUnit('minute')}`}
          </span>
        )}
      </div>
    </div>
  );
}
