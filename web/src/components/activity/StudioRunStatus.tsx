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
  not_run: 'bg-shell-status-idle',
  cannot_run: 'bg-shell-status-failure',
  stale: 'bg-shell-status-warning',
  queued: 'bg-shell-status-neutral',
  awaiting_authorization: 'bg-shell-status-warning',
  blocked: 'bg-shell-status-failure',
  starting: 'bg-shell-status-running',
  running: 'bg-shell-status-running',
  stopping: 'bg-shell-status-warning',
  collecting: 'bg-shell-status-running',
  succeeded: 'bg-shell-status-success',
  failed: 'bg-shell-status-failure',
  stopped: 'bg-shell-status-neutral',
  interrupted: 'bg-shell-status-warning',
  corrupt: 'bg-shell-status-failure',
};

const integer = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

export function studioRunMarkClass(status: StudioRunStatusId) {
  return statusColors[status];
}

export function StudioRunStatus({ status }: { status: StudioRunStatusId }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-sm text-shell-muted" aria-live="polite">
      <span aria-hidden="true" className={`size-2 rounded-full ${studioRunMarkClass(status)}`} />
      {studioRunStatus(status)}
    </span>
  );
}

export function StudioRunReason({ job }: { job: StudioActivityJob }) {
  const reason = job.reason ?? (job.status === 'starting' ? studioRunPhase(job.phase) : null);
  return (
    <>
      {reason && <p className="text-xs text-shell-muted">{studioReason('observed', reason)}</p>}
      {job.can_resume && <p className="text-xs text-shell-muted">{studioReason('canResume', studioName('run'))}</p>}
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
        className="studio-running-progress block h-1.5 w-full accent-shell-status-running"
        value={total > 0 ? completed : undefined}
        max={total > 0 ? total : undefined}
      />
      <div className="flex flex-wrap gap-x-2 text-xs tabular-nums text-shell-muted">
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
