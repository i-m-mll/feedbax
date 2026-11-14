import type { CSSProperties } from 'react';
import type { StudioActivityJob } from '@/generated/studioContracts';
import {
  studioName,
  studioReason,
  studioRunPhase,
  studioRunStatus,
  studioUnit,
  type StudioRunStatusId,
} from './studioVocabulary';

const statusMarks = {
  not_run: ['bg-shell-status-idle', 'M3 6h6'],
  cannot_run: ['bg-shell-status-failure', 'M6 2v5m0 2v1'],
  stale: ['bg-shell-status-warning', 'M3 7a3 3 0 1 0 1-3m-1-2v3h3'],
  queued: ['bg-shell-status-neutral', 'M2 4h8M2 8h8'],
  awaiting_authorization: ['bg-shell-status-warning', 'M4 5V4a2 2 0 0 1 4 0v1M3 5h6v5H3z'],
  blocked: ['bg-shell-status-failure', 'M4 2v8M8 2v8'],
  starting: ['bg-shell-status-running', 'M3 8l3-4 3 4'],
  running: ['bg-shell-status-running', 'M4 3l5 3-5 3z'],
  stopping: ['bg-shell-status-warning', 'M3 3h6v6H3z'],
  collecting: ['bg-shell-status-running', 'M6 2v6M3 5l3 3 3-3M3 10h6'],
  succeeded: ['bg-shell-status-success', 'M2.5 6l2.5 2.5 4.5-5'],
  failed: ['bg-shell-status-failure', 'M3 3l6 6m0-6L3 9'],
  stopped: ['bg-shell-status-neutral', 'M3 6h6M6 3v6'],
  interrupted: ['bg-shell-status-warning', 'M4 8l4-4'],
  corrupt: ['bg-shell-status-failure', 'M3 9l3-6 3 6z'],
} satisfies Record<StudioRunStatusId, readonly [string, string]>;

const integer = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

export function StudioRunMark({ status, count }: { status: StudioRunStatusId; count?: number }) {
  const [color, path] = statusMarks[status];
  return (
    <span aria-hidden="true" data-run-status-mark={status} className={`flex h-4 min-w-4 items-center justify-center rounded-sm text-white ${color}`}
      style={count === undefined ? undefined : { flexGrow: count } satisfies CSSProperties}>
      <svg className="size-4 shrink-0" viewBox="0 0 12 12">
        <path d={path} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </span>
  );
}

export function StudioRunStatus({ status }: { status: StudioRunStatusId }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-sm text-shell-muted">
      <span aria-hidden="true" className={`size-2 rounded-full ${statusMarks[status][0]}`} />
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
