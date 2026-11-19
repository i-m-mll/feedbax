import type { HTMLAttributes } from 'react';
import { StatusMark } from '@/components/ui/StatusMark';
import type { StoreTransferProgress, StudioActivityJob } from '@/generated/studioContracts';
import {
  studioName,
  studioFormat,
  studioReason,
  studioRunPhase,
  studioRunStatus,
  studioUnit,
  type StudioResultState,
} from './studioVocabulary';

const integer = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

export function StudioRunStatus({ status, ...props }: { status: StudioResultState } & HTMLAttributes<HTMLSpanElement>) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-sm text-shell-muted" {...props}>
      <StatusMark status={status} />
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

const bytes = new Intl.NumberFormat(undefined, { style: 'unit', unit: 'byte', unitDisplay: 'narrow' });

export function StudioCollectionProgress({ progress }: { progress?: StoreTransferProgress | null }) {
  return <div className="space-y-1">
    <progress aria-label={studioRunStatus('collecting')} value={progress?.verified_bytes} max={progress?.total_bytes} />
    {progress && <div className="text-xs text-shell-muted">
      {studioName('verifiedBytes')} {studioFormat('ofCount', {
        count: bytes.format(progress.verified_bytes), total: bytes.format(progress.total_bytes),
      })}
    </div>}
  </div>;
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
