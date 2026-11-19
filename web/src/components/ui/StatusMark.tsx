import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import type { HTMLAttributes } from 'react';
import type { ReceiptValidity, StudioActivityJob } from '@/generated/studioContracts';

/** A run's status with its result's validity folded in. */
export type ResultStatus =
  | StudioActivityJob['status']
  | Exclude<ReceiptValidity['state'], 'current' | 'stale'>
  | 'checking'
  | 'environment';

/** A machine's state that draws a mark; a ready machine draws none. */
export type MachineMarkStatus = 'busy' | 'unreachable' | 'off';

export type StatusMarkStatus = ResultStatus | MachineMarkStatus;

type Fill = 'solid' | 'hollow';

const marks = {
  not_run: ['text-shell-status-idle', 'solid', 'M3 6h6'],
  cannot_run: ['text-shell-status-failure', 'solid', 'M6 2v5m0 2v1'],
  stale: ['text-shell-status-warning', 'solid', 'M3 7a3 3 0 1 0 1-3m-1-2v3h3'],
  queued: ['text-shell-status-neutral', 'solid', 'M2 4h8M2 8h8'],
  awaiting_authorization: ['text-shell-status-warning', 'solid', 'M4 5V4a2 2 0 0 1 4 0v1M3 5h6v5H3z'],
  blocked: ['text-shell-status-failure', 'solid', 'M4 2v8M8 2v8'],
  starting: ['text-shell-status-running', 'solid', 'M3 8l3-4 3 4'],
  running: ['text-shell-status-running', 'solid', 'M4 3l5 3-5 3z'],
  stopping: ['text-shell-status-warning', 'solid', 'M3 3h6v6H3z'],
  collecting: ['text-shell-status-running', 'solid', 'M6 2v6M3 5l3 3 3-3M3 10h6'],
  succeeded: ['text-shell-status-success', 'solid', 'M2.5 6l2.5 2.5 4.5-5'],
  failed: ['text-shell-status-failure', 'solid', 'M3 3l6 6m0-6L3 9'],
  stopped: ['text-shell-status-neutral', 'solid', 'M3 6h6M6 3v6'],
  interrupted: ['text-shell-status-warning', 'solid', 'M4 8l4-4'],
  corrupt: ['text-shell-status-failure', 'solid', 'M3 9l3-6 3 6z'],
  unchecked: ['text-shell-status-success', 'hollow', 'M2.5 6l2.5 2.5 4.5-5'],
  checking: ['text-shell-status-success', 'hollow', 'M6 2a4 4 0 1 1-4 4'],
  environment: ['text-shell-status-neutral', 'hollow', 'M6 2v8M2 6h8M3.5 3.5l5 5m0-5-5 5'],
  busy: ['text-shell-status-running', 'hollow', 'M6 2a4 4 0 1 1-4 4'],
  unreachable: ['text-shell-status-failure', 'hollow', 'M2 6h2.5m3 0H10M5 4l2 4'],
  off: ['text-shell-status-idle', 'hollow', 'M6 2v3.5M3.8 4a3.2 3.2 0 1 0 4.4 0'],
} as const satisfies Record<StatusMarkStatus, readonly [string, Fill, string]>;

/** The status's colour class, for a mark too small to carry its shape. */
export function statusTone(status: StatusMarkStatus) {
  return marks[status][0];
}

export interface StatusMarkProps extends Omit<HTMLAttributes<HTMLSpanElement>, 'children'> {
  status: StatusMarkStatus;
  /** The status's word, read aloud and shown as the tooltip; omit where the mark sits beside it. */
  label?: string;
  /** The tooltip in place of the label, when it says more than the word, such as the reason. */
  tip?: string;
}

const spinning: ReadonlySet<StatusMarkStatus> = new Set(['checking', 'busy']);

export function StatusMark({ status, label, tip, className, ...props }: StatusMarkProps) {
  const [tone, fill, path] = marks[status];
  const unchecked = status === 'unchecked';
  return (
    <span
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      {...(label ? appTip(tip ?? label) : {})}
      data-status-mark={status}
      className={clsx(
        'relative inline-flex size-4 shrink-0 items-center justify-center rounded-sm border-[1.5px] border-current',
        tone,
        unchecked && '[:root[data-unchecked=badge]_&]:bg-current',
        fill === 'solid' && 'bg-current',
        className,
      )}
      {...props}
    >
      <svg
        className={clsx(
          'size-3.5 shrink-0',
          fill === 'solid' ? 'text-shell-status-ink' : 'text-current',
          unchecked && '[:root[data-unchecked=badge]_&]:text-shell-status-ink',
          spinning.has(status) && 'motion-safe:animate-spin',
        )}
        viewBox="0 0 12 12"
      >
        <path d={path} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {unchecked ? (
        <span className={clsx(
          'absolute -right-[3px] -top-[3px] hidden size-[7px] rounded-full border-[1.5px] border-shell-surface bg-shell-status-warning',
          '[:root[data-unchecked=badge]_&]:block',
        )} />
      ) : null}
    </span>
  );
}

/** Statuses from the one that most needs attention to the one that needs none. */
const severity: readonly ResultStatus[] = [
  'failed', 'corrupt', 'cannot_run', 'blocked', 'interrupted', 'awaiting_authorization', 'stale', 'environment',
  'stopping', 'running', 'collecting', 'starting', 'checking', 'queued', 'unchecked', 'stopped', 'not_run', 'succeeded',
];

/** The status of a set that most needs attention, with how many members have it. */
export function worstStatus(counts: Readonly<Partial<Record<ResultStatus, number>>>) {
  const status = severity.find((candidate) => (counts[candidate] ?? 0) > 0);
  return status && { status, count: counts[status]! };
}
