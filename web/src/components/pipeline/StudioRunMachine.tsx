import { useId, type ReactNode } from 'react';
import { studioDateTime, studioMachineStatus, studioMoney, studioName, studioRunStatus, studioUnit } from '@/components/activity/studioVocabulary';
import { Dialog } from '@/components/ui/Dialog';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { ExecutionTargets, StudioMachineQuote } from '@/generated/studioContracts';

export function StudioRunMachine({ catalog, value, disabled, onChange }: {
  catalog?: ExecutionTargets;
  value?: string;
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  const local = catalog?.default ?? 'local';
  const selection = value ?? local;
  const id = useId();
  const current = catalog?.targets.find(({ profile }) => profile.id === selection);
  const unavailable = current && current.availability.state !== 'ready' ? current.availability : undefined;
  return <label className="flex shrink-0 items-center gap-1">
    {studioName('machine')}
    <select value={selection} disabled={disabled}
      aria-describedby={unavailable ? id : undefined}
      className="max-w-48 rounded border border-shell-divider bg-shell-surface px-2 py-1"
      onChange={(event) => onChange(event.target.value)}>
      {catalog?.targets.map(({ profile, availability }) => (
        <option key={profile.id} value={profile.id} disabled={availability.state !== 'ready'}>
          {profile.label} · {studioMachineStatus(availability.state)}
        </option>
      )) ?? <option value={local}>{studioName('thisMac')}</option>}
      {selection !== local && !current &&
        <option value={selection} disabled>{studioMachineStatus('off')}</option>}
    </select>
    {unavailable && <span id={id} className="sr-only">
      {unavailable.reason}
    </span>}
  </label>;
}

function valueLabel(value: unknown) {
  return typeof value === 'string' ? value : JSON.stringify(value);
}

export function StudioMachineApproval({ quote, busy, error, onApprove, onCancel }: {
  quote: StudioMachineQuote;
  busy: boolean;
  error?: string;
  onApprove: () => void;
  onCancel: () => void;
}) {
  const status = studioRunStatus(quote.status);
  const rate = studioMoney(quote.hourly_microusd);
  const values: [string, ReactNode][] = [
    [studioName('machine'), quote.machine],
    [studioName('resources'), <dl>
      {Object.entries(quote.resources).map(([key, value]) => (
        <div key={key} className="flex flex-wrap gap-x-2">
          <dt>{humanizeLabel(key)}</dt><dd>{valueLabel(value)}</dd>
        </div>
      ))}
    </dl>],
    [studioName('hourlyRate'), rate],
    [studioName('runs'), <ol className="list-decimal ps-5">
      {quote.rows.map(({ key, label, axes }) => (
        <li key={JSON.stringify(key)}>
          {label}
          {Object.entries(axes).map(([axis, value]) => <span key={axis} className="ms-2 text-shell-muted">
            {axis} {valueLabel(value)}
          </span>)}
        </li>
      ))}
    </ol>],
    [studioName('costCeiling'), studioMoney(quote.ceiling_microusd)],
    [studioName('expiry'), <time dateTime={new Date(quote.expires_at_ms).toISOString()}>
      {studioDateTime(quote.expires_at_ms)}
    </time>],
    [studioName('idleTimeout'), quote.idle_timeout_seconds === null
      ? studioName('noIdleTimeout') : `${quote.idle_timeout_seconds} ${studioUnit('second')}`],
    [studioName('atExpiry'), studioName('undownloadedResultsDestroyed')],
    [studioName('idleWithResults'), `${studioName(quote.idle_with_results)} · ${rate}/h`],
  ];
  return <Dialog ariaLabel={status} manageFocus onClose={onCancel}
    closeOnEscape={!busy} closeOnBackdrop={false}
    panelClassName="max-h-[90vh] w-[min(32rem,90vw)] overflow-auto rounded bg-shell-surface p-4 shadow">
    <h2 className="font-semibold">{status}</h2>
    <dl className="my-3 space-y-2 text-sm">
      {values.map(([label, value]) => <div key={label} className="grid grid-cols-[9rem_minmax(0,1fr)] gap-2">
        <dt className="font-medium">{label}</dt><dd className="break-words">{value}</dd>
      </div>)}
    </dl>
    {error && <p className="text-shell-status-failure" role="alert">{error}</p>}
    <div className="mt-3 flex justify-end gap-2">
      <button type="button" data-dialog-autofocus className="rounded border border-shell-divider px-3 py-1"
        onClick={onCancel} disabled={busy}>{studioName('cancel')}</button>
      <button type="button" className="rounded border border-shell-divider px-3 py-1"
        onClick={onApprove} disabled={busy}>{studioName('approve')}</button>
    </div>
  </Dialog>;
}
