import { useId, type ReactNode } from 'react';
import { studioDateTime, studioMachineStatus, studioMoney, studioName, studioRunStatus, studioUnit } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { Select } from '@/components/ui/Select';
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
    <Select value={selection} disabled={disabled} aria-label={studioName('machine')}
      aria-describedby={unavailable ? id : undefined} className="max-w-48"
      onValueChange={onChange}
      items={[
        ...catalog?.targets.map(({ profile, availability }) => ({
          value: profile.id, label: `${profile.label} · ${studioMachineStatus(availability.state)}`,
          disabled: availability.state !== 'ready',
        })) ?? [{ value: local, label: studioName('thisMac') }],
        ...selection !== local && !current ? [{ value: selection, label: studioMachineStatus('off'), disabled: true }] : [],
      ]} />
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
    panelClassName="max-h-[90vh] w-[min(32rem,90vw)] overflow-auto">
    <h2 className="font-semibold">{status}</h2>
    <dl className="my-3 space-y-2 text-sm">
      {values.map(([label, value]) => <div key={label} className="grid grid-cols-[9rem_minmax(0,1fr)] gap-2">
        <dt className="font-medium">{label}</dt><dd className="break-words">{value}</dd>
      </div>)}
    </dl>
    {error && <p className="text-shell-status-failure" role="alert">{error}</p>}
    <div className="mt-3 flex justify-end gap-2">
      <Button data-dialog-autofocus onClick={onCancel} disabled={busy}>{studioName('cancel')}</Button>
      <Button variant="primary" onClick={onApprove} disabled={busy}>{studioName('approve')}</Button>
    </div>
  </Dialog>;
}
