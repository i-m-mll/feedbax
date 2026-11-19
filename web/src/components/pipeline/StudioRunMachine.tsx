import { appTip } from '@window-shell/browser/tooltip';
import { Monitor } from 'lucide-react';
import { useId, type ReactNode } from 'react';
import { studioDateTime, studioMachineStatus, studioMoney, studioName, studioOpeningStep, studioRunStatus, studioUnit } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import type { BarControl } from '@/components/ui/CommandBar';
import { Dialog } from '@/components/ui/Dialog';
import { MenuGroup, MenuGroupLabel, MenuRadioGroup, MenuRadioItem } from '@/components/ui/Menu';
import { Select } from '@/components/ui/Select';
import { StatusMark, type MachineMarkStatus } from '@/components/ui/StatusMark';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { ExecutionTargets, StudioMachineQuote, StudioRemoteSession, TargetOption } from '@/generated/studioContracts';
import { useStudioJobs } from '@/hooks/useStudioJobs';

interface MachineProps {
  catalog?: ExecutionTargets;
  value?: string;
  disabled: boolean;
  onChange: (value: string) => void;
}

type MachineState = 'ready' | 'queued' | 'opening' | 'closing' | 'unreachable' | 'off';

const machineMarks = {
  ready: undefined, queued: 'busy', opening: 'busy', closing: 'busy', unreachable: 'unreachable', off: 'off',
} as const satisfies Record<MachineState, MachineMarkStatus | undefined>;

const busyStates: readonly MachineState[] = ['opening', 'queued', 'closing'];

interface MachineChoice {
  value: string;
  label: string;
  state: MachineState;
  /** The opening step or the reason, when there is one. */
  detail?: string;
  disabled: boolean;
}

function profileState(profileId: string, availability: TargetOption['availability'], sessions: readonly StudioRemoteSession[]) {
  if (availability.state !== 'ready') return { state: availability.state, detail: availability.reason };
  const mine = sessions.filter((session) => session.profile === profileId);
  const session = busyStates.map((state) => mine.find((candidate) => candidate.state === state)).find(Boolean);
  if (!session) return { state: 'ready' as const };
  const step = session.state === 'opening' ? session.opening_observations?.at(-1)?.step : undefined;
  return { state: session.state as MachineState, detail: step && studioOpeningStep(step) };
}

function useMachineChoices({ catalog, value }: Pick<MachineProps, 'catalog' | 'value'>) {
  const sessions = useStudioJobs().data?.sessions
    .filter((session): session is StudioRemoteSession => 'profile' in session) ?? [];
  const local = catalog?.default ?? 'local';
  const selection = value ?? local;
  const known = catalog?.targets.some(({ profile }) => profile.id === selection);
  const items: MachineChoice[] = [
    ...catalog?.targets.map(({ profile, availability }) => ({
      value: profile.id, label: profile.label, disabled: availability.state !== 'ready',
      ...profileState(profile.id, availability, sessions),
    })) ?? [{ value: local, label: studioName('thisMac'), state: 'ready' as const, disabled: false }],
    ...selection !== local && !known ? [{ value: selection, label: selection, state: 'off' as const, disabled: true }] : [],
  ];
  return { selection, items, current: items.find((item) => item.value === selection) };
}

function MachineMark({ state, detail }: Pick<MachineChoice, 'state' | 'detail'>) {
  const mark = machineMarks[state];
  if (!mark) return null;
  const word = studioMachineStatus(state);
  return <StatusMark status={mark} label={word} tip={detail ? `${word} · ${detail}` : word} />;
}

function StudioRunMachine({ catalog, value, disabled, onChange }: MachineProps) {
  const { selection, items, current } = useMachineChoices({ catalog, value });
  const id = useId();
  const described = current && current.state !== 'ready';
  return <div className="flex shrink-0" {...appTip(studioName('machine'))}>
    <Select value={selection} disabled={disabled} aria-label={studioName('machine')}
      aria-describedby={described ? id : undefined} className="max-w-48"
      leading={<Monitor aria-hidden="true" />} onValueChange={onChange}
      items={items.map((item) => ({ ...item, mark: <MachineMark {...item} /> }))} />
    {described && <span id={id} className="sr-only">
      {[studioMachineStatus(current.state), current.detail].filter(Boolean).join(' · ')}
    </span>}
  </div>;
}

function StudioRunMachineMenu({ catalog, value, disabled, onChange }: MachineProps) {
  const { selection, items } = useMachineChoices({ catalog, value });
  return <MenuGroup>
    <MenuGroupLabel>{studioName('machine')}</MenuGroupLabel>
    <MenuRadioGroup value={selection} onValueChange={(next: string) => onChange(next)} disabled={disabled}>
      {items.map((item) => <MenuRadioItem key={item.value} value={item.value} disabled={item.disabled}
        mark={<MachineMark {...item} />}>{item.label}</MenuRadioItem>)}
    </MenuRadioGroup>
  </MenuGroup>;
}

/** The Machine picker as a bar entry: a select in the bar, a group of choices in its trailing menu. */
export function studioRunMachineControl(props: MachineProps): BarControl {
  return { id: 'machine', bar: <StudioRunMachine {...props} />, menu: <StudioRunMachineMenu {...props} /> };
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
