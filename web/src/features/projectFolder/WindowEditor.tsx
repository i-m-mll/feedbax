import { plainText, referentName } from '@/components/ui/ShownText';
import { Plus, X } from 'lucide-react';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { ChangeMark as ChangedMark } from '@/components/ui/ChangeMark';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover';
import { Select } from '@/components/ui/Select';
import { fieldBox } from '@/components/ui/styles';
import type { ChangeMark } from '@/components/values/fieldModel';
import type { FieldModel } from '@/components/values/fieldModel';
import { changeTip, ValueField } from '@/components/values/ValueField';
import type {
  Window, WindowEpoch, WindowEpochAnchor, WindowInterval, WindowMoment, WindowPosition, WindowRelativePosition, WindowTrialAnchor,
} from '@/generated/studioContracts';

type Span = Window['spans'][number];
type Anchor = WindowTrialAnchor | WindowEpochAnchor;
type Edge = Anchor['edge'];

/** An epoch a window can name, and how many times it occurs in the trial. */
export interface WindowEpochChoice {
  readonly id: string;
  readonly name: string;
  readonly occurrences: number;
}

/** Where a span's offset is stored, relative to the window. */
export type OffsetSite = readonly ['spans', number, 'start' | 'end' | 'at', 'offset'];

/**
 * What hosts a Window: the epochs it can name, the field of each offset (whose own commit stores it), and `change`,
 * which stores the window's structure, null being the whole trial.
 */
export interface WindowHost {
  readonly epochs: readonly WindowEpochChoice[];
  offset(site: OffsetSite): FieldModel;
  readonly change?: (window: Window | null) => unknown;
  readonly changed?: ChangeMark;
}

const zero = { value: 0, unit: 's' } as const;

/** An epoch, or one zero-based occurrence of it, by a choice's value. */
const epochKey = ({ epoch, occurrence }: { epoch: string; occurrence?: number | null }) =>
  JSON.stringify([epoch, occurrence ?? null]);
const fromEpochKey = (key: string): WindowEpoch => {
  const [epoch, occurrence] = JSON.parse(key) as [string, number | null];
  return occurrence === null ? { kind: 'epoch', epoch } : { kind: 'epoch', epoch, occurrence };
};

/** Each epoch, and each occurrence of one that occurs more than once, numbered from 1. */
function epochItems(epochs: readonly WindowEpochChoice[]) {
  return epochs.flatMap(({ id, name, occurrences }) => [
    { value: epochKey({ epoch: id }), label: name },
    ...occurrences > 1 ? Array.from({ length: occurrences }, (_, occurrence) =>
      ({ value: epochKey({ epoch: id, occurrence }), label: `${name} ${occurrence + 1}` })) : [],
  ]);
}

function epochName(epochs: readonly WindowEpochChoice[], { epoch, occurrence }: { epoch: string; occurrence?: number | null }) {
  const name = plainText(referentName(epochs.find(({ id }) => id === epoch), epoch));
  return occurrence == null ? name : `${name} ${occurrence + 1}`;
}

const trialKey = 'trial';
const lengthKey = 'length';

const anchorKey = (anchor: Anchor) => anchor.kind === 'trial' ? trialKey : epochKey(anchor);
function anchorAt(key: string, edge: Edge): Anchor {
  if (key === trialKey) return { kind: 'trial', edge };
  const { epoch, occurrence } = fromEpochKey(key);
  return occurrence == null ? { kind: 'epoch', epoch, edge } : { kind: 'epoch', epoch, occurrence, edge };
}

const firstEpoch = (epochs: readonly WindowEpochChoice[]): WindowEpoch | undefined =>
  epochs[0] && { kind: 'epoch', epoch: epochs[0].id };
const edgeOf = (epoch: WindowEpoch | undefined, edge: Edge): Anchor =>
  epoch ? { ...epoch, edge } : { kind: 'trial', edge };
const epochOfAnchor = (anchor: Anchor): WindowEpoch | undefined => anchor.kind === 'epoch'
  ? anchor.occurrence == null ? { kind: 'epoch', epoch: anchor.epoch } : { kind: 'epoch', epoch: anchor.epoch, occurrence: anchor.occurrence }
  : undefined;

/** A new span of `kind`; converting `from` keeps the epoch or anchor it names. */
function spanOf(kind: Span['kind'], epochs: readonly WindowEpochChoice[], from?: Span): Span | undefined {
  const epoch = from?.kind === 'epoch' ? from : from?.kind === 'interval' ? epochOfAnchor(from.start.anchor)
    : from?.kind === 'moment' ? epochOfAnchor(from.at.anchor) : undefined;
  const start: WindowPosition = from?.kind === 'interval' ? from.start : from?.kind === 'moment' ? from.at
    : { anchor: edgeOf(epoch ?? firstEpoch(epochs), 'start'), offset: zero };
  switch (kind) {
    case 'epoch': {
      const chosen = epoch ?? firstEpoch(epochs);
      return chosen && { ...chosen };
    }
    case 'interval': return {
      kind: 'interval', start,
      end: from?.kind === 'moment' ? { offset: { value: 1, unit: 'st' } }
        : { anchor: start.anchor.kind === 'trial' ? { kind: 'trial', edge: 'end' } : { ...start.anchor, edge: 'end' }, offset: zero },
    };
    case 'moment': return { kind: 'moment', at: start };
  }
}

/** The text of a window: its spans, `name` an epoch, `[a, b)` an interval, `@a` a moment; empty for the whole trial. */
export function windowText(window: Window | null | undefined, host: Pick<WindowHost, 'epochs' | 'offset'>) {
  const position = (position: WindowPosition | WindowRelativePosition, site: OffsetSite) => {
    const offset = position.offset == null ? '' : host.offset(site).text ?? '';
    const shown = offset && !offset.startsWith('-') ? `+ ${offset}` : offset.replace(/^-\s*/, '- ');
    if (!('anchor' in position)) return shown;
    const { anchor } = position;
    const name = anchor.kind === 'trial' ? studioName('trial') : epochName(host.epochs, anchor);
    const at = anchor.edge === 'end' ? `${name}.${studioName('end').toLowerCase()}` : name;
    return shown && !/^[+-] 0(\.0*)? ?(s|st)?$/.test(shown) ? `${at} ${shown}` : at;
  };
  return (window?.spans ?? []).map((span, index) => {
    switch (span.kind) {
      case 'epoch': return epochName(host.epochs, span);
      case 'interval':
        return `[${position(span.start, ['spans', index, 'start', 'offset'])}, ${position(span.end, ['spans', index, 'end', 'offset'])})`;
      case 'moment': return `@${position(span.at, ['spans', index, 'at', 'offset'])}`;
    }
  }).join(', ');
}

const kinds = ['epoch', 'interval', 'moment'] as const;

function PositionFields({ host, position, site, relative, disabled, set }: {
  host: WindowHost; position: WindowPosition | WindowRelativePosition; site: OffsetSite;
  /** The end of an interval, which may be a length after its start. */
  relative?: boolean;
  disabled: boolean;
  set(position: WindowPosition | WindowRelativePosition): void;
}) {
  const anchored = 'anchor' in position ? position.anchor : undefined;
  const label = studioName(site[2]);
  const places = [
    { value: trialKey, label: studioName('trial') },
    ...epochItems(host.epochs),
    ...relative ? [{ value: lengthKey, label: studioName('length') }] : [],
  ];
  const edges = [{ value: 'start', label: studioName('start') }, { value: 'end', label: studioName('end') }] as const;
  return <div role="group" aria-label={studioName(site[2])} className="flex min-w-0 flex-wrap items-center gap-1">
    <span className="w-8 shrink-0 text-xs text-shell-muted">{label}</span>
    <Select className="min-w-16 flex-1" aria-label={studioAccessibleName('anchor', label)} items={places} disabled={disabled}
      value={anchored ? anchorKey(anchored) : lengthKey}
      onValueChange={(key) => set(key === lengthKey ? { offset: position.offset ?? zero }
        : { anchor: anchorAt(key, anchored?.edge ?? 'start'), offset: position.offset ?? zero })} />
    {anchored && <Select className="min-w-16 flex-1" aria-label={studioAccessibleName('edge', label)} items={edges} disabled={disabled}
      value={anchored.edge} onValueChange={(edge) => set({ ...position, anchor: { ...anchored, edge } })} />}
    <div className="min-w-28 flex-1"><ValueField model={host.offset(site)} form="cell" /></div>
  </div>;
}

function SpanFields({ host, span, index, set, remove }: {
  host: WindowHost; span: Span; index: number; set(span: Span): void; remove(): void;
}) {
  const disabled = !host.change;
  const site = (at: 'start' | 'end' | 'at'): OffsetSite => ['spans', index, at, 'offset'];
  const name = studioAccessibleName('span', index + 1);
  return <li role="group" aria-label={studioAccessibleName('span', index + 1)} className="grid gap-1 border-b border-shell-divider pb-2 last:border-b-0 last:pb-0">
    <div className="flex min-w-0 items-center gap-1">
      <Select className="w-28 shrink-0" aria-label={studioAccessibleName('kind', name)} disabled={disabled}
        items={kinds.map((kind) => ({ value: kind, label: studioName(kind), disabled: kind === 'epoch' && !host.epochs.length }))}
        value={span.kind} onValueChange={(kind) => {
          const next = kind === span.kind ? undefined : spanOf(kind, host.epochs, span);
          if (next) set(next);
        }} />
      {span.kind === 'epoch' && <Select className="min-w-0 flex-1" aria-label={studioAccessibleName('epoch', name)} disabled={disabled}
        items={epochItems(host.epochs)} value={epochKey(span)} onValueChange={(key) => set(fromEpochKey(key))} />}
      {!disabled && <IconButton size="mark" className="ml-auto" label={studioAccessibleName('remove', name)} icon={<X />} onClick={remove} />}
    </div>
    {span.kind === 'interval' && <>
      <PositionFields host={host} position={span.start} site={site('start')} disabled={disabled}
        set={(start) => set({ ...span, start: start as WindowPosition } satisfies WindowInterval)} />
      <PositionFields host={host} position={span.end} site={site('end')} relative disabled={disabled}
        set={(end) => set({ ...span, end } satisfies WindowInterval)} />
    </>}
    {span.kind === 'moment' && <PositionFields host={host} position={span.at} site={site('at')} disabled={disabled}
      set={(at) => set({ ...span, at: at as WindowPosition } satisfies WindowMoment)} />}
  </li>;
}

/**
 * A Window's spans, each a whole epoch, a half-open interval between two anchored positions, or a moment; no span is
 * the whole trial. Each offset is its host's field; the structure is stored through the host's `change`.
 */
export function WindowEditor({ window, host, labelId }: { window: Window | null | undefined; host: WindowHost; labelId?: string }) {
  const spans = window?.spans ?? [];
  const store = (next: readonly Span[]) => { void host.change?.(next.length ? { spans: [...next] } : null); };
  const add = (kind: Span['kind']) => {
    const span = spanOf(kind, host.epochs);
    if (span) store([...spans, span]);
  };
  return <div role="group" aria-labelledby={labelId} aria-label={labelId ? undefined : studioName('window')} className="grid gap-2">
    {spans.length ? <ol className="grid gap-2">
      {spans.map((span, index) => <SpanFields key={index} host={host} span={span} index={index}
        set={(next) => store(spans.map((old, position) => position === index ? next : old))}
        remove={() => store(spans.filter((_, position) => position !== index))} />)}
    </ol> : <span className="text-shell-muted">{studioName('wholeTrial')}</span>}
    {host.change && <Menu>
      <MenuTrigger render={<IconButton size="mark" label={studioName('addSpan')} icon={<Plus />} />} />
      <MenuContent aria-label={studioName('addSpan')}>
        {kinds.map((kind) => <MenuItem key={kind} disabled={kind === 'epoch' && !host.epochs.length} onClick={() => add(kind)}>
          {studioName(kind)}
        </MenuItem>)}
      </MenuContent>
    </Menu>}
  </div>;
}

/** A Window in a table cell: its text, which opens the editor when the host can change it. */
export function WindowCell({ window, host }: { window: Window | null | undefined; host: WindowHost }) {
  const text = windowText(window, host);
  const { changed } = host;
  if (!host.change) {
    return <span className="truncate px-control-inline font-mono text-xs">
      {text || <span className="font-sans text-shell-muted">{studioName('wholeTrial')}</span>}
    </span>;
  }
  return <Popover>
    <div data-field-box="" className={fieldBox}>
      <PopoverTrigger aria-label={studioAccessibleName('window', text || studioName('wholeTrial'))}
        className="flex h-full min-w-0 flex-1 items-center truncate px-control-inline text-left font-mono text-xs outline-none">
        {text || <span className="font-sans text-shell-muted">{studioName('wholeTrial')}</span>}
      </PopoverTrigger>
      {changed && <span className="flex shrink-0 pr-1">
        <ChangedMark tip={changeTip(changed.from)} reset={changed.reset && (() => { void changed.reset!(); })} />
      </span>}
    </div>
    <PopoverContent aria-label={studioName('window')} className="w-[26rem] max-w-[var(--available-width)]">
      <WindowEditor window={window} host={host} />
    </PopoverContent>
  </Popover>;
}
