import clsx from 'clsx';
import { Check, ChevronDown } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { studioFormat, studioName, studioRunStatus } from '@/components/activity/studioVocabulary';
import { StatusMark } from '@/components/ui/StatusMark';
import { Button } from '@/components/ui/Button';
import { FieldRow } from '@/components/ui/FieldRow';
import { NumberField } from '@/components/ui/NumberField';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { popupItem, primaryText, secondaryText } from '@/components/ui/styles';
import { plainText, Shown, type ShownText } from '@/components/ui/ShownText';
import {
  aggregates, averages, offeredPicks, pickText, samePick, sampleCount, setSize, type TrialClock, type TrialOffer, type TrialPick,
  type TrialSet, type TrialShowing,
} from './trialClock';

export interface TrialPickerProps {
  readonly offer: TrialOffer;
  readonly clock: TrialClock;
  readonly shown: TrialPick;
  /** A recorded trial's condition, by its index, drawn after its entry. */
  readonly recordedLabels?: readonly ShownText[];
  /** Whether the recorded trials are a result an edit put out of date, which their entries and the shown text mark. */
  readonly recordedOutOfDate?: boolean;
  /** Mean's By options, once the Service has answered a Mean. */
  readonly meanOptions?: MeanChoices;
  readonly onChange: (clock: TrialClock) => void;
}

/** A whole number, committed when its edit ends; a value outside `least` to `most` is brought within them. */
function WholeField({ label, value, least, most, onCommit }: {
  label: string; value: number; least: number; most: number; onCommit(value: number): void;
}) {
  const [draft, setDraft] = useState<number | null>(value);
  return <FieldRow label={label}>
    <NumberField aria-label={label} value={draft} min={least} max={most} step={1} onValueChange={setDraft}
      onValueCommitted={(next) => {
        const whole = next === null || !Number.isFinite(next) ? value : Math.min(most, Math.max(least, Math.round(next)));
        setDraft(whole);
        if (whole !== value) onCommit(whole);
      }} />
  </FieldRow>;
}

function Entry({ checked, role, onClick, children }: {
  checked: boolean; role: 'radio' | 'checkbox'; onClick(): void; children: ReactNode;
}) {
  return <Button variant="bare" size="content" align="start" role={role} aria-checked={checked}
    className={clsx(popupItem, 'w-full text-left tabular-nums hover:bg-shell-hover')} onClick={onClick}>
    <span className="flex size-[1em] shrink-0 items-center justify-center">{checked && <Check aria-hidden="true" />}</span>
    {children}
  </Button>;
}

/** A section's name in the popup, above its entries. */
const sectionLabel = 'px-row-inline pb-0.5 text-xs text-shell-muted';

const sets: readonly TrialSet[] = ['nominal', 'samples', 'recorded'];

/** The trials under Mean's By: the labels of the variables chosen, all of them unless chosen. */
/** What a Mean can be taken by and aligned on: Task variables and epochs, each by its id with its current name. */
export interface MeanChoices {
  readonly by: readonly { readonly id: string; readonly label: ShownText }[];
  readonly align: readonly { readonly id: string; readonly label: ShownText }[];
}

const byLabels = (options: MeanChoices | undefined, by: readonly string[] | null) =>
  (options?.by ?? []).filter(({ id }) => !by || by.includes(id)).map(({ label }) => plainText(label));

/**
 * The one statement of which trials are shown and how. Its text is what is shown. Its popup chooses One, All or Mean,
 * then what that showing is of: under One a single trial, every one that exists for the selection in groups; under All
 * the set shown, when more than one can be; under Mean the variables whose values make a condition and the epoch whose start is zero (the trial's
 * start unless chosen). The seed and the
 * count the samples are drawn with are at its foot while samples are offered.
 */
export function TrialPicker({ offer, clock, shown, recordedLabels, recordedOutOfDate, meanOptions, onChange }: TrialPickerProps) {
  const [open, setOpen] = useState(false);
  const count = sampleCount(clock, offer);
  const text = (pick: TrialPick) => pickText(pick, clock, offer);
  const kind = shown.showing.kind;
  const by = byLabels(meanOptions, clock.mean.by);
  const shownText = kind !== 'mean' ? text(shown)
    : meanOptions && by.length ? studioFormat('meanByText', { by: by.join(', ') })
      : studioFormat('meanOfCount', { count: setSize('recorded', clock, offer) });
  const outOfDate = recordedOutOfDate && shown.trials === 'recorded';
  const aggregated = sets.filter((trials) => aggregates(trials) && setSize(trials, clock, offer) > 0);
  const pick = (picked: TrialPick) => onChange({ ...clock, picked });
  const showAs = (next: TrialShowing['kind']) => {
    if (next === kind) return;
    const trials = next === 'one' ? shown.trials : next === 'mean' ? sets.find(averages)!
      : aggregated.includes(shown.trials) ? shown.trials : aggregated[0];
    pick({ trials, showing: { kind: next, index: trials === shown.trials ? shown.showing.index : 0 } });
  };
  const ones = offeredPicks(offer, count);
  const groups = sets.map((trials) => ones.filter((one) => one.trials === trials)).filter((group) => group.length);
  const list = (children: ReactNode) => <div role="radiogroup" aria-label={studioName('trial')} className="grid max-h-80 grid-cols-[minmax(0,1fr)] overflow-y-auto">
    {children}
  </div>;
  const second = kind === 'one' ? list(groups.map((group, index) => <div key={group[0].trials} role="presentation"
    className={clsx('grid grid-cols-[minmax(0,1fr)]', index > 0 && 'mt-1 border-t border-shell-divider pt-1')}>
    {group.map((one) => {
      const at = one.showing.index;
      return <Entry key={text(one)} role="radio" checked={samePick(one, shown)} onClick={() => { pick(one); setOpen(false); }}>
        <span className={clsx(primaryText, 'flex-none')}>{text(one)}</span>
        {recordedOutOfDate && one.trials === 'recorded' && <StatusMark status="stale" label={studioRunStatus('stale')} />}
        {one.trials === 'recorded' && recordedLabels?.[at] && <span className={secondaryText}><Shown text={recordedLabels[at]} /></span>}
      </Entry>;
    })}
  </div>))
    : kind === 'all' ? aggregated.length > 1 && list(aggregated.map((trials) => {
      const all: TrialPick = { trials, showing: { kind: 'all', index: trials === shown.trials ? shown.showing.index : 0 } };
      return <Entry key={trials} role="radio" checked={trials === shown.trials} onClick={() => pick(all)}>
        <span className={primaryText}>{text(all)}</span>
        <span className={secondaryText}>{studioName(trials === 'samples' ? 'sampledTrials' : 'recordedTrials')}</span>
      </Entry>;
    }))
      : meanOptions && <>
        <div role="group" aria-label={studioName('meanBy')} className="grid grid-cols-[minmax(0,1fr)]">
        <span aria-hidden="true" className={sectionLabel}>{studioName('meanBy')}</span>
        {meanOptions.by.map(({ id, label }) => {
          const chosen = !clock.mean.by || clock.mean.by.includes(id);
          return <Entry key={id} role="checkbox" checked={chosen} onClick={() => {
            const next = meanOptions.by.map((option) => option.id)
              .filter((option) => option === id ? !chosen : !clock.mean.by || clock.mean.by.includes(option));
            onChange({ ...clock, mean: { ...clock.mean, by: next.length === meanOptions.by.length ? null : next } });
          }}><span className={primaryText}><Shown text={label} /></span></Entry>;
        })}
        </div>
        <div role="radiogroup" aria-label={studioName('meanAlign')}
          className="mt-1 grid grid-cols-[minmax(0,1fr)] border-t border-shell-divider pt-1">
          <span aria-hidden="true" className={sectionLabel}>{studioName('meanAlign')}</span>
          {[{ id: null, label: studioName('trialStart') as ShownText }, ...meanOptions.align].map(({ id, label }) =>
            <Entry key={id ?? ''} role="radio" checked={clock.mean.align === id}
              onClick={() => onChange({ ...clock, mean: { ...clock.mean, align: id } })}>
              <span className={primaryText}><Shown text={label} /></span>
            </Entry>)}
        </div>
      </>;
  const sampled = offer.samples > 0 && (kind === 'one' || shown.trials === 'samples');
  return <Popover open={open} onOpenChange={setOpen}>
    <PopoverTrigger render={<Button variant="ghost" className="w-44 shrink-0 gap-1 px-1 tabular-nums" />}
      aria-label={`${studioName('trial')} ${shownText}`} aria-description={outOfDate ? studioRunStatus('stale') : undefined}>
      {outOfDate && <StatusMark status="stale" />}
      <span className="grow truncate text-left">{shownText}</span>
      <ChevronDown aria-hidden="true" />
    </PopoverTrigger>
    <PopoverContent side="top" aria-label={studioName('trial')}
      className="grid w-72 grid-cols-[minmax(0,1fr)] gap-1 !p-1 [&>*+*]:border-t [&>*+*]:border-shell-divider [&>*+*]:pt-1">
      {second && <div>{second}</div>}
      {sampled && <div className="grid gap-1 px-1">
        <WholeField label={studioName('seed')} value={clock.seed} least={Number.MIN_SAFE_INTEGER} most={Number.MAX_SAFE_INTEGER}
          onCommit={(seed) => onChange({ ...clock, seed })} />
        <WholeField label={studioName('count')} value={count} least={1} most={offer.samples}
          onCommit={(count) => onChange({ ...clock, count })} />
      </div>}
      <div><SegmentedControl aria-label={studioName('trials')} value={kind} className="w-full [&>*]:flex-1"
        items={[
          { value: 'one', label: studioName('oneTrial') },
          { value: 'all', label: studioName('allTrials'), disabled: !aggregated.length },
          { value: 'mean', label: studioName('mean'), disabled: !sets.some((trials) => averages(trials) && setSize(trials, clock, offer) > 0) },
        ]}
        onValueChange={showAs} /></div>
    </PopoverContent>
  </Popover>;
}
