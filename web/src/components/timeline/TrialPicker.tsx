import clsx from 'clsx';
import { Check, ChevronDown } from 'lucide-react';
import { useState } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { FieldRow } from '@/components/ui/FieldRow';
import { NumberField } from '@/components/ui/NumberField';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover';
import { popupItem, primaryText, secondaryText } from '@/components/ui/styles';
import {
  entryText, maxSampleCount, offeredEntries, sameEntry, type TrialClock, type TrialEntry, type TrialOffer,
} from './trialClock';

export interface TrialPickerProps {
  readonly offer: TrialOffer;
  readonly clock: TrialClock;
  readonly shown: TrialEntry;
  /** A recorded trial's condition, by its index, drawn after its entry. */
  readonly recordedLabels?: readonly string[];
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

/**
 * The one statement of which trial is shown. Its text is the shown entry; its list holds every entry that exists for
 * the selection, in groups, with the seed and the count the samples are drawn with at its foot.
 */
export function TrialPicker({ offer, clock, shown, recordedLabels, onChange }: TrialPickerProps) {
  const [open, setOpen] = useState(false);
  const text = (entry: TrialEntry) => entryText(entry, clock.count, offer.recorded);
  const entries = offeredEntries(offer, clock.count);
  const groups = (['nominal', 'sample', 'recorded'] as const)
    .map((kind) => entries.filter((entry) => entry.kind === kind)).filter((group) => group.length);
  return <Popover open={open} onOpenChange={setOpen}>
    <PopoverTrigger render={<Button variant="ghost" className="min-w-0 gap-1 px-1 tabular-nums" />}
      aria-label={`${studioName('trial')} ${text(shown)}`}>
      <span className="truncate">{text(shown)}</span>
      <ChevronDown aria-hidden="true" />
    </PopoverTrigger>
    <PopoverContent side="top" aria-label={studioName('trial')} className="grid w-56 gap-1 !p-1">
      <div role="radiogroup" aria-label={studioName('trial')} className="grid max-h-80 overflow-y-auto">
        {groups.map((group, index) => <div key={group[0].kind} role="presentation"
          className={clsx('grid', index > 0 && 'mt-1 border-t border-shell-divider pt-1')}>
          {group.map((entry) => {
            const checked = sameEntry(entry, shown);
            return <Button key={text(entry)} variant="bare" size="content" align="start" role="radio" aria-checked={checked}
              className={clsx(popupItem, 'w-full text-left tabular-nums hover:bg-shell-hover')}
              onClick={() => { onChange({ ...clock, picked: entry }); setOpen(false); }}>
              <span className="flex size-[1em] shrink-0 items-center justify-center">{checked && <Check aria-hidden="true" />}</span>
              <span className={primaryText}>{text(entry)}</span>
              {entry.kind === 'recorded' && recordedLabels?.[entry.index] &&
                <span className={secondaryText}>{recordedLabels[entry.index]}</span>}
            </Button>;
          })}
        </div>)}
      </div>
      {offer.samples && <div className="grid gap-1 border-t border-shell-divider px-1 pt-1">
        <WholeField label={studioName('seed')} value={clock.seed} least={Number.MIN_SAFE_INTEGER} most={Number.MAX_SAFE_INTEGER}
          onCommit={(seed) => onChange({ ...clock, seed })} />
        <WholeField label={studioName('count')} value={clock.count} least={1} most={maxSampleCount}
          onCommit={(count) => onChange({ ...clock, count })} />
      </div>}
    </PopoverContent>
  </Popover>;
}
