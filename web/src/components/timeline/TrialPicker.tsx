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
  offeredPicks, pickText, samePick, sampleCount, type TrialClock, type TrialOffer, type TrialPick,
} from './trialClock';

export interface TrialPickerProps {
  readonly offer: TrialOffer;
  readonly clock: TrialClock;
  readonly shown: TrialPick;
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
 * The one statement of which trials are shown. Its text is what is shown; its list holds every single trial that exists
 * for the selection, in groups, with the seed and the count the samples are drawn with at its foot. Picking a trial
 * shows that trial alone.
 */
export function TrialPicker({ offer, clock, shown, recordedLabels, onChange }: TrialPickerProps) {
  const [open, setOpen] = useState(false);
  const count = sampleCount(clock, offer);
  const text = (pick: TrialPick) => pickText(pick, clock, offer);
  const shownText = text(shown);
  const picks = offeredPicks(offer, count);
  const groups = (['nominal', 'samples', 'recorded'] as const)
    .map((trials) => picks.filter((pick) => pick.trials === trials)).filter((group) => group.length);
  return <Popover open={open} onOpenChange={setOpen}>
    <PopoverTrigger render={<Button variant="ghost" className="min-w-0 gap-1 px-1 tabular-nums" />}
      aria-label={`${studioName('trial')} ${shownText}`}>
      <span className="truncate">{shownText}</span>
      <ChevronDown aria-hidden="true" />
    </PopoverTrigger>
    <PopoverContent side="top" aria-label={studioName('trial')} className="grid w-56 gap-1 !p-1">
      <div role="radiogroup" aria-label={studioName('trial')} className="grid max-h-80 overflow-y-auto">
        {groups.map((group, index) => <div key={group[0].trials} role="presentation"
          className={clsx('grid', index > 0 && 'mt-1 border-t border-shell-divider pt-1')}>
          {group.map((pick) => {
            const checked = samePick(pick, shown);
            const at = pick.showing.kind === 'one' ? pick.showing.index : undefined;
            return <Button key={text(pick)} variant="bare" size="content" align="start" role="radio" aria-checked={checked}
              className={clsx(popupItem, 'w-full text-left tabular-nums hover:bg-shell-hover')}
              onClick={() => { onChange({ ...clock, picked: pick }); setOpen(false); }}>
              <span className="flex size-[1em] shrink-0 items-center justify-center">{checked && <Check aria-hidden="true" />}</span>
              <span className={primaryText}>{text(pick)}</span>
              {pick.trials === 'recorded' && at !== undefined && recordedLabels?.[at] &&
                <span className={secondaryText}>{recordedLabels[at]}</span>}
            </Button>;
          })}
        </div>)}
      </div>
      {offer.samples > 0 && <div className="grid gap-1 border-t border-shell-divider px-1 pt-1">
        <WholeField label={studioName('seed')} value={clock.seed} least={Number.MIN_SAFE_INTEGER} most={Number.MAX_SAFE_INTEGER}
          onCommit={(seed) => onChange({ ...clock, seed })} />
        <WholeField label={studioName('count')} value={count} least={1} most={offer.samples}
          onCommit={(count) => onChange({ ...clock, count })} />
      </div>}
    </PopoverContent>
  </Popover>;
}
