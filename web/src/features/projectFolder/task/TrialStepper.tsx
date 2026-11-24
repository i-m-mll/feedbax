import { ChevronLeft, ChevronRight, Dices, RefreshCw } from 'lucide-react';
import { studioFormat, studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';

/** Which sampled trial the view shows: none (the definition), the batch as one aggregate, or one trial by its index. */
export type TrialChoice = 'definition' | 'all' | number;

/**
 * The trial stepper at the strip's trailing end: Sample shows sampled trials; All, before trial 1, shows the batch at
 * once; Resample draws a new set under a new seed.
 */
export function TrialStepper({ choice, count, onChoose, onResample }: {
  choice: TrialChoice; count: number; onChoose(choice: TrialChoice): void; onResample(): void;
}) {
  const sampling = choice !== 'definition';
  const position = choice === 'all' ? 0 : choice === 'definition' ? -1 : choice + 1;
  const to = (next: number) => onChoose(next === 0 ? 'all' : next - 1);
  return <div role="group" aria-label={studioName('trial')} className="flex items-center gap-0.5">
    <IconButton variant="ghost" label={studioName('sample')} icon={<Dices aria-hidden="true" />} pressed={sampling}
      onClick={() => onChoose(sampling ? 'definition' : 0)} />
    {sampling && <>
      <IconButton variant="ghost" label={studioName('previousTrial')} icon={<ChevronLeft aria-hidden="true" />}
        disabled={position === 0} onClick={() => to(position - 1)} />
      <output className="min-w-[4.5rem] text-center text-xs tabular-nums" aria-live="polite">
        {choice === 'all' ? studioName('allTrials') : studioFormat('ofCount', { count: position, total: count })}
      </output>
      <IconButton variant="ghost" label={studioName('nextTrial')} icon={<ChevronRight aria-hidden="true" />}
        disabled={position === count} onClick={() => to(position + 1)} />
      <IconButton variant="ghost" label={studioName('resample')} icon={<RefreshCw aria-hidden="true" />} onClick={onResample} />
    </>}
  </div>;
}
