import { PanelBottomOpen } from 'lucide-react';
import type { ReactNode } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { TrialPicker, type TrialPickerProps } from './TrialPicker';

/**
 * The bottom bar's clock: the trial picker, the transport and scrubber when the shown trial can be played, and the
 * shelf's switch when there are rows to show.
 */
export function ClockGroup({ transport, shelf, ...picker }: TrialPickerProps & {
  readonly transport?: ReactNode;
  /** Whether the shelf has rows for the shown trial. */
  readonly shelf: boolean;
}) {
  return <div role="group" aria-label={studioName('clock')} className="flex min-w-0 items-center gap-2">
    <TrialPicker {...picker} />
    {transport}
    {shelf && <IconButton label={studioName('shelf')} icon={<PanelBottomOpen aria-hidden="true" />} pressed={picker.clock.shelf}
      className="studio-window-control" onClick={() => picker.onChange({ ...picker.clock, shelf: !picker.clock.shelf })} />}
  </div>;
}
