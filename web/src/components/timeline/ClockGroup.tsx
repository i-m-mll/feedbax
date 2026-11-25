import { PanelBottomOpen, Pause, Play, Repeat, RotateCcw, StepBack, StepForward, type LucideIcon } from 'lucide-react';
import { useRef, useState, type ReactNode } from 'react';
import { studioName, studioUnit, type StudioNameId } from '@/components/activity/studioVocabulary';
import { SPEED_OPTIONS, usePlaybackTransport } from '@/components/viewer/usePlaybackTransport';
import { IconButton } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Slider } from '@/components/ui/Slider';
import { usePublishedAxis, type TimeAxisStore } from './timeAxis';
import { TrialPicker, type TrialPickerProps } from './TrialPicker';

/**
 * The shown time, in seconds: where it starts (0, the trial's start, or under Mean the earliest time from Align's
 * epoch) and its length, its frame times, and its epochs' spans, drawn as bands on the scrubber.
 */
export interface ClockTimeline {
  readonly start: number;
  readonly duration: number;
  readonly frameTimes: readonly number[];
  readonly bands: readonly { readonly start: number; readonly end: number }[];
}

const speedItems = SPEED_OPTIONS.map((option) => ({ value: String(option), label: `${option}×` }));

function ClockButton({ name, icon: Icon, onClick, pressed, disabled }: {
  name: StudioNameId; icon: LucideIcon; onClick: () => void; pressed?: boolean; disabled?: boolean;
}) {
  return <IconButton label={studioName(name)} icon={<Icon aria-hidden="true" />} pressed={pressed} disabled={disabled} onClick={onClick} />;
}

const percent = (value: number, total: number) => `${(100 * value / total).toFixed(3)}%`;
const seconds = (value: number) => `${value.toFixed(2)} ${studioUnit('second')}`;

/**
 * The bottom bar's clock, on two lines. The upper holds the scrubber, with the shown trial's epochs as bands, and the
 * readout; the scrubber's track is the extent the shelf's time axis spans. The lower holds the trial picker and how
 * its trials are shown (One, All, Mean, with Mean's By and Align), the transport, Loop and Speed, and the shelf's switch.
 */
export function ClockGroup({ timeline, time, onTimeChange, axis, shelfOpen, onShelfChange, showing, ...picker }: TrialPickerProps & {
  readonly showing?: ReactNode;
  readonly timeline: ClockTimeline;
  readonly time: number;
  readonly onTimeChange: (time: number) => void;
  readonly axis: TimeAxisStore;
  readonly shelfOpen: boolean;
  readonly onShelfChange: (open: boolean) => void;
}) {
  const { start, duration, frameTimes, bands } = timeline;
  const [speed, setSpeed] = useState(1);
  const transport = usePlaybackTransport({
    position: time, start, end: start + duration, speed, frameTimes, onPositionChange: onTimeChange,
  });
  const track = useRef<HTMLDivElement>(null);
  usePublishedAxis(axis, track);
  const still = !transport.hasTimeline;
  return <div role="group" aria-label={studioName('clock')} className="studio-clock">
    <div className="studio-clock__line">
      <div ref={track} className="relative min-w-0" data-clock-track="">
        {duration > 0 && bands.map((band, index) => <span key={`${band.start}:${band.end}`} aria-hidden="true"
          className={`pointer-events-none absolute inset-y-0.5 rounded-sm bg-shell-foreground ${index % 2 ? 'opacity-[0.04]' : 'opacity-[0.09]'}`}
          style={{ left: percent(band.start - start, duration), width: percent(band.end - band.start, duration) }} />)}
        <Slider size="line" min={start} max={start + duration} step={duration / 1000 || 1} value={transport.bounded} disabled={still}
          aria-label={studioName('playbackPosition')} onValueChange={transport.seek} />
      </div>
      <span className="w-[17ch] whitespace-nowrap pl-2 text-right text-xs tabular-nums text-shell-muted">
        {seconds(transport.bounded)} / {seconds(start + duration)}
      </span>
    </div>
    <div className="studio-clock__controls">
      <TrialPicker {...picker} />
      {showing}
      <div className="flex shrink-0 items-center">
        <ClockButton name="restart" icon={RotateCcw} disabled={still} onClick={() => transport.seek(start)} />
        <ClockButton name="previousFrame" icon={StepBack} disabled={still} onClick={() => transport.step(-1)} />
        <ClockButton name={transport.playing ? 'pause' : 'play'} icon={transport.playing ? Pause : Play} disabled={still}
          onClick={transport.togglePlay} />
        <ClockButton name="nextFrame" icon={StepForward} disabled={still} onClick={() => transport.step(1)} />
      </div>
      <div className="flex shrink-0 items-center gap-1">
        <ClockButton name="loop" icon={Repeat} onClick={transport.toggleLoop} pressed={transport.looping} />
        <Select items={speedItems} value={String(speed)} onValueChange={(value) => setSpeed(Number(value))}
          aria-label={studioName('playbackSpeed')} className="w-20" />
        <IconButton label={studioName('shelf')} icon={<PanelBottomOpen aria-hidden="true" />} pressed={shelfOpen}
          className="studio-window-control" onClick={() => onShelfChange(!shelfOpen)} />
      </div>
    </div>
  </div>;
}
