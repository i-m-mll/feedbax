import { PanelBottomOpen, Pause, Play, Repeat, RotateCcw, StepBack, StepForward, type LucideIcon } from 'lucide-react';
import { useRef, useState } from 'react';
import { studioName, studioUnit, type StudioNameId } from '@/components/activity/studioVocabulary';
import { SPEED_OPTIONS, usePlaybackTransport } from '@/components/viewer/usePlaybackTransport';
import { IconButton } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Slider } from '@/components/ui/Slider';
import { usePublishedAxis, type TimeAxisStore } from './timeAxis';
import { TrialPicker, type TrialPickerProps } from './TrialPicker';

/** The shown trial's time: its length and frame times in seconds, and its epochs' spans, drawn as bands on the scrubber. */
export interface ClockTimeline {
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
 * The bottom bar's clock, in order: the trial picker, the transport, the scrubber with the shown trial's epochs as
 * bands, the readout, Loop and Speed, and the shelf's switch. The scrubber's track is the extent the shelf's time axis
 * spans.
 */
export function ClockGroup({ timeline, time, onTimeChange, axis, shelfOpen, onShelfChange, ...picker }: TrialPickerProps & {
  readonly timeline: ClockTimeline;
  readonly time: number;
  readonly onTimeChange: (time: number) => void;
  readonly axis: TimeAxisStore;
  readonly shelfOpen: boolean;
  readonly onShelfChange: (open: boolean) => void;
}) {
  const { duration, frameTimes, bands } = timeline;
  const [speed, setSpeed] = useState(1);
  const transport = usePlaybackTransport({ position: time, duration, speed, frameTimes, onPositionChange: onTimeChange });
  const track = useRef<HTMLDivElement>(null);
  usePublishedAxis(axis, track);
  const still = !transport.hasTimeline;
  return <div role="group" aria-label={studioName('clock')} className="flex min-w-0 items-center gap-2">
    <TrialPicker {...picker} />
    <div className="flex shrink-0 items-center">
      <ClockButton name="restart" icon={RotateCcw} disabled={still} onClick={() => transport.seek(0)} />
      <ClockButton name="previousFrame" icon={StepBack} disabled={still} onClick={() => transport.step(-1)} />
      <ClockButton name={transport.playing ? 'pause' : 'play'} icon={transport.playing ? Pause : Play} disabled={still}
        onClick={transport.togglePlay} />
      <ClockButton name="nextFrame" icon={StepForward} disabled={still} onClick={() => transport.step(1)} />
    </div>
    <div ref={track} className="relative min-w-16 grow" data-clock-track="">
      {duration > 0 && bands.map(({ start, end }, index) => <span key={`${start}:${end}`} aria-hidden="true"
        className={`pointer-events-none absolute inset-y-1.5 rounded-sm bg-shell-foreground ${index % 2 ? 'opacity-[0.04]' : 'opacity-[0.09]'}`}
        style={{ left: percent(start, duration), width: percent(end - start, duration) }} />)}
      <Slider min={0} max={duration} step={duration / 1000 || 1} value={transport.bounded} disabled={still}
        aria-label={studioName('playbackPosition')} onValueChange={transport.seek} />
    </div>
    <output className="shrink-0 whitespace-nowrap text-xs tabular-nums text-shell-muted">
      {seconds(transport.bounded)} / {seconds(duration)}
    </output>
    <div className="flex shrink-0 items-center gap-1">
      <ClockButton name="loop" icon={Repeat} onClick={transport.toggleLoop} pressed={transport.looping} />
      <Select items={speedItems} value={String(speed)} onValueChange={(value) => setSpeed(Number(value))}
        aria-label={studioName('playbackSpeed')} />
      <IconButton label={studioName('shelf')} icon={<PanelBottomOpen aria-hidden="true" />} pressed={shelfOpen}
        className="studio-window-control" onClick={() => onShelfChange(!shelfOpen)} />
    </div>
  </div>;
}
