import { Pause, Play, Repeat, RotateCcw, StepBack, StepForward, type LucideIcon } from 'lucide-react';
import { studioName, studioUnit, type StudioNameId } from '@/components/activity/studioVocabulary';
import { SPEED_OPTIONS, timelineOffset, usePlaybackTransport } from '@/components/viewer/usePlaybackTransport';
import { IconButton } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Slider } from '@/components/ui/Slider';
import type { SceneTimeline } from './sceneView';

const speedItems = SPEED_OPTIONS.map((option) => ({ value: String(option), label: `${option}×` }));

interface SceneClockProps {
  timeline: SceneTimeline;
  time: number;
  speed: number;
  onTimeChange: (time: number) => void;
  onSpeedChange: (speed: number) => void;
}

function ClockButton({ name, icon: Icon, onClick, pressed }: {
  name: StudioNameId;
  icon: LucideIcon;
  onClick: () => void;
  pressed?: boolean;
}) {
  return (
    <IconButton label={studioName(name)} icon={<Icon aria-hidden="true" />} pressed={pressed} onClick={onClick} />
  );
}

function SceneClock({ timeline, time, speed, onTimeChange, onSpeedChange }: SceneClockProps) {
  const { duration, frameTimes, units, bands, eventTicks } = timeline;
  const transport = usePlaybackTransport({ position: time, duration, speed, frameTimes, onPositionChange: onTimeChange });
  return (
    <div role="group" aria-label={studioName('clock')} className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
      <div className="flex items-center">
        <ClockButton name="restart" icon={RotateCcw} onClick={() => transport.seek(0)} />
        <ClockButton name="previousFrame" icon={StepBack} onClick={() => transport.step(-1)} />
        <ClockButton name={transport.playing ? 'pause' : 'play'} icon={transport.playing ? Pause : Play} onClick={transport.togglePlay} />
        <ClockButton name="nextFrame" icon={StepForward} onClick={() => transport.step(1)} />
        <ClockButton name="loop" icon={Repeat} onClick={transport.toggleLoop} pressed={transport.looping} />
      </div>
      <div className="min-w-16 grow">
        {(bands.length > 0 || eventTicks.length > 0) && (
          <div className="relative h-1.5 overflow-hidden rounded-sm bg-shell-subtle" aria-hidden="true">
            {bands.map((band) => (
              <span key={band.id} className={band.kind === 'loss_window' ? 'absolute h-full bg-shell-scene-guide' : 'absolute h-full bg-shell-scene-feature'}
                style={{ left: timelineOffset(band.start, duration), right: timelineOffset(duration - band.end, duration) }} />
            ))}
            {eventTicks.map((tick) => (
              <span key={tick.id} className="absolute h-full w-px bg-shell-foreground" style={{ left: timelineOffset(tick.time, duration) }} />
            ))}
          </div>
        )}
        <Slider min={0} max={duration} step={duration / 1000 || 1} value={transport.bounded}
          aria-label={studioName('playbackPosition')} onValueChange={transport.seek} />
      </div>
      <output className="whitespace-nowrap text-xs tabular-nums text-shell-muted">
        {transport.bounded.toFixed(2)} / {duration.toFixed(2)} {units ?? studioUnit('second')}
      </output>
      <Select items={speedItems} value={String(speed)} onValueChange={(value) => onSpeedChange(Number(value))}
        aria-label={studioName('playbackSpeed')} />
    </div>
  );
}

export function SceneClockControls({ timeline, ...props }: Omit<SceneClockProps, 'timeline'> & { timeline: SceneTimeline | null }) {
  return timeline ? <SceneClock key={timeline.id} timeline={timeline} {...props} /> : null;
}
