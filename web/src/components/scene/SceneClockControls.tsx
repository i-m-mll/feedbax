import { Pause, Play, Repeat, RotateCcw, StepBack, StepForward, type LucideIcon } from 'lucide-react';
import { studioName, studioUnit, type StudioNameId } from '@/components/activity/studioVocabulary';
import { SPEED_OPTIONS, timelineOffset, usePlaybackTransport } from '@/components/viewer/usePlaybackTransport';
import type { SceneTimeline } from './sceneView';

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
    <button type="button" aria-label={studioName(name)} aria-pressed={pressed} onClick={onClick}
      className="rounded p-1.5 text-slate-700 hover:bg-slate-100 aria-pressed:bg-violet-100">
      <Icon className="size-4" aria-hidden="true" />
    </button>
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
          <div className="relative h-1.5 overflow-hidden rounded-sm bg-slate-100" aria-hidden="true">
            {bands.map((band) => (
              <span key={band.id} className={band.kind === 'loss_window' ? 'absolute h-full bg-violet-300' : 'absolute h-full bg-teal-300'}
                style={{ left: timelineOffset(band.start, duration), right: timelineOffset(duration - band.end, duration) }} />
            ))}
            {eventTicks.map((tick) => (
              <span key={tick.id} className="absolute h-full w-px bg-slate-800" style={{ left: timelineOffset(tick.time, duration) }} />
            ))}
          </div>
        )}
        <input type="range" min={0} max={duration} step="any" value={transport.bounded}
          aria-label={studioName('playbackPosition')} onChange={(event) => transport.seek(event.currentTarget.valueAsNumber)}
          className="block w-full accent-violet-700" />
      </div>
      <output className="whitespace-nowrap text-xs tabular-nums text-slate-700">
        {transport.bounded.toFixed(2)} / {duration.toFixed(2)} {units ?? studioUnit('second')}
      </output>
      <select value={speed} onChange={(event) => onSpeedChange(Number(event.currentTarget.value))}
        aria-label={studioName('playbackSpeed')} className="rounded border border-slate-300 bg-white px-1 py-0.5 text-xs">
        {SPEED_OPTIONS.map((option) => <option key={option} value={option}>{option}×</option>)}
      </select>
    </div>
  );
}

export function SceneClockControls({ timeline, ...props }: Omit<SceneClockProps, 'timeline'> & { timeline: SceneTimeline | null }) {
  return timeline ? <SceneClock key={timeline.id} timeline={timeline} {...props} /> : null;
}
