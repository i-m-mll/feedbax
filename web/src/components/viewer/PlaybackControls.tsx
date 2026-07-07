import { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import {
  Pause,
  Play,
  Repeat,
  RotateCcw,
  StepBack,
  StepForward,
} from 'lucide-react';
import type {
  WorkspaceReplayEventTick,
  WorkspaceReplayTimelineBand,
} from '@/features/scenario/workspaceReplay';

const SPEED_OPTIONS = [0.25, 0.5, 1, 1.5, 2, 4] as const;

function boundedPosition(position: number, duration: number): number {
  if (!Number.isFinite(position)) return 0;
  return Math.max(0, Math.min(Math.max(duration, 0), position));
}

function timelineOffset(value: number, duration: number): string {
  if (!Number.isFinite(value) || !Number.isFinite(duration) || duration <= 0) return '0%';
  return `${Math.max(0, Math.min(1, value / duration)) * 100}%`;
}

function nextFramePosition(frameTimes: number[], position: number, direction: -1 | 1): number {
  if (frameTimes.length === 0) return 0;
  const sorted = [...frameTimes].sort((left, right) => left - right);
  if (direction > 0) {
    return sorted.find((time) => time > position + 1e-8) ?? sorted[sorted.length - 1];
  }
  return [...sorted].reverse().find((time) => time < position - 1e-8) ?? sorted[0];
}

function shouldIgnoreShortcut(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(target.tagName);
}

export function PlaybackControls({
  position,
  duration,
  speed,
  frameTimes,
  bands,
  eventTicks,
  cursorLabel,
  disabled = false,
  onPositionChange,
  onSpeedChange,
}: {
  position: number;
  duration: number;
  speed: number;
  frameTimes: number[];
  bands: WorkspaceReplayTimelineBand[];
  eventTicks: WorkspaceReplayEventTick[];
  cursorLabel: string;
  disabled?: boolean;
  onPositionChange: (position: number) => void;
  onSpeedChange: (speed: number) => void;
}) {
  const [playing, setPlaying] = useState(false);
  const [looping, setLooping] = useState(true);
  const lastFrameRef = useRef<number | null>(null);
  const bounded = boundedPosition(position, duration);
  const hasTimeline = !disabled && duration > 0 && frameTimes.length > 0;

  useEffect(() => {
    if (!playing || !hasTimeline) {
      lastFrameRef.current = null;
      return;
    }
    let rafId = 0;
    const tick = (time: number) => {
      const last = lastFrameRef.current ?? time;
      lastFrameRef.current = time;
      const deltaSeconds = Math.max(0, (time - last) / 1000);
      const next = bounded + deltaSeconds * speed;
      if (next >= duration) {
        if (looping) {
          onPositionChange(0);
        } else {
          onPositionChange(duration);
          setPlaying(false);
        }
      } else {
        onPositionChange(next);
      }
      rafId = window.requestAnimationFrame(tick);
    };
    rafId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(rafId);
  }, [bounded, duration, hasTimeline, looping, onPositionChange, playing, speed]);

  useEffect(() => {
    if (!hasTimeline) setPlaying(false);
  }, [hasTimeline]);

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (disabled || shouldIgnoreShortcut(event.target)) return;
      if (event.key === ' ') {
        event.preventDefault();
        setPlaying((current) => !current);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        setPlaying(false);
        onPositionChange(nextFramePosition(frameTimes, bounded, 1));
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        setPlaying(false);
        onPositionChange(nextFramePosition(frameTimes, bounded, -1));
      } else if (event.key === 'Home') {
        event.preventDefault();
        setPlaying(false);
        onPositionChange(0);
      } else if (event.key === 'End') {
        event.preventDefault();
        setPlaying(false);
        onPositionChange(duration);
      } else if (event.key.toLowerCase() === 'l') {
        event.preventDefault();
        setLooping((current) => !current);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [bounded, disabled, duration, frameTimes, onPositionChange]);

  const step = (direction: -1 | 1) => {
    setPlaying(false);
    onPositionChange(nextFramePosition(frameTimes, bounded, direction));
  };

  const timeText = `${bounded.toFixed(2)} / ${Math.max(duration, 0).toFixed(2)} s`;

  return (
    <div className="grid h-28 grid-rows-[2rem_1fr_2rem] gap-2 rounded-md border border-slate-200 bg-white/95 p-2 shadow-sm backdrop-blur">
      <div className="flex min-w-0 items-center justify-between gap-2">
        <div className="min-w-0 truncate text-xs font-semibold text-slate-800">
          {cursorLabel}
        </div>
        <div className="shrink-0 tabular-nums text-[11px] font-medium text-slate-500">
          {timeText}
        </div>
      </div>
      <div className="relative h-9">
        <div className="absolute inset-x-0 top-1 h-3 overflow-hidden rounded-full bg-slate-100">
          {bands.map((band, index) => (
            <span
              key={band.id}
              className={clsx(
                'absolute top-0 h-3',
                band.kind === 'loss_window'
                  ? 'bg-violet-500/35'
                  : index % 2 === 0
                    ? 'bg-teal-500/20'
                    : 'bg-sky-500/20'
              )}
              style={{
                left: timelineOffset(band.start, duration),
                width: `calc(${timelineOffset(band.end, duration)} - ${timelineOffset(band.start, duration)})`,
              }}
              title={`${band.label}: ${band.start.toFixed(2)}-${band.end.toFixed(2)} s`}
            />
          ))}
          {eventTicks.map((tick) => (
            <span
              key={tick.id}
              className="absolute top-0 h-3 w-1 rounded-full bg-amber-500"
              style={{ left: timelineOffset(tick.time, duration) }}
              title={`${tick.label}: ${tick.time.toFixed(2)} s`}
            />
          ))}
        </div>
        <div
          className="pointer-events-none absolute bottom-0 top-0 w-px bg-slate-900"
          style={{ left: timelineOffset(bounded, duration) }}
        />
        <input
          type="range"
          min={0}
          max={Math.max(duration, 0)}
          step="any"
          value={bounded}
          disabled={!hasTimeline}
          onChange={(event) => {
            setPlaying(false);
            onPositionChange(Number.parseFloat(event.target.value));
          }}
          className="absolute inset-x-0 bottom-0 h-5 accent-slate-900 disabled:opacity-40"
          aria-label="Playback position"
          title="Playback position"
        />
      </div>
      <div className="flex min-w-0 items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => onPositionChange(0)}
            disabled={!hasTimeline}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300"
            title="Restart"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => step(-1)}
            disabled={!hasTimeline}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300"
            title="Previous frame"
          >
            <StepBack className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setPlaying((current) => !current)}
            disabled={!hasTimeline}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md bg-slate-900 text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            title={playing ? 'Pause' : 'Play'}
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </button>
          <button
            type="button"
            onClick={() => step(1)}
            disabled={!hasTimeline}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300"
            title="Next frame"
          >
            <StepForward className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setLooping((current) => !current)}
            disabled={!hasTimeline}
            className={clsx(
              'inline-flex h-8 w-8 items-center justify-center rounded-md disabled:cursor-not-allowed disabled:text-slate-300',
              looping
                ? 'bg-teal-50 text-teal-700 hover:bg-teal-100'
                : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
            )}
            title={looping ? 'Loop enabled' : 'Loop disabled'}
          >
            <Repeat className="h-4 w-4" />
          </button>
        </div>
        <select
          value={speed}
          onChange={(event) => onSpeedChange(Number.parseFloat(event.target.value))}
          disabled={disabled}
          className="h-8 rounded-md border border-slate-200 bg-white px-2 text-xs font-medium text-slate-700 disabled:opacity-40"
          aria-label="Playback speed"
          title="Playback speed"
        >
          {SPEED_OPTIONS.map((option) => (
            <option key={option} value={option}>
              {option}x
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
