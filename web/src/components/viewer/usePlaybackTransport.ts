import { useEffect, useRef, useState } from 'react';

export const SPEED_OPTIONS = [0.25, 0.5, 1, 1.5, 2, 4] as const;

function boundedPosition(position: number, duration: number): number {
  if (!Number.isFinite(position)) return 0;
  return Math.max(0, Math.min(Math.max(duration, 0), position));
}

export function timelineOffset(value: number, duration: number): string {
  if (!Number.isFinite(value) || !Number.isFinite(duration) || duration <= 0) return '0%';
  return `${Math.max(0, Math.min(1, value / duration)) * 100}%`;
}

function nextFramePosition(frameTimes: readonly number[], position: number, direction: -1 | 1): number {
  if (frameTimes.length === 0) return 0;
  const sorted = [...frameTimes].sort((left, right) => left - right);
  if (direction > 0) {
    return sorted.find((time) => time > position + 1e-8) ?? sorted[sorted.length - 1];
  }
  return [...sorted].reverse().find((time) => time < position - 1e-8) ?? sorted[0];
}

export function usePlaybackTransport({
  position,
  duration,
  speed,
  frameTimes,
  onPositionChange,
  enabled = true,
}: {
  position: number;
  duration: number;
  speed: number;
  frameTimes: readonly number[];
  onPositionChange: (position: number) => void;
  enabled?: boolean;
}) {
  const [playing, setPlaying] = useState(false);
  const [looping, setLooping] = useState(true);
  const lastFrameRef = useRef<number | null>(null);
  const bounded = boundedPosition(position, duration);
  const hasTimeline = enabled && duration > 0 && frameTimes.length > 0;

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

  const seek = (time: number) => {
    setPlaying(false);
    onPositionChange(boundedPosition(time, duration));
  };
  const step = (direction: -1 | 1) => seek(nextFramePosition(frameTimes, bounded, direction));
  const togglePlay = () => setPlaying((current) => !current);
  const toggleLoop = () => setLooping((current) => !current);

  return { bounded, hasTimeline, playing, looping, seek, step, togglePlay, toggleLoop };
}
