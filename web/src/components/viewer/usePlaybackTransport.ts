import { useEffect, useRef, useState } from 'react';

export const SPEED_OPTIONS = [0.25, 0.5, 1, 1.5, 2, 4] as const;

function boundedPosition(position: number, start: number, end: number): number {
  if (!Number.isFinite(position)) return start;
  return Math.max(start, Math.min(Math.max(end, start), position));
}

function nextFramePosition(frameTimes: readonly number[], position: number, direction: -1 | 1, start: number): number {
  if (frameTimes.length === 0) return start;
  const sorted = [...frameTimes].sort((left, right) => left - right);
  if (direction > 0) {
    return sorted.find((time) => time > position + 1e-8) ?? sorted[sorted.length - 1];
  }
  return [...sorted].reverse().find((time) => time < position - 1e-8) ?? sorted[0];
}

/** Playback of a span of time from `start` to `end`, in seconds. */
export function usePlaybackTransport({
  position,
  start,
  end,
  speed,
  frameTimes,
  onPositionChange,
  enabled = true,
}: {
  position: number;
  start: number;
  end: number;
  speed: number;
  frameTimes: readonly number[];
  onPositionChange: (position: number) => void;
  enabled?: boolean;
}) {
  const [playing, setPlaying] = useState(false);
  const [looping, setLooping] = useState(true);
  const lastFrameRef = useRef<number | null>(null);
  const bounded = boundedPosition(position, start, end);
  const hasTimeline = enabled && end > start && frameTimes.length > 0;

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
      if (next >= end) {
        if (looping) {
          onPositionChange(start);
        } else {
          onPositionChange(end);
          setPlaying(false);
        }
      } else {
        onPositionChange(next);
      }
      rafId = window.requestAnimationFrame(tick);
    };
    rafId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(rafId);
  }, [bounded, start, end, hasTimeline, looping, onPositionChange, playing, speed]);

  useEffect(() => {
    if (!hasTimeline) setPlaying(false);
  }, [hasTimeline]);

  const seek = (time: number) => {
    setPlaying(false);
    onPositionChange(boundedPosition(time, start, end));
  };
  const step = (direction: -1 | 1) => seek(nextFramePosition(frameTimes, bounded, direction, start));
  const togglePlay = () => setPlaying((current) => !current);
  const toggleLoop = () => setLooping((current) => !current);

  return { bounded, hasTimeline, playing, looping, seek, step, togglePlay, toggleLoop };
}
