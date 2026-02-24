import { useCallback, useEffect, useRef } from 'react';
import clsx from 'clsx';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import { useTrajectoryStore } from '@/stores/trajectoryStore';

const SPEED_OPTIONS = [0.25, 0.5, 1, 2];

export function PlaybackControls() {
  const playback = useTrajectoryStore((s) => s.playback);
  const togglePlay = useTrajectoryStore((s) => s.togglePlay);
  const setFrame = useTrajectoryStore((s) => s.setFrame);
  const setSpeed = useTrajectoryStore((s) => s.setSpeed);
  const stepForward = useTrajectoryStore((s) => s.stepForward);
  const stepBackward = useTrajectoryStore((s) => s.stepBackward);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);

  // Animation loop
  useEffect(() => {
    if (!playback.playing || playback.totalFrames === 0) {
      lastTimeRef.current = null;
      return;
    }

    const animate = (timestamp: number) => {
      if (lastTimeRef.current === null) {
        lastTimeRef.current = timestamp;
        rafRef.current = requestAnimationFrame(animate);
        return;
      }

      const dt = timestamp - lastTimeRef.current;
      lastTimeRef.current = timestamp;

      // Advance frame: speed * (dt / frame_duration_at_60fps)
      const frameDuration = 1000 / 60;
      const advance = playback.speed * (dt / frameDuration);

      const store = useTrajectoryStore.getState();
      const currentFrame = store.playback.frame;
      const total = store.playback.totalFrames;

      let nextFrame = currentFrame + advance;
      if (nextFrame >= total) {
        nextFrame = 0; // Wrap around
      }

      store.setFrame(nextFrame);
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [playback.playing, playback.totalFrames, playback.speed]);

  // Keyboard shortcuts
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Don't capture keyboard events when user is focused on an input
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA') {
        return;
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          togglePlay();
          break;
        case 'ArrowRight':
          e.preventDefault();
          stepForward();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          stepBackward();
          break;
        case 'ArrowUp':
          e.preventDefault();
          {
            const currentIdx = SPEED_OPTIONS.indexOf(playback.speed);
            if (currentIdx < SPEED_OPTIONS.length - 1) {
              setSpeed(SPEED_OPTIONS[currentIdx + 1]);
            }
          }
          break;
        case 'ArrowDown':
          e.preventDefault();
          {
            const currentIdx = SPEED_OPTIONS.indexOf(playback.speed);
            if (currentIdx > 0) {
              setSpeed(SPEED_OPTIONS[currentIdx - 1]);
            }
          }
          break;
      }
    },
    [togglePlay, stepForward, stepBackward, setSpeed, playback.speed],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const displayFrame = Math.floor(playback.frame);

  return (
    <div className="flex items-center gap-3 px-3 py-2 border-t border-slate-100 bg-slate-50/70">
      {/* Transport controls */}
      <div className="flex items-center gap-1">
        <button
          onClick={stepBackward}
          className="p-1 rounded text-slate-500 hover:text-slate-700 hover:bg-slate-100"
          title="Step backward (ArrowLeft)"
        >
          <SkipBack className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={togglePlay}
          className="p-1.5 rounded-full text-white bg-brand-500 hover:bg-brand-600"
          title={playback.playing ? 'Pause (Space)' : 'Play (Space)'}
        >
          {playback.playing ? (
            <Pause className="w-3.5 h-3.5" />
          ) : (
            <Play className="w-3.5 h-3.5" />
          )}
        </button>
        <button
          onClick={stepForward}
          className="p-1 rounded text-slate-500 hover:text-slate-700 hover:bg-slate-100"
          title="Step forward (ArrowRight)"
        >
          <SkipForward className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Speed selector */}
      <div className="flex items-center gap-0.5">
        {SPEED_OPTIONS.map((s) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            className={clsx(
              'text-[10px] font-semibold px-1.5 py-0.5 rounded',
              playback.speed === s
                ? 'bg-brand-500/10 text-brand-600 border border-brand-500'
                : 'text-slate-400 hover:text-slate-600 border border-transparent',
            )}
          >
            {s}x
          </button>
        ))}
      </div>

      {/* Timeline slider */}
      <input
        type="range"
        min={0}
        max={Math.max(0, playback.totalFrames - 1)}
        value={displayFrame}
        onChange={(e) => setFrame(Number(e.target.value))}
        className="flex-1 h-1.5 accent-brand-500 cursor-pointer"
      />

      {/* Frame counter */}
      <span className="text-xs text-slate-400 tabular-nums whitespace-nowrap min-w-[100px] text-right">
        Frame {displayFrame} / {playback.totalFrames}
      </span>
    </div>
  );
}
