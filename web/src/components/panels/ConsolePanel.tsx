// Console tab: displays streamed training log lines
// Shows scrolling log output with level-based coloring
import { useEffect, useRef } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import type { TrainingLogLine } from '@/types/training';

function levelClass(level: TrainingLogLine['level']): string {
  switch (level) {
    case 'warning':
      return 'text-amber-400';
    case 'error':
      return 'text-red-400';
    default:
      return 'text-slate-300';
  }
}

export function ConsolePanel() {
  const consoleLogs = useTrainingStore((state) => state.consoleLogs);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [consoleLogs]);

  return (
    <div className="h-full flex flex-col bg-slate-900 text-xs font-mono">
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto px-4 py-3 space-y-0.5"
      >
        {consoleLogs.length === 0 ? (
          <div className="text-slate-500 italic">No logs yet</div>
        ) : (
          consoleLogs.map((line, idx) => (
            <div key={idx} className={`leading-5 ${levelClass(line.level)}`}>
              <span className="text-slate-500 mr-2">[{line.batch}]</span>
              {line.message}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
