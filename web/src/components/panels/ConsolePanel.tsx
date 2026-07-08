// Console tab: displays streamed training log lines
// Shows scrolling log output with level-based coloring
import { useEffect, useRef } from 'react';
import { formatTrainingDiagnostic } from '@/hooks/useTraining';
import { useTrainingStore } from '@/stores/trainingStore';
import type { TrainingLogLine } from '@/types/training';
import type { DomainDiagnostic } from '@/generated/studioContracts';

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

function diagnosticLevelClass(severity: DomainDiagnostic['severity']): string {
  if (severity === 'warning') return levelClass('warning');
  if (severity === 'info') return levelClass('info');
  return levelClass('error');
}

export function ConsolePanel() {
  const consoleLogs = useTrainingStore((state) => state.consoleLogs);
  const diagnostics = useTrainingStore((state) => state.trainingDiagnostics);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [consoleLogs, diagnostics]);

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
        {diagnostics.map((diagnostic, idx) => (
          <div
            key={`${diagnostic.code}-${idx}`}
            className={`leading-5 ${diagnosticLevelClass(diagnostic.severity)}`}
          >
            <span className="text-slate-500 mr-2">[diagnostic]</span>
            {formatTrainingDiagnostic(diagnostic)}
          </div>
        ))}
      </div>
    </div>
  );
}
