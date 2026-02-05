import { CheckCircle2, Circle, AlertTriangle } from 'lucide-react';
import { useMemo } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import { validateGraph } from '@/features/graph/validation';
import { useTrainingStore } from '@/stores/trainingStore';

export function StatusBar() {
  const graph = useGraphStore(state => state.graph);
  const validation = useMemo(() => validateGraph(graph), [graph]);
  const status = useTrainingStore((state) => state.status);

  return (
    <footer className="h-6 px-4 border-t border-slate-100 bg-white/80 text-xs text-slate-500 flex items-center gap-4">
      <div className="flex items-center gap-1">
        <Circle className="w-2.5 h-2.5 text-mint-500" fill="currentColor" />
        Connected
      </div>
      <div className="flex items-center gap-1">
        {validation.valid ? (
          <CheckCircle2 className="w-3 h-3 text-mint-500" />
        ) : (
          <AlertTriangle className="w-3 h-3 text-amber-500" />
        )}
        {validation.valid ? 'Valid graph' : `${validation.errors.length} issue(s)`}
      </div>
      <div className="ml-auto capitalize">{status}</div>
    </footer>
  );
}
