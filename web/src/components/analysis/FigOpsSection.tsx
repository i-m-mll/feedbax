import { useCallback, useRef, useState } from 'react';
import type { FigOpSpec, FinalOpSpec, AnalysisNodeMeta } from '@/types/analysis';
import { GripVertical, ChevronDown, ChevronRight, Settings2, Layers, Combine } from 'lucide-react';
import clsx from 'clsx';

/** Human-readable label for a fig op type. */
function figOpLabel(name: string): string {
  const labels: Record<string, string> = {
    map_figs_at_level: 'Map at level',
    combine_figs_by_level: 'Combine by level',
    combine_figs_by_axis: 'Combine by axis',
    then_transform_figs: 'Transform figs',
    then_transform_result: 'Transform result',
  };
  return labels[name] ?? name;
}

/** Icon for a fig op type. */
function FigOpIcon({ name }: { name: string }) {
  if (name.startsWith('map_figs')) return <Layers className="w-3 h-3 text-brand-500" />;
  if (name.startsWith('combine_figs')) return <Combine className="w-3 h-3 text-indigo-500" />;
  return <Settings2 className="w-3 h-3 text-slate-400" />;
}

/** Renders a single fig op item in the ordered list. */
function FigOpItem({
  op,
  index,
  isDragging,
  isDragOver,
  onDragStart,
  onDragOver,
  onDragEnd,
}: {
  op: FigOpSpec;
  index: number;
  isDragging: boolean;
  isDragOver: boolean;
  onDragStart: (index: number) => void;
  onDragOver: (index: number) => void;
  onDragEnd: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const paramEntries = Object.entries(op.params).filter(
    ([, v]) => v !== null && v !== undefined
  );

  return (
    <div
      className={clsx(
        'rounded-lg border bg-white transition-all duration-150',
        isDragging ? 'border-brand-300 opacity-50 scale-95' : 'border-slate-200',
        isDragOver && !isDragging ? 'border-brand-400 shadow-sm' : '',
      )}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.effectAllowed = 'move';
        onDragStart(index);
      }}
      onDragOver={(e) => {
        e.preventDefault();
        onDragOver(index);
      }}
      onDragEnd={onDragEnd}
    >
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs"
        onClick={() => setExpanded(!expanded)}
      >
        <GripVertical className="w-3 h-3 text-slate-300 cursor-grab shrink-0" />
        <span className="text-slate-400 font-mono w-4 text-right shrink-0">{index + 1}</span>
        <FigOpIcon name={op.name} />
        <span className="font-medium text-slate-700 truncate">{figOpLabel(op.name)}</span>
        {paramEntries.length > 0 && (
          <span className="ml-auto text-slate-400">
            {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </span>
        )}
      </button>
      {expanded && paramEntries.length > 0 && (
        <div className="px-3 pb-2 space-y-1 border-t border-slate-100 pt-2">
          {paramEntries.map(([key, value]) => (
            <div key={key} className="flex items-center gap-2 text-xs">
              <span className="text-slate-400 min-w-[80px]">{key}</span>
              <span className="text-slate-600 font-mono truncate">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Renders a final op item. */
function FinalOpItem({ op }: { op: FinalOpSpec }) {
  const [expanded, setExpanded] = useState(false);
  const paramEntries = Object.entries(op.params ?? {});

  return (
    <div className="rounded-lg border border-slate-150 bg-slate-50/50">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs"
        onClick={() => setExpanded(!expanded)}
      >
        <Settings2 className="w-3 h-3 text-slate-400 shrink-0" />
        <span className="text-slate-600 truncate">{op.fn_name}</span>
        {paramEntries.length > 0 && (
          <span className="ml-auto text-slate-400">
            {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </span>
        )}
      </button>
      {expanded && paramEntries.length > 0 && (
        <div className="px-3 pb-2 space-y-1 border-t border-slate-100 pt-1">
          {paramEntries.map(([key, value]) => (
            <div key={key} className="flex items-center gap-2 text-xs">
              <span className="text-slate-400 min-w-[60px]">{key}</span>
              <span className="text-slate-500 font-mono truncate">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Section showing fig ops and final ops for an analysis node in the properties panel. */
export function FigOpsSection({
  meta,
  onReorderFigOps,
}: {
  meta: AnalysisNodeMeta;
  onReorderFigOps?: (reordered: FigOpSpec[]) => void;
}) {
  const [orderedOps, setOrderedOps] = useState<FigOpSpec[]>(meta.fig_ops);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  // Sync with incoming meta when it changes
  const prevOpsRef = useRef(meta.fig_ops);
  if (prevOpsRef.current !== meta.fig_ops) {
    prevOpsRef.current = meta.fig_ops;
    setOrderedOps(meta.fig_ops);
  }

  const handleDragStart = useCallback((index: number) => {
    setDragIndex(index);
  }, []);

  const handleDragOver = useCallback((index: number) => {
    setDragOverIndex(index);
  }, []);

  const handleDragEnd = useCallback(() => {
    if (dragIndex !== null && dragOverIndex !== null && dragIndex !== dragOverIndex) {
      const reordered = [...orderedOps];
      const [moved] = reordered.splice(dragIndex, 1);
      reordered.splice(dragOverIndex, 0, moved);
      setOrderedOps(reordered);
      onReorderFigOps?.(reordered);
    }
    setDragIndex(null);
    setDragOverIndex(null);
  }, [dragIndex, dragOverIndex, orderedOps, onReorderFigOps]);

  const finalOps = meta.final_ops_by_type;
  const hasFigOps = orderedOps.length > 0;
  const hasFinalOps = Object.values(finalOps).some((ops) => ops.length > 0);

  if (!hasFigOps && !hasFinalOps) {
    return null;
  }

  return (
    <>
      {hasFigOps && (
        <div className="border-t border-slate-100 pt-4 space-y-2">
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Fig Ops Pipeline</div>
          <div className="space-y-1.5">
            {orderedOps.map((op, index) => (
              <FigOpItem
                key={`${op.name}-${index}`}
                op={op}
                index={index}
                isDragging={dragIndex === index}
                isDragOver={dragOverIndex === index}
                onDragStart={handleDragStart}
                onDragOver={handleDragOver}
                onDragEnd={handleDragEnd}
              />
            ))}
          </div>
        </div>
      )}

      {hasFinalOps && (
        <div className="border-t border-slate-100 pt-4 space-y-3">
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Final Ops</div>
          {Object.entries(finalOps).map(([opType, ops]) => {
            if (ops.length === 0) return null;
            return (
              <div key={opType} className="space-y-1.5">
                <div className="text-[10px] uppercase tracking-wider text-slate-400 font-medium">
                  {opType}
                </div>
                {ops.map((op, index) => (
                  <FinalOpItem key={`${op.fn_name}-${index}`} op={op} />
                ))}
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}

/** Section showing dependency port wiring for an analysis node. */
export function DependencyPortsSection({
  ports,
  nodeId,
}: {
  ports: string[];
  nodeId: string;
}) {
  if (ports.length === 0) return null;

  return (
    <div className="border-t border-slate-100 pt-4 space-y-2">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Dependencies</div>
      <ul className="space-y-1 text-xs text-slate-600">
        {ports.map((port) => (
          <li key={port} className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-300 shrink-0" />
            <span className="font-mono">{port}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
