/**
 * TransformNode — a small pill-shaped node representing a prep op (transform)
 * on an edge in the analysis DAG.
 *
 * Visually lighter than full analysis nodes: pill shape, smaller footprint,
 * subtle styling. Sits inline on edges. Click to see config.
 *
 * Follows the TapNode pattern (small inline nodes) but with a distinct
 * visual style for transforms vs probes/interventions.
 */

import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { TransformNodeData } from '@/stores/analysisStore';
import { useAnalysisStore } from '@/stores/analysisStore';
import { SlidersHorizontal } from 'lucide-react';
import clsx from 'clsx';
import { useCallback, useState } from 'react';

const WIDTH = 120;
const HEIGHT = 32;

export function TransformNode({ id, data, selected }: NodeProps) {
  const nodeData = data as TransformNodeData;
  const transform = nodeData.transform;
  const setSelectedTransform = useAnalysisStore((s) => s.setSelectedTransform);
  const [showPopup, setShowPopup] = useState(false);

  const handleClick = useCallback(() => {
    setSelectedTransform(id);
    setShowPopup((prev) => !prev);
  }, [id, setSelectedTransform]);

  const params = transform.params;
  const paramEntries = Object.entries(params);

  return (
    <div className="relative">
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="in"
        style={{ left: -5, top: '50%', transform: 'translateY(-50%)' }}
        className="w-1.5 h-1.5 bg-slate-300 border border-white shadow-soft"
      />

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="out"
        style={{ right: -5, top: '50%', transform: 'translateY(-50%)' }}
        className="w-1.5 h-1.5 bg-slate-300 border border-white shadow-soft"
      />

      {/* Pill body */}
      <button
        onClick={handleClick}
        className={clsx(
          'flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-medium',
          'bg-white/80 backdrop-blur shadow-soft transition-all duration-150 cursor-pointer',
          selected
            ? 'border-amber-400 ring-1 ring-amber-300/40 text-amber-700'
            : 'border-slate-200/80 text-slate-500 hover:border-slate-300 hover:text-slate-600'
        )}
        style={{ width: WIDTH, height: HEIGHT }}
      >
        <SlidersHorizontal className="w-3 h-3 shrink-0" />
        <span className="truncate">{transform.label}</span>
      </button>

      {/* Config popup on click */}
      {showPopup && paramEntries.length > 0 && (
        <div className="absolute z-50 top-full mt-2 left-1/2 -translate-x-1/2 min-w-[160px] rounded-lg border border-slate-200 bg-white p-3 shadow-lift text-xs">
          <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">
            Config
          </div>
          <div className="space-y-1">
            {paramEntries.map(([key, value]) => (
              <div key={key} className="flex justify-between gap-2">
                <span className="text-slate-500">{key}</span>
                <span className="text-slate-700 font-medium">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
