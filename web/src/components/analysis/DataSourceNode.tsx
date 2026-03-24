/**
 * DataSourceNode — the implicit data source on the left edge of the analysis DAG.
 *
 * Renders the hierarchical state field tree (states, model, task) with
 * expand/collapse controls. Each tree node is a connectable React Flow Handle,
 * allowing users to wire specific sub-fields (e.g. "states.net.hidden") to
 * analysis nodes. The node auto-resizes when branches are expanded/collapsed.
 *
 * Top-level items are always visible with chevrons. Connecting to a branch
 * node sends the full subtree; connecting to a leaf sends that specific field.
 */

import { useEffect } from 'react';
import { useUpdateNodeInternals, type NodeProps } from '@xyflow/react';
import type { DataSourceNodeData } from '@/stores/analysisStore';
import { STATE_FIELD_TREE } from '@/types/analysis';
import { StateFieldTree, useFieldTreeExpansion, FIELD_ROW_HEIGHT } from './StateFieldTree';
import { Database } from 'lucide-react';
import clsx from 'clsx';

const WIDTH = 200;
const HEADER_HEIGHT = 36;
const BODY_PADDING = 8;

export function DataSourceNode({ id, data, selected }: NodeProps) {
  const nodeData = data as DataSourceNodeData;
  const { expandedPaths, toggleExpand, visibleCount } = useFieldTreeExpansion(STATE_FIELD_TREE);

  const bodyHeight = BODY_PADDING * 2 + visibleCount * FIELD_ROW_HEIGHT;
  const totalHeight = HEADER_HEIGHT + bodyHeight;

  // Notify React Flow that handles changed when expansion state changes
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, visibleCount, updateNodeInternals]);

  return (
    <div
      className={clsx(
        'relative rounded-lg border bg-slate-50/80 backdrop-blur shadow-soft transition-all duration-150',
        selected
          ? 'border-brand-500 ring-1 ring-brand-500/40'
          : 'border-slate-200/80',
      )}
      style={{ width: WIDTH, height: totalHeight }}
    >
      {/* Header */}
      <div className="px-3 py-2 flex items-center justify-center border-b border-slate-100/80 rounded-t-lg">
        <Database className="w-4 h-4 text-slate-400" />
      </div>

      {/* Hierarchical field tree */}
      <div
        className="relative text-[11px] text-slate-400"
        style={{ height: bodyHeight, padding: BODY_PADDING }}
      >
        <StateFieldTree
          nodes={STATE_FIELD_TREE}
          expandedPaths={expandedPaths}
          onToggle={toggleExpand}
          bodyPadding={BODY_PADDING}
        />
      </div>
    </div>
  );
}
