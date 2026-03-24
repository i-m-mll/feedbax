/**
 * DataSourceNode — the implicit data source on the left edge of the analysis DAG.
 *
 * Represents AnalysisInputData: the states, inputs, outputs, targets, and
 * metadata that flow into analysis nodes. Styled as a muted, rounded rectangle
 * with output handles only, visually distinct from full analysis nodes.
 */

import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { DataSourceNodeData } from '@/stores/analysisStore';
import { Database } from 'lucide-react';
import clsx from 'clsx';

const WIDTH = 180;
const HEADER_HEIGHT = 36;
const ROW_HEIGHT = 24;
const BODY_PADDING = 8;
const HANDLE_OFFSET = -6;
const LABEL_OFFSET = 18;

export function DataSourceNode({ data, selected }: NodeProps) {
  const nodeData = data as DataSourceNodeData;
  const outputs = nodeData.outputs ?? [];
  const rowCount = Math.max(1, outputs.length);
  const bodyHeight = BODY_PADDING * 2 + rowCount * ROW_HEIGHT;
  const totalHeight = HEADER_HEIGHT + bodyHeight;

  const rowCenter = (index: number) => BODY_PADDING + ROW_HEIGHT * (index + 0.5);

  return (
    <div
      className={clsx(
        'relative rounded-lg border bg-slate-50/80 backdrop-blur shadow-soft transition-all duration-150',
        selected
          ? 'border-brand-500 ring-1 ring-brand-500/40'
          : 'border-slate-200/80'
      )}
      style={{ width: WIDTH, height: totalHeight }}
    >
      {/* Header */}
      <div className="px-3 py-2 flex items-center justify-center border-b border-slate-100/80 rounded-t-lg">
        <Database className="w-4 h-4 text-slate-400" />
      </div>

      {/* Output ports */}
      <div
        className="relative text-[11px] text-slate-400"
        style={{ height: bodyHeight, padding: BODY_PADDING }}
      >
        {outputs.map((port, index) => (
          <Handle
            key={`handle-out-${port}`}
            type="source"
            position={Position.Right}
            id={port}
            style={{
              top: rowCenter(index),
              right: HANDLE_OFFSET,
              transform: 'translateY(-50%)',
              width: '7px',
              height: '7px',
            }}
            className="border border-white shadow-soft bg-slate-300"
          />
        ))}
        {outputs.map((port, index) => (
          <div
            key={`label-${port}`}
            className="absolute flex items-center justify-end text-slate-400"
            style={{
              top: rowCenter(index),
              right: LABEL_OFFSET,
              transform: 'translateY(-50%)',
            }}
          >
            {port}
          </div>
        ))}
      </div>
    </div>
  );
}
