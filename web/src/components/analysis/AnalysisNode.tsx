/**
 * AnalysisNode — custom React Flow node for analysis DAG entries.
 *
 * Full analysis nodes are rendered as rounded rectangles with input/output
 * handles, matching the visual language of CustomNode in the model canvas
 * but with a distinct teal/emerald accent to differentiate the analysis
 * context from the model-editing context.
 *
 * Dependency nodes (role === 'dependency') are rendered smaller and muted.
 */

import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { AnalysisNodeData } from '@/stores/analysisStore';
import clsx from 'clsx';

// Layout constants — aligned with CustomNode for consistency
const WIDTH_FULL = 200;
const WIDTH_DEP = 160;
const HEADER_HEIGHT_FULL = 36;
const HEADER_HEIGHT_DEP = 30;
const BODY_PADDING = 10;
const ROW_HEIGHT_FULL = 24;
const ROW_HEIGHT_DEP = 20;
const HANDLE_OFFSET = -6;
const LABEL_OFFSET = 18;

export function AnalysisNode({ id, data, selected }: NodeProps) {
  const nodeData = data as AnalysisNodeData;
  const spec = nodeData.spec;
  const isDep = spec.role === 'dependency';

  const width = isDep ? WIDTH_DEP : WIDTH_FULL;
  const headerHeight = isDep ? HEADER_HEIGHT_DEP : HEADER_HEIGHT_FULL;
  const rowHeight = isDep ? ROW_HEIGHT_DEP : ROW_HEIGHT_FULL;

  const inputCount = spec.inputPorts.length;
  const outputCount = spec.outputPorts.length;
  const rowCount = Math.max(1, inputCount, outputCount);
  const bodyHeight = BODY_PADDING * 2 + rowCount * rowHeight;
  const totalHeight = headerHeight + bodyHeight;

  const rowCenter = (index: number) => BODY_PADDING + rowHeight * (index + 0.5);

  return (
    <div
      className={clsx(
        'relative rounded-xl border shadow-soft backdrop-blur transition-all duration-150',
        isDep ? 'bg-white/70' : 'bg-white/90',
        selected
          ? 'border-brand-500 ring-1 ring-brand-500/40'
          : isDep
            ? 'border-slate-200/60'
            : 'border-emerald-200'
      )}
      style={{ width, height: totalHeight }}
    >
      {/* Header */}
      <div
        className={clsx(
          'px-3 py-1.5 flex items-center justify-between gap-2 overflow-hidden border-b rounded-t-xl',
          isDep
            ? 'bg-slate-50/60 border-slate-100/60'
            : 'bg-emerald-50/60 border-emerald-100/60'
        )}
      >
        <div
          className={clsx(
            'font-medium truncate',
            isDep ? 'text-xs text-slate-600' : 'text-sm text-slate-800'
          )}
          title={spec.label}
        >
          {spec.label}
        </div>
        <div
          className={clsx(
            'text-[10px] uppercase tracking-wide shrink-0',
            isDep ? 'text-slate-400' : 'text-emerald-500'
          )}
        >
          {spec.type}
        </div>
      </div>

      {/* Ports */}
      <div
        className={clsx('relative', isDep ? 'text-[10px]' : 'text-xs')}
        style={{ height: bodyHeight, padding: BODY_PADDING }}
      >
        {/* Input handles */}
        {spec.inputPorts.map((port, index) => (
          <Handle
            key={`in-${port}`}
            type="target"
            position={Position.Left}
            id={port}
            style={{
              top: rowCenter(index),
              left: HANDLE_OFFSET,
              transform: 'translateY(-50%)',
              clipPath: 'polygon(0% 0%, 100% 50%, 0% 100%)',
              width: isDep ? '6px' : '8px',
              height: isDep ? '6px' : '8px',
            }}
            className={clsx(
              'border border-white shadow-soft',
              isDep ? 'bg-slate-300' : 'bg-emerald-400'
            )}
          />
        ))}

        {/* Output handles */}
        {spec.outputPorts.map((port, index) => (
          <Handle
            key={`out-${port}`}
            type="source"
            position={Position.Right}
            id={port}
            style={{
              top: rowCenter(index),
              right: HANDLE_OFFSET,
              transform: 'translateY(-50%)',
              clipPath: 'polygon(0% 0%, 100% 50%, 0% 100%)',
              width: isDep ? '6px' : '8px',
              height: isDep ? '6px' : '8px',
            }}
            className={clsx(
              'border border-white shadow-soft',
              isDep ? 'bg-slate-300' : 'bg-emerald-400'
            )}
          />
        ))}

        {/* Input labels */}
        {spec.inputPorts.map((port, index) => (
          <div
            key={`label-in-${port}`}
            className={clsx(
              'absolute flex items-center',
              isDep ? 'text-slate-400' : 'text-slate-600'
            )}
            style={{
              top: rowCenter(index),
              left: LABEL_OFFSET,
              transform: 'translateY(-50%)',
            }}
          >
            {port}
          </div>
        ))}

        {/* Output labels */}
        {spec.outputPorts.map((port, index) => (
          <div
            key={`label-out-${port}`}
            className={clsx(
              'absolute flex items-center justify-end',
              isDep ? 'text-slate-400' : 'text-slate-600'
            )}
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

/** Smaller variant used for dependency nodes. Same component, just aliased for clarity. */
export const AnalysisDepNode = AnalysisNode;
