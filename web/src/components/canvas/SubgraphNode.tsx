/**
 * SubgraphNode — a React Flow custom node that renders a CDE controller (or
 * any other named subgraph) as a collapsible nested canvas preview.
 *
 * Collapsed: looks like any other node with input/output handles and a
 * "Subgraph" type badge.
 * Expanded: shows a read-only nested ReactFlow canvas of the internal nodes
 * and edges drawn from data.subgraph.
 */

import { useCallback, useState } from 'react';
import {
  Handle,
  Position,
  ReactFlow,
  Background,
  BackgroundVariant,
  type NodeProps,
} from '@xyflow/react';
import { CustomNode } from './CustomNode';
import type { GraphNodeData } from '@/types/graph';
import clsx from 'clsx';
import { ChevronDown, ChevronRight, Layers } from 'lucide-react';

// ---------------------------------------------------------------------------
// Layout constants — match CustomNode for visual consistency
// ---------------------------------------------------------------------------

const DEFAULT_WIDTH = 260;
const HEADER_HEIGHT = 40;
const HANDLE_OFFSET = -6;
const ROW_HEIGHT = 26;
const BODY_PADDING = 12;
const LABEL_OFFSET = 22;
const PREVIEW_HEIGHT = 300;
const PREVIEW_MIN_WIDTH = 500;

// Node types for the nested preview canvas — only 'component' nodes appear
// inside a subgraph template.
const NESTED_NODE_TYPES = {
  component: CustomNode,
};

// ---------------------------------------------------------------------------
// SubgraphNode component
// ---------------------------------------------------------------------------

export function SubgraphNode({ data, selected }: NodeProps) {
  const nodeData = data as GraphNodeData;
  const { label, spec, subgraph } = nodeData;

  const [expanded, setExpanded] = useState(false);

  const toggleExpanded = useCallback(() => setExpanded((v) => !v), []);

  const inputPorts = subgraph?.inputPorts ?? spec.input_ports;
  const outputPorts = subgraph?.outputPorts ?? spec.output_ports;
  const inputCount = inputPorts.length;
  const outputCount = outputPorts.length;
  const rowCount = Math.max(1, inputCount, outputCount);
  const bodyHeight = BODY_PADDING * 2 + rowCount * ROW_HEIGHT;

  const width = nodeData.size?.width ?? DEFAULT_WIDTH;

  const expandedTotalHeight = HEADER_HEIGHT + bodyHeight + PREVIEW_HEIGHT + 8;
  const collapsedTotalHeight = HEADER_HEIGHT + bodyHeight;
  const totalHeight = expanded ? expandedTotalHeight : collapsedTotalHeight;

  const rowCenterInBody = (index: number) => BODY_PADDING + ROW_HEIGHT * (index + 0.5);

  const templateName = spec.params?._template as string | undefined;

  const nestedNodes = subgraph?.nodes as Parameters<typeof ReactFlow>[0]['nodes'] | undefined;
  const nestedEdges = subgraph?.edges as Parameters<typeof ReactFlow>[0]['edges'] | undefined;

  return (
    <div
      className={clsx(
        'relative rounded-xl border shadow-soft bg-white/90 backdrop-blur transition-all duration-150',
        selected ? 'border-brand-500 ring-1 ring-brand-500/40' : 'border-violet-200',
      )}
      style={{ width, height: totalHeight }}
    >
      {/* State handles (hidden but present for connection compatibility) */}
      <Handle
        type="target"
        position={Position.Left}
        id="__state_in"
        style={{ top: HEADER_HEIGHT / 2, left: HANDLE_OFFSET - 2, transform: 'translateY(-50%)' }}
        className="w-4 h-4 rounded-full border-2 border-white shadow-soft cursor-crosshair bg-slate-300"
      />
      <Handle
        type="source"
        position={Position.Right}
        id="__state_out"
        style={{ top: HEADER_HEIGHT / 2, right: HANDLE_OFFSET - 2, transform: 'translateY(-50%)' }}
        className="w-4 h-4 rounded-full border-2 border-white shadow-soft cursor-crosshair bg-slate-300"
      />

      {/* Header */}
      <div
        className={clsx(
          'px-3 py-2 bg-violet-50/80 flex items-center justify-between gap-3 overflow-hidden border-b border-violet-100',
          'rounded-t-xl',
        )}
      >
        <div className="min-w-0 flex-1 flex items-center gap-2 pr-2">
          <Layers className="w-3.5 h-3.5 text-violet-400 shrink-0" />
          <div className="text-sm font-medium text-slate-800 truncate w-full" title={label}>
            {label}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <div className="text-[10px] text-violet-500 uppercase tracking-wide shrink-0">
            {templateName ?? 'Subgraph'}
          </div>
          <button
            className="text-slate-400 hover:text-violet-600 transition-colors"
            onClick={(event) => {
              event.stopPropagation();
              toggleExpanded();
            }}
            title={expanded ? 'Collapse preview' : 'Expand preview'}
          >
            {expanded ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Port body */}
      <div
        className="relative text-xs text-slate-600"
        style={{ height: bodyHeight, padding: BODY_PADDING }}
      >
        {/* Input handles and labels */}
        {inputPorts.map((port, index) => (
          <Handle
            key={`handle-in-${port}`}
            type="target"
            position={Position.Left}
            id={port}
            style={{
              top: rowCenterInBody(index),
              left: HANDLE_OFFSET,
              transform: 'translateY(-50%)',
            }}
            className="w-3 h-3 z-20 border border-white shadow-soft bg-slate-300"
          />
        ))}
        {outputPorts.map((port, index) => (
          <Handle
            key={`handle-out-${port}`}
            type="source"
            position={Position.Right}
            id={port}
            style={{
              top: rowCenterInBody(index),
              right: HANDLE_OFFSET,
              transform: 'translateY(-50%)',
            }}
            className="w-3 h-3 z-20 border border-white shadow-soft bg-slate-300"
          />
        ))}
        {inputPorts.map((port, index) => (
          <div
            key={`label-in-${port}`}
            className="absolute left-0 flex items-center text-slate-600"
            style={{
              top: rowCenterInBody(index),
              left: LABEL_OFFSET,
              transform: 'translateY(-50%)',
            }}
          >
            {port}
          </div>
        ))}
        {outputPorts.map((port, index) => (
          <div
            key={`label-out-${port}`}
            className="absolute right-0 flex items-center justify-end text-slate-600"
            style={{
              top: rowCenterInBody(index),
              right: LABEL_OFFSET,
              transform: 'translateY(-50%)',
            }}
          >
            {port}
          </div>
        ))}
      </div>

      {/* Nested preview canvas (only when expanded and subgraph data exists) */}
      {expanded && subgraph && nestedNodes && nestedEdges && (
        <div
          className="mx-2 mb-2 rounded-lg border border-violet-100 overflow-hidden"
          style={{ height: PREVIEW_HEIGHT }}
        >
          <div
            className="nodrag nowheel"
            style={{ width: '100%', height: '100%', minWidth: PREVIEW_MIN_WIDTH }}
          >
            <ReactFlow
              nodes={nestedNodes}
              edges={nestedEdges}
              nodeTypes={NESTED_NODE_TYPES}
              fitView
              fitViewOptions={{ padding: 0.15 }}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable={false}
              panOnDrag={false}
              zoomOnScroll={false}
              zoomOnPinch={false}
              zoomOnDoubleClick={false}
              preventScrolling={false}
              proOptions={{ hideAttribution: true }}
            >
              <Background
                variant={BackgroundVariant.Dots}
                gap={12}
                size={0.8}
                color="#e2d9f3"
              />
            </ReactFlow>
          </div>
        </div>
      )}

      {/* Fallback when expanded but no subgraph data */}
      {expanded && !subgraph && (
        <div
          className="mx-2 mb-2 rounded-lg border border-violet-100 bg-violet-50/40 flex items-center justify-center"
          style={{ height: PREVIEW_HEIGHT }}
        >
          <span className="text-xs text-violet-400">No internal graph data</span>
        </div>
      )}
    </div>
  );
}
