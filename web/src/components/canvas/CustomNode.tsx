import { Handle, NodeResizer, Position, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '@/types/graph';
import clsx from 'clsx';
import { useGraphStore } from '@/stores/graphStore';
import { useEffect, useState } from 'react';
import { ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';

const DEFAULT_WIDTH = 220;
const HEADER_HEIGHT = 40;
const BODY_PADDING = 12;
const ROW_HEIGHT = 26;
const MIN_WIDTH = 180;
const MIN_HEIGHT = 96;

export function CustomNode({ data, selected }: NodeProps<GraphNodeData>) {
  const { spec, label, collapsed } = data;
  const toggleNodeCollapse = useGraphStore((state) => state.toggleNodeCollapse);
  const renameNode = useGraphStore((state) => state.renameNode);
  const enterSubgraph = useGraphStore((state) => state.enterSubgraph);
  const [isEditing, setIsEditing] = useState(false);
  const [nameValue, setNameValue] = useState(label);
  const isComposite = spec.type === 'Network';
  const inputCount = spec.input_ports.length;
  const outputCount = spec.output_ports.length;
  const rowCount = Math.max(1, inputCount, outputCount);
  const canCollapse = rowCount > 1;
  const collapsedEffective = collapsed && canCollapse;
  const defaultHeight = HEADER_HEIGHT + BODY_PADDING * 2 + rowCount * ROW_HEIGHT;
  const baseHeight = data.size?.height ?? defaultHeight;
  const height = collapsedEffective ? Math.min(baseHeight, defaultHeight) : baseHeight;
  const width = data.size?.width ?? DEFAULT_WIDTH;
  const bodyHeight = Math.max(ROW_HEIGHT, height - HEADER_HEIGHT - BODY_PADDING * 2);
  const rowMidpoint = (index: number) =>
    HEADER_HEIGHT + BODY_PADDING + ((index + 0.5) / rowCount) * bodyHeight;

  useEffect(() => {
    if (!isEditing) {
      setNameValue(label);
    }
  }, [label, isEditing]);

  return (
    <div
      className={clsx(
        'relative rounded-xl border shadow-soft bg-white/90 backdrop-blur',
        selected ? 'border-brand-500 ring-1 ring-brand-500/40' : 'border-slate-200'
      )}
      style={{ width, height }}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={MIN_WIDTH}
        minHeight={MIN_HEIGHT}
        keepAspectRatio={false}
        handleClassName="bg-white border border-slate-300 shadow-soft"
        lineClassName="border border-dashed border-slate-200"
      />
      <div
        className="px-3 py-2 border-b border-slate-100 bg-slate-50/70 rounded-t-xl flex items-center justify-between gap-3 overflow-hidden"
        onDoubleClick={(event) => {
          event.stopPropagation();
          if (isComposite) {
            enterSubgraph(label);
          } else if (canCollapse) {
            toggleNodeCollapse(label);
          }
        }}
      >
        <div className="min-w-0 flex-1 flex items-center gap-2">
          {canCollapse && (
            <button
              className="text-slate-400 hover:text-slate-600"
              onClick={(event) => {
                event.stopPropagation();
                toggleNodeCollapse(label);
              }}
              title={collapsed ? 'Expand node' : 'Collapse node'}
            >
              {collapsed ? (
                <ChevronRight className="w-3 h-3" />
              ) : (
                <ChevronDown className="w-3 h-3" />
              )}
            </button>
          )}
          {isEditing ? (
            <input
              value={nameValue}
              onChange={(event) => setNameValue(event.target.value)}
              onBlur={() => {
                renameNode(label, nameValue);
                setIsEditing(false);
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  renameNode(label, nameValue);
                  setIsEditing(false);
                }
                if (event.key === 'Escape') {
                  setIsEditing(false);
                  setNameValue(label);
                }
              }}
              className="w-full bg-white/70 border border-slate-200 rounded-md px-2 py-1 text-sm text-slate-800"
              autoFocus
              onClick={(event) => event.stopPropagation()}
            />
          ) : (
            <button
              className="text-sm font-medium text-slate-800 hover:text-brand-600 truncate w-full text-left"
              onClick={(event) => {
                event.stopPropagation();
                setIsEditing(true);
              }}
              onDoubleClick={(event) => event.stopPropagation()}
              title={label}
            >
              {label}
            </button>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isComposite && (
            <button
              className="text-slate-400 hover:text-brand-600"
              onClick={(event) => {
                event.stopPropagation();
                enterSubgraph(label);
              }}
              title="Open subgraph"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </button>
          )}
          <div className="text-[11px] text-slate-500 truncate max-w-[110px]" title={spec.type}>
            {spec.type}
          </div>
        </div>
      </div>

      <div
        className="relative px-3 py-2 text-xs text-slate-600"
        style={{ height: Math.max(ROW_HEIGHT, height - HEADER_HEIGHT) }}
      >
        {spec.input_ports.map((port, index) => (
          <div
            key={port}
            className={clsx(
              'absolute left-3 flex items-center gap-2',
              collapsedEffective ? 'text-slate-400' : 'text-slate-600'
            )}
            style={{ top: rowMidpoint(index) - 10 }}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={port}
              style={{ top: rowMidpoint(index) }}
              className="w-3 h-3 bg-brand-500"
            />
            {!collapsedEffective && <span>{port}</span>}
          </div>
        ))}
        {spec.output_ports.map((port, index) => (
          <div
            key={port}
            className={clsx(
              'absolute right-3 flex items-center gap-2 justify-end',
              collapsedEffective ? 'text-slate-400' : 'text-slate-600'
            )}
            style={{ top: rowMidpoint(index) - 10 }}
          >
            {!collapsedEffective && <span>{port}</span>}
            <Handle
              type="source"
              position={Position.Right}
              id={port}
              style={{ top: rowMidpoint(index) }}
              className="w-3 h-3 bg-mint-500"
            />
          </div>
        ))}
        {collapsedEffective && (
          <div className="absolute left-3 bottom-2 text-[10px] text-slate-400">
            {inputCount} in • {outputCount} out
          </div>
        )}
      </div>
    </div>
  );
}
