import { Handle, NodeResizer, Position, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '@/types/graph';
import clsx from 'clsx';
import { useGraphStore } from '@/stores/graphStore';
import { useLayoutStore } from '@/stores/layoutStore';
import { useEffect, useState } from 'react';
import { ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';

const DEFAULT_WIDTH = 220;
const HEADER_HEIGHT = 40;
const BODY_PADDING = 12;
const COLLAPSED_BODY_PADDING = 4;
const COLLAPSED_BODY_HEIGHT = 24;
const ROW_HEIGHT = 26;
const LABEL_OFFSET = 18;
const MIN_WIDTH = 180;
const MIN_HEIGHT = 96;

export function CustomNode({ data, selected }: NodeProps) {
  const nodeData = data as GraphNodeData;
  const { spec, label, collapsed } = nodeData;
  const resizeMode = useLayoutStore((state) => state.resizeMode);
  const toggleNodeCollapse = useGraphStore((state) => state.toggleNodeCollapse);
  const renameNode = useGraphStore((state) => state.renameNode);
  const enterSubgraph = useGraphStore((state) => state.enterSubgraph);
  const [isEditing, setIsEditing] = useState(false);
  const [nameValue, setNameValue] = useState(label);
  const hasSubgraph = useGraphStore((state) => Boolean(state.graph.subgraphs?.[label]));
  const isComposite =
    spec.type === 'Network' || spec.type === 'Subgraph' || hasSubgraph;
  const inputCount = spec.input_ports.length;
  const outputCount = spec.output_ports.length;
  const totalPorts = inputCount + outputCount;
  const canCollapse = totalPorts > 1;
  const collapsedEffective = collapsed && canCollapse;
  const rowCount = collapsedEffective ? 1 : Math.max(1, inputCount, outputCount);
  const bodyPadding = collapsedEffective ? COLLAPSED_BODY_PADDING : BODY_PADDING;
  const rowHeightTarget = ROW_HEIGHT;
  const defaultHeight = HEADER_HEIGHT + bodyPadding * 2 + rowCount * rowHeightTarget;
  const width = nodeData.size?.width ?? DEFAULT_WIDTH;
  const baseHeight = nodeData.size?.height ?? defaultHeight;
  const height = collapsedEffective ? HEADER_HEIGHT + COLLAPSED_BODY_HEIGHT : baseHeight;
  const bodyHeight = collapsedEffective
    ? COLLAPSED_BODY_HEIGHT
    : Math.max(rowHeightTarget + bodyPadding * 2, height - HEADER_HEIGHT);
  const contentHeight = collapsedEffective
    ? COLLAPSED_BODY_HEIGHT
    : Math.max(rowHeightTarget, bodyHeight - bodyPadding * 2);
  const rowHeight = contentHeight / rowCount;
  const rowCenterInBody = (index: number) =>
    collapsedEffective ? bodyHeight / 2 : bodyPadding + rowHeight * (index + 0.5);

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
        isVisible={selected && resizeMode}
        minWidth={MIN_WIDTH}
        minHeight={MIN_HEIGHT}
        keepAspectRatio={false}
        handleClassName="bg-white border border-slate-300 shadow-soft z-10"
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
        <div className="min-w-0 flex-1 flex items-center gap-2 pr-2">
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
        <div className="flex items-center gap-2 shrink-0">
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
        className="relative text-xs text-slate-600"
        style={{ height: bodyHeight, padding: bodyPadding }}
      >
        {spec.input_ports.map((port, index) => {
          const isVisible = !collapsedEffective || index === 0;
          return (
            <Handle
              key={`handle-in-${port}`}
              type="target"
              position={Position.Left}
              id={port}
              style={{
                top: rowCenterInBody(collapsedEffective ? 0 : index),
                transform: 'translateY(-50%)',
              }}
              className={clsx(
                'w-3 h-3 z-20',
                isVisible
                  ? 'bg-brand-500'
                  : 'bg-transparent opacity-0 pointer-events-none border border-transparent'
              )}
            />
          );
        })}
        {spec.output_ports.map((port, index) => {
          const isVisible = !collapsedEffective || index === 0;
          return (
            <Handle
              key={`handle-out-${port}`}
              type="source"
              position={Position.Right}
              id={port}
              style={{
                top: rowCenterInBody(collapsedEffective ? 0 : index),
                transform: 'translateY(-50%)',
              }}
              className={clsx(
                'w-3 h-3 z-20',
                isVisible
                  ? 'bg-mint-500'
                  : 'bg-transparent opacity-0 pointer-events-none border border-transparent'
              )}
            />
          );
        })}
        {!collapsedEffective &&
          spec.input_ports.map((port, index) => (
            <div
              key={`label-in-${port}`}
              className="absolute left-0 flex items-center gap-2 text-slate-600"
              style={{
                top: rowCenterInBody(index),
                left: LABEL_OFFSET,
                transform: 'translateY(-50%)',
              }}
            >
              <span>{port}</span>
            </div>
          ))}
        {!collapsedEffective &&
          spec.output_ports.map((port, index) => (
            <div
              key={`label-out-${port}`}
              className="absolute right-0 flex items-center gap-2 justify-end text-slate-600"
              style={{
                top: rowCenterInBody(index),
                right: LABEL_OFFSET,
                transform: 'translateY(-50%)',
              }}
            >
              <span>{port}</span>
            </div>
          ))}
      </div>
    </div>
  );
}
