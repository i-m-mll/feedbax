import { Handle, NodeResizer, Position, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '@/types/graph';
import clsx from 'clsx';
import { useGraphStore } from '@/stores/graphStore';
import { useLayoutStore } from '@/stores/layoutStore';
import { useTrainingStore } from '@/stores/trainingStore';
import { ChevronDown, ChevronRight, ExternalLink, Crosshair } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { PortContextMenu } from './PortContextMenu';

const DEFAULT_WIDTH = 220;
const HEADER_HEIGHT = 40;
const BODY_PADDING = 12;
const ROW_HEIGHT = 26;
const LABEL_OFFSET = 22;
const HANDLE_OFFSET = -6;
const MIN_WIDTH = 180;
const MIN_HEIGHT = 96;

export function CustomNode({ data, selected }: NodeProps) {
  const nodeData = data as GraphNodeData;
  const { spec, label, collapsed } = nodeData;
  const resizeMode = useLayoutStore((state) => state.resizeMode);
  const toggleNodeCollapse = useGraphStore((state) => state.toggleNodeCollapse);
  const enterSubgraph = useGraphStore((state) => state.enterSubgraph);
  const hasSubgraph = useGraphStore((state) => Boolean(state.graph.subgraphs?.[label]));
  const highlightedProbeSelector = useTrainingStore((state) => state.highlightedProbeSelector);

  // Context menu state for port right-click
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    portName: string;
    portType: 'input' | 'output';
  } | null>(null);

  const compositeTypes = useGraphStore((state) => state._compositeTypes);
  const isComposite =
    compositeTypes.has(spec.type) || hasSubgraph;
  const inputCount = spec.input_ports.length;
  const outputCount = spec.output_ports.length;
  const totalPorts = inputCount + outputCount;
  const canCollapse = totalPorts > 1;
  const collapsedEffective = collapsed && canCollapse;
  const connectedInputs = new Set(nodeData.connected_inputs ?? []);
  const connectedOutputs = new Set(nodeData.connected_outputs ?? []);
  const hasStateIn = nodeData.state_in ?? false;
  const hasStateOut = nodeData.state_out ?? false;
  const rowCount = Math.max(1, inputCount, outputCount);
  const defaultHeight = HEADER_HEIGHT + BODY_PADDING * 2 + rowCount * ROW_HEIGHT;
  const width = nodeData.size?.width ?? DEFAULT_WIDTH;
  const baseHeight = nodeData.size?.height ?? defaultHeight;
  const expandedHeight = Math.max(baseHeight, defaultHeight);
  const height = collapsedEffective ? HEADER_HEIGHT : expandedHeight;
  const bodyHeight = Math.max(ROW_HEIGHT + BODY_PADDING * 2, height - HEADER_HEIGHT);
  const contentHeight = Math.max(ROW_HEIGHT, bodyHeight - BODY_PADDING * 2);
  const rowHeight = contentHeight / rowCount;
  const rowCenterInBody = (index: number) => BODY_PADDING + rowHeight * (index + 0.5);

  // Check if this node has any highlighted ports
  const highlightedPorts = useMemo(() => {
    if (!highlightedProbeSelector) return new Set<string>();
    const ports = new Set<string>();
    // Check if selector matches this node's ports
    if (highlightedProbeSelector.startsWith('port:')) {
      const portRef = highlightedProbeSelector.slice(5);
      if (portRef.startsWith(`${label}.`)) {
        const portName = portRef.slice(label.length + 1);
        ports.add(portName);
      }
    }
    return ports;
  }, [highlightedProbeSelector, label]);

  const isNodeHighlighted = highlightedPorts.size > 0;

  const handlePortContextMenu = useCallback(
    (event: React.MouseEvent, portName: string, portType: 'input' | 'output') => {
      event.preventDefault();
      event.stopPropagation();
      setContextMenu({
        x: event.clientX,
        y: event.clientY,
        portName,
        portType,
      });
    },
    []
  );

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);
  return (
    <div
      className={clsx(
        'relative rounded-xl border shadow-soft bg-white/90 backdrop-blur transition-all duration-150',
        selected ? 'border-brand-500 ring-1 ring-brand-500/40' : 'border-slate-200',
        isNodeHighlighted && !selected && 'border-amber-400 ring-2 ring-amber-200'
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
      <Handle
        type="target"
        position={Position.Left}
        id="__state_in"
        style={{ top: HEADER_HEIGHT / 2, left: HANDLE_OFFSET - 2, transform: 'translateY(-50%)' }}
        className={clsx(
          'w-4 h-4 rounded-full border-2 border-white shadow-soft cursor-crosshair',
          hasStateIn ? 'bg-slate-500' : 'bg-slate-300'
        )}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="__state_out"
        style={{
          top: HEADER_HEIGHT / 2,
          right: HANDLE_OFFSET - 2,
          transform: 'translateY(-50%)',
        }}
        className={clsx(
          'w-4 h-4 rounded-full border-2 border-white shadow-soft cursor-crosshair',
          hasStateOut ? 'bg-slate-500' : 'bg-slate-300'
        )}
      />
      <div
        className={clsx(
          'px-3 py-2 bg-slate-50/70 flex items-center justify-between gap-3 overflow-hidden',
          collapsedEffective ? 'rounded-xl' : 'border-b border-slate-100 rounded-t-xl'
        )}
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
          <div className="text-sm font-medium text-slate-800 truncate w-full" title={label}>
            {label}
          </div>
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
          {!collapsedEffective && (
            <div className="text-[11px] text-slate-500 truncate max-w-[110px]" title={spec.type}>
              {spec.type}
            </div>
          )}
        </div>
      </div>

      {collapsedEffective ? null : (
        <div className="relative text-xs text-slate-600" style={{ height: bodyHeight, padding: BODY_PADDING }}>
          {spec.input_ports.map((port, index) => (
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
              className={clsx(
                'w-3 h-3 z-20 border border-white shadow-soft',
                connectedInputs.has(port) ? 'bg-brand-500' : 'bg-slate-300'
              )}
            />
          ))}
          {spec.output_ports.map((port, index) => (
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
              className={clsx(
                'w-3 h-3 z-20 border border-white shadow-soft transition-all duration-150',
                connectedOutputs.has(port) ? 'bg-mint-500' : 'bg-slate-300',
                highlightedPorts.has(port) && 'bg-amber-400 ring-2 ring-amber-200 scale-125'
              )}
              onContextMenu={(e) => handlePortContextMenu(e, port, 'output')}
            />
          ))}
          {spec.input_ports.map((port, index) => (
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
          {spec.output_ports.map((port, index) => (
            <div
              key={`label-out-${port}`}
              className={clsx(
                'absolute right-0 flex items-center gap-1 justify-end',
                highlightedPorts.has(port) ? 'text-amber-600 font-medium' : 'text-slate-600'
              )}
              style={{
                top: rowCenterInBody(index),
                right: LABEL_OFFSET,
                transform: 'translateY(-50%)',
              }}
            >
              {highlightedPorts.has(port) && (
                <Crosshair className="w-3 h-3 text-amber-500" />
              )}
              <span>{port}</span>
            </div>
          ))}
        </div>
      )}

      {/* Port context menu */}
      {contextMenu && (
        <PortContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          nodeName={label}
          portName={contextMenu.portName}
          portType={contextMenu.portType}
          onClose={closeContextMenu}
        />
      )}
    </div>
  );
}
