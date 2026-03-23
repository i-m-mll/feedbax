import { Handle, NodeResizer, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '@/types/graph';
import type { AnalysisNodeMeta } from '@/types/analysis';
import clsx from 'clsx';
import { useGraphStore } from '@/stores/graphStore';
import { useLayoutStore } from '@/stores/layoutStore';
import { useTrainingStore } from '@/stores/trainingStore';
import { ArrowLeftRight, ExternalLink, Crosshair } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { PortContextMenu } from './PortContextMenu';
import { FigureOutputPin } from '@/components/analysis/FigureOutputPin';

const DEFAULT_WIDTH = 220;
const HEADER_HEIGHT = 40;
const BODY_PADDING = 12;
const ROW_HEIGHT = 26;
const LABEL_OFFSET = 22;
const HANDLE_OFFSET = -6;
const MIN_WIDTH = 180;
const MIN_HEIGHT = 96;

export function CustomNode({ id, data, selected }: NodeProps) {
  const nodeData = data as GraphNodeData;
  const { spec, label, collapsed } = nodeData;
  const resizeMode = useLayoutStore((state) => state.resizeMode);
  const toggleNodeCollapse = useGraphStore((state) => state.toggleNodeCollapse);
  const toggleNodeReversed = useGraphStore((state) => state.toggleNodeReversed);
  const enterSubgraph = useGraphStore((state) => state.enterSubgraph);
  const hasSubgraph = useGraphStore((state) => Boolean(state.graph.subgraphs?.[label]));
  const highlightedProbeSelector = useTrainingStore((state) => state.highlightedProbeSelector);

  const reversed = nodeData.reversed ?? false;

  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, reversed, updateNodeInternals]);

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

  // Check for analysis metadata indicating this node produces figures
  const analysisMeta = spec.params?._analysis_meta as AnalysisNodeMeta | undefined;
  const hasFigureOutput = analysisMeta?.has_make_figs ?? false;
  // Position the figure pin below the last output port
  const figPinOffset = HEADER_HEIGHT + BODY_PADDING + (outputCount > 0 ? outputCount * ROW_HEIGHT : ROW_HEIGHT) + 8;
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
        position={reversed ? Position.Right : Position.Left}
        id="__state_in"
        style={{
          top: HEADER_HEIGHT / 2,
          [reversed ? 'right' : 'left']: HANDLE_OFFSET - 2,
          transform: 'translateY(-50%)',
          clipPath: reversed
            ? 'polygon(100% 0%, 0% 50%, 100% 100%)'
            : 'polygon(0% 0%, 100% 50%, 0% 100%)',
          width: '8px',
          height: '8px',
        }}
        className="w-2 h-2 border-2 border-white shadow-soft cursor-crosshair bg-slate-600"
      />
      <Handle
        type="source"
        position={reversed ? Position.Left : Position.Right}
        id="__state_out"
        style={{
          top: HEADER_HEIGHT / 2,
          [reversed ? 'left' : 'right']: HANDLE_OFFSET - 2,
          transform: 'translateY(-50%)',
          clipPath: reversed
            ? 'polygon(100% 0%, 0% 50%, 100% 100%)'
            : 'polygon(0% 0%, 100% 50%, 0% 100%)',
          width: '8px',
          height: '8px',
        }}
        className="w-2 h-2 border-2 border-white shadow-soft cursor-crosshair bg-slate-600"
      />
      <div
        className={clsx(
          'px-3 py-2 bg-slate-50/70 flex items-center gap-3 overflow-hidden',
          collapsedEffective ? 'rounded-xl' : 'border-b border-slate-100 rounded-t-xl'
        )}
        onDoubleClick={(event) => {
          event.stopPropagation();
          if (isComposite) {
            enterSubgraph(label);
          }
        }}
      >
        {/* Left slot: name (normal) or type string (reversed) */}
        {reversed ? (
          !collapsedEffective && (
            <div className="text-[11px] text-slate-500 shrink-0 truncate max-w-[110px]" title={spec.type}>
              {spec.type}
            </div>
          )
        ) : (
          <div className="min-w-0 flex-1 flex items-center gap-2 pr-2">
            <div className="text-sm font-medium text-slate-800 truncate w-full" title={label}>
              {label}
            </div>
          </div>
        )}
        {/* Right slot: action icons + type (normal) or name+chevron (reversed) */}
        <div className={clsx('flex items-center gap-2 shrink-0', reversed && 'ml-auto')}>
          <button
            className="text-slate-400 hover:text-brand-600"
            onClick={(event) => {
              event.stopPropagation();
              toggleNodeReversed(label);
            }}
            title={reversed ? 'Restore default direction' : 'Reverse node direction'}
          >
            <ArrowLeftRight className="w-3.5 h-3.5" />
          </button>
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
          {reversed ? (
            <>
              <div className="text-sm font-medium text-slate-800 truncate" title={label}>
                {label}
              </div>
            </>
          ) : (
            !collapsedEffective && (
              <div className="text-[11px] text-slate-500 truncate max-w-[110px]" title={spec.type}>
                {spec.type}
              </div>
            )
          )}
        </div>
      </div>

      {collapsedEffective ? null : (
        <div className="relative text-xs text-slate-600" style={{ height: bodyHeight, padding: BODY_PADDING }}>
          {spec.input_ports.map((port, index) => (
            <Handle
              key={`handle-in-${port}`}
              type="target"
              position={reversed ? Position.Right : Position.Left}
              id={port}
              style={{
                top: rowCenterInBody(index),
                [reversed ? 'right' : 'left']: HANDLE_OFFSET,
                transform: 'translateY(-50%)',
                clipPath: reversed
                  ? 'polygon(100% 0%, 0% 50%, 100% 100%)'
                  : 'polygon(0% 0%, 100% 50%, 0% 100%)',
                width: '8px',
                height: '8px',
              }}
              className="w-2 h-2 z-20 border border-white shadow-soft bg-slate-400"
            />
          ))}
          {spec.output_ports.map((port, index) => (
            <Handle
              key={`handle-out-${port}`}
              type="source"
              position={reversed ? Position.Left : Position.Right}
              id={port}
              style={{
                top: rowCenterInBody(index),
                [reversed ? 'left' : 'right']: HANDLE_OFFSET,
                transform: 'translateY(-50%)',
                clipPath: reversed
                  ? 'polygon(100% 0%, 0% 50%, 100% 100%)'
                  : 'polygon(0% 0%, 100% 50%, 0% 100%)',
                width: '8px',
                height: '8px',
              }}
              className={clsx(
                'w-2 h-2 z-20 border border-white shadow-soft transition-all duration-150 bg-slate-400',
                highlightedPorts.has(port) && 'bg-amber-400 ring-2 ring-amber-200 scale-125'
              )}
              onContextMenu={(e) => handlePortContextMenu(e, port, 'output')}
            />
          ))}
          {spec.input_ports.map((port, index) => (
            <div
              key={`label-in-${port}`}
              className={clsx(
                'absolute flex items-center gap-2 text-slate-600',
                reversed && 'flex-row-reverse'
              )}
              style={{
                top: rowCenterInBody(index),
                [reversed ? 'right' : 'left']: LABEL_OFFSET,
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
                'absolute flex items-center gap-1',
                reversed ? 'justify-start' : 'justify-end',
                highlightedPorts.has(port) ? 'text-amber-600 font-medium' : 'text-slate-600'
              )}
              style={{
                top: rowCenterInBody(index),
                [reversed ? 'left' : 'right']: LABEL_OFFSET,
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

      {/* Figure output pin for analysis nodes */}
      {hasFigureOutput && !collapsedEffective && (
        <FigureOutputPin
          nodeId={label}
          topOffset={figPinOffset}
          reversed={reversed}
        />
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
