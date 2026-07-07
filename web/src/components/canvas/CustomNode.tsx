import { Handle, NodeResizer, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '@/types/graph';
import type { AnalysisNodeMeta } from '@/types/analysis';
import clsx from 'clsx';
import { useGraphStore } from '@/stores/graphStore';
import { useLayoutStore } from '@/stores/layoutStore';
import { useTrainingStore } from '@/stores/trainingStore';
import {
  getActiveStage,
  getScenario,
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { graphPortEntityId } from '@/features/scenario/entities';
import { objectiveGraphPortTarget } from '@/features/scenario/objectives';
import { ensureTaskBindingSpec, scopedTaskBindingSpec } from '@/features/scenario/taskBindings';
import { visibleMuxInputPorts } from '@/features/graph/dynamicPorts';
import { ArrowLeftRight, ExternalLink, Crosshair } from 'lucide-react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
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
const TASK_SOURCE_NODE_WIDTH = 112;
const TASK_SOURCE_NODE_GAP = 24;
const COLLAPSED_TASK_SOURCE_ROW_HEIGHT = 20;
const COLLAPSED_TASK_SOURCE_ROW_GAP = 4;
const COLLAPSED_STATE_HANDLE_CENTER_OFFSET = 4;

function CustomNodeComponent({ id, data, selected }: NodeProps) {
  const nodeData = data as GraphNodeData;
  const { spec, label, collapsed } = nodeData;
  const resizeMode = useLayoutStore((state) => state.resizeMode);
  const toggleNodeCollapse = useGraphStore((state) => state.toggleNodeCollapse);
  const toggleNodeReversed = useGraphStore((state) => state.toggleNodeReversed);
  const enterSubgraph = useGraphStore((state) => state.enterSubgraph);
  const graph = useGraphStore((state) => state.graph);
  const graphStack = useGraphStore((state) => state.graphStack);
  const hasSubgraph = useGraphStore((state) => Boolean(state.graph.subgraphs?.[label]));
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const setSelectedTap = useGraphStore((state) => state.setSelectedTap);
  const setSelectedEdge = useGraphStore((state) => state.setSelectedEdge);
  const highlightedProbeSelector = useTrainingStore((state) => state.highlightedProbeSelector);
  const workspace = useWorkspaceStore((state) => state.workspace);
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const hoverTopPaneEntity = useWorkspaceStore((state) => state.hoverTopPaneEntity);
  const topPane = getTopPaneState(workspace);
  const trainingScenario = getTrainingScenario(workspace);
  const currentGraphPath = useMemo(
    () => graphStack.map((layer) => layer.childNodeId).filter((item): item is string => Boolean(item)),
    [graphStack]
  );
  const rootGraph = graphStack.length > 0 ? graphStack[0].graph : graph;
  const allTaskBindingSpec = useMemo(
    () =>
      ensureTaskBindingSpec(
        trainingScenario?.task_binding_spec,
        rootGraph,
        trainingScenario?.task_spec
      ),
    [rootGraph, trainingScenario?.task_binding_spec, trainingScenario?.task_spec]
  );
  const taskBindingSpec = useMemo(
    () => scopedTaskBindingSpec(allTaskBindingSpec, currentGraphPath),
    [allTaskBindingSpec, currentGraphPath]
  );
  const taskBoundInputs = useMemo(() => {
    const taskDataById = new Map(taskBindingSpec.exposed_data.map((item) => [item.id, item]));
    const boundInputs = new Map<string, string>();
    for (const binding of taskBindingSpec.bindings) {
      if (binding.target_node_id !== label) continue;
      const taskData = taskDataById.get(binding.source_data_id);
      boundInputs.set(binding.target_port, taskData?.label ?? binding.source_data_id);
    }
    return boundInputs;
  }, [label, taskBindingSpec]);
  const showTaskSourceHints = topPane.active_projection === 'model';
  const objectivePorts = useMemo(() => {
    const activeScenario = getScenario(workspace, getActiveStage(workspace)?.scenario_id);
    const objectiveSpec = activeScenario?.objective_spec;
    if (!objectiveSpec || typeof objectiveSpec !== 'object' || !Array.isArray(objectiveSpec.terms)) {
      return new Set<string>();
    }
    const ports = new Set<string>();
    for (const term of objectiveSpec.terms) {
      const portTarget = objectiveGraphPortTarget(term?.source_selector);
      if (!portTarget || portTarget.nodeId !== label) continue;
      ports.add(`${portTarget.direction}:${portTarget.port}`);
    }
    return ports;
  }, [label, workspace]);

  const reversed = nodeData.reversed ?? false;

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
  const muxInputs = useMemo(
    () => visibleMuxInputPorts(graph, label, taskBindingSpec),
    [graph, label, taskBindingSpec]
  );
  const inputPorts = muxInputs?.ports ?? spec.input_ports;
  const nextMuxPort = muxInputs?.nextPort ?? null;
  const taskHintEntries = useMemo(
    () =>
      inputPorts.flatMap((port) => {
        const taskLabel = taskBoundInputs.get(port);
        return taskLabel ? [{ port, taskLabel }] : [];
      }),
    [inputPorts, taskBoundInputs]
  );
  const inputCount = inputPorts.length;
  const outputCount = spec.output_ports.length;
  const totalPorts = inputCount + outputCount;
  const canCollapse = totalPorts > 1;
  const collapsedEffective = collapsed && canCollapse;
  const collapsedTaskHintHeight =
    taskHintEntries.length * COLLAPSED_TASK_SOURCE_ROW_HEIGHT +
    Math.max(0, taskHintEntries.length - 1) * COLLAPSED_TASK_SOURCE_ROW_GAP;
  const collapsedTaskHintEndpointX = reversed
    ? COLLAPSED_STATE_HANDLE_CENTER_OFFSET
    : TASK_SOURCE_NODE_WIDTH + TASK_SOURCE_NODE_GAP - COLLAPSED_STATE_HANDLE_CENTER_OFFSET;
  const collapsedTaskHintStartX = reversed ? TASK_SOURCE_NODE_GAP : TASK_SOURCE_NODE_WIDTH;
  const connectedInputs = new Set(nodeData.connected_inputs ?? []);
  const connectedOutputs = new Set(nodeData.connected_outputs ?? []);
  const hasRecurrentSlots = (nodeData.state_slots ?? []).length > 0;
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

  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals(id);
  }, [collapsedEffective, height, id, inputPorts, reversed, updateNodeInternals, width]);

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
      setSelectedNode(label);
      setSelectedTap(null);
      setSelectedEdge(null);
      selectTopPaneEntity(graphPortEntityId(label, portType, portName), 'graph_port_selected');
      setContextMenu({
        x: event.clientX,
        y: event.clientY,
        portName,
        portType,
      });
    },
    [label, selectTopPaneEntity, setSelectedEdge, setSelectedNode, setSelectedTap]
  );

  const selectPort = useCallback(
    (
      event: React.MouseEvent,
      portName: string,
      direction: 'input' | 'output'
    ) => {
      event.preventDefault();
      event.stopPropagation();
      setSelectedNode(label);
      setSelectedTap(null);
      setSelectedEdge(null);
      selectTopPaneEntity(graphPortEntityId(label, direction, portName), 'graph_port_selected');
    },
    [label, selectTopPaneEntity, setSelectedEdge, setSelectedNode, setSelectedTap]
  );

  const hoverPort = useCallback(
    (portName: string, direction: 'input' | 'output') => {
      hoverTopPaneEntity(graphPortEntityId(label, direction, portName));
    },
    [hoverTopPaneEntity, label]
  );

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // Check for analysis metadata indicating this node produces figures
  const analysisMeta = spec.params?._analysis_meta as unknown as AnalysisNodeMeta | undefined;
  const hasFigureOutput = analysisMeta?.has_make_figs ?? false;
  // Position the figure pin below the last output port
  const figPinOffset = HEADER_HEIGHT + BODY_PADDING + (outputCount > 0 ? outputCount * ROW_HEIGHT : ROW_HEIGHT) + 8;
  return (
    <div
      className={clsx(
        'relative rounded-xl border-2 shadow-soft bg-white/90 backdrop-blur transition-all duration-150',
        selected ? 'border-brand-500 ring-2 ring-brand-500/30' : 'border-slate-200',
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
          zIndex: 40,
        }}
        className="w-2 h-2 z-20 border-2 border-white shadow-soft cursor-crosshair bg-slate-600"
      />
      {collapsedEffective && showTaskSourceHints && taskHintEntries.length > 0 && (
        <div
          className="pointer-events-none absolute z-[1]"
          style={{
            top: HEADER_HEIGHT / 2,
            [reversed ? 'right' : 'left']: -(TASK_SOURCE_NODE_WIDTH + TASK_SOURCE_NODE_GAP),
            width: TASK_SOURCE_NODE_WIDTH + TASK_SOURCE_NODE_GAP,
            height: collapsedTaskHintHeight,
            transform: 'translateY(-50%)',
          }}
          aria-hidden="true"
        >
          <svg
            className="absolute inset-0 overflow-visible"
            width={TASK_SOURCE_NODE_WIDTH + TASK_SOURCE_NODE_GAP}
            height={collapsedTaskHintHeight}
            viewBox={`0 0 ${TASK_SOURCE_NODE_WIDTH + TASK_SOURCE_NODE_GAP} ${collapsedTaskHintHeight}`}
            aria-hidden="true"
          >
            {taskHintEntries.map(({ port }, index) => {
              const y =
                index * (COLLAPSED_TASK_SOURCE_ROW_HEIGHT + COLLAPSED_TASK_SOURCE_ROW_GAP) +
                COLLAPSED_TASK_SOURCE_ROW_HEIGHT / 2;
              const controlOffset = Math.max(14, TASK_SOURCE_NODE_GAP * 0.8);
              const path = reversed
                ? `M ${collapsedTaskHintStartX} ${y} C ${TASK_SOURCE_NODE_GAP * 0.45} ${y}, ${controlOffset} ${collapsedTaskHintHeight / 2}, ${collapsedTaskHintEndpointX} ${collapsedTaskHintHeight / 2}`
                : `M ${collapsedTaskHintStartX} ${y} C ${
                    TASK_SOURCE_NODE_WIDTH + TASK_SOURCE_NODE_GAP * 0.55
                  } ${y}, ${
                    collapsedTaskHintEndpointX - controlOffset
                  } ${collapsedTaskHintHeight / 2}, ${collapsedTaskHintEndpointX} ${
                    collapsedTaskHintHeight / 2
                  }`;
              return (
                <path
                  key={`collapsed-task-source-path-${port}`}
                  d={path}
                  fill="none"
                  stroke="#6ee7b7"
                  strokeWidth={1.5}
                  strokeLinecap="round"
                />
              );
            })}
          </svg>
          {taskHintEntries.map(({ port, taskLabel }, index) => (
            <div
              key={`collapsed-task-source-${port}`}
              className="absolute max-w-[112px] truncate rounded-md border border-emerald-300 bg-white px-2 py-1 text-[10px] font-semibold leading-none text-emerald-700 shadow-soft"
              style={{
                top: index * (COLLAPSED_TASK_SOURCE_ROW_HEIGHT + COLLAPSED_TASK_SOURCE_ROW_GAP),
                [reversed ? 'right' : 'left']: 0,
                width: TASK_SOURCE_NODE_WIDTH,
              }}
            >
              {taskLabel}
            </div>
          ))}
        </div>
      )}
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
          zIndex: 40,
        }}
        className="w-2 h-2 z-20 border-2 border-white shadow-soft cursor-crosshair bg-slate-600"
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
          <div className="min-w-0 flex-1 flex items-center gap-2 pr-2">
            {hasRecurrentSlots && (
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full bg-sky-500"
                title="Has recurrent value slots"
              />
            )}
            <div className="text-sm font-medium text-slate-800 truncate w-full" title={label}>
              {label}
            </div>
          </div>
        ) : (
          <div className="min-w-0 flex-1 flex items-center gap-2 pr-2">
            {hasRecurrentSlots && (
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full bg-sky-500"
                title="Has recurrent value slots"
              />
            )}
            <div className="text-sm font-medium text-slate-800 truncate w-full" title={label}>
              {label}
            </div>
          </div>
        )}
        {/* Right slot: action icons + type (normal) or name+chevron (reversed) */}
        <div className={clsx('flex items-center gap-2 shrink-0', reversed && 'ml-auto')}>
          <button
            className="shrink-0 text-slate-400 hover:text-brand-600"
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
              className="shrink-0 text-slate-400 hover:text-brand-600"
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
            !collapsedEffective && (
              <div className="text-[11px] text-slate-500 truncate max-w-[110px]" title={spec.type}>
                {spec.type}
              </div>
            )
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
          {inputPorts.map((port, index) => {
            const isDynamicMuxPort = port === nextMuxPort;
            return (
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
                  zIndex: 40,
                }}
                className={clsx(
                  'w-2 h-2 z-20 border border-white shadow-soft transition-all duration-150 bg-slate-400',
                  isDynamicMuxPort && 'bg-white ring-1 ring-slate-300 border-slate-300',
                  taskBoundInputs.has(port) && 'bg-emerald-500 ring-2 ring-emerald-200',
                  objectivePorts.has(`input:${port}`) && 'bg-violet-500 ring-2 ring-violet-200',
                  topPane.selected_entity_id === graphPortEntityId(label, 'input', port) &&
                    'bg-brand-600 ring-4 ring-brand-300 scale-150'
                )}
                onContextMenu={(e) => handlePortContextMenu(e, port, 'input')}
              />
            );
          })}
          {showTaskSourceHints &&
            inputPorts.map((port, index) => {
              const taskLabel = taskBoundInputs.get(port);
              if (!taskLabel) return null;
              const side = reversed ? 'right' : 'left';
              return (
                <div
                  key={`task-source-${port}`}
                  className="pointer-events-none absolute z-[1] flex items-center"
                  style={{
                    top: rowCenterInBody(index),
                    [side]: -(TASK_SOURCE_NODE_WIDTH + TASK_SOURCE_NODE_GAP),
                    transform: 'translateY(-50%)',
                    width: TASK_SOURCE_NODE_WIDTH + TASK_SOURCE_NODE_GAP,
                  }}
                  aria-hidden="true"
                >
                  {reversed ? (
                    <>
                      <div className="h-px flex-1 bg-emerald-400" />
                      <div className="max-w-[112px] truncate rounded-md border border-emerald-300 bg-white px-2 py-1 text-[10px] font-semibold leading-none text-emerald-700 shadow-soft">
                        {taskLabel}
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="max-w-[112px] truncate rounded-md border border-emerald-300 bg-white px-2 py-1 text-[10px] font-semibold leading-none text-emerald-700 shadow-soft">
                        {taskLabel}
                      </div>
                      <div className="h-px flex-1 bg-emerald-400" />
                    </>
                  )}
                </div>
              );
            })}
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
                zIndex: 40,
              }}
              className={clsx(
                'w-2 h-2 z-20 border border-white shadow-soft transition-all duration-150 bg-slate-400',
                objectivePorts.has(`output:${port}`) && 'bg-violet-500 ring-2 ring-violet-200',
                highlightedPorts.has(port) && 'bg-amber-400 ring-2 ring-amber-200 scale-125',
                topPane.selected_entity_id === graphPortEntityId(label, 'output', port) &&
                  'bg-brand-600 ring-4 ring-brand-300 scale-150'
              )}
              onContextMenu={(e) => handlePortContextMenu(e, port, 'output')}
            />
          ))}
          {inputPorts.map((port, index) => {
            const isDynamicMuxPort = port === nextMuxPort;
            return (
              <button
                key={`label-in-${port}`}
                type="button"
                onClick={(event) => selectPort(event, port, 'input')}
                onMouseEnter={() => hoverPort(port, 'input')}
                onMouseLeave={() => hoverTopPaneEntity(null)}
                className={clsx(
                  'nodrag nopan absolute flex items-center gap-2 rounded px-1 py-0.5 text-slate-600 hover:bg-brand-50 hover:text-brand-700',
                  isDynamicMuxPort && 'text-slate-400',
                  objectivePorts.has(`input:${port}`) && 'font-semibold text-violet-700',
                  topPane.selected_entity_id === graphPortEntityId(label, 'input', port) &&
                    'bg-brand-100 font-semibold text-brand-800 ring-[3px] ring-brand-300 shadow-sm',
                  reversed && 'flex-row-reverse'
                )}
                style={{
                  top: rowCenterInBody(index),
                  [reversed ? 'right' : 'left']: LABEL_OFFSET,
                  transform: 'translateY(-50%)',
                }}
                title={
                  isDynamicMuxPort
                    ? `Connect to add ${label}.${port}`
                    : `Select ${label}.${port}`
                }
                onContextMenu={(event) => handlePortContextMenu(event, port, 'input')}
              >
                {objectivePorts.has(`input:${port}`) && (
                  <Crosshair className="w-3 h-3 text-violet-500" />
                )}
                <span>{port}</span>
              </button>
            );
          })}
          {spec.output_ports.map((port, index) => (
            <button
              key={`label-out-${port}`}
              type="button"
              onClick={(event) => selectPort(event, port, 'output')}
              onMouseEnter={() => hoverPort(port, 'output')}
              onMouseLeave={() => hoverTopPaneEntity(null)}
              className={clsx(
                'nodrag nopan absolute flex items-center gap-1 rounded px-1 py-0.5 hover:bg-brand-50 hover:text-brand-700',
                reversed ? 'justify-start' : 'justify-end',
                objectivePorts.has(`output:${port}`) && 'font-semibold text-violet-700',
                highlightedPorts.has(port) ? 'text-amber-600 font-medium' : 'text-slate-600',
                topPane.selected_entity_id === graphPortEntityId(label, 'output', port) &&
                  'bg-brand-100 font-semibold text-brand-800 ring-[3px] ring-brand-300 shadow-sm'
              )}
              style={{
                top: rowCenterInBody(index),
                [reversed ? 'left' : 'right']: LABEL_OFFSET,
                transform: 'translateY(-50%)',
              }}
              title={`Select ${label}.${port}`}
              onContextMenu={(event) => handlePortContextMenu(event, port, 'output')}
            >
              {(highlightedPorts.has(port) || objectivePorts.has(`output:${port}`)) && (
                <Crosshair
                  className={clsx(
                    'w-3 h-3',
                    highlightedPorts.has(port) ? 'text-amber-500' : 'text-violet-500'
                  )}
                />
              )}
              <span>{port}</span>
            </button>
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

function areCustomNodePropsEqual(previous: NodeProps, next: NodeProps) {
  return (
    previous.id === next.id &&
    previous.data === next.data &&
    previous.selected === next.selected &&
    previous.dragging === next.dragging &&
    previous.isConnectable === next.isConnectable
  );
}

export const CustomNode = memo(CustomNodeComponent, areCustomNodePropsEqual);
