import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type RefObject,
} from 'react';
import {
  Background,
  Controls,
  ControlButton,
  MiniMap,
  ReactFlow,
  Panel,
  useNodesInitialized,
  useReactFlow,
  type Connection,
  BackgroundVariant,
} from '@xyflow/react';
import { useGraphStore } from '@/stores/graphStore';
import { useLayoutStore } from '@/stores/layoutStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import {
  graphEdgeEntityId,
  graphNodeEntityId,
  probeEntityId,
  taskBindingEntityId,
} from '@/features/scenario/entities';
import {
  ensureTaskBindingSpec,
  targetInputOccupied,
} from '@/features/scenario/taskBindings';
import {
  hasBlockingSchemaIssue,
  projectStudioSchema,
  validateConnectionAgainstSchema,
} from '@/features/schema/project';
import { CustomNode } from './CustomNode';
import { SubgraphNode } from './SubgraphNode';
import { RoutedEdge } from './RoutedEdge';
import { StateFlowEdge } from './StateFlowEdge';
import { TapNode } from './TapNode';
import { useComponents } from '@/hooks/useComponents';
import clsx from 'clsx';
import type { GraphNodeData } from '@/types/graph';
import type { StudioTaskBinding } from '@/types/workspace';
import { ChevronsDown, ChevronsUp, Map as MapIcon, MoveDiagonal } from 'lucide-react';

const nodeTypes = {
  component: CustomNode,
  subgraph: SubgraphNode,
  tap: TapNode,
};

const edgeTypes = {
  routed: RoutedEdge,
  'state-flow': StateFlowEdge,
};

const DEFAULT_FIT_VIEW_OPTIONS = { padding: 0.22, maxZoom: 1 } as const;

export function Canvas() {
  const {
    graphId,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNodeFromComponent,
    setSelectedNode,
    setSelectedTap,
    setSelectedEdge,
    addTapForEdge,
    setAllNodesCollapsed,
    pendingStateMerge,
    confirmStateMerge,
    cancelStateMerge,
    graph,
    graphStack,
    currentGraphLabel,
    exitToBreadcrumb,
    wrapInParentGraph,
  } = useGraphStore();
  const { resizeMode, toggleResizeMode } = useLayoutStore();
  const showMinimap = useSettingsStore((state) => state.showMinimap);
  const toggleMinimap = useSettingsStore((state) => state.toggleMinimap);
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const hoverTopPaneEntity = useWorkspaceStore((state) => state.hoverTopPaneEntity);
  const workspace = useWorkspaceStore((state) => state.workspace);
  const { components } = useComponents();
  const reactFlow = useReactFlow();
  const nodesInitialized = useNodesInitialized();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const lastSize = useRef<{ width: number; height: number } | null>(null);
  const fittedGraphKey = useRef<string | null>(null);
  const trainingScenario = getTrainingScenario(workspace);
  const taskBindingSpec = useMemo(
    () => ensureTaskBindingSpec(trainingScenario?.task_binding_spec, graph),
    [graph, trainingScenario?.task_binding_spec]
  );
  const displayEdges = useMemo(
    () => edges,
    [edges]
  );

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) return;
      const { width, height } = entries[0].contentRect;
      const prev = lastSize.current;
      lastSize.current = { width, height };
      if (!prev || prev.width === 0 || prev.height === 0) return;
      if (width === prev.width && height === prev.height) return;
      const scaleX = width / prev.width;
      const scaleY = height / prev.height;
      const scale = Math.sqrt(scaleX * scaleY);
      if (!Number.isFinite(scale) || Math.abs(scale - 1) < 0.01) return;
      const viewport = reactFlow.getViewport();
      const newZoom = Math.max(0.1, Math.min(2.5, viewport.zoom * scale));
      const centerFlow = {
        x: (prev.width / 2 - viewport.x) / viewport.zoom,
        y: (prev.height / 2 - viewport.y) / viewport.zoom,
      };
      const nextX = width / 2 - centerFlow.x * newZoom;
      const nextY = height / 2 - centerFlow.y * newZoom;
      reactFlow.setViewport({ x: nextX, y: nextY, zoom: newZoom }, { duration: 0 });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [reactFlow]);

  const schemaRegistry = useMemo(
    () => projectStudioSchema(graph, components, taskBindingSpec),
    [components, graph, taskBindingSpec]
  );

  const breadcrumbs = useMemo(
    () => [...graphStack.map((layer) => layer.label), currentGraphLabel],
    [graphStack, currentGraphLabel]
  );
  const collapsibleNodes = useMemo(() => nodes.filter((node) => node.type !== 'tap'), [nodes]);
  const allNodesCollapsed =
    collapsibleNodes.length > 0 &&
    collapsibleNodes.every((node) => Boolean((node.data as GraphNodeData).collapsed));
  const CollapseIcon = allNodesCollapsed ? ChevronsUp : ChevronsDown;

  const graphViewKey = useMemo(
    () =>
      [graphId ?? 'inline', ...graphStack.map((layer) => layer.graphId ?? layer.label), currentGraphLabel].join(
        '/'
      ),
    [graphId, graphStack, currentGraphLabel]
  );

  useEffect(() => {
    if (!nodesInitialized || nodes.length === 0 || fittedGraphKey.current === graphViewKey) {
      return;
    }
    fittedGraphKey.current = graphViewKey;
    requestAnimationFrame(() => {
      reactFlow.fitView({ ...DEFAULT_FIT_VIEW_OPTIONS, duration: 0 });
    });
  }, [graphViewKey, nodes.length, nodesInitialized, reactFlow]);

  const isStateHandle = (handleId?: string | null) =>
    typeof handleId === 'string' && handleId.startsWith('__state');

  const isValidConnection = useCallback(
    (connection: Connection) => {
      if (!connection.target || !connection.targetHandle) return false;
      if (!connection.source || !connection.sourceHandle) return false;
      const sourceIsState = isStateHandle(connection.sourceHandle);
      const targetIsState = isStateHandle(connection.targetHandle);
      if (sourceIsState || targetIsState) {
        return (
          sourceIsState &&
          targetIsState &&
          connection.sourceHandle === '__state_out' &&
          connection.targetHandle === '__state_in'
        );
      }
      const inputTaken = targetInputOccupied(
        graph,
        taskBindingSpec,
        connection.target,
        connection.targetHandle
      );
      if (inputTaken) return false;
      return !hasBlockingSchemaIssue(
        validateConnectionAgainstSchema(
          schemaRegistry,
          connection.source,
          connection.sourceHandle,
          connection.target,
          connection.targetHandle
        )
      );
    },
    [graph, schemaRegistry, taskBindingSpec]
  );

  const handleConnect = useCallback(
    (connection: Connection) => {
      if (
        connection.target &&
        connection.targetHandle &&
        targetInputOccupied(graph, taskBindingSpec, connection.target, connection.targetHandle)
      ) {
        return;
      }
      if (
        connection.source &&
        connection.sourceHandle &&
        connection.target &&
        connection.targetHandle &&
        hasBlockingSchemaIssue(
          validateConnectionAgainstSchema(
            schemaRegistry,
            connection.source,
            connection.sourceHandle,
            connection.target,
            connection.targetHandle
          )
        )
      ) {
        return;
      }
      onConnect(connection);
    },
    [graph, onConnect, schemaRegistry, taskBindingSpec]
  );

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();
      const componentName = event.dataTransfer.getData('application/feedbax-component');
      if (!componentName) return;
      const component = components.find((item) => item.name === componentName);
      if (!component) return;
      if (component.category === 'Tasks') return;

      const position = reactFlow.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      addNodeFromComponent(component, position);
    },
    [addNodeFromComponent, reactFlow, components]
  );

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden bg-[radial-gradient(circle_at_top,_#ffffff_0%,_#f4f5f7_45%,_#eef1f6_100%)]"
    >
      <TaskBindingOverlay
        bindings={taskBindingSpec.bindings}
        containerRef={containerRef}
        onSelect={(binding) => {
          setSelectedEdge(null);
          setSelectedNode(null);
          setSelectedTap(null);
          selectTopPaneEntity(taskBindingEntityId(binding.id));
        }}
        onHover={(binding) => hoverTopPaneEntity(binding ? taskBindingEntityId(binding.id) : null)}
      />
      <ReactFlow
        className="relative z-10"
        style={{ zIndex: 10 }}
        nodes={nodes}
        edges={displayEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={handleConnect}
        isValidConnection={isValidConnection}
        onPaneClick={() => {
          setSelectedNode(null);
          setSelectedTap(null);
          setSelectedEdge(null);
          selectTopPaneEntity(null);
        }}
        onNodeClick={(_, node) => {
          if (node.type === 'tap') {
            const tapId = node.id.replace(/^tap:/, '');
            setSelectedTap(tapId);
            selectTopPaneEntity(probeEntityId(tapId));
          } else {
            setSelectedTap(null);
            setSelectedNode(node.id);
            setSelectedEdge(null);
            selectTopPaneEntity(graphNodeEntityId(node.id));
          }
        }}
        onEdgeClick={(_, edge) => {
          setSelectedEdge(edge.id);
          setSelectedNode(null);
          setSelectedTap(null);
          selectTopPaneEntity(graphEdgeEntityId(edge.id));
        }}
        onNodeMouseEnter={(_, node) => {
          hoverTopPaneEntity(
            node.type === 'tap'
              ? probeEntityId(node.id.replace(/^tap:/, ''))
              : graphNodeEntityId(node.id)
          );
        }}
        onNodeMouseLeave={() => hoverTopPaneEntity(null)}
        onEdgeMouseEnter={(_, edge) => {
          hoverTopPaneEntity(graphEdgeEntityId(edge.id));
        }}
        onEdgeMouseLeave={() => hoverTopPaneEntity(null)}
        onEdgeDoubleClick={(_, edge) => {
          if (edge.type === 'state-flow') {
            addTapForEdge(edge.id, 'probe');
          }
        }}
        onDrop={onDrop}
        onDragOver={onDragOver}
        snapToGrid
        snapGrid={[16, 16]}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#cbd5f5" />
        <Controls>
          <ControlButton
            onClick={() => setAllNodesCollapsed(!allNodesCollapsed)}
            title={allNodesCollapsed ? 'Expand all nodes' : 'Collapse all nodes'}
            aria-label={allNodesCollapsed ? 'Expand all nodes' : 'Collapse all nodes'}
            disabled={collapsibleNodes.length === 0}
            className="text-slate-500"
          >
            <CollapseIcon className="h-4 w-4" aria-hidden="true" />
          </ControlButton>
          <ControlButton
            onClick={toggleResizeMode}
            title={resizeMode ? 'Exit resize mode' : 'Enter resize mode'}
            aria-label={resizeMode ? 'Exit resize mode' : 'Enter resize mode'}
            className={resizeMode ? 'text-brand-600' : 'text-slate-500'}
          >
            <MoveDiagonal className="h-4 w-4" aria-hidden="true" />
          </ControlButton>
          <ControlButton
            onClick={toggleMinimap}
            title={showMinimap ? 'Hide minimap' : 'Show minimap'}
            aria-label={showMinimap ? 'Hide minimap' : 'Show minimap'}
            className={showMinimap ? 'text-brand-600' : 'text-slate-500'}
          >
            <MapIcon className="h-4 w-4" aria-hidden="true" />
          </ControlButton>
        </Controls>
        {showMinimap && <MiniMap nodeColor="#9ca3af" />}
        <Panel position="top-left" className="nodrag">
          <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-xs text-slate-500 shadow-soft">
            <button
              className="flex items-center justify-center rounded-full text-slate-500 hover:text-slate-700"
              onClick={wrapInParentGraph}
              title="Wrap this graph in a new parent"
            >
              <span className="text-xs font-mono">..</span>
            </button>
            <div className="h-4 w-px bg-slate-200" />
            {breadcrumbs.map((crumb, index) => {
              const isLast = index === breadcrumbs.length - 1;
              return (
                <div key={`${crumb}-${index}`} className="flex items-center gap-2">
                  <button
                    className={clsx(
                      'text-xs font-medium',
                      isLast ? 'text-slate-700' : 'text-brand-600 hover:text-brand-700'
                    )}
                    onClick={() => {
                      if (!isLast) {
                        exitToBreadcrumb(index);
                      }
                    }}
                    disabled={isLast}
                  >
                    {crumb}
                  </button>
                  {!isLast && <span className="text-slate-300">/</span>}
                </div>
              );
            })}
          </div>
        </Panel>
      </ReactFlow>
      {pendingStateMerge && (
        <StateMergeDialog
          request={pendingStateMerge}
          onCancel={cancelStateMerge}
          onConfirm={confirmStateMerge}
        />
      )}
    </div>
  );
}

function TaskBindingOverlay({
  bindings,
  containerRef,
  onSelect,
  onHover,
}: {
  bindings: StudioTaskBinding[];
  containerRef: RefObject<HTMLDivElement | null>;
  onSelect: (binding: StudioTaskBinding) => void;
  onHover: (binding: StudioTaskBinding | null) => void;
}) {
  const [paths, setPaths] = useState<
    Array<{
      binding: StudioTaskBinding;
      sourceX: number;
      sourceY: number;
      targetX: number;
      targetY: number;
    }>
  >([]);

  const selectorValue = useCallback(
    (value: string) => value.replace(/\\/g, '\\\\').replace(/"/g, '\\"'),
    []
  );

  const updatePaths = useCallback(() => {
    const container = containerRef.current;
    if (!container || bindings.length === 0) {
      setPaths([]);
      return;
    }

    const containerRect = container.getBoundingClientRect();
    const nextPaths = bindings.flatMap((binding, index) => {
      const nodeSelector = selectorValue(binding.target_node_id);
      const targetHandle =
        container.querySelector<HTMLElement>(
          `.react-flow__handle[data-nodeid="${nodeSelector}"][data-handleid="${selectorValue(
            binding.target_port
          )}"]`
        ) ??
        container.querySelector<HTMLElement>(
          `.react-flow__handle[data-nodeid="${nodeSelector}"][data-handleid="__state_in"]`
        );
      if (!targetHandle) return [];

      const targetRect = targetHandle.getBoundingClientRect();
      const sourcePort = document.querySelector<HTMLElement>(
        `[data-task-data-port-id="${selectorValue(binding.source_data_id)}"]`
      );
      const sourceRect = sourcePort?.getBoundingClientRect();
      const sourceY = sourceRect
        ? sourceRect.top + sourceRect.height / 2 - containerRect.top
        : 88 + index * 28;
      return [
        {
          binding,
          sourceX: 0,
          sourceY,
          targetX: targetRect.left + targetRect.width / 2 - containerRect.left,
          targetY: targetRect.top + targetRect.height / 2 - containerRect.top,
        },
      ];
    });
    setPaths(nextPaths);
  }, [bindings, containerRef, selectorValue]);

  useEffect(() => {
    updatePaths();
    const container = containerRef.current;
    if (!container) return undefined;

    let frame = 0;
    const schedule = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        frame = 0;
        updatePaths();
      });
    };

    const resizeObserver = new ResizeObserver(schedule);
    resizeObserver.observe(container);
    document
      .querySelectorAll<HTMLElement>('[data-task-data-port-id]')
      .forEach((element) => resizeObserver.observe(element));

    const viewportObserver = new MutationObserver(schedule);
    let observedViewport: HTMLElement | null = null;
    const attachViewportObserver = () => {
      const viewport = container.querySelector<HTMLElement>('.react-flow__viewport');
      if (!viewport || viewport === observedViewport) return;
      viewportObserver.disconnect();
      observedViewport = viewport;
      viewportObserver.observe(viewport, {
        attributes: true,
        childList: true,
        subtree: true,
        attributeFilter: ['style', 'class', 'transform'],
      });
    };
    attachViewportObserver();

    const rootObserver = new MutationObserver(() => {
      attachViewportObserver();
      schedule();
    });
    rootObserver.observe(container, {
      childList: true,
      subtree: true,
    });

    let startupPolls = 0;
    const startupInterval = window.setInterval(() => {
      attachViewportObserver();
      schedule();
      startupPolls += 1;
      if (startupPolls >= 20) {
        window.clearInterval(startupInterval);
      }
    }, 50);

    window.addEventListener('resize', schedule);
    window.addEventListener('scroll', schedule, true);
    container.addEventListener('pointermove', schedule);
    container.addEventListener('pointerup', schedule);
    container.addEventListener('wheel', schedule, { passive: true });

    return () => {
      if (frame) cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      viewportObserver.disconnect();
      rootObserver.disconnect();
      window.clearInterval(startupInterval);
      window.removeEventListener('resize', schedule);
      window.removeEventListener('scroll', schedule, true);
      container.removeEventListener('pointermove', schedule);
      container.removeEventListener('pointerup', schedule);
      container.removeEventListener('wheel', schedule);
    };
  }, [containerRef, updatePaths]);

  if (paths.length === 0) return null;

  return (
    <svg
      className="task-binding-overlay pointer-events-none absolute inset-0 h-full w-full"
      style={{ zIndex: 0 }}
    >
      {paths.map(({ binding, sourceX, sourceY, targetX, targetY }) => {
        const midX = sourceX + Math.max(48, (targetX - sourceX) / 2);
        const path = `M ${sourceX} ${sourceY} C ${midX} ${sourceY}, ${midX} ${targetY}, ${targetX} ${targetY}`;
        return (
          <g key={binding.id}>
            <path
              className="task-binding-edge"
              d={path}
              fill="none"
              stroke="#10b981"
              strokeWidth={3}
              pointerEvents="stroke"
              style={{ pointerEvents: 'stroke' }}
              onClick={(event) => {
                event.stopPropagation();
                onSelect(binding);
              }}
              onMouseEnter={() => onHover(binding)}
              onMouseLeave={() => onHover(null)}
            />
          </g>
        );
      })}
    </svg>
  );
}

function StateMergeDialog({
  request,
  onCancel,
  onConfirm,
}: {
  request: {
    sourceNode: string;
    targetNode: string;
    sourceOutputs: string[];
    targetInputs: string[];
    currentSources: Record<string, { source_node: string; source_port: string } | null>;
    suggested: Record<string, string | null>;
    hasExistingConnections: boolean;
  };
  onCancel: () => void;
  onConfirm: (mapping: Record<string, string>) => void;
}) {
  const buildInitial = useCallback(() => {
    const next: Record<string, { selected: boolean; output: string }> = {};
    for (const input of request.targetInputs) {
      const suggested = request.suggested[input];
      const defaultOutput = suggested ?? request.sourceOutputs[0] ?? '';
      const selected = !request.hasExistingConnections && Boolean(suggested);
      next[input] = { selected, output: defaultOutput };
    }
    return next;
  }, [request]);

  const [rows, setRows] = useState(buildInitial);

  useEffect(() => {
    setRows(buildInitial());
  }, [buildInitial]);

  const hasSelection = Object.values(rows).some((row) => row.selected && row.output);

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-900/20 backdrop-blur-sm">
      <div className="w-full max-w-2xl rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">State Merge</div>
        <div className="mt-2 text-lg font-semibold text-slate-800">
          Connect {request.sourceNode} → {request.targetNode}
        </div>
        <p className="mt-2 text-sm text-slate-500">
          Select which inputs should receive state from {request.sourceNode}. Unselected
          inputs keep their current wiring.
        </p>
        <div className="mt-4 space-y-2">
          <div className="grid grid-cols-[1.2fr_1fr_1.2fr_auto] gap-2 text-xs uppercase tracking-[0.2em] text-slate-400">
            <div>Input</div>
            <div>Current Source</div>
            <div>Wire From</div>
            <div />
          </div>
          {request.targetInputs.map((input) => {
            const current = request.currentSources[input];
            const row = rows[input];
            return (
              <div
                key={input}
                className="grid grid-cols-[1.2fr_1fr_1.2fr_auto] gap-2 items-center text-sm"
              >
                <div className="text-slate-700">{input}</div>
                <div className="text-slate-500">
                  {current ? `${current.source_node}.${current.source_port}` : '—'}
                </div>
                <select
                  value={row.output}
                  disabled={!row.selected || request.sourceOutputs.length === 0}
                  onChange={(event) =>
                    setRows((prev) => ({
                      ...prev,
                      [input]: { ...prev[input], output: event.target.value },
                    }))
                  }
                  className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700 disabled:bg-slate-50"
                >
                  {request.sourceOutputs.length === 0 && <option value="">No outputs</option>}
                  {request.sourceOutputs.map((output) => (
                    <option key={output} value={output}>
                      {request.sourceNode}.{output}
                    </option>
                  ))}
                </select>
                <label className="flex items-center gap-2 text-xs text-slate-500">
                  <input
                    type="checkbox"
                    checked={row.selected}
                    onChange={(event) =>
                      setRows((prev) => ({
                        ...prev,
                        [input]: { ...prev[input], selected: event.target.checked },
                      }))
                    }
                    className="h-4 w-4 rounded border-slate-300 text-brand-500 focus:ring-brand-500"
                  />
                  Use
                </label>
              </div>
            );
          })}
        </div>
        <div className="mt-6 flex items-center justify-end gap-3">
          <button
            className="rounded-full border border-slate-200 px-4 py-1.5 text-sm text-slate-600 hover:text-slate-800"
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            className="rounded-full bg-brand-600 px-4 py-1.5 text-sm text-white shadow-soft disabled:bg-slate-300"
            disabled={!hasSelection}
            onClick={() => {
              const mapping: Record<string, string> = {};
              for (const [input, row] of Object.entries(rows)) {
                if (row.selected && row.output) {
                  mapping[input] = row.output;
                }
              }
              onConfirm(mapping);
            }}
          >
            Connect
          </button>
        </div>
      </div>
    </div>
  );
}
