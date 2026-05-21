import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type RefObject,
} from 'react';
import { flushSync } from 'react-dom';
import {
  Background,
  Controls,
  ControlButton,
  Handle,
  MiniMap,
  ReactFlow,
  Panel,
  Position,
  useNodesInitialized,
  useReactFlow,
  useViewport,
  type Connection,
  type Edge,
  type EdgeChange,
  type EdgeProps,
  type Node,
  type NodeProps,
  getBezierPath,
  BackgroundVariant,
} from '@xyflow/react';
import { useGraphStore } from '@/stores/graphStore';
import { useLayoutStore } from '@/stores/layoutStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { getTopPaneState, getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import {
  graphEdgeEntityId,
  graphNodeEntityId,
  probeEntityId,
  taskBindingEntityId,
} from '@/features/scenario/entities';
import {
  ensureTaskBindingSpec,
  taskBindingId,
  targetInputOccupied,
} from '@/features/scenario/taskBindings';
import {
  hasBlockingSchemaIssue,
  projectStudioSchema,
  validateConnectionAgainstSchema,
} from '@/features/schema/project';
import { isNextMuxInputPort } from '@/features/graph/dynamicPorts';
import { CustomNode } from './CustomNode';
import { SubgraphNode } from './SubgraphNode';
import { RoutedEdge } from './RoutedEdge';
import { StateFlowEdge } from './StateFlowEdge';
import { TapNode } from './TapNode';
import { useComponents } from '@/hooks/useComponents';
import clsx from 'clsx';
import type { GraphEdgeData, GraphNodeData, TapNodeData } from '@/types/graph';
import type { StudioTaskBinding } from '@/types/workspace';
import { ChevronsDown, ChevronsUp, Map as MapIcon, MoveDiagonal } from 'lucide-react';

interface TaskSourceNodeData extends Record<string, unknown> {
  label: string;
  handleSize: number;
  beginTaskConnection?: () => void;
  prepareTaskConnection?: () => void;
  releaseTaskConnection?: () => void;
}

interface TaskBindingEdgeData extends Record<string, unknown> {
  task_binding_id?: string;
  source_data_id?: string;
  target_node_id?: string;
  target_port?: string;
}

function TaskSourceNode({ data }: NodeProps) {
  const nodeData = data as TaskSourceNodeData;
  const handleSize = Number(nodeData.handleSize) || TASK_SOURCE_HANDLE_SCREEN_SIZE;
  return (
    <div
      className="relative"
      style={{ width: handleSize, height: handleSize }}
      aria-label={`${nodeData.label} task data source`}
      title={`${nodeData.label} task data source`}
      onPointerDownCapture={nodeData.beginTaskConnection}
      onPointerEnter={nodeData.prepareTaskConnection}
      onPointerLeave={nodeData.releaseTaskConnection}
    >
      <Handle
        type="source"
        position={Position.Right}
        id="out"
        className="cursor-crosshair border-0 bg-transparent opacity-0"
        style={{
          left: 0,
          top: 0,
          width: handleSize,
          height: handleSize,
          transform: 'none',
        }}
      />
    </div>
  );
}

function TaskBindingEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });
  const edgeData = data as TaskBindingEdgeData | undefined;
  const [taskPath, setTaskPath] = useState<string | null>(null);
  const bindingKey =
    edgeData?.task_binding_id &&
    edgeData.source_data_id &&
    edgeData.target_node_id &&
    edgeData.target_port
      ? [
          edgeData.task_binding_id,
          edgeData.source_data_id,
          edgeData.target_node_id,
          edgeData.target_port,
        ].join(':')
      : null;

  useEffect(() => {
    if (
      !bindingKey ||
      !edgeData?.source_data_id ||
      !edgeData.target_node_id ||
      !edgeData.target_port
    ) {
      setTaskPath(null);
      return undefined;
    }
    const binding: Pick<
      StudioTaskBinding,
      'source_data_id' | 'target_node_id' | 'target_port'
    > = {
      source_data_id: edgeData.source_data_id,
      target_node_id: edgeData.target_node_id,
      target_port: edgeData.target_port,
    };
    const container = document.querySelector<HTMLElement>('[data-studio-canvas-root="true"]');
    if (!container) {
      setTaskPath(null);
      return undefined;
    }

    let frame = 0;
    let active = true;
    let lastPath: string | null = null;
    const update = () => {
      const nextPath = taskBindingFlowPath(container, binding);
      if (nextPath !== lastPath) {
        lastPath = nextPath;
        setTaskPath(nextPath);
      }
      if (active) frame = requestAnimationFrame(update);
    };
    update();
    return () => {
      active = false;
      if (frame) cancelAnimationFrame(frame);
    };
  }, [bindingKey, edgeData?.source_data_id, edgeData?.target_node_id, edgeData?.target_port]);

  return (
    <path
      className="react-flow__edge-path"
      d={taskPath ?? path}
      fill="none"
      stroke="rgba(0, 0, 0, 0)"
      style={{
        stroke: 'rgba(0, 0, 0, 0)',
        strokeWidth: 22,
        pointerEvents: 'stroke',
      }}
      strokeWidth={22}
      strokeLinecap="round"
      pointerEvents="stroke"
    />
  );
}

const nodeTypes = {
  component: CustomNode,
  subgraph: SubgraphNode,
  tap: TapNode,
  taskSource: TaskSourceNode,
};

const edgeTypes = {
  routed: RoutedEdge,
  'state-flow': StateFlowEdge,
  taskBinding: TaskBindingEdge,
};

const DEFAULT_FIT_VIEW_OPTIONS = { padding: 0.22, maxZoom: 1 } as const;
const TASK_BINDING_ENTITY_PREFIX = 'task_binding:';
const TASK_SOURCE_NODE_PREFIX = '__task_data_source__:';
const TASK_BINDING_EDGE_PREFIX = '__task_binding_edge__:';
const TASK_SOURCE_HANDLE_SCREEN_SIZE = 10;
const TASK_CONNECT_AUTOPAN_EDGE_DISTANCE = 36;
// Keep arming outside the edge pan band so leaving the task-port dead-zone cannot twitch.
const TASK_CONNECT_AUTOPAN_ARM_DISTANCE = TASK_CONNECT_AUTOPAN_EDGE_DISTANCE + 24;
const TASK_CONNECT_AUTOPAN_MAX_SPEED = 16;

function taskSourceNodeId(dataId: string): string {
  return `${TASK_SOURCE_NODE_PREFIX}${dataId}`;
}

function taskDataIdFromSourceNodeId(nodeId: string | null | undefined): string | null {
  return nodeId?.startsWith(TASK_SOURCE_NODE_PREFIX)
    ? nodeId.slice(TASK_SOURCE_NODE_PREFIX.length)
    : null;
}

function taskBindingEdgeId(bindingId: string): string {
  return `${TASK_BINDING_EDGE_PREFIX}${bindingId}`;
}

function taskBindingIdFromEdgeId(edgeId: string | null | undefined): string | null {
  return edgeId?.startsWith(TASK_BINDING_EDGE_PREFIX)
    ? edgeId.slice(TASK_BINDING_EDGE_PREFIX.length)
    : null;
}

function cssSelectorValue(value: string): string {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(value);
  }
  return value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

function screenToCanvasFlowPosition(
  point: { x: number; y: number },
  container: HTMLElement | null
): { x: number; y: number } {
  const canvas =
    container ?? document.querySelector<HTMLElement>('[data-studio-canvas-root="true"]');
  const viewportElement = canvas?.querySelector<HTMLElement>('.react-flow__viewport');
  const canvasRect = canvas?.getBoundingClientRect();
  if (!canvasRect || !viewportElement) return point;

  const transform = getComputedStyle(viewportElement).transform;
  const matrix =
    transform && transform !== 'none'
      ? new DOMMatrixReadOnly(transform)
      : new DOMMatrixReadOnly();
  const scaleX = matrix.a || 1;
  const scaleY = matrix.d || 1;
  return {
    x: (point.x - canvasRect.left - matrix.e) / scaleX,
    y: (point.y - canvasRect.top - matrix.f) / scaleY,
  };
}

type TaskBindingEndpoint = Pick<
  StudioTaskBinding,
  'source_data_id' | 'target_node_id' | 'target_port'
>;

interface TaskBindingCurve {
  source: { x: number; y: number };
  control1: { x: number; y: number };
  control2: { x: number; y: number };
  target: { x: number; y: number };
}

function taskBindingCurve(
  container: HTMLElement,
  binding: TaskBindingEndpoint
): TaskBindingCurve | null {
  const containerRect = container.getBoundingClientRect();
  const sourcePort = document.querySelector<HTMLElement>(
    `[data-task-data-port-id="${cssSelectorValue(binding.source_data_id)}"]`
  );
  const targetHandle =
    container.querySelector<HTMLElement>(
      `.react-flow__handle[data-nodeid="${cssSelectorValue(
        binding.target_node_id
      )}"][data-handleid="${cssSelectorValue(binding.target_port)}"]`
    ) ??
    container.querySelector<HTMLElement>(
      `.react-flow__handle[data-nodeid="${cssSelectorValue(
        binding.target_node_id
      )}"][data-handleid="__state_in"]`
    );
  if (!sourcePort || !targetHandle) return null;

  const sourceRect = sourcePort.getBoundingClientRect();
  const targetRect = targetHandle.getBoundingClientRect();
  const sourceX = Math.max(
    containerRect.left + 1,
    Math.min(containerRect.right - 1, sourceRect.left + sourceRect.width / 2)
  );
  const sourceY = sourceRect.top + sourceRect.height / 2;
  const targetX = targetRect.left + targetRect.width / 2;
  const targetY = targetRect.top + targetRect.height / 2;
  const controlOffset = Math.max(48, Math.abs(targetX - sourceX) * 0.45);
  return {
    source: { x: sourceX, y: sourceY },
    control1: { x: sourceX + controlOffset, y: sourceY },
    control2: { x: targetX - controlOffset, y: targetY },
    target: { x: targetX, y: targetY },
  };
}

function curvePath(curve: TaskBindingCurve): string {
  return `M ${curve.source.x} ${curve.source.y} C ${curve.control1.x} ${curve.control1.y}, ${
    curve.control2.x
  } ${curve.control2.y}, ${curve.target.x} ${curve.target.y}`;
}

function taskBindingPath(
  container: HTMLElement,
  binding: TaskBindingEndpoint
): string | null {
  const containerRect = container.getBoundingClientRect();
  const curve = taskBindingCurve(container, binding);
  if (!curve) return null;
  return curvePath({
    source: {
      x: curve.source.x - containerRect.left,
      y: curve.source.y - containerRect.top,
    },
    control1: {
      x: curve.control1.x - containerRect.left,
      y: curve.control1.y - containerRect.top,
    },
    control2: {
      x: curve.control2.x - containerRect.left,
      y: curve.control2.y - containerRect.top,
    },
    target: {
      x: curve.target.x - containerRect.left,
      y: curve.target.y - containerRect.top,
    },
  });
}

function taskBindingFlowPath(
  container: HTMLElement,
  binding: TaskBindingEndpoint
): string | null {
  const curve = taskBindingCurve(container, binding);
  if (!curve) return null;
  return curvePath({
    source: screenToCanvasFlowPosition(curve.source, container),
    control1: screenToCanvasFlowPosition(curve.control1, container),
    control2: screenToCanvasFlowPosition(curve.control2, container),
    target: screenToCanvasFlowPosition(curve.target, container),
  });
}

function TaskBindingVisualOverlay({
  bindings,
  selectedBindingId,
  containerRef,
}: {
  bindings: StudioTaskBinding[];
  selectedBindingId: string | null;
  containerRef: RefObject<HTMLDivElement | null>;
}) {
  const bindingKey = bindings
    .map(
      (binding) =>
        `${binding.id}:${binding.source_data_id}:${binding.target_node_id}:${binding.target_port}`
    )
    .join('|');

  useEffect(() => {
    const container = containerRef.current;
    if (!container || bindings.length === 0) return undefined;

    let frame = 0;
    let active = true;
    const update = () => {
      for (const binding of bindings) {
        const path = taskBindingPath(container, binding);
        container
          .querySelectorAll<SVGPathElement>(
            `[data-task-binding-visual-id="${cssSelectorValue(binding.id)}"]`
          )
          .forEach((element) => {
            if (path) {
              element.setAttribute('d', path);
              element.style.display = '';
            } else {
              element.style.display = 'none';
            }
          });
      }
    };
    const tick = () => {
      update();
      if (active) frame = requestAnimationFrame(tick);
    };
    update();
    const interval = window.setInterval(update, 100);
    frame = requestAnimationFrame(tick);
    return () => {
      active = false;
      if (frame) cancelAnimationFrame(frame);
      window.clearInterval(interval);
    };
  }, [bindingKey, bindings, containerRef]);

  if (bindings.length === 0) return null;

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    >
      {bindings.map((binding) => {
        const selected = selectedBindingId === binding.id;
        return (
          <g key={binding.id}>
            <path
              data-task-binding-visual-id={binding.id}
              fill="none"
              stroke={selected ? '#bbf7d0' : 'transparent'}
              strokeWidth={9}
              strokeLinecap="round"
            />
            <path
              data-task-binding-visual-id={binding.id}
              fill="none"
              stroke={selected ? '#059669' : '#10b981'}
              strokeWidth={selected ? 4 : 3}
              strokeLinecap="round"
            />
          </g>
        );
      })}
    </svg>
  );
}

export function Canvas() {
  const {
    graphId,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNodeFromComponent,
    markDirty,
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
  const updateTaskBindingSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioTaskBindingSpec
  );
  const workspace = useWorkspaceStore((state) => state.workspace);
  const topPane = getTopPaneState(workspace);
  const selectedTaskBindingId = topPane.selected_entity_id?.startsWith(TASK_BINDING_ENTITY_PREFIX)
    ? topPane.selected_entity_id.slice(TASK_BINDING_ENTITY_PREFIX.length)
    : null;
  const { components } = useComponents();
  const reactFlow = useReactFlow();
  const viewport = useViewport();
  const nodesInitialized = useNodesInitialized();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const lastSize = useRef<{ width: number; height: number } | null>(null);
  const fittedGraphKey = useRef<string | null>(null);
  const taskConnectionActive = useRef(false);
  const taskConnectionPanArmed = useRef(false);
  const taskConnectionPanFrame = useRef(0);
  const taskConnectionPointer = useRef<{ x: number; y: number } | null>(null);
  const [taskSourcePositions, setTaskSourcePositions] = useState<
    Record<string, { x: number; y: number }>
  >({});
  const [taskConnectionAutoPan, setTaskConnectionAutoPan] = useState(true);
  const trainingScenario = getTrainingScenario(workspace);
  const taskBindingSpec = useMemo(
    () =>
      ensureTaskBindingSpec(
        trainingScenario?.task_binding_spec,
        graph,
        trainingScenario?.task_spec
      ),
    [graph, trainingScenario?.task_binding_spec, trainingScenario?.task_spec]
  );
  const taskDataSignature = taskBindingSpec.exposed_data
    .map((data) => `${data.id}:${data.label}:${data.role}:${data.bindable ? '1' : '0'}`)
    .join('|');
  const taskBindingSignature = taskBindingSpec.bindings
    .map(
      (binding) =>
        `${binding.id}:${binding.source_data_id}:${binding.target_node_id}:${binding.target_port}:${binding.role}`
    )
    .join('|');
  const taskBindings = useMemo(() => taskBindingSpec.bindings, [taskBindingSignature]);
  const bindableTaskData = useMemo(
    () => taskBindingSpec.exposed_data.filter((data) => data.bindable),
    [taskDataSignature]
  );
  const bindableTaskDataKey = bindableTaskData.map((data) => data.id).join('|');
  const taskSourceHandleSize =
    TASK_SOURCE_HANDLE_SCREEN_SIZE / Math.max(0.1, viewport.zoom || 1);

  const updateTaskSourcePositions = useCallback(() => {
    if (topPane.active_projection !== 'task') {
      setTaskSourcePositions({});
      return;
    }
    const container = containerRef.current;
    if (!container || bindableTaskData.length === 0) {
      setTaskSourcePositions({});
      return;
    }

    const containerRect = container.getBoundingClientRect();
    const nextPositions: Record<string, { x: number; y: number }> = {};
    bindableTaskData.forEach((data, index) => {
      const sourcePort = document.querySelector<HTMLElement>(
        `[data-task-data-port-id="${cssSelectorValue(data.id)}"]`
      );
      const sourceRect = sourcePort?.getBoundingClientRect();
      const handleScreenX = sourceRect
        ? Math.max(
            containerRect.left + 1,
            Math.min(
              containerRect.right - 1,
              sourceRect.left + sourceRect.width / 2
            )
          )
        : containerRect.left + 1;
      const handleScreenY = sourceRect
        ? sourceRect.top + sourceRect.height / 2
        : containerRect.top + 88 + index * 28;
      nextPositions[data.id] = screenToCanvasFlowPosition(
        {
          x: handleScreenX - TASK_SOURCE_HANDLE_SCREEN_SIZE / 2,
          y: handleScreenY - TASK_SOURCE_HANDLE_SCREEN_SIZE / 2,
        },
        container
      );
    });
    setTaskSourcePositions((previous) => {
      const previousKeys = Object.keys(previous);
      const nextKeys = Object.keys(nextPositions);
      if (
        previousKeys.length === nextKeys.length &&
        nextKeys.every((key) => {
          const before = previous[key];
          const after = nextPositions[key];
          return (
            before &&
            Math.abs(before.x - after.x) < 0.5 &&
            Math.abs(before.y - after.y) < 0.5
          );
        })
      ) {
        return previous;
      }
      return nextPositions;
    });
  }, [bindableTaskData, topPane.active_projection]);

  const runTaskConnectionPanFrame = useCallback(() => {
    taskConnectionPanFrame.current = 0;
    if (!taskConnectionActive.current || !taskConnectionPanArmed.current) return;
    const pointer = taskConnectionPointer.current;
    const containerRect = containerRef.current?.getBoundingClientRect();
    if (!pointer || !containerRect) return;

    let dx = 0;
    let dy = 0;
    if (pointer.x < containerRect.left + TASK_CONNECT_AUTOPAN_EDGE_DISTANCE) {
      const distance = Math.max(0, pointer.x - containerRect.left);
      dx =
        TASK_CONNECT_AUTOPAN_MAX_SPEED *
        (1 - distance / TASK_CONNECT_AUTOPAN_EDGE_DISTANCE);
    } else if (pointer.x > containerRect.right - TASK_CONNECT_AUTOPAN_EDGE_DISTANCE) {
      const distance = Math.max(0, containerRect.right - pointer.x);
      dx =
        -TASK_CONNECT_AUTOPAN_MAX_SPEED *
        (1 - distance / TASK_CONNECT_AUTOPAN_EDGE_DISTANCE);
    }
    if (pointer.y < containerRect.top + TASK_CONNECT_AUTOPAN_EDGE_DISTANCE) {
      const distance = Math.max(0, pointer.y - containerRect.top);
      dy =
        TASK_CONNECT_AUTOPAN_MAX_SPEED *
        (1 - distance / TASK_CONNECT_AUTOPAN_EDGE_DISTANCE);
    } else if (pointer.y > containerRect.bottom - TASK_CONNECT_AUTOPAN_EDGE_DISTANCE) {
      const distance = Math.max(0, containerRect.bottom - pointer.y);
      dy =
        -TASK_CONNECT_AUTOPAN_MAX_SPEED *
        (1 - distance / TASK_CONNECT_AUTOPAN_EDGE_DISTANCE);
    }

    if (dx === 0 && dy === 0) return;
    const currentViewport = reactFlow.getViewport();
    reactFlow.setViewport(
      {
        x: currentViewport.x + dx,
        y: currentViewport.y + dy,
        zoom: currentViewport.zoom,
      },
      { duration: 0 }
    );
    updateTaskSourcePositions();
    taskConnectionPanFrame.current = requestAnimationFrame(runTaskConnectionPanFrame);
  }, [reactFlow, updateTaskSourcePositions]);

  const scheduleTaskConnectionPan = useCallback(() => {
    if (taskConnectionPanFrame.current) return;
    taskConnectionPanFrame.current = requestAnimationFrame(runTaskConnectionPanFrame);
  }, [runTaskConnectionPanFrame]);

  const stopTaskConnection = useCallback(() => {
    taskConnectionActive.current = false;
    taskConnectionPanArmed.current = false;
    taskConnectionPointer.current = null;
    if (taskConnectionPanFrame.current) {
      cancelAnimationFrame(taskConnectionPanFrame.current);
      taskConnectionPanFrame.current = 0;
    }
    setTaskConnectionAutoPan(true);
  }, []);

  const prepareTaskConnection = useCallback(() => {
    setTaskConnectionAutoPan(false);
  }, []);

  const beginTaskConnection = useCallback(() => {
    flushSync(() => {
      taskConnectionActive.current = true;
      taskConnectionPanArmed.current = false;
      taskConnectionPointer.current = null;
      if (taskConnectionPanFrame.current) {
        cancelAnimationFrame(taskConnectionPanFrame.current);
        taskConnectionPanFrame.current = 0;
      }
      setTaskConnectionAutoPan(false);
    });
  }, []);

  const releaseTaskConnection = useCallback(() => {
    if (!taskConnectionActive.current) {
      setTaskConnectionAutoPan(true);
    }
  }, []);

  const updateTaskConnectionPointer = useCallback(
    (event: PointerEvent | MouseEvent) => {
      if (!taskConnectionActive.current) return;
      taskConnectionPointer.current = { x: event.clientX, y: event.clientY };
      const containerRect = containerRef.current?.getBoundingClientRect();
      if (
        containerRect &&
        event.clientX > containerRect.left + TASK_CONNECT_AUTOPAN_ARM_DISTANCE
      ) {
        taskConnectionPanArmed.current = true;
      }
      if (taskConnectionPanArmed.current) {
        scheduleTaskConnectionPan();
      }
    },
    [scheduleTaskConnectionPan]
  );

  useLayoutEffect(() => {
    if (topPane.active_projection !== 'task') {
      setTaskSourcePositions({});
      return undefined;
    }
    const container = containerRef.current;
    if (!container || bindableTaskData.length === 0) {
      setTaskSourcePositions({});
      return undefined;
    }

    let frame = 0;
    const schedule = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        frame = 0;
        updateTaskSourcePositions();
      });
    };

    updateTaskSourcePositions();
    schedule();
    const resizeObserver = new ResizeObserver(schedule);
    resizeObserver.observe(container);
    document
      .querySelectorAll<HTMLElement>('[data-task-data-port-id]')
      .forEach((element) => resizeObserver.observe(element));
    window.addEventListener('resize', schedule);
    window.addEventListener('scroll', schedule, true);
    return () => {
      if (frame) cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      window.removeEventListener('resize', schedule);
      window.removeEventListener('scroll', schedule, true);
    };
  }, [
    bindableTaskDataKey,
    topPane.active_projection,
    updateTaskSourcePositions,
  ]);

  useEffect(() => {
    if (taskConnectionAutoPan) return undefined;
    window.addEventListener('pointermove', updateTaskConnectionPointer, true);
    window.addEventListener('mousemove', updateTaskConnectionPointer, true);
    return () => {
      window.removeEventListener('pointermove', updateTaskConnectionPointer, true);
      window.removeEventListener('mousemove', updateTaskConnectionPointer, true);
    };
  }, [taskConnectionAutoPan, updateTaskConnectionPointer]);

  const taskSourceNodes = useMemo<Node<GraphNodeData | TapNodeData | TaskSourceNodeData>[]>(
    () =>
      topPane.active_projection === 'task'
        ? bindableTaskData.map((data, index) => {
            let position = taskSourcePositions[data.id];
            if (!position) {
              const container = containerRef.current;
              const sourcePort = document.querySelector<HTMLElement>(
                `[data-task-data-port-id="${cssSelectorValue(data.id)}"]`
              );
              if (container && sourcePort) {
                const containerRect = container.getBoundingClientRect();
                const sourceRect = sourcePort.getBoundingClientRect();
                const handleScreenX = Math.max(
                  containerRect.left + 1,
                  Math.min(
                    containerRect.right - 1,
                    sourceRect.left + sourceRect.width / 2
                  )
                );
                const handleScreenY = sourceRect.top + sourceRect.height / 2;
                position = screenToCanvasFlowPosition(
                  {
                    x: handleScreenX - TASK_SOURCE_HANDLE_SCREEN_SIZE / 2,
                    y: handleScreenY - TASK_SOURCE_HANDLE_SCREEN_SIZE / 2,
                  },
                  container
                );
              } else {
                position = {
                  x: -taskSourceHandleSize,
                  y: 88 + index * 28,
                };
              }
            }
            return {
              id: taskSourceNodeId(data.id),
              type: 'taskSource',
              position,
              data: {
                label: data.label,
                handleSize: taskSourceHandleSize,
                beginTaskConnection,
                prepareTaskConnection,
                releaseTaskConnection,
              },
              draggable: false,
              selectable: false,
              deletable: false,
              focusable: false,
              style: {
                width: taskSourceHandleSize,
                height: taskSourceHandleSize,
                opacity: 0,
                pointerEvents: 'all',
              },
              zIndex: 3,
            };
          })
        : [],
    [
      beginTaskConnection,
      bindableTaskData,
      prepareTaskConnection,
      releaseTaskConnection,
      taskSourceHandleSize,
      taskSourcePositions,
      topPane.active_projection,
    ]
  );

  const taskBindingEdges = useMemo<Edge<GraphEdgeData>[]>(
    () =>
      topPane.active_projection === 'task'
        ? taskBindings.map((binding) => ({
            id: taskBindingEdgeId(binding.id),
            source: taskSourceNodeId(binding.source_data_id),
            sourceHandle: 'out',
            target: binding.target_node_id,
            targetHandle: binding.target_port,
            type: 'taskBinding',
            selectable: true,
            deletable: true,
            zIndex: 0,
            data: {
              task_binding_id: binding.id,
              source_data_id: binding.source_data_id,
              target_node_id: binding.target_node_id,
              target_port: binding.target_port,
            },
            selected: selectedTaskBindingId === binding.id,
          }))
        : [],
    [selectedTaskBindingId, taskBindingSignature, taskBindings, topPane.active_projection]
  );

  const displayNodes = useMemo(
    () => [...nodes, ...taskSourceNodes],
    [nodes, taskSourceNodes]
  );
  const displayEdges = useMemo(
    () => [...edges, ...taskBindingEdges],
    [edges, taskBindingEdges]
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

  const upsertTaskBinding = useCallback(
    (dataId: string, targetNodeId: string, targetPort: string) => {
      const taskData = taskBindingSpec.exposed_data.find((data) => data.id === dataId);
      if (!taskData?.bindable) return;
      const nextBindingId = taskBindingId(dataId, targetNodeId, targetPort);
      const existingBinding = taskBindingSpec.bindings.find(
        (binding) => binding.id === nextBindingId
      );
      if (existingBinding) {
        setSelectedEdge(null);
        setSelectedNode(null);
        setSelectedTap(null);
        selectTopPaneEntity(taskBindingEntityId(existingBinding.id));
        return;
      }
      if (targetInputOccupied(graph, taskBindingSpec, targetNodeId, targetPort)) {
        return;
      }
      const nextBinding: StudioTaskBinding = {
        id: nextBindingId,
        source_data_id: dataId,
        target_node_id: targetNodeId,
        target_port: targetPort,
        role: taskData.role,
        metadata: {},
      };
      updateTaskBindingSpec({
        ...taskBindingSpec,
        bindings: [
          ...taskBindingSpec.bindings.filter(
            (binding) => binding.id !== nextBinding.id
          ),
          nextBinding,
        ],
      });
      markDirty();
      setSelectedEdge(null);
      setSelectedNode(null);
      setSelectedTap(null);
      selectTopPaneEntity(taskBindingEntityId(nextBinding.id));
    },
    [
      graph,
      markDirty,
      selectTopPaneEntity,
      setSelectedEdge,
      setSelectedNode,
      setSelectedTap,
      taskBindingSpec,
      updateTaskBindingSpec,
    ]
  );

  const hasBlockingCanvasConnectionIssue = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.sourceHandle) return true;
      if (!connection.target || !connection.targetHandle) return true;
      const issues = validateConnectionAgainstSchema(
        schemaRegistry,
        connection.source,
        connection.sourceHandle,
        connection.target,
        connection.targetHandle
      );
      const dynamicMuxTarget = isNextMuxInputPort(
        graph,
        connection.target,
        connection.targetHandle,
        taskBindingSpec
      );
      if (!dynamicMuxTarget) return hasBlockingSchemaIssue(issues);
      return hasBlockingSchemaIssue(
        issues.filter((issue) => issue.type !== 'unknown_target_port')
      );
    },
    [graph, schemaRegistry, taskBindingSpec]
  );

  const isValidConnection = useCallback(
    (connection: Connection) => {
      if (!connection.target || !connection.targetHandle) return false;
      if (!connection.source || !connection.sourceHandle) return false;
      const taskDataId = taskDataIdFromSourceNodeId(connection.source);
      if (taskDataId) {
        if (isStateHandle(connection.targetHandle)) return false;
        return !targetInputOccupied(
          graph,
          taskBindingSpec,
          connection.target,
          connection.targetHandle
        );
      }
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
      return !hasBlockingCanvasConnectionIssue(connection);
    },
    [graph, hasBlockingCanvasConnectionIssue, taskBindingSpec]
  );

  const handleConnect = useCallback(
    (connection: Connection) => {
      const taskDataId = taskDataIdFromSourceNodeId(connection.source);
      if (taskDataId) {
        if (!connection.target || !connection.targetHandle) return;
        if (isStateHandle(connection.targetHandle)) return;
        upsertTaskBinding(taskDataId, connection.target, connection.targetHandle);
        return;
      }
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
        hasBlockingCanvasConnectionIssue(connection)
      ) {
        return;
      }
      onConnect(connection);
    },
    [graph, hasBlockingCanvasConnectionIssue, onConnect, taskBindingSpec, upsertTaskBinding]
  );

  const handleNodesChange = useCallback(
    (changes: Parameters<typeof onNodesChange>[0]) => {
      const graphChanges = changes.filter(
        (change) => !('id' in change) || !taskDataIdFromSourceNodeId(change.id)
      );
      if (graphChanges.length > 0) onNodesChange(graphChanges);
    },
    [onNodesChange]
  );

  const removeTaskBindings = useCallback(
    (bindingIds: Set<string>) => {
      if (bindingIds.size === 0) return;
      updateTaskBindingSpec({
        ...taskBindingSpec,
        bindings: taskBindingSpec.bindings.filter((binding) => !bindingIds.has(binding.id)),
      });
      markDirty();
      if (selectedTaskBindingId && bindingIds.has(selectedTaskBindingId)) {
        selectTopPaneEntity(null);
      }
    },
    [
      markDirty,
      selectTopPaneEntity,
      selectedTaskBindingId,
      taskBindingSpec,
      updateTaskBindingSpec,
    ]
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const graphChanges: EdgeChange[] = [];
      const removedTaskBindingIds = new Set<string>();
      for (const change of changes) {
        const taskBindingId = taskBindingIdFromEdgeId('id' in change ? change.id : null);
        if (!taskBindingId) {
          graphChanges.push(change);
          continue;
        }
        if (change.type === 'remove') {
          removedTaskBindingIds.add(taskBindingId);
        }
        if (change.type === 'select') {
          if (change.selected) {
            setSelectedEdge(null);
            setSelectedNode(null);
            setSelectedTap(null);
            selectTopPaneEntity(taskBindingEntityId(taskBindingId));
          } else if (selectedTaskBindingId === taskBindingId) {
            selectTopPaneEntity(null);
          }
        }
      }
      removeTaskBindings(removedTaskBindingIds);
      if (graphChanges.length > 0) onEdgesChange(graphChanges);
    },
    [
      onEdgesChange,
      removeTaskBindings,
      selectTopPaneEntity,
      selectedTaskBindingId,
      setSelectedEdge,
      setSelectedNode,
      setSelectedTap,
    ]
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
      data-studio-canvas-root="true"
      className="relative w-full h-full overflow-hidden bg-[radial-gradient(circle_at_top,_#ffffff_0%,_#f4f5f7_45%,_#eef1f6_100%)]"
    >
      {topPane.active_projection === 'task' && (
        <TaskBindingVisualOverlay
          bindings={taskBindings}
          selectedBindingId={selectedTaskBindingId}
          containerRef={containerRef}
        />
      )}
      <ReactFlow
        className="relative z-10"
        style={{ zIndex: 10 }}
        nodes={displayNodes}
        edges={displayEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={handleConnect}
        isValidConnection={isValidConnection}
        autoPanOnConnect={taskConnectionAutoPan}
        onConnectStart={(_, params) => {
          if (taskDataIdFromSourceNodeId(params.nodeId)) {
            beginTaskConnection();
            return;
          }
          stopTaskConnection();
        }}
        onConnectEnd={stopTaskConnection}
        onMoveEnd={() => updateTaskSourcePositions()}
        onPaneClick={() => {
          setSelectedNode(null);
          setSelectedTap(null);
          setSelectedEdge(null);
          selectTopPaneEntity(null);
        }}
        onNodeClick={(_, node) => {
          if (taskDataIdFromSourceNodeId(node.id)) return;
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
          const bindingId = taskBindingIdFromEdgeId(edge.id);
          if (bindingId) {
            setSelectedEdge(null);
            setSelectedNode(null);
            setSelectedTap(null);
            selectTopPaneEntity(taskBindingEntityId(bindingId));
            return;
          }
          setSelectedEdge(edge.id);
          setSelectedNode(null);
          setSelectedTap(null);
          selectTopPaneEntity(graphEdgeEntityId(edge.id));
        }}
        onNodeMouseEnter={(_, node) => {
          if (taskDataIdFromSourceNodeId(node.id)) return;
          hoverTopPaneEntity(
            node.type === 'tap'
              ? probeEntityId(node.id.replace(/^tap:/, ''))
              : graphNodeEntityId(node.id)
          );
        }}
        onNodeMouseLeave={() => hoverTopPaneEntity(null)}
        onEdgeMouseEnter={(_, edge) => {
          const bindingId = taskBindingIdFromEdgeId(edge.id);
          if (bindingId) {
            hoverTopPaneEntity(taskBindingEntityId(bindingId));
            return;
          }
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
