import { useCallback, useEffect, useMemo, useRef, type DragEvent } from 'react';
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  Panel,
  useReactFlow,
  type Connection,
  BackgroundVariant,
} from '@xyflow/react';
import { useGraphStore } from '@/stores/graphStore';
import { useLayoutStore } from '@/stores/layoutStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { CustomNode } from './CustomNode';
import { RoutedEdge } from './RoutedEdge';
import { StateFlowEdge } from './StateFlowEdge';
import { TapNode } from './TapNode';
import { useComponents } from '@/hooks/useComponents';
import clsx from 'clsx';
import type { PortType } from '@/types/components';
import { ChevronsDown, ChevronsUp, MoveDiagonal, Plus } from 'lucide-react';

const nodeTypes = {
  component: CustomNode,
  tap: TapNode,
};

const edgeTypes = {
  routed: RoutedEdge,
  'state-flow': StateFlowEdge,
};

export function Canvas() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNodeFromComponent,
    setSelectedNode,
    setSelectedTap,
    setAllNodesCollapsed,
    graph,
    graphStack,
    currentGraphLabel,
    exitToBreadcrumb,
    wrapInParentGraph,
  } = useGraphStore();
  const { resizeMode, toggleResizeMode } = useLayoutStore();
  const showMinimap = useSettingsStore((state) => state.showMinimap);
  const { components } = useComponents();
  const reactFlow = useReactFlow();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const lastSize = useRef<{ width: number; height: number } | null>(null);

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

  const componentMap = useMemo(
    () => new Map(components.map((component) => [component.name, component])),
    [components]
  );

  const breadcrumbs = useMemo(
    () => [...graphStack.map((layer) => layer.label), currentGraphLabel],
    [graphStack, currentGraphLabel]
  );

  const getPortType = useCallback(
    (nodeId: string, port: string, direction: 'inputs' | 'outputs') => {
      const nodeSpec = graph.nodes[nodeId];
      if (!nodeSpec) return undefined;
      const component = componentMap.get(nodeSpec.type);
      return component?.port_types?.[direction]?.[port];
    },
    [componentMap, graph.nodes]
  );

  const isCompatible = useCallback((sourceType?: PortType, targetType?: PortType) => {
    if (!sourceType || !targetType) return true;
    if (sourceType.dtype === 'any' || targetType.dtype === 'any') return true;
    if (sourceType.dtype !== targetType.dtype) return false;
    if (
      sourceType.rank !== undefined &&
      targetType.rank !== undefined &&
      sourceType.rank !== targetType.rank
    ) {
      return false;
    }
    if (sourceType.shape && targetType.shape) {
      if (sourceType.shape.length !== targetType.shape.length) return false;
      for (let i = 0; i < sourceType.shape.length; i += 1) {
        const sourceDim = sourceType.shape[i];
        const targetDim = targetType.shape[i];
        const sourceWildcard = sourceDim < 0;
        const targetWildcard = targetDim < 0;
        if (!sourceWildcard && !targetWildcard && sourceDim !== targetDim) return false;
      }
    }
    return true;
  }, []);

  const isStateHandle = (handleId?: string | null) =>
    typeof handleId === 'string' && handleId.startsWith('__state');

  const isValidConnection = useCallback(
    (connection: Connection) => {
      if (!connection.target || !connection.targetHandle) return false;
      if (!connection.source || !connection.sourceHandle) return false;
      if (isStateHandle(connection.sourceHandle) || isStateHandle(connection.targetHandle)) {
        return false;
      }
      const inputTaken = edges.some(
        (edge) => edge.target === connection.target && edge.targetHandle === connection.targetHandle
      );
      if (inputTaken) return false;
      const sourceType = getPortType(connection.source, connection.sourceHandle, 'outputs');
      const targetType = getPortType(connection.target, connection.targetHandle, 'inputs');
      return isCompatible(sourceType, targetType);
    },
    [edges, getPortType, isCompatible]
  );

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();
      const componentName = event.dataTransfer.getData('application/feedbax-component');
      if (!componentName) return;
      const component = components.find((item) => item.name === componentName);
      if (!component) return;

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
      className="w-full h-full bg-[radial-gradient(circle_at_top,_#ffffff_0%,_#f4f5f7_45%,_#eef1f6_100%)]"
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={(connection) => onConnect(connection)}
        isValidConnection={isValidConnection}
        onPaneClick={() => {
          setSelectedNode(null);
          setSelectedTap(null);
        }}
        onNodeClick={(_, node) => {
          if (node.type === 'tap') {
            setSelectedTap(node.id.replace(/^tap:/, ''));
          } else {
            setSelectedTap(null);
            setSelectedNode(node.id);
          }
        }}
        onDrop={onDrop}
        onDragOver={onDragOver}
        fitView
        snapToGrid
        snapGrid={[16, 16]}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#cbd5f5" />
        <Controls />
        {showMinimap && <MiniMap nodeColor="#9ca3af" />}
        <Panel position="top-left" className="nodrag">
          <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-xs text-slate-500 shadow-soft">
            <button
              className="flex items-center justify-center rounded-full text-slate-500 hover:text-slate-700"
              onClick={wrapInParentGraph}
              title="Wrap this graph in a new parent"
            >
              <Plus className="w-3.5 h-3.5" />
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
        <Panel position="top-right" className="nodrag">
          <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-xs shadow-soft">
            <button
              className="flex items-center justify-center text-slate-500 hover:text-slate-700"
              onClick={() => setAllNodesCollapsed(true)}
              title="Collapse all nodes"
            >
              <ChevronsDown className="w-3.5 h-3.5" />
            </button>
            <button
              className="flex items-center justify-center text-slate-500 hover:text-slate-700"
              onClick={() => setAllNodesCollapsed(false)}
              title="Expand all nodes"
            >
              <ChevronsUp className="w-3.5 h-3.5" />
            </button>
            <div className="h-4 w-px bg-slate-200" />
            <button
              className={clsx(
                'flex items-center gap-2 rounded-full px-2 py-0.5',
                resizeMode
                  ? 'text-brand-600 bg-brand-500/10'
                  : 'text-slate-500 hover:text-slate-700'
              )}
              onClick={toggleResizeMode}
              title={resizeMode ? 'Exit resize mode' : 'Enter resize mode'}
            >
              <MoveDiagonal className="w-3.5 h-3.5" />
              Resize
            </button>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
}
