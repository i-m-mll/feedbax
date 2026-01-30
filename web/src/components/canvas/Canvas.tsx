import { useCallback, useEffect, useMemo, useState, type DragEvent } from 'react';
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  useReactFlow,
  type Connection,
} from '@xyflow/react';
import { useGraphStore } from '@/stores/graphStore';
import { CustomNode } from './CustomNode';
import { RoutedEdge } from './RoutedEdge';
import { useComponents } from '@/hooks/useComponents';
import clsx from 'clsx';
import type { PortType } from '@/types/components';

const nodeTypes = {
  component: CustomNode,
};

const edgeTypes = {
  routed: RoutedEdge,
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
    edgeStyle,
    setEdgeStyle,
    graph,
  } = useGraphStore();
  const { components } = useComponents();
  const reactFlow = useReactFlow();
  const [isShiftDown, setIsShiftDown] = useState(false);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Shift') setIsShiftDown(true);
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === 'Shift') setIsShiftDown(false);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  const componentMap = useMemo(
    () => new Map(components.map((component) => [component.name, component])),
    [components]
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

  const isValidConnection = useCallback(
    (connection: Connection) => {
      if (!connection.target || !connection.targetHandle) return false;
      if (!connection.source || !connection.sourceHandle) return false;
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
    <div className="w-full h-full bg-[radial-gradient(circle_at_top,_#ffffff_0%,_#f4f5f7_45%,_#eef1f6_100%)]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={(connection) =>
          onConnect(connection, isShiftDown ? 'elbow' : edgeStyle)
        }
        isValidConnection={isValidConnection}
        onPaneClick={() => setSelectedNode(null)}
        onNodeClick={(_, node) => setSelectedNode(node.id)}
        onDrop={onDrop}
        onDragOver={onDragOver}
        fitView
        snapToGrid
        snapGrid={[16, 16]}
      >
        <Background variant="dots" gap={16} size={1} color="#cbd5f5" />
        <Controls />
        <MiniMap nodeColor="#9ca3af" />
        <div className="absolute right-4 top-4 flex items-center gap-2 rounded-full border border-slate-200 bg-white/80 px-2 py-1 text-xs text-slate-500 shadow-soft">
          <span className="px-2 text-[10px] uppercase tracking-[0.2em] text-slate-400">Default</span>
          <button
            className={clsx(
              'px-2 py-1 rounded-full',
              edgeStyle === 'bezier' ? 'bg-brand-500/10 text-brand-600' : 'hover:bg-slate-100'
            )}
            onClick={() => setEdgeStyle('bezier')}
          >
            Curved
          </button>
          <button
            className={clsx(
              'px-2 py-1 rounded-full',
              edgeStyle === 'elbow' ? 'bg-brand-500/10 text-brand-600' : 'hover:bg-slate-100'
            )}
            onClick={() => setEdgeStyle('elbow')}
          >
            Elbow
          </button>
          <span className="px-2 text-[10px] text-slate-400">Shift = elbow</span>
        </div>
      </ReactFlow>
    </div>
  );
}
