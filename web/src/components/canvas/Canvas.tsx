import { useCallback, type DragEvent } from 'react';
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
import { CustomEdge } from './CustomEdge';
import { ElbowEdge } from './ElbowEdge';
import { useComponents } from '@/hooks/useComponents';
import clsx from 'clsx';

const nodeTypes = {
  component: CustomNode,
};

const edgeTypes = {
  wire: CustomEdge,
  elbow: ElbowEdge,
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
  } = useGraphStore();
  const { components } = useComponents();
  const reactFlow = useReactFlow();

  const isValidConnection = useCallback(
    (connection: Connection) => {
      if (!connection.target || !connection.targetHandle) return false;
      return !edges.some(
        (edge) => edge.target === connection.target && edge.targetHandle === connection.targetHandle
      );
    },
    [edges]
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
        onConnect={onConnect}
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
        </div>
      </ReactFlow>
    </div>
  );
}
