import { useCallback, type DragEvent } from 'react';
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  useReactFlow,
} from '@xyflow/react';
import { useGraphStore } from '@/stores/graphStore';
import { CustomNode } from './CustomNode';
import { CustomEdge } from './CustomEdge';
import { useComponents } from '@/hooks/useComponents';

const nodeTypes = {
  component: CustomNode,
};

const edgeTypes = {
  wire: CustomEdge,
};

export function Canvas() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNodeFromComponent, setSelectedNode } =
    useGraphStore();
  const { components } = useComponents();
  const reactFlow = useReactFlow();

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
    [addNodeFromComponent, reactFlow]
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
      </ReactFlow>
    </div>
  );
}
