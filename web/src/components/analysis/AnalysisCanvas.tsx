/**
 * AnalysisCanvas — React Flow canvas for the analysis DAG.
 *
 * Renders analysis nodes, data source nodes, transform nodes, and their
 * connecting edges in a left-to-right layout. Supports drag-to-add from
 * the sidebar analysis palette.
 *
 * Follows the same structural patterns as the model Canvas component
 * but with analysis-specific node types, edge types, and a distinct
 * visual scheme (emerald accent vs brand blue).
 */

import { useCallback, useEffect, useMemo, useRef, type DragEvent } from 'react';
import {
  Background,
  Controls,
  ReactFlow,
  Panel,
  useReactFlow,
  BackgroundVariant,
  type Connection,
} from '@xyflow/react';
import { useAnalysisStore } from '@/stores/analysisStore';
import { AnalysisNode, AnalysisDepNode } from './AnalysisNode';
import { DataSourceNode } from './DataSourceNode';
import { TransformNode } from './TransformNode';
import { AnalysisExplicitEdge, AnalysisImplicitEdge } from './AnalysisEdges';
import type { AnalysisClassDef } from '@/types/analysis';

// ---------------------------------------------------------------------------
// Node and edge type registrations
// ---------------------------------------------------------------------------

const nodeTypes = {
  analysis: AnalysisNode,
  analysisDep: AnalysisDepNode,
  dataSource: DataSourceNode,
  transform: TransformNode,
};

const edgeTypes = {
  analysisExplicit: AnalysisExplicitEdge,
  analysisImplicit: AnalysisImplicitEdge,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AnalysisCanvas() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    setSelectedNode,
    setSelectedTransform,
    addAnalysisNode,
    connectNodes,
    analysisClasses,
  } = useAnalysisStore();

  const reactFlow = useReactFlow();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const lastSize = useRef<{ width: number; height: number } | null>(null);

  // Viewport tracking on resize — same pattern as model Canvas
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

  // Build a lookup for analysis classes by name
  const classMap = useMemo(
    () => new Map(analysisClasses.map((c) => [c.name, c])),
    [analysisClasses]
  );

  // Drag-and-drop from the analysis palette
  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();
      const analysisName = event.dataTransfer.getData('application/feedbax-analysis');
      if (!analysisName) return;
      const classDef = classMap.get(analysisName);
      if (!classDef) return;

      const position = reactFlow.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      addAnalysisNode(classDef, position);
    },
    [addAnalysisNode, reactFlow, classMap]
  );

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onConnect = useCallback(
    (connection: Connection) => {
      connectNodes(connection);
    },
    [connectNodes]
  );

  const hasNodes = nodes.length > 0;

  return (
    <div
      ref={containerRef}
      className="w-full h-full bg-[radial-gradient(circle_at_top,_#f0fdf4_0%,_#f7f7f8_45%,_#f0f1f3_100%)]"
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onPaneClick={() => {
          setSelectedNode(null);
          setSelectedTransform(null);
        }}
        onNodeClick={(_, node) => {
          if (node.type === 'transform') {
            setSelectedTransform(node.id);
          } else {
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
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#bbf7d0" />
        <Controls />

        {/* Empty state */}
        {!hasNodes && (
          <Panel position="top-center" className="nodrag">
            <div className="mt-16 text-center">
              <div className="text-sm text-slate-400">
                Drag analyses from the sidebar to build a pipeline
              </div>
              <div className="mt-1 text-xs text-slate-300">
                Connect the data source to analysis nodes with wires
              </div>
            </div>
          </Panel>
        )}
      </ReactFlow>
    </div>
  );
}
