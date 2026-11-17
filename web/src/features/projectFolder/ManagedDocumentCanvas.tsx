import dagre from '@dagrejs/dagre';
import { ReactFlowProvider, useReactFlow, type Edge, type Node } from '@xyflow/react';
import { useEffect, useMemo, useRef, useState, type ReactNode, type Ref } from 'react';
import { studioAccessibleName } from '@/components/activity/studioVocabulary';
import { StudioCanvas, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import type { PresentationPosition } from '@/generated/studioContracts';

interface CanvasNode {
  id: string;
  selectionId: string;
  label: ReactNode;
  problem?: string;
}

type CanvasProps = {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  items: readonly CanvasNode[];
  connections: readonly Edge[];
  positions?: Record<string, PresentationPosition>;
  openAt?: (point: PresentationPosition) => void;
  rememberPoint?: (point: PresentationPosition | null) => void;
  placing?: boolean;
  viewportRef?: Ref<CanvasViewport>;
  /** An armed reference pick over the items' selection ids: an accepted item fills the reference when clicked. */
  pick?: { accepts(id: string): boolean; choose(id: string): void } | null;
};

export function ManagedDocumentCanvas(props: CanvasProps) {
  return <ReactFlowProvider><Canvas {...props} /></ReactFlowProvider>;
}

function Canvas({ items, connections, positions, selectedId, onSelect, openAt, rememberPoint, placing, viewportRef, pick }: CanvasProps) {
  const { screenToFlowPosition } = useReactFlow();
  const { nodes, edges } = useMemo(() => {
    const layout = new dagre.graphlib.Graph().setGraph({ rankdir: 'LR' }).setDefaultEdgeLabel(() => ({}));
    items.forEach(({ id }) => layout.setNode(id, { width: 150, height: 50 }));
    const edges = connections.filter(({ source, target }) => layout.hasNode(source) && layout.hasNode(target));
    edges.forEach(({ source, target }) => layout.setEdge(source, target));
    dagre.layout(layout);
    const nodes = items.map(({ id, selectionId, label, problem }): Node => {
      const { x, y } = layout.node(id);
      return {
        id, ariaLabel: studioAccessibleName('select', id),
        data: { selectionId, label: <span className="break-words" title={problem}>{label}{problem &&
          <span aria-label={problem} className="text-shell-status-failure"> !</span>}</span> },
        position: positions?.[id] ?? { x: x - 75, y: y - 25 },
      };
    });
    return { nodes, edges };
  }, [items, connections, positions]);
  const shownNodes = useMemo(() => nodes.map((node) => ({
    ...node, selected: !pick && selectedId === node.data.selectionId,
    ...pick && {
      selectable: false,
      domAttributes: { 'data-pick': pick.accepts(node.data.selectionId as string) ? 'allowed' : 'refused' } as Node['domAttributes'],
    },
  })), [nodes, selectedId, pick]);
  const shownEdges = useMemo(() => edges.map((edge) => ({ ...edge, selected: selectedId === edge.id })), [edges, selectedId]);
  return <div className="h-full min-h-0"
    onPointerMove={(event) => {
      if (openAt) rememberPoint?.(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
    }}
    onPointerLeave={() => rememberPoint?.(null)}
    onDoubleClick={(event) => {
      if (openAt && event.target instanceof Element && event.target.classList.contains('react-flow__pane')) {
        openAt(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
      }
    }}>
    <StudioCanvas nodes={shownNodes} edges={shownEdges} fitNewContent lockAutoFit={placing} viewportRef={viewportRef}
      zoomOnDoubleClick={!openAt}
      onNodeClick={pick ? (_, node) => {
        const id = node.data.selectionId;
        if (typeof id === 'string' && pick.accepts(id)) pick.choose(id);
      } : undefined}
      onNodesChange={(changes) => {
        const change = changes.find((item) => item.type === 'select' && item.selected);
        const node = change?.type === 'select' ? nodes.find(({ id }) => id === change.id) : undefined;
        if (typeof node?.data.selectionId === 'string') onSelect(node.data.selectionId);
      }}
      onEdgesChange={(changes) => {
        const change = changes.find((item) => item.type === 'select' && item.selected);
        if (change?.type === 'select') onSelect(change.id);
      }}
      onPaneClick={() => onSelect(null)} />
  </div>;
}
