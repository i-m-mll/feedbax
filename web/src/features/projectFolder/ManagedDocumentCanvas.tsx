import dagre from '@dagrejs/dagre';
import { ReactFlow, ReactFlowProvider, getNodesBounds, useReactFlow, useStore, type Edge, type Node } from '@xyflow/react';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { studioAccessibleName } from '@/components/activity/studioVocabulary';
import type { PresentationPosition } from '@/generated/studioContracts';

export function useElementSelection(scope: string, reveal?: object, revealedId?: string) {
  const [selection, setSelection] = useState<{ scope: string; id: string | null }>();
  const handled = useRef<object>();
  useEffect(() => {
    if (!reveal || revealedId === undefined || handled.current === reveal) return;
    handled.current = reveal;
    setSelection({ scope, id: revealedId });
  }, [scope, reveal, revealedId]);
  return {
    selectedId: selection?.scope === scope ? selection.id : null,
    onSelect: (id: string | null) => setSelection({ scope, id }),
  };
}

interface CanvasNode {
  id: string;
  selectionId: string;
  label: ReactNode;
  problem?: string;
}

type CanvasProps = ReturnType<typeof useElementSelection> & {
  items: readonly CanvasNode[];
  connections: readonly Edge[];
  positions?: Record<string, PresentationPosition>;
  openAt?: (point: PresentationPosition) => void;
  rememberPoint?: (point: PresentationPosition | null) => void;
  placing?: boolean;
};

export function ManagedDocumentCanvas(props: CanvasProps) {
  return <ReactFlowProvider><Canvas {...props} /></ReactFlowProvider>;
}

function Canvas({ items, connections, positions, selectedId, onSelect, openAt, rememberPoint, placing }: CanvasProps) {
  const { fitBounds, viewportInitialized, screenToFlowPosition } = useReactFlow();
  const width = useStore((state) => state.width);
  const height = useStore((state) => state.height);
  const bounds = useStore(({ nodeLookup }) => {
    const rendered = [...nodeLookup.values()];
    return rendered.length > 0 && rendered.every((node) => !!node.measured.width && !!node.measured.height)
      ? getNodesBounds(rendered) : null;
  }, (first, second) => first?.x === second?.x && first?.y === second?.y &&
    first?.width === second?.width && first?.height === second?.height);
  const navigated = useRef(false);
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
          <span aria-label={problem} className="text-red-700"> !</span>}</span> },
        position: positions?.[id] ?? { x: x - 75, y: y - 25 },
      };
    });
    return { nodes, edges };
  }, [items, connections, positions]);
  const shownNodes = useMemo(() => nodes.map((node) => ({
    ...node, selected: selectedId === node.data.selectionId,
  })), [nodes, selectedId]);
  const shownEdges = useMemo(() => edges.map((edge) => ({ ...edge, selected: selectedId === edge.id })), [edges, selectedId]);
  useEffect(() => {
    if (placing) navigated.current = true;
    if (!bounds || !viewportInitialized || !width || !height) return;
    const frame = requestAnimationFrame(() => {
      if (!navigated.current) void fitBounds(bounds, { padding: 0.14 });
    });
    return () => cancelAnimationFrame(frame);
  }, [fitBounds, bounds, viewportInitialized, width, height, placing]);
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
    <ReactFlow nodes={shownNodes} edges={shownEdges} nodesDraggable={false} nodesConnectable={false}
      deleteKeyCode={null} zoomOnDoubleClick={!openAt} proOptions={{ hideAttribution: true }}
      onMove={(event) => { if (event) navigated.current = true; }}
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
