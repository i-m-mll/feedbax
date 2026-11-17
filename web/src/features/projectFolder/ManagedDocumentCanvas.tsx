import dagre from '@dagrejs/dagre';
import { Handle, Position, ReactFlowProvider, useReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react';
import { memo, useMemo, type ReactNode, type Ref } from 'react';
import { appTip } from '@window-shell/browser/tooltip';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { StudioCanvas, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { Card, CardHead } from '@/components/ui/Card';
import { StatusMark } from '@/components/ui/StatusMark';
import type { PresentationPosition } from '@/generated/studioContracts';

export type CanvasNode = {
  id: string;
  selectionId: string;
  name: string;
  glyph: ReactNode;
  /** What the element is, where its name does not say it: an operation's label. */
  caption?: string;
  problem?: string;
};

type CardData = CanvasNode & { selected: boolean };

const ElementCard = memo(function ElementCard({ data }: NodeProps<Node<CardData>>) {
  return <Card selected={data.selected} className="w-[150px] text-sm">
    <Handle type="target" position={Position.Left} isConnectable={false} className="!opacity-0" />
    <CardHead className="flex min-w-0 items-center gap-1.5 px-2 py-1">
      <span className="studio-card__glyph text-shell-muted">{data.glyph}</span>
      <span className="min-w-0 flex-1 truncate font-medium">{data.name}</span>
      {data.problem && <span className="inline-flex" {...appTip(data.problem)}>
        <StatusMark status="cannot_run" label={studioName('problems')} />
      </span>}
    </CardHead>
    {data.caption && <div className="truncate px-2 pb-1 text-shell-muted">{data.caption}</div>}
    <Handle type="source" position={Position.Right} isConnectable={false} className="!opacity-0" />
  </Card>;
});

const nodeTypes = { elementCard: ElementCard };

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
  /** A library row dropped on the canvas: its data under `dropType`, and where. */
  dropType?: string;
  onDropItem?: (data: string, point: PresentationPosition) => void;
};

export function ManagedDocumentCanvas(props: CanvasProps) {
  return <ReactFlowProvider><Canvas {...props} /></ReactFlowProvider>;
}

function Canvas({
  items, connections, positions, selectedId, onSelect, openAt, rememberPoint, placing, viewportRef, dropType, onDropItem,
}: CanvasProps) {
  const { screenToFlowPosition } = useReactFlow();
  const { nodes, edges } = useMemo(() => {
    const layout = new dagre.graphlib.Graph().setGraph({ rankdir: 'LR' }).setDefaultEdgeLabel(() => ({}));
    items.forEach(({ id }) => layout.setNode(id, { width: 150, height: 50 }));
    const edges = connections.filter(({ source, target }) => layout.hasNode(source) && layout.hasNode(target));
    edges.forEach(({ source, target }) => layout.setEdge(source, target));
    dagre.layout(layout);
    const nodes = items.map((item): Node<CardData> => {
      const { x, y } = layout.node(item.id);
      return {
        id: item.id, type: 'elementCard', ariaLabel: studioAccessibleName('select', item.id),
        data: { ...item, selected: false },
        position: positions?.[item.id] ?? { x: x - 75, y: y - 25 },
      };
    });
    return { nodes, edges };
  }, [items, connections, positions]);
  const shownNodes = useMemo(() => nodes.map((node) => {
    const selected = selectedId === node.data.selectionId;
    return { ...node, selected, data: { ...node.data, selected } };
  }), [nodes, selectedId]);
  const shownEdges = useMemo(() => edges.map((edge) => ({ ...edge, selected: selectedId === edge.id })), [edges, selectedId]);
  return <div className="h-full min-h-0"
    onPointerMove={(event) => {
      if (openAt) rememberPoint?.(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
    }}
    onPointerLeave={() => rememberPoint?.(null)}
    onDragOver={(event) => {
      if (dropType && onDropItem && event.dataTransfer.types.includes(dropType)) event.preventDefault();
    }}
    onDrop={(event) => {
      const data = dropType && event.dataTransfer.getData(dropType);
      if (!data || !onDropItem) return;
      event.preventDefault();
      onDropItem(data, screenToFlowPosition({ x: event.clientX, y: event.clientY }));
    }}
    onDoubleClick={(event) => {
      if (openAt && event.target instanceof Element && event.target.classList.contains('react-flow__pane')) {
        openAt(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
      }
    }}>
    <StudioCanvas nodes={shownNodes} edges={shownEdges} nodeTypes={nodeTypes} fitNewContent lockAutoFit={placing} viewportRef={viewportRef}
      zoomOnDoubleClick={!openAt}
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
