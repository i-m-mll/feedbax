import dagre from '@dagrejs/dagre';
import { Handle, Position, ReactFlowProvider, useReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react';
import { memo, useMemo, useRef, type ReactNode, type Ref } from 'react';
import { studioAccessibleName } from '@/components/activity/studioVocabulary';
import { StudioCanvas, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { CommandContextMenu } from '@/components/pipeline/CommandContextMenu';
import { Card, CardHead } from '@/components/ui/Card';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { StatusMark } from '@/components/ui/StatusMark';
import type { StudioCommand } from '@/features/projectWindow/studioCommands';
import type { PresentationPosition } from '@/generated/studioContracts';

export type CanvasNode = {
  id: string;
  selectionId: string;
  /** The shown name. */
  name: string;
  glyph: ReactNode;
  problem?: string;
  /** The element's context menu. */
  menu?: () => readonly StudioCommand[];
  /** The name is being edited in place: `commit` takes the new text, `end` leaves the edit. */
  naming?: { commit(name: string): void; end(): void };
};

type ElementData = CanvasNode & { selected: boolean };

const ElementCard = memo(function ElementCard({ data }: NodeProps<Node<ElementData>>) {
  const head = useRef<HTMLElement>(null);
  const card = <Card selected={data.selected} className="text-sm" style={{ width: cardWidth(data.name) }}>
    <Handle type="target" position={Position.Left} isConnectable={false} className="!opacity-0" />
    <CardHead ref={head} className="flex min-w-0 items-center gap-1.5 px-2 py-1.5">
      <span aria-hidden="true" className="studio-card__glyph text-shell-muted [&>svg]:size-3.5">{data.glyph}</span>
      {data.naming ? <InlineEdit value={data.name} editing className="nodrag min-w-0 flex-1 font-medium"
        aria-label={studioAccessibleName('rename', data.name)}
        onEditingChange={(editing) => { if (!editing) data.naming?.end(); }}
        onCommit={(name) => { data.naming?.commit(name); }} />
        : <span className="min-w-0 flex-1 whitespace-nowrap font-medium">{data.name}</span>}
      {data.problem && <StatusMark status="cannot_run" label={data.problem} />}
    </CardHead>
    <Handle type="source" position={Position.Right} isConnectable={false} className="!opacity-0" />
  </Card>;
  return data.menu ? <CommandContextMenu commands={data.menu} trigger={card} focusAfter={() => head.current?.querySelector('input') ?? null} /> : card;
});

const nodeTypes = { element: ElementCard };
const cardHeight = 34;
/** Wide enough for the glyph, the name and a problem mark, so the name is never cut. */
const cardWidth = (name: string) => Math.max(150, 56 + name.length * 8);

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
  /** A library drag of this data type places the dragged item where it is dropped. */
  drop?: { type: string; place(id: string, point: PresentationPosition): void };
};

export function ManagedDocumentCanvas(props: CanvasProps) {
  return <ReactFlowProvider><Canvas {...props} /></ReactFlowProvider>;
}

const labelPlate: [number, number] = [8, 3];

function Canvas({ items, connections, positions, selectedId, onSelect, openAt, rememberPoint, placing, viewportRef, pick, drop }: CanvasProps) {
  const { screenToFlowPosition } = useReactFlow();
  const { nodes, edges } = useMemo(() => {
    const layout = new dagre.graphlib.Graph().setGraph({ rankdir: 'LR', ranksep: 40, nodesep: 24 }).setDefaultEdgeLabel(() => ({}));
    items.forEach(({ id, name }) => layout.setNode(id, { width: cardWidth(name), height: cardHeight }));
    const edges = connections.filter(({ source, target }) => layout.hasNode(source) && layout.hasNode(target));
    edges.forEach(({ source, target, label }) => layout.setEdge(source, target, typeof label === 'string'
      ? { width: label.length * 7 + 16, height: 20, labelpos: 'c' } : {}));
    dagre.layout(layout);
    const nodes = items.map((item): Node<ElementData> => {
      const { x, y } = layout.node(item.id);
      return {
        id: item.id, type: 'element', ariaLabel: studioAccessibleName('select', item.name),
        data: { ...item, selected: false },
        position: positions?.[item.id] ?? { x: x - cardWidth(item.name) / 2, y: y - cardHeight / 2 },
      };
    });
    return { nodes, edges };
  }, [items, connections, positions]);
  const shownNodes = useMemo(() => nodes.map((node) => ({
    ...node, selected: !pick && selectedId === node.data.selectionId,
    data: { ...node.data, selected: !pick && selectedId === node.data.selectionId },
    ...pick && {
      selectable: false,
      domAttributes: { 'data-pick': pick.accepts(node.data.selectionId) ? 'allowed' : 'refused' } as Node['domAttributes'],
    },
  })), [nodes, selectedId, pick]);
  // A label sits on a canvas-coloured plate wide enough that the edge visibly stops short of the text.
  const shownEdges = useMemo(() => edges.map((edge) => ({
    ...edge, selected: selectedId === edge.id, labelBgPadding: labelPlate,
  })), [edges, selectedId]);
  return <div className="h-full min-h-0"
    onPointerMove={(event) => {
      if (openAt) rememberPoint?.(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
    }}
    onPointerLeave={() => rememberPoint?.(null)}
    onDragOver={(event) => {
      if (drop && event.dataTransfer.types.includes(drop.type)) event.preventDefault();
    }}
    onDrop={(event) => {
      const id = drop && event.dataTransfer.getData(drop.type);
      if (!drop || !id) return;
      event.preventDefault();
      drop.place(id, screenToFlowPosition({ x: event.clientX, y: event.clientY }));
    }}
    onDoubleClick={(event) => {
      if (openAt && event.target instanceof Element && event.target.classList.contains('react-flow__pane')) {
        openAt(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
      }
    }}>
    <StudioCanvas nodes={shownNodes} nodeTypes={nodeTypes} edges={shownEdges} fitNewContent lockAutoFit={placing} viewportRef={viewportRef}
      zoomOnDoubleClick={!openAt}
      onNodeClick={pick ? (_, node) => {
        const id = nodes.find((item) => item.id === node.id)?.data.selectionId;
        if (id !== undefined && pick.accepts(id)) pick.choose(id);
      } : undefined}
      onNodesChange={(changes) => {
        const change = changes.find((item) => item.type === 'select' && item.selected);
        const node = change?.type === 'select' ? nodes.find(({ id }) => id === change.id) : undefined;
        if (node) onSelect(node.data.selectionId);
      }}
      onEdgesChange={(changes) => {
        const change = changes.find((item) => item.type === 'select' && item.selected);
        if (change?.type === 'select') onSelect(change.id);
      }}
      onPaneClick={() => onSelect(null)} />
  </div>;
}
