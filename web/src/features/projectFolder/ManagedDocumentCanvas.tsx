import dagre from '@dagrejs/dagre';
import { Handle, Position, ReactFlowProvider, useReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react';
import { memo, useMemo, useRef, type ReactNode, type Ref } from 'react';
import { studioAccessibleName, studioRunStatus } from '@/components/activity/studioVocabulary';
import { StudioCanvas, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { CommandContextMenu } from '@/components/pipeline/CommandContextMenu';
import { useCarryTarget, type CarryKind } from '@/components/ui/carry';
import { Card, CardHead } from '@/components/ui/Card';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { StatusMark } from '@/components/ui/StatusMark';
import { extendsSelection } from '@/features/pipeline/selection';
import type { LibraryItem } from '@/features/projectWindow/LibraryList';
import type { StudioCommand } from '@/features/projectWindow/studioCommands';
import type { PresentationPosition, StudioProjectWindowMapElement } from '@/generated/studioContracts';

export type CanvasNode = {
  id: string;
  selectionId: string;
  /** The shown name. */
  name: string;
  glyph: ReactNode;
  problem?: string;
  /** The element's planned status: a mapped element's is the summary of its requests'. */
  status?: StudioProjectWindowMapElement['status'];
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
      {data.problem ? <StatusMark status="cannot_run" label={data.problem} />
        : data.status && <StatusMark status={data.status} label={studioRunStatus(data.status)} />}
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
  /** The selected items' selection ids. */
  selectedIds: readonly string[];
  /** Selects an item, or with `extend` toggles it among the selected items; null clears the selection. */
  onSelect: (id: string | null, extend?: boolean) => void;
  items: readonly CanvasNode[];
  connections: readonly Edge[];
  positions?: Record<string, PresentationPosition>;
  openAt?: (point: PresentationPosition) => void;
  rememberPoint?: (point: PresentationPosition | null) => void;
  placing?: boolean;
  viewportRef?: Ref<CanvasViewport>;
  /** An armed reference pick over the items' selection ids: an accepted item fills the reference when clicked. */
  pick?: { accepts(id: string): boolean; choose(id: string): void } | null;
  /** A library drag of this kind is drawn where it would be placed, and placed where it is dropped. */
  drop?: { kind: CarryKind<LibraryItem>; place(id: string, point: PresentationPosition): void };
};

export function ManagedDocumentCanvas(props: CanvasProps) {
  return <ReactFlowProvider><Canvas {...props} /></ReactFlowProvider>;
}

const labelPlate: [number, number] = [8, 3];
/** The id of the card a library drag draws where it would be placed. */
const carriedNode = 'carried:library';
const multiSelectionKeys = ['Meta', 'Control', 'Shift'];

function Canvas({ items, connections, positions, selectedIds, onSelect, openAt, rememberPoint, placing, viewportRef, pick, drop }: CanvasProps) {
  const { screenToFlowPosition, getInternalNode } = useReactFlow();
  // A selection change carries no modifiers; the gesture that made it records them.
  const extend = useRef(false);
  const surface = useRef<HTMLDivElement>(null);
  const carried = useCarryTarget(surface, drop?.kind, { drop: (item, at) => drop?.place(item.id, screenToFlowPosition(at)) });
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
        // React Flow hides a node it has no size for; a hidden card would drop the focus of a name being edited in it.
        initialWidth: cardWidth(item.name), initialHeight: cardHeight,
        data: { ...item, selected: false },
        position: positions?.[item.id] ?? { x: x - cardWidth(item.name) / 2, y: y - cardHeight / 2 },
      };
    });
    return { nodes, edges };
  }, [items, connections, positions]);
  // A node given without its measured size loses its handles until React Flow measures it again, and a second change
  // that lands before that measurement leaves it unmeasured, so its edges are never drawn.
  const shownNodes = useMemo(() => nodes.map((node) => ({
    ...node, measured: getInternalNode(node.id)?.measured, selected: !pick && selectedIds.includes(node.data.selectionId),
    data: { ...node.data, selected: !pick && selectedIds.includes(node.data.selectionId) },
    ...pick && {
      selectable: false,
      domAttributes: { 'data-pick': pick.accepts(node.data.selectionId) ? 'allowed' : 'refused' } as Node['domAttributes'],
    },
  })), [nodes, selectedIds, pick, getInternalNode]);
  const carriedAt = carried && screenToFlowPosition(carried.at);
  const drawnNodes = carried && carriedAt ? [...shownNodes, {
    id: carriedNode, type: 'element', position: carriedAt, className: 'studio-carried', selectable: false, draggable: false,
    focusable: false, initialWidth: cardWidth(carried.value.name), initialHeight: cardHeight,
    data: { id: carriedNode, selectionId: carriedNode, name: carried.value.name, glyph: carried.value.glyph, selected: false },
  } satisfies Node<ElementData>] : shownNodes;
  // A label sits on a canvas-coloured plate wide enough that the edge visibly stops short of the text.
  const shownEdges = useMemo(() => edges.map((edge) => ({
    ...edge, selected: selectedIds.includes(edge.id), labelBgPadding: labelPlate,
  })), [edges, selectedIds]);
  return <div ref={surface} className="h-full min-h-0"
    onPointerDownCapture={(event) => { extend.current = extendsSelection(event); }}
    onKeyDownCapture={(event) => { extend.current = extendsSelection(event); }}
    onPointerMove={(event) => {
      if (openAt) rememberPoint?.(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
    }}
    onPointerLeave={() => rememberPoint?.(null)}
    onDoubleClick={(event) => {
      if (openAt && event.target instanceof Element && event.target.classList.contains('react-flow__pane')) {
        openAt(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
      }
    }}>
    <StudioCanvas nodes={drawnNodes} nodeTypes={nodeTypes} edges={shownEdges} fitNewContent lockAutoFit={placing || !!carried} viewportRef={viewportRef}
      zoomOnDoubleClick={!openAt} multiSelectionKeyCode={multiSelectionKeys}
      onNodeClick={pick ? (_, node) => {
        const id = nodes.find((item) => item.id === node.id)?.data.selectionId;
        if (id !== undefined && pick.accepts(id)) pick.choose(id);
      } : undefined}
      onNodesChange={(changes) => {
        // An extending gesture toggles the one node it reaches; a plain one selects the node it reaches.
        const change = changes.find((item) => item.type === 'select' && (extend.current || item.selected));
        const node = change?.type === 'select' ? nodes.find(({ id }) => id === change.id) : undefined;
        if (node) onSelect(node.data.selectionId, extend.current);
      }}
      onEdgesChange={(changes) => {
        const change = changes.find((item) => item.type === 'select' && item.selected);
        if (change?.type === 'select') onSelect(change.id);
      }}
      onPaneClick={() => onSelect(null)} />
  </div>;
}
