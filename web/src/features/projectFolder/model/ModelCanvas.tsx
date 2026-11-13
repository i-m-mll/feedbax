import {
  Background, BaseEdge, ConnectionMode, EdgeLabelRenderer, Handle, Position, ReactFlowProvider,
  getBezierPath, useReactFlow, useStore, useUpdateNodeInternals,
  type Edge, type EdgeProps, type Node, type NodeProps, type OnConnectStart,
} from '@xyflow/react';
import { memo, useEffect, useImperativeHandle, useRef, useState, type MouseEvent as ReactMouseEvent, type Ref } from 'react';
import { Scan, Lock, Minus, Plus, Unlock, ChevronsDownUp } from 'lucide-react';
import { StudioCanvas, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import type { ModelAddress, ModelPosition } from '@/generated/studioContracts';
import { portId, type Port } from './modelGraph';
import './modelCanvas.css';

export type ModelCardData = {
  address: ModelAddress;
  name: string;
  caption?: string;
  ports: readonly Port[];
  flipped?: boolean;
  collapsed?: boolean;
  stacked?: boolean;
  locked: boolean;
  selected?: boolean;
  holdsSelection?: boolean;
  problem?: string;
  pickPort?: (port: Port) => void;
  canPick?: (port: Port) => boolean;
  portReasons?: ReadonlyMap<string, string | null>;
  onPort(port: Port): void;
  enter?: () => void;
  rename?: (name: string) => void;
  renaming?: boolean;
  endRename?: () => void;
  addPort?: () => void;
};
export type ModelCard = Node<ModelCardData, 'modelCard'>;
export type ModelEdgeData = {
  address: ModelAddress;
  selected?: boolean;
  waypoints?: readonly ModelPosition[];
  attachments?: readonly { id: string; label: string; mark: string; delay: boolean; select(): void }[];
  self?: number;
};
export type ModelEdge = Edge<ModelEdgeData, 'signal' | 'physical'>;

function side(port: Port, flipped: boolean) {
  return (port.direction === 'input') !== flipped ? Position.Left : Position.Right;
}

const Card = memo(function Card({ id, data }: NodeProps<ModelCard>) {
  const update = useUpdateNodeInternals();
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { update(id); }, [id, update, data.ports, data.flipped, data.collapsed]);
  useEffect(() => { if (data.renaming) input.current?.select(); }, [data.renaming]);
  return <article className="studio-model-card" data-stacked={data.stacked} data-selected={data.selected}
    data-holds-selection={data.holdsSelection} data-boundary={data.address.element.kind === 'graph'}
    aria-label={data.name} onKeyDown={(event) => {
      if (event.target === event.currentTarget || event.target === event.currentTarget.parentElement) {
        if (event.key === 'Enter' && data.enter) { event.stopPropagation(); data.enter(); }
      }
    }}>
    <header className="studio-model-card-header">
      {data.renaming ? <form className="nodrag" onSubmit={(event) => {
        event.preventDefault();
        const name = input.current?.value.trim();
        if (name) data.rename?.(name);
      }}>
        <input ref={input} autoFocus defaultValue={data.name} aria-label="Name"
          onBlur={data.endRename} onKeyDown={(event) => {
            if (event.key === 'Escape') { event.stopPropagation(); data.endRename?.(); }
          }} />
      </form> : <strong>{data.name}</strong>}
      {data.caption && <span>{data.caption}</span>}
      {data.problem && <span role="img" title={data.problem} aria-label={data.problem}>◆</span>}
    </header>
    {data.ports.map((port) => {
      const location = side(port, Boolean(data.flipped));
      const physical = port.direction === 'physical';
      const reason = data.portReasons?.get(portId(port));
      const available = data.canPick?.(port);
      return <div key={portId(port)} className="studio-model-port" data-collapsed={data.collapsed}
        data-direction={port.direction} data-side={location} data-verdict={reason ? 'refused' : reason === null || available ? 'allowed' : available === false ? 'refused' : undefined}>
        <Handle id={portId(port)} type={port.direction === 'input' ? 'target' : 'source'}
          position={location} isConnectable={!data.locked && !data.canPick} tabIndex={data.collapsed ? -1 : 0}
          aria-label={port.name} className="nodrag" onClick={(event) => {
            event.stopPropagation();
            if (available) data.pickPort?.(port);
            else if (!data.canPick) data.onPort(port);
          }} onKeyDown={(event) => {
            if (event.key !== 'Enter') return;
            event.stopPropagation();
            if (available) data.pickPort?.(port);
            else if (!data.canPick) data.onPort(port);
          }} data-physical={physical} />
        {!data.collapsed && <span>{port.name}</span>}
        {!data.collapsed && reason && <span className="studio-model-port-reason" role="status">{reason}</span>}
      </div>;
    })}
    {data.addPort && <button className="nodrag studio-model-add-port" type="button" aria-label="Add Port"
      onClick={(event) => { event.stopPropagation(); data.addPort?.(); }}>+</button>}
  </article>;
});

function rounded(points: readonly ModelPosition[]) {
  let path = `M ${points[0].x},${points[0].y}`;
  for (let index = 1; index < points.length - 1; index += 1) {
    const a = points[index - 1], b = points[index], c = points[index + 1];
    const before = Math.min(6, Math.hypot(b.x - a.x, b.y - a.y) / 2);
    const after = Math.min(6, Math.hypot(c.x - b.x, c.y - b.y) / 2);
    path += ` L ${b.x + Math.sign(a.x - b.x) * before},${b.y + Math.sign(a.y - b.y) * before}`;
    path += ` Q ${b.x},${b.y} ${b.x + Math.sign(c.x - b.x) * after},${b.y + Math.sign(c.y - b.y) * after}`;
  }
  const last = points.at(-1)!;
  return `${path} L ${last.x},${last.y}`;
}

const Wire = memo(function Wire({ id, data, type, source, sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition, markerEnd }: EdgeProps<ModelEdge>) {
  const physical = type === 'physical';
  const start = { x: sourceX, y: sourceY }, end = { x: targetX, y: targetY };
  const a = { x: sourceX + (sourcePosition === Position.Left ? -16 : 16), y: sourceY };
  const b = { x: targetX + (targetPosition === Position.Left ? -16 : 16), y: targetY };
  const self = data?.self;
  const bottom = useStore((state) => {
    if (self === undefined) return 0;
    const node = state.nodeLookup.get(source);
    return node ? node.internals.positionAbsolute.y + (node.measured.height ?? 0) : 0;
  });
  const bend = self === undefined ? (sourceY + targetY) / 2 : Math.max(sourceY, targetY, bottom) + 20 + self * 10;
  const pivots = data?.waypoints?.length ? data.waypoints
    : [{ x: a.x, y: bend }, { x: b.x, y: bend }];
  const points = [start, a];
  for (const pivot of pivots) {
    const prior = points.at(-1)!;
    points.push({ x: prior.x, y: pivot.y }, pivot);
  }
  points.push({ x: b.x, y: points.at(-1)!.y }, b, end);
  const turns = points.filter((point, index) => index === 0 || point.x !== points[index - 1].x || point.y !== points[index - 1].y);
  const path = physical ? getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition })[0] : rounded(turns);
  const mid = { x: (a.x + b.x) / 2, y: bend };
  const delayed = data?.attachments?.some((item) => item.delay);
  return <>
    <BaseEdge id={id} path={path} markerEnd={markerEnd} className="studio-model-wire"
      style={{ strokeWidth: data?.selected ? 3 : physical ? 2.4 : 1.3, strokeDasharray: delayed ? '5 3' : undefined }} />
    {data?.attachments?.length ? <EdgeLabelRenderer>
      <div className="studio-model-attachments nodrag nopan" style={{ transform: `translate(-50%, -50%) translate(${mid.x}px, ${mid.y}px)` }}>
        {data.attachments.map(({ id, label, mark, select }) =>
          <button key={id} type="button" aria-label={label} title={label}
            onClick={(event) => { event.stopPropagation(); select(); }}>{mark}</button>)}
      </div>
    </EdgeLabelRenderer> : null}
  </>;
});

const nodeTypes = { modelCard: Card };
const edgeTypes = { signal: Wire, physical: Wire };
const origin = { x: 32, y: 30, zoom: 0.85 };

type Props = {
  nodes: ModelCard[];
  edges: ModelEdge[];
  physical: boolean;
  locked: boolean;
  collapsed: boolean;
  canEdit: boolean;
  viewportRef: Ref<CanvasViewport>;
  onToggleLock(): void;
  onToggleCollapse(): void;
  onCanvasPoint(point: ModelPosition | undefined): void;
  onSearch(point: ModelPosition): void;
  onNode(node: ModelCard, extend: boolean): void;
  onEdge(edge: ModelEdge): void;
  onClear(): void;
  onBox(nodes: ModelCard[]): void;
  onMove(nodes: ModelCard[]): void;
  onContext(event: ReactMouseEvent, address: ModelAddress, point: ModelPosition): void;
  onConnectStart: OnConnectStart;
  onConnect(source: string, sourceHandle: string | null, target: string, targetHandle: string | null): void;
  onConnectEnd(point: ModelPosition, valid: boolean): void;
  onLibraryDrop(type: string, point: ModelPosition): void;
};

export function ModelCanvas(props: Props) {
  return <ReactFlowProvider><Canvas {...props} /></ReactFlowProvider>;
}

function Canvas({ nodes, edges, viewportRef, physical, locked, collapsed, canEdit, onToggleLock, onToggleCollapse, onCanvasPoint,
  onSearch, onNode, onEdge, onClear, onBox, onMove, onContext, onConnect, onConnectStart, onConnectEnd, onLibraryDrop }: Props) {
  const { screenToFlowPosition, setNodes, setEdges } = useReactFlow<ModelCard, ModelEdge>();
  const viewport = useRef<CanvasViewport>(null);
  useImperativeHandle(viewportRef, () => ({
    fitView: () => viewport.current?.fitView(),
    zoomIn: () => viewport.current?.zoomIn(),
    zoomOut: () => viewport.current?.zoomOut(),
  }), []);
  const [initial] = useState(() => ({ nodes, edges }));
  const [boxSelecting, setBoxSelecting] = useState(false);
  const point = (event: { clientX: number; clientY: number }) => screenToFlowPosition({ x: event.clientX, y: event.clientY });
  useEffect(() => {
    setNodes((prior) => {
      const current = new Map(prior.map((node) => [node.id, node]));
      return nodes.map((node) => {
        const live = current.get(node.id);
        return live?.dragging ? { ...node, position: live.position, dragging: true } : node;
      });
    });
  }, [nodes, setNodes]);
  useEffect(() => { setEdges(edges); }, [edges, setEdges]);
  return <div className="studio-model-canvas">
    <div className="studio-model-strip">
      <button type="button" title="Collapse All" aria-label="Collapse All" aria-pressed={collapsed} onClick={onToggleCollapse}><ChevronsDownUp /></button>
      <button type="button" title="Lock" aria-label="Lock" aria-pressed={locked} disabled={!canEdit} onClick={onToggleLock}>{locked ? <Lock /> : <Unlock />}</button>
      <div />
      <button type="button" title="Fit View" aria-label="Fit View" onClick={() => viewport.current?.fitView()}><Scan /></button>
      <button type="button" title="Zoom In" aria-label="Zoom In" onClick={() => viewport.current?.zoomIn()}><Plus /></button>
      <button type="button" title="Zoom Out" aria-label="Zoom Out" onClick={() => viewport.current?.zoomOut()}><Minus /></button>
    </div>
    <div className="studio-model-surface" onPointerMove={(event) => onCanvasPoint(point(event))}
      onPointerLeave={() => onCanvasPoint(undefined)}
      onDoubleClick={(event) => {
        if (event.target instanceof Element && event.target.classList.contains('react-flow__pane')) onSearch(point(event));
      }} onDragOver={(event) => {
        if (event.dataTransfer.types.includes('application/x-feedbax-component')) event.preventDefault();
      }} onDrop={(event) => {
        const type = event.dataTransfer.getData('application/x-feedbax-component');
        if (type) { event.preventDefault(); onLibraryDrop(type, point(event)); }
      }}>
      <StudioCanvas defaultNodes={initial.nodes} defaultEdges={initial.edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
        viewportRef={viewport} fitReady={false} defaultViewport={origin} minZoom={0.06}
        zoomOnDoubleClick={false} nodesDraggable={canEdit && !locked} nodesConnectable={canEdit && !locked}
        connectionMode={physical ? ConnectionMode.Loose : ConnectionMode.Strict}
        multiSelectionKeyCode="Shift" selectionKeyCode="Shift"
        onSelectionStart={() => setBoxSelecting(true)} onSelectionEnd={() => setBoxSelecting(false)}
        onSelectionChange={boxSelecting ? (selection) => onBox(selection.nodes as ModelCard[]) : undefined}
        onNodeClick={(event, node) => onNode(node as ModelCard, event.shiftKey)}
        onNodeDoubleClick={(_, node) => (node as ModelCard).data.enter?.()}
        onNodeContextMenu={(event, node) => onContext(event, (node as ModelCard).data.address, point(event))}
        onEdgeClick={(_, edge) => onEdge(edge as ModelEdge)}
        onEdgeContextMenu={(event, edge) => { if (edge.data) onContext(event, (edge as ModelEdge).data!.address, point(event)); }}
        onPaneClick={onClear} onNodeDragStop={(_, _node, moved) => onMove(moved as ModelCard[])}
        onConnectStart={onConnectStart}
        onConnect={(connection) => onConnect(connection.source, connection.sourceHandle, connection.target, connection.targetHandle)}
        onConnectEnd={(event, connection) => onConnectEnd(point('changedTouches' in event ? event.changedTouches[0] : event), connection.isValid === true)}>
        {physical && <Background gap={16} size={1} />}
      </StudioCanvas>
    </div>
  </div>;
}
