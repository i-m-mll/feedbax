import {
  Background, BaseEdge, ConnectionMode, EdgeLabelRenderer, Handle, Position, ReactFlowProvider,
  getBezierPath, getSmoothStepPath, useConnection, useReactFlow, useStore, useUpdateNodeInternals,
  type Edge, type EdgeProps, type Node, type NodeProps, type OnConnectStart, type OnSelectionChangeParams,
} from '@xyflow/react';
import {
  createContext, memo, useCallback, useContext, useEffect, useId, useImperativeHandle, useLayoutEffect, useRef, useState,
  type MouseEvent as ReactMouseEvent, type ReactNode, type Ref,
} from 'react';
import { Lock, Unlock, ChevronsDownUp } from 'lucide-react';
import { appTip } from '@window-shell/browser/tooltip';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { CanvasStrip } from '@/components/canvas/CanvasStrip';
import { StudioCanvas, type CanvasViewport, type CanvasViewportCommand } from '@/components/canvas/StudioCanvas';
import type { ModelAddress, ModelPosition } from '@/generated/studioContracts';
import { commandAttributes, type StudioCommand } from '@/features/projectWindow/studioCommands';
import { Button, IconButton } from '@/components/ui/Button';
import { Card as SurfaceCard, CardHead } from '@/components/ui/Card';
import { CarryChip, carry, carryKind, useCarryTarget, type CarryKind } from '@/components/ui/carry';
import type { Point } from '@/components/ui/pointerPress';
import type { LibraryItem } from '@/features/projectWindow/LibraryList';
import { ChangeMark, type ChangeMarkProps } from '@/components/ui/ChangeMark';
import { TextField } from '@/components/ui/TextField';
import { portId, portTypeLabel, portTypeName, type Port } from './modelGraph';
import type { ModelLocation } from '@/features/projectWindow/elementSelection';
import './modelCanvas.css';

export type ModelCardData = {
  address: ModelAddress;
  name: string;
  ports: readonly Port[];
  flipped?: boolean;
  collapsed?: boolean;
  stacked?: boolean;
  locked: boolean;
  physicalConnections: boolean;
  selected?: boolean;
  holdsSelection?: boolean;
  problem?: string;
  /** What the shown document or value removed, drawn where it was: it cannot be wired to or edited. */
  ghost?: boolean;
  change?: ChangeMarkProps;
  /** The marks of the ports whose connection changed, by port id. */
  portChanges?: ReadonlyMap<string, ChangeMarkProps>;
  pickPort?: (port: Port) => void;
  canPick?: (port: Port) => boolean;
  cancelPick?: () => void;
  /**
   * Whether each input is offered to the connection being drawn, by port id; `note` is why it is not, or the Problem the
   * offered wire is stored with.
   */
  portVerdicts?: ReadonlyMap<string, { offered: boolean; note?: string }>;
  /** Previews the connection being drawn ending at `port`, or at none. */
  previewPort?: (port: Port | undefined) => void;
  portProblems?: ReadonlyMap<string, string>;
  /** The side each connected physical port is drawn on, by port id, which Flip does not change. */
  portSides?: ReadonlyMap<string, 'left' | 'right'>;
  onPort(port: Port): void;
  keyboardConnect(port: Port): void;
  enter?: () => void;
  rename?: (name: string) => void;
  renaming?: boolean;
  endRename?: () => void;
  /** The type's short name and headline value, or a composite's kind, shown at the header's right. */
  headline?: string;
  /** Opens, at the empty handle, the list of signals outside that can be brought in as a new port. */
  addPort?: (anchor: DOMRect) => void;
  /** Writes an Objective term on `port`, for a port a term can read. */
  addTerm?: (port: Port) => ((anchor: DOMRect) => void) | undefined;
};
export type ModelCard = Node<ModelCardData, 'modelCard'>;
export type ModelEdgeData = {
  address: ModelAddress;
  selected?: boolean;
  waypoints?: readonly ModelPosition[];
  attachments?: readonly {
    id: string; label: string; description: string; mark: string; delay: boolean; select(): void; drag?: ModelLocation;
    ghost?: boolean; change?: ChangeMarkProps;
  }[];
  ghost?: boolean;
  change?: ChangeMarkProps;
  /** Nesting order among the wires between the same cards, innermost first, for a wire routed under them. */
  rank?: number;
  bundleCount?: number;
  moveWaypoint?: (index: number, point: ModelPosition) => void;
  problem?: string;
};
export type ModelEdge = Edge<ModelEdgeData, 'signal' | 'physical'>;
type WireAttachment = NonNullable<ModelEdgeData['attachments']>[number];

/** An attachment carried from the wire it is on. */
type CarriedAttachment = { wire: string; attachment: WireAttachment };
const attachmentCarry = carryKind<CarriedAttachment>('attachment');

/** What a Library item dropped would make: its card, and the wires into and out of it when it splits a wire. */
export type LibraryPreview = { cards: ModelCard[]; wires: ModelEdge[] };

/**
 * Inputs are on the left and outputs on the right; of several physical ports the first is on the left, as an element
 * that joins two others is drawn between them. Flip swaps the sides.
 */
function side(port: Port, ports: readonly Port[], flipped: boolean) {
  const physical = ports.filter((each) => each.direction === 'physical');
  const left = port.direction === 'physical' ? physical.length > 1 && physical[0] === port : port.direction === 'input';
  return left !== flipped ? Position.Left : Position.Right;
}

const HandleUpdates = createContext<(id: string) => void>(() => {
  throw new Error('Model cards require a handle update queue');
});

function ModelHandleUpdates({ children }: { children: ReactNode }) {
  const updateNodeInternals = useUpdateNodeInternals();
  const pendingHandles = useRef(new Set<string>());
  const updateHandles = useCallback((id: string) => {
    const pending = pendingHandles.current;
    if (!pending.size) queueMicrotask(() => {
      if (!pending.size) return;
      const ids = [...pending];
      pending.clear();
      updateNodeInternals(ids);
    });
    pending.add(id);
  }, [updateNodeInternals]);
  useEffect(() => () => pendingHandles.current.clear(), []);
  return <HandleUpdates.Provider value={updateHandles}>{children}</HandleUpdates.Provider>;
}

const Card = memo(function Card({ id, data }: NodeProps<ModelCard>) {
  const update = useContext(HandleUpdates);
  const nameId = useId();
  const input = useRef<HTMLInputElement>(null);
  const ports = data.ports.map((port) => {
    const key = portId(port);
    const offer = data.portVerdicts?.get(key);
    const reason = offer?.note;
    const available = data.canPick?.(port);
    const problem = data.portProblems?.get(key);
    const verdict = offer ? offer.offered ? reason ? 'marked' : 'allowed' : 'refused'
      : available ? 'allowed' : available === false ? 'refused' : undefined;
    return { port, key, reason, available, problem, verdict };
  });
  const geometry = JSON.stringify([!!data.collapsed, !!data.flipped, [...data.portSides ?? []], ports.map(({ port, key, verdict, problem }) =>
    [key, port.name, port.mark, verdict === 'allowed' || verdict === 'marked', !!problem])]);
  useEffect(() => { update(id); }, [id, update, geometry]);
  useEffect(() => { if (data.renaming) input.current?.select(); }, [data.renaming]);
  return <SurfaceCard as="article" className="studio-model-card" data-stacked={data.stacked} selected={data.selected}
    data-ghost={data.ghost || undefined}
    data-holds-selection={data.holdsSelection} data-boundary={data.address.element.kind === 'inputs' || data.address.element.kind === 'outputs'}
    aria-labelledby={nameId}>
    <CardHead as="header" className="studio-model-card-header">
      <strong id={nameId} className={data.renaming ? 'sr-only' : undefined}>{data.name}</strong>
      {data.renaming && <form className="nodrag min-w-0 flex-1" onSubmit={(event) => {
        event.preventDefault();
        const name = input.current?.value.trim();
        if (name) data.rename?.(name);
      }}>
        <TextField ref={input} autoFocus defaultValue={data.name} aria-label={studioName('name')}
          onBlur={data.endRename} onKeyDown={(event) => {
            if (event.key === 'Escape') { event.stopPropagation(); data.endRename?.(); }
          }} />
      </form>}
      {data.headline && <span>{data.headline}</span>}
      {data.problem && <span role="img" aria-label={studioName('problems')} aria-description={data.problem}>◆</span>}
      {data.change && <ChangeMark className="nodrag nopan" {...data.change} />}
    </CardHead>
    {ports.map(({ port, key, reason, available, problem, verdict }) => {
      const change = data.portChanges?.get(key);
      const facing = data.portSides?.get(key);
      const location = facing ? facing === 'left' ? Position.Left : Position.Right : side(port, data.ports, Boolean(data.flipped));
      const physical = port.direction === 'physical';
      const interactive = !data.locked && !data.canPick;
      const connectable = interactive && (!physical || data.physicalConnections);
      const addTerm = data.collapsed || data.canPick ? undefined : data.addTerm?.(port);
      const typeName = portTypeName(port);
      return <div key={key} className="studio-model-port" data-collapsed={data.collapsed}
        data-direction={port.direction} data-side={location} data-hollow={!!port.taskSignal || !!port.growing}
        data-verdict={verdict}>
        <Handle id={key} type={port.direction === 'input' ? 'target' : 'source'}
          position={location} isConnectable={connectable} isConnectableStart={connectable} isConnectableEnd={connectable}
          role="button" tabIndex={data.collapsed ? -1 : 0}
          aria-disabled={!interactive && !available}
          aria-description={[studioName(port.direction), portTypeLabel(port), reason ?? problem].filter(Boolean).join(' · ')}
          aria-label={studioAccessibleName('port', port.name)} data-model-port className="nodrag" style={{ pointerEvents: 'auto' }}
          onFocus={available ? () => data.previewPort?.(port) : undefined}
          onBlur={data.canPick ? () => data.previewPort?.(undefined) : undefined} onClick={(event) => {
            event.stopPropagation();
            if (available) data.pickPort?.(port);
            else if (!data.canPick) data.onPort(port);
          }} onKeyDown={(event) => {
            if (data.canPick && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
              event.preventDefault();
              event.stopPropagation();
              const canvas = event.currentTarget.closest('.studio-model-surface');
              const ports = [...(canvas?.querySelectorAll<HTMLElement>('[data-model-port]') ?? [])];
              const start = ports.indexOf(event.currentTarget);
              const direction = event.key === 'ArrowUp' || event.key === 'ArrowLeft' ? -1 : 1;
              for (let offset = 1; offset < ports.length; offset += 1) {
                const candidate = ports[(start + direction * offset + ports.length) % ports.length];
                if (candidate.getAttribute('aria-disabled') === 'false' && candidate.tabIndex >= 0) {
                  candidate.focus();
                  break;
                }
              }
              return;
            }
            if (data.canPick && event.key === 'Escape') {
              event.preventDefault();
              data.cancelPick?.();
              event.stopPropagation();
              return;
            }
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            event.stopPropagation();
            if (available) data.pickPort?.(port);
            else if (!data.canPick && port.direction === 'output') data.keyboardConnect(port);
            else if (!data.canPick) data.onPort(port);
          }} data-physical={physical} />
        {!data.collapsed && <span>{port.name}</span>}
        {!data.collapsed && typeName && <span className="studio-model-port-type">{typeName}</span>}
        {addTerm && <Button variant="bare" size="mark" className="studio-model-port-term nodrag nopan"
          aria-label={studioAccessibleName('addTerm', port.name)} {...appTip(studioName('addTerm'))}
          onClick={(event) => { event.stopPropagation(); addTerm(event.currentTarget.getBoundingClientRect()); }}>+</Button>}
        {!data.collapsed && port.mark && <span role="img" aria-label={studioName(port.mark)}
          {...appTip(studioName(port.mark))}>{port.mark === 'task' ? '▥' : '◇'}</span>}
        {!data.collapsed && problem && <span role="img" aria-description={problem} aria-label={studioName('problems')}>◆</span>}
        {!data.collapsed && change && <ChangeMark className="nodrag nopan" {...change} />}
        {!data.collapsed && reason && verdict === 'refused' && <span className="studio-model-port-reason" role="status">{reason}</span>}
      </div>;
    })}
    {data.addPort && !data.collapsed && <div className="studio-model-port" data-hollow="true" data-side={data.flipped ? 'left' : 'right'}>
      <Button variant="bare" size="content" aria-label={studioName('addPort')} aria-haspopup="menu"
        className={`nodrag react-flow__handle react-flow__handle-${data.flipped ? 'left' : 'right'}`}
        onClick={(event) => { event.stopPropagation(); data.addPort?.(event.currentTarget.getBoundingClientRect()); }} />
      <span>{studioName('addPort')}</span>
    </div>}
  </SurfaceCard>;
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

const Wire = memo(function Wire({ id, data, type, source, target, sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition, markerEnd }: EdgeProps<ModelEdge>) {
  const { screenToFlowPosition } = useReactFlow();
  const [moving, setMoving] = useState<{ index: number; point: ModelPosition }>();
  const physical = type === 'physical';
  const start = { x: sourceX, y: sourceY }, end = { x: targetX, y: targetY };
  const exit = sourcePosition === Position.Left ? -1 : 1;
  const entry = targetPosition === Position.Left ? 1 : -1;
  /** A wire whose ends face the same way, or whose target lies behind its source's side, runs under both cards rather than through them. */
  const under = !physical && (exit !== entry || (targetX - sourceX) * exit < 32);
  const rank = data?.rank ?? 0;
  const reach = under ? 16 + rank * 10 : 16;
  const a = { x: sourceX + exit * reach, y: sourceY };
  const b = { x: targetX - entry * reach, y: targetY };
  const bottom = useStore((state) => under ? Math.max(...[source, target].map((card) => {
    const node = state.nodeLookup.get(card);
    return node ? node.internals.positionAbsolute.y + (node.measured.height ?? 0) : 0;
  })) : 0);
  const bend = under ? Math.max(sourceY, targetY, bottom) + 20 + rank * 10 : (sourceY + targetY) / 2;
  const waypoints = data?.waypoints?.map((point, index) => moving?.index === index ? moving.point : point);
  const pivots = waypoints?.length ? waypoints
    : [{ x: a.x, y: bend }, { x: b.x, y: bend }];
  const points = [start, a];
  for (const pivot of pivots) {
    const prior = points.at(-1)!;
    points.push({ x: prior.x, y: pivot.y }, pivot);
  }
  points.push({ x: b.x, y: points.at(-1)!.y }, b, end);
  const turns = points.filter((point, index) => index === 0 || point.x !== points[index - 1].x || point.y !== points[index - 1].y);
  const geometry = { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition };
  const [path, labelX, labelY] = physical ? getBezierPath(geometry)
    : !under && !waypoints?.length ? getSmoothStepPath({ ...geometry, borderRadius: 6, offset: 16 })
      : [rounded(turns), (a.x + b.x) / 2, bend];
  const mid = { x: labelX, y: labelY };
  const delayed = data?.attachments?.some((item) => item.delay);
  const pathRef = useRef<SVGPathElement>(null);
  const [dash, setDash] = useState<string>();
  useEffect(() => {
    if (!delayed) return;
    const length = pathRef.current?.getTotalLength() ?? 0;
    const half = length / 2;
    setDash(`${half} 0 ${'5 3 '.repeat(Math.min(1000, Math.ceil(half / 8)))}0 ${length}`);
  }, [delayed, path]);
  return <>
    {delayed && <path d={path} ref={pathRef} visibility="hidden" aria-hidden="true" />}
    <BaseEdge id={id} path={path} markerEnd={markerEnd} className="studio-model-wire"
      style={{ strokeWidth: data?.selected || data?.bundleCount ? 3 : physical ? 2.4 : 1.3,
        strokeDasharray: data?.ghost ? '4 3' : delayed ? dash : undefined }} />
    {data?.attachments?.length || data?.bundleCount || data?.problem || data?.change || data?.selected && waypoints?.length ? <EdgeLabelRenderer>
      {data.selected && waypoints?.map((point, index) => <Button key={index} size="mark" aria-label={studioAccessibleName('waypoint', index + 1)}
        className="studio-model-waypoint nodrag nopan aspect-square" style={{ transform: `translate(-50%, -50%) translate(${point.x}px, ${point.y}px)` }}
        disabled={!data.moveWaypoint}
        onPointerDown={(event) => {
          event.stopPropagation();
          event.currentTarget.setPointerCapture(event.pointerId);
          setMoving({ index, point });
        }} onPointerMove={(event) => {
          if (moving?.index === index) setMoving({ index, point: screenToFlowPosition({ x: event.clientX, y: event.clientY }) });
        }} onPointerUp={(event) => {
          if (moving?.index !== index) return;
          event.stopPropagation();
          setMoving(undefined);
          data.moveWaypoint?.(index, screenToFlowPosition({ x: event.clientX, y: event.clientY }));
        }} onPointerCancel={() => setMoving(undefined)} />)}
      <div className="studio-model-attachments nodrag nopan" style={{ transform: `translate(-50%, -50%) translate(${mid.x}px, ${mid.y}px)` }}>
        {data.problem && <span role="img" aria-label={studioName('problems')} aria-description={data.problem}>◆</span>}
        {data.bundleCount && <span className="rounded-row border border-shell-divider bg-shell-chrome px-1 text-xs">{data.bundleCount}</span>}
        {data.change && <ChangeMark {...data.change} />}
        {data.attachments?.map((attachment) => <span key={attachment.id} className="contents">
          <Button size="mark" aria-label={studioAccessibleName('attachment', attachment.label)}
            aria-description={attachment.description} data-ghost={attachment.ghost || undefined}
            {...appTip(studioAccessibleName('attachment', attachment.label))}
            onPointerDown={attachment.drag ? (event) => carry(event, attachmentCarry, { wire: id, attachment }) : undefined}
            onClick={(event) => { event.stopPropagation(); attachment.select(); }}>{attachment.mark}</Button>
          {attachment.change && <ChangeMark {...attachment.change} />}
        </span>)}
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
  lock: StudioCommand;
  collapse: StudioCommand;
  viewportCommands: readonly CanvasViewportCommand[];
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
  onConnectEnd(point: ModelPosition, valid: boolean, empty: boolean): void;
  /** The handle a pointer drawing a connection is over changed. */
  onConnectHover(node: string, handle: string | null): void;
  /** What a Library drag carries; without it the canvas takes none. */
  libraryCarry?: CarryKind<LibraryItem>;
  /** What dropping the Library item `type` at `point` would make, splitting `edge` when it is over one. */
  previewLibrary(type: string, point: ModelPosition, edge: ModelEdge | undefined, reversed: boolean): LibraryPreview | undefined;
  onLibraryDrop(type: string, point: ModelPosition, edge: ModelEdge | undefined, reversed: boolean): void;
  onAttachmentDrop(attachment: ModelLocation, edge: ModelEdge): void;
};

export function ModelCanvas(props: Props) {
  return <ReactFlowProvider><ModelHandleUpdates><Canvas {...props} /></ModelHandleUpdates></ReactFlowProvider>;
}

function Canvas({ nodes, edges, viewportRef, physical, locked, collapsed, canEdit, lock, collapse, viewportCommands, onCanvasPoint,
  onSearch, onNode, onEdge, onClear, onBox, onMove, onContext, onConnect, onConnectStart, onConnectEnd, onConnectHover,
  libraryCarry, previewLibrary, onLibraryDrop, onAttachmentDrop }: Props) {
  const { screenToFlowPosition, setNodes, setEdges, getNode, getInternalNode } = useReactFlow<ModelCard, ModelEdge>();
  const viewport = useRef<CanvasViewport>(null);
  useImperativeHandle(viewportRef, () => ({
    fitView: () => viewport.current?.fitView(),
    zoomIn: () => viewport.current?.zoomIn(),
    zoomOut: () => viewport.current?.zoomOut(),
  }), []);
  const [initial] = useState(() => ({ nodes, edges }));
  const over = useConnection((connection) => connection.inProgress && connection.toHandle
    ? [connection.toHandle.nodeId, connection.toHandle.id ?? null] as const : undefined);
  const hovered = useRef(onConnectHover);
  hovered.current = onConnectHover;
  const [overNode, overHandle] = over ?? [];
  useEffect(() => { hovered.current(overNode ?? '', overHandle ?? null); }, [overNode, overHandle]);
  /** React Flow answers every new selection handler with the current selection, so the handler stays one function. */
  const boxing = useRef(false);
  const boxed = useRef(onBox);
  boxed.current = onBox;
  const boxChange = useCallback((selection: OnSelectionChangeParams) => {
    if (boxing.current) boxed.current(selection.nodes as ModelCard[]);
  }, []);
  const point = (event: { clientX: number; clientY: number }) => screenToFlowPosition({ x: event.clientX, y: event.clientY });
  const surface = useRef<HTMLDivElement>(null);
  /** The shown wire under the pointer at `at`; the wires a drag draws are not among them. */
  const wireAt = (at: Point) => {
    const id = document.elementFromPoint(at.x, at.y)?.closest('.react-flow__edge')?.getAttribute('data-id');
    return id ? edges.find((edge) => edge.id === id) : undefined;
  };
  const reversed = (edge: ModelEdge | undefined) =>
    !!edge && (getNode(edge.source)?.position.x ?? 0) > (getNode(edge.target)?.position.x ?? 0);
  const library = useCarryTarget(surface, libraryCarry, {
    drop: ({ id }, at) => {
      const edge = wireAt(at);
      onLibraryDrop(id, screenToFlowPosition(at), edge, reversed(edge));
    },
  });
  const attachment = useCarryTarget(surface, attachmentCarry, {
    takes: (_, at) => wireAt(at)?.type === 'signal',
    drop: ({ attachment: { drag } }, at) => {
      const edge = wireAt(at);
      if (drag && edge) onAttachmentDrop(drag, edge);
    },
  });
  const splitting = library && wireAt(library.at);
  const preview = library && previewLibrary(library.value.id, screenToFlowPosition(library.at), splitting, reversed(splitting));
  const split = preview?.wires.length ? splitting?.id : undefined;
  const receiving = attachment && wireAt(attachment.at);
  const drawnNodes = preview ? [...nodes, ...preview.cards.map((card) => ({ ...card, className: 'studio-carried' }))] : nodes;
  const drawnEdges = !preview && !receiving ? edges : [...edges.map((edge): ModelEdge => {
    if (edge.id === split) return { ...edge, className: 'studio-carried-split' };
    if (!attachment || !receiving || !edge.data) return edge;
    const { wire, attachment: moved } = attachment.value;
    let list = edge.data.attachments ?? [];
    if (edge.id === wire) list = list.filter((item) => item.id !== moved.id);
    if (edge.id === receiving.id) list = [...list, { ...moved, drag: undefined }];
    return list === edge.data.attachments ? edge : { ...edge, data: { ...edge.data, attachments: list } };
  }), ...(preview?.wires.map((wire) => ({ ...wire, className: 'studio-carried', interactionWidth: 0, focusable: false, selectable: false })) ?? [])];
  /** The cards follow their props before the browser paints, so a card added by a gesture is drawn in the next frame. */
  useLayoutEffect(() => {
    setNodes((prior) => {
      const current = new Map(prior.map((node) => [node.id, node]));
      return drawnNodes.map((node) => {
        const live = current.get(node.id);
        if (!live) return node;
        return {
          ...node,
          measured: getInternalNode(node.id)?.measured,
          ...(live.dragging ? { position: live.position, dragging: true } : {}),
        };
      });
    });
  }, [drawnNodes, setNodes, getInternalNode]);
  useLayoutEffect(() => { setEdges(drawnEdges); }, [drawnEdges, setEdges]);
  return <CanvasStrip viewport={viewportCommands} tools={<>
    <IconButton label={collapse.label} icon={<ChevronsDownUp />} {...commandAttributes(collapse, 'right')} pressed={collapsed}
      disabled={!collapse.enabled} onClick={collapse.invoke} />
    <IconButton label={lock.label} icon={locked ? <Lock /> : <Unlock />} {...commandAttributes(lock, 'right')} pressed={locked}
      disabled={!lock.enabled} onClick={lock.invoke} />
  </>}>
    <div className="studio-model-surface" onPointerMove={(event) => onCanvasPoint(point(event))}
      onPointerLeave={() => onCanvasPoint(undefined)}
      onDoubleClick={(event) => {
        if (event.target instanceof Element && event.target.classList.contains('react-flow__pane')) onSearch(point(event));
      }} ref={surface}>
      <StudioCanvas defaultNodes={initial.nodes} defaultEdges={initial.edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
        viewportRef={viewport} fitReady={false} defaultViewport={origin} minZoom={0.06}
        zoomOnDoubleClick={false} nodesDraggable={canEdit && !locked} nodesConnectable={canEdit && !locked}
        connectionMode={physical ? ConnectionMode.Loose : ConnectionMode.Strict}
        multiSelectionKeyCode="Shift" selectionKeyCode="Shift"
        onSelectionStart={() => { boxing.current = true; }} onSelectionEnd={() => { boxing.current = false; }}
        onSelectionChange={boxChange}
        onNodeClick={(event, node) => onNode(node as ModelCard, event.shiftKey)}
        onNodeDoubleClick={(_, node) => (node as ModelCard).data.enter?.()}
        onNodeContextMenu={(event, node) => onContext(event, (node as ModelCard).data.address, point(event))}
        onEdgeClick={(_, edge) => onEdge(edge as ModelEdge)}
        onEdgeContextMenu={(event, edge) => { if (edge.data) onContext(event, (edge as ModelEdge).data!.address, point(event)); }}
        onPaneClick={onClear} onNodeDragStop={(_, _node, moved) => onMove(moved as ModelCard[])}
        onConnectStart={onConnectStart}
        onConnect={(connection) => onConnect(connection.source, connection.sourceHandle, connection.target, connection.targetHandle)}
        onConnectEnd={(event, connection) => {
          const pointer = 'changedTouches' in event ? event.changedTouches[0] : event;
          const target = document.elementFromPoint(pointer.clientX, pointer.clientY);
          onConnectEnd(point(pointer), connection.isValid === true, !!target?.classList.contains('react-flow__pane'));
        }}>
        {physical && <Background gap={16} size={1} />}
      </StudioCanvas>
      <CarryChip kind={attachmentCarry}>
        {({ attachment: carried }) => <Button size="mark" tabIndex={-1} aria-hidden="true">{carried.mark}</Button>}
      </CarryChip>
    </div>
  </CanvasStrip>;
}
