import dagre from '@dagrejs/dagre';
import {
  Background, BaseEdge, Handle, Position, ReactFlow, getBezierPath, useNodesState, type Edge, type EdgeProps,
  type Node, type NodeProps, type NodeTypes,
} from '@xyflow/react';
import { installTooltips } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { Box, ChartLine, Ellipsis, FlaskConical } from 'lucide-react';
import { useId, useLayoutEffect, type ReactNode } from 'react';
import { setAppearance } from '@/appearance/appearance';
import { useRootAttribute } from '@/appearance/useRootAttribute';
import { IconButton } from '@/components/ui/Button';
import { Card, CardHead } from '@/components/ui/Card';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { StatusMark, type StatusMarkStatus } from '@/components/ui/StatusMark';
import './nodeStyles.css';

type Theme = 'light' | 'dark';
type Kind = 'train' | 'evaluate' | 'analyze';
type Step = { label: string; kind: Kind; runs: StatusMarkStatus[] };
type StepNode = Node<Step>;

const kinds = {
  train: { name: 'Train', Icon: Box },
  evaluate: { name: 'Evaluate', Icon: FlaskConical },
  analyze: { name: 'Analyze', Icon: ChartLine },
} as const;

const steps: Record<string, Step> = {
  train: { label: 'Reaching', kind: 'train', runs: ['succeeded', 'succeeded', 'running', 'queued'] },
  nominal: { label: 'Nominal', kind: 'evaluate', runs: ['succeeded', 'succeeded', 'stale'] },
  perturbed: { label: 'Perturbed', kind: 'evaluate', runs: ['succeeded', 'failed', 'not_run'] },
  trajectories: { label: 'Trajectories', kind: 'analyze', runs: ['succeeded', 'succeeded', 'succeeded'] },
  losses: { label: 'Losses', kind: 'analyze', runs: ['not_run'] },
};

const links: [string, string][] = [
  ['train', 'nominal'], ['train', 'perturbed'], ['nominal', 'trajectories'], ['perturbed', 'trajectories'], ['train', 'losses'],
];

const nodeWidth = 200;
const nodeHeight = 80;

function layout(type: string, selected: string): Node[] {
  const graph = new dagre.graphlib.Graph().setGraph({ rankdir: 'LR', nodesep: 24, ranksep: 64 }).setDefaultEdgeLabel(() => ({}));
  Object.keys(steps).forEach((id) => graph.setNode(id, { width: nodeWidth, height: nodeHeight }));
  links.forEach(([source, target]) => graph.setEdge(source, target));
  dagre.layout(graph);
  return Object.entries(steps).map(([id, data]) => {
    const { x, y } = graph.node(id);
    return {
      id, type, selected: id === selected,
      data: type === 'default' ? { label: <StockLabel step={data} /> } : data,
      position: { x: x - nodeWidth / 2, y: y - nodeHeight / 2 },
      sourcePosition: Position.Right, targetPosition: Position.Left,
    };
  });
}

function edges(type?: string): Edge[] {
  return links.map(([source, target]) => ({ id: `${source}-${target}`, source, target, type }));
}

function Runs({ runs }: { runs: readonly StatusMarkStatus[] }) {
  return <span className="flex gap-1">{runs.map((status, index) => <StatusMark key={index} status={status} />)}</span>;
}

function Done({ runs }: { runs: readonly StatusMarkStatus[] }) {
  return <span className="tabular-nums">{runs.filter((status) => status === 'succeeded').length}/{runs.length}</span>;
}

function StepName({ step }: { step: Step }) {
  const { name, Icon } = kinds[step.kind];
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <Icon aria-hidden="true" className="size-4 shrink-0" />
      <span className="truncate font-medium">{step.label}</span>
      <span className="shrink-0 text-xs text-shell-muted">{name}</span>
    </span>
  );
}

function Handles({ className }: { className?: string }) {
  return (
    <>
      <Handle type="target" position={Position.Left} className={className} />
      <Handle type="source" position={Position.Right} className={className} />
    </>
  );
}

function StockLabel({ step }: { step: Step }) {
  return (
    <span className="grid justify-items-center gap-1.5">
      <StepName step={step} />
      <Runs runs={step.runs} />
    </span>
  );
}

type Indicator = 'loading' | 'success' | 'error' | 'idle';

function indicator(runs: readonly StatusMarkStatus[]): Indicator {
  if (runs.includes('running')) return 'loading';
  if (runs.includes('failed')) return 'error';
  if (runs.every((status) => status === 'succeeded')) return 'success';
  return 'idle';
}

function FlowUiNode({ data, selected }: NodeProps<StepNode>) {
  const { name, Icon } = kinds[data.kind];
  return (
    <div className="node-flow-ui__status" data-status={indicator(data.runs)}>
      <div className="node-flow-ui" data-selected={selected || undefined}>
        <header className="node-flow-ui__header">
          <Icon aria-hidden="true" className="size-4 shrink-0" />
          <h3 className="min-w-0 flex-1 truncate font-medium">{data.label}</h3>
          <Menu>
            <MenuTrigger render={<IconButton size="mark" label="Step actions" icon={<Ellipsis />} className="nodrag" />} />
            <MenuContent aria-label="Step actions">
              <MenuItem>Run</MenuItem>
              <MenuItem>Duplicate</MenuItem>
              <MenuItem>Remove</MenuItem>
            </MenuContent>
          </Menu>
        </header>
        <div className="flex items-center justify-between gap-2 px-3 py-2 text-xs text-shell-muted">
          <Runs runs={data.runs} />
          <span>{name} · <Done runs={data.runs} /></span>
        </div>
        <Handles className="node-flow-ui__handle" />
      </div>
    </div>
  );
}

function TurboNode({ data, selected }: NodeProps<StepNode>) {
  const { name, Icon } = kinds[data.kind];
  return (
    <div className="node-turbo" data-selected={selected || undefined}>
      <div className="node-turbo__glow" aria-hidden="true" />
      <div className="node-turbo__body">
        <span className="node-turbo__icon"><Icon aria-hidden="true" className="size-5" /></span>
        <span className="grid min-w-0 gap-1">
          <span className="truncate font-medium">{data.label}</span>
          <span className="flex items-center gap-2 text-xs text-shell-muted">{name}<Runs runs={data.runs} /></span>
        </span>
      </div>
      <Handles className="node-turbo__handle" />
    </div>
  );
}

function TurboEdge(props: EdgeProps) {
  const [path] = getBezierPath(props);
  return <BaseEdge id={props.id} path={path} className="node-turbo__edge" style={{ stroke: `url(#${props.data?.gradient})` }} />;
}

function CardNode({ data, selected }: NodeProps<StepNode>) {
  return (
    <Card selected={selected} className="text-sm" style={{ width: nodeWidth }}>
      <CardHead className="px-3 py-2"><StepName step={data} /></CardHead>
      <div className="flex items-center justify-between gap-2 border-t border-shell-divider px-3 py-1.5 text-xs text-shell-muted">
        <Runs runs={data.runs} />
        <Done runs={data.runs} />
      </div>
      <Handles className="node-card__handle" />
    </Card>
  );
}

const nodeTypes: NodeTypes = { flowUi: FlowUiNode, turbo: TurboNode, card: CardNode };
const edgeTypes = { turbo: TurboEdge };

function Canvas({ name, type, edgeType, className, children, wrapper }: {
  name: string;
  type: string;
  edgeType?: string;
  className: string;
  children?: (gradient: string) => ReactNode;
  wrapper?: Record<string, string>;
}) {
  const [nodes, , onNodesChange] = useNodesState(layout(type, 'nominal'));
  const gradient = useId().replace(/:/g, '');
  const lines = edges(edgeType).map((edge) => ({ ...edge, data: { gradient } }));
  return (
    <section className="grid gap-2">
      <h2 className="font-medium">{name}</h2>
      <div className={clsx('node-styles__canvas', className)} {...wrapper}>
        {children?.(gradient)}
        <ReactFlow nodes={nodes} edges={lines} onNodesChange={onNodesChange} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
          fitView fitViewOptions={{ padding: 0.12 }} proOptions={{ hideAttribution: true }} minZoom={0.3}>
          <Background gap={16} />
        </ReactFlow>
      </div>
    </section>
  );
}

function TurboGradient(id: string) {
  return (
    <svg className="absolute size-0" aria-hidden="true">
      <defs>
        <linearGradient id={id}>
          <stop offset="0%" className="node-turbo__stop-start" />
          <stop offset="100%" className="node-turbo__stop-end" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export function NodeStyles() {
  useLayoutEffect(() => installTooltips(), []);
  const theme = useRootAttribute<Theme>('data-theme');
  return (
    <main className="min-h-screen bg-shell-surface text-shell-foreground">
      <header className="sticky top-0 z-10 flex h-panel-header items-center gap-3 border-b border-shell-divider bg-shell-chrome px-panel-header-inline">
        <h1 className="flex-1 font-medium">Node styles</h1>
        <SegmentedControl<Theme> aria-label="Theme" value={theme} onValueChange={(value) => setAppearance(window, 'theme', value)}
          items={[{ value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }]} />
      </header>
      <div className="grid max-w-5xl gap-6 p-6">
        <Canvas name="React Flow default" type="default" className="node-styles--stock" />
        <Canvas name="React Flow UI" type="flowUi" className="node-styles--flow-ui" />
        <Canvas name="Turbo Flow" type="turbo" edgeType="turbo" className="node-styles--turbo">{TurboGradient}</Canvas>
        <Canvas name="Card quiet" type="card" className="node-styles--card" />
        <Canvas name="Card ring" type="card" className="node-styles--card" wrapper={{ 'data-card': 'ring' }} />
      </div>
    </main>
  );
}
