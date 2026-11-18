import dagre from '@dagrejs/dagre';
import {
  Background, Handle, Position, ReactFlow, useNodesState, type Edge, type Node, type NodeProps, type NodeTypes,
} from '@xyflow/react';
import { appTip } from '@window-shell/browser/tooltip';
import { Box, ChartLine, FlaskConical } from 'lucide-react';
import type { AppearanceValue } from '@/appearance/appearance';
import { Card, CardHead, type CardStatus } from '@/components/ui/Card';
import { StatusMark, type StatusMarkStatus } from '@/components/ui/StatusMark';
import '@xyflow/react/dist/style.css';
import './nodeStyles.css';

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

function layout(selected: string): Node[] {
  const graph = new dagre.graphlib.Graph().setGraph({ rankdir: 'LR', nodesep: 24, ranksep: 64 }).setDefaultEdgeLabel(() => ({}));
  Object.keys(steps).forEach((id) => graph.setNode(id, { width: nodeWidth, height: nodeHeight }));
  links.forEach(([source, target]) => graph.setEdge(source, target));
  dagre.layout(graph);
  return Object.entries(steps).map(([id, data]) => {
    const { x, y } = graph.node(id);
    return {
      id, type: 'card', selected: id === selected, data,
      position: { x: x - nodeWidth / 2, y: y - nodeHeight / 2 },
      sourcePosition: Position.Right, targetPosition: Position.Left,
    };
  });
}

const edges: Edge[] = links.map(([source, target]) => ({ id: `${source}-${target}`, source, target, className: 'studio-map-edge' }));

function Runs({ runs }: { runs: readonly StatusMarkStatus[] }) {
  return <span className="flex gap-1">{runs.map((status, index) => <StatusMark key={index} status={status} />)}</span>;
}

function status(runs: readonly StatusMarkStatus[]): CardStatus | undefined {
  if (runs.includes('running')) return 'busy';
  if (runs.includes('failed')) return 'failure';
  return runs.every((state) => state === 'succeeded') ? 'success' : undefined;
}

function CardNode({ data, selected }: NodeProps<StepNode>) {
  const { name, Icon } = kinds[data.kind];
  return (
    <Card selected={selected} status={status(data.runs)} className="text-sm" style={{ width: nodeWidth }}>
      <CardHead className="px-3 py-2">
        <span className="studio-card__title flex min-w-0 items-center gap-1.5">
          <span className="studio-card__glyph" {...appTip(name)}><Icon aria-hidden="true" className="size-4" /></span>
          <span className="truncate font-medium">{data.label}</span>
        </span>
      </CardHead>
      <div className="px-3 pb-2"><Runs runs={data.runs} /></div>
      <Handle type="target" position={Position.Left} className="node-card__handle" />
      <Handle type="source" position={Position.Right} className="node-card__handle" />
    </Card>
  );
}

const nodeTypes: NodeTypes = { card: CardNode };

function Canvas({ name, style }: { name: string; style: AppearanceValue<'card'> }) {
  const [nodes, , onNodesChange] = useNodesState(layout('nominal'));
  return (
    <section className="grid gap-2">
      <h2 className="font-medium">{name}</h2>
      <div className="node-styles__canvas" data-card={style}>
        <svg className="absolute size-0" aria-hidden="true">
          <defs>
            <linearGradient id="studio-map-turbo-edge">
              <stop offset="0%" style={{ stopColor: 'var(--turbo-start)' }} />
              <stop offset="100%" style={{ stopColor: 'var(--turbo-end)' }} />
            </linearGradient>
          </defs>
        </svg>
        <ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} nodeTypes={nodeTypes}
          fitView fitViewOptions={{ padding: 0.12 }} proOptions={{ hideAttribution: true }} minZoom={0.3}>
          <Background gap={16} />
        </ReactFlow>
      </div>
    </section>
  );
}

const styles = [
  ['React Flow', 'flow'], ['React Flow UI', 'flow-ui'], ['Turbo', 'turbo'], ['Quiet', 'quiet'], ['Ring', 'ring'],
] as const;

/** The shared Card under each card style, one small pipeline per style. */
export function CardStyles() {
  return <div className="grid w-full max-w-5xl gap-6">
    {styles.map(([name, style]) => <Canvas key={style} name={name} style={style} />)}
  </div>;
}
