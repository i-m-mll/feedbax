import { useState, type DragEvent } from 'react';
import {
  CircuitBoard,
  Activity,
  AudioWaveform,
  Brain,
  BrainCircuit,
  BrainCog,
  Circle,
  Minus,
  Sigma,
  Signal,
  Plus,
  X,
  SlidersHorizontal,
  SlidersVertical,
  Clock,
  Sparkles,
  TrendingUp,
  HeartPulse,
  Move,
  MoveHorizontal,
  Wind,
  Flag,
  Pin,
  Magnet,
  Shield,
  Zap,
  Asterisk,
  Copy,
  Target,
  Timer,
  Radar,
  Anchor,
  Pause,
  Eye,
  Layers,
  Hexagon,
  Cog,
} from 'lucide-react';
import { useComponents } from '@/hooks/useComponents';
import type { ComponentDefinition } from '@/types/components';

const iconMap = {
  CircuitBoard,
  Activity,
  AudioWaveform,
  HeartPulse,
  Brain,
  BrainCircuit,
  BrainCog,
  Move,
  MoveHorizontal,
  Circle,
  Signal,
  Minus,
  Sigma,
  Plus,
  X,
  SlidersHorizontal,
  SlidersVertical,
  Clock,
  Sparkles,
  TrendingUp,
  Wind,
  Flag,
  Pin,
  Magnet,
  Shield,
  Zap,
  Asterisk,
  Copy,
  Target,
  Timer,
  Radar,
  Anchor,
  Pause,
  Eye,
  Layers,
  Hexagon,
  Cog,
};

export function TaskLibrary() {
  const [search, setSearch] = useState('');
  const { components, isLoading, error } = useComponents();

  const taskComponents = components.filter((c) => c.category === 'Tasks');

  const filtered = search
    ? taskComponents.filter(
        (c) =>
          c.name.toLowerCase().includes(search.toLowerCase()) ||
          c.description.toLowerCase().includes(search.toLowerCase())
      )
    : taskComponents;

  return (
    <div className="flex flex-col h-full overflow-x-hidden">
      <div className="px-4 pb-4">
        <input
          type="text"
          placeholder="Search tasks..."
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/40"
        />
      </div>
      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-2">
        {isLoading && taskComponents.length === 0 && (
          <div className="text-xs text-slate-400">Loading tasks...</div>
        )}
        {error && (
          <div className="text-xs text-amber-500">Using local component catalog.</div>
        )}
        {!isLoading && filtered.length === 0 && (
          <div className="text-xs text-slate-400">No tasks found.</div>
        )}
        {filtered.map((component) => (
          <TaskCard key={component.name} component={component} />
        ))}
      </div>
    </div>
  );
}

function TaskCard({ component }: { component: ComponentDefinition }) {
  const Icon = iconMap[component.icon as keyof typeof iconMap] ?? CircuitBoard;

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData('application/feedbax-component', component.name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className="rounded-xl bg-white/90 p-3 shadow-soft cursor-grab transition border border-slate-100 hover:border-slate-200 hover:-translate-y-0.5"
    >
      <div className="flex items-center gap-2">
        <div className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 bg-slate-100">
          <Icon className="w-5 h-5 text-slate-600" />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            <div className="text-sm font-semibold text-slate-800 truncate">{component.name}</div>
            {component.is_composite && (
              <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-500">
                Composite
              </span>
            )}
          </div>
          <div className="text-xs text-slate-500 line-clamp-2">{component.description}</div>
        </div>
      </div>
    </div>
  );
}
