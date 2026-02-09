import { useMemo, useState, type DragEvent } from 'react';
import {
  Activity,
  AudioWaveform,
  Brain,
  BrainCircuit,
  BrainCog,
  Circle,
  CircuitBoard,
  Minus,
  Sigma,
  Signal,
  ChevronDown,
  ChevronRight,
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
} from 'lucide-react';
import { useComponents } from '@/hooks/useComponents';
import { useGraphStore } from '@/stores/graphStore';
import type { ComponentDefinition } from '@/types/components';
import { groupComponentsByCategory } from '@/utils/components';
import clsx from 'clsx';

const CONTEXT_SUGGESTED_CATEGORIES: Record<string, string[]> = {
  'top-level': [],  // no filtering at top level
  'network': ['Neural Networks', 'Math', 'Signal Processing'],
  'penzai': ['Neural Networks', 'Math', 'Signal Processing'],
  'muscle': ['Muscles', 'Math', 'Signal Processing'],
  'acausal': ['Mechanics', 'Control', 'Math', 'Signal Processing'],
  'generic': [],
};

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
};

export function ComponentLibrary() {
  const [search, setSearch] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['Neural Networks'])
  );
  const { components, isLoading, error } = useComponents();
  const currentContext = useGraphStore((state) => state.currentContext);
  const coreComponents = useMemo(
    () => components.filter((component) => component.name === 'Subgraph'),
    [components]
  );

  const { suggestedCategories, otherCategories } = useMemo(() => {
    const filtered = search
      ? components.filter((component) =>
          component.name.toLowerCase().includes(search.toLowerCase()) ||
          component.description.toLowerCase().includes(search.toLowerCase())
        )
      : components;

    const withoutPinned = filtered.filter((component) => component.name !== 'Subgraph');
    const all = groupComponentsByCategory(withoutPinned);

    const suggested = CONTEXT_SUGGESTED_CATEGORIES[currentContext] ?? [];
    if (suggested.length === 0) {
      return { suggestedCategories: {} as Record<string, ComponentDefinition[]>, otherCategories: all };
    }

    const suggestedCategories: Record<string, ComponentDefinition[]> = {};
    const otherCategories: Record<string, ComponentDefinition[]> = {};

    for (const [category, comps] of Object.entries(all)) {
      if (suggested.includes(category)) {
        suggestedCategories[category] = comps;
      } else {
        otherCategories[category] = comps;
      }
    }

    return { suggestedCategories, otherCategories };
  }, [components, search, currentContext]);

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  return (
    <div className="flex flex-col h-full overflow-x-hidden">
      <div className="px-4 pb-4">
        <input
          type="text"
          placeholder="Search components..."
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/40"
        />
      </div>
      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-3">
        {isLoading && (
          <div className="text-xs text-slate-400">Loading components...</div>
        )}
        {error && (
          <div className="text-xs text-amber-500">Using local component catalog.</div>
        )}
        {coreComponents.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-[0.2em]">
              Structure
            </div>
            <div className="space-y-2">
              {coreComponents.map((component) => (
                <ComponentCard key={component.name} component={component} />
              ))}
            </div>
          </div>
        )}
        {/* Suggested for context */}
        {Object.keys(suggestedCategories).length > 0 && (
          <>
            <div className="text-[10px] text-brand-500 uppercase tracking-widest">
              Suggested
            </div>
            {Object.entries(suggestedCategories).map(([category, comps]) => (
              <CategorySection
                key={category}
                category={category}
                components={comps}
                expanded={expandedCategories.has(category)}
                onToggle={() => toggleCategory(category)}
              />
            ))}
            <div className="border-t border-slate-100 my-1" />
          </>
        )}
        {/* Other categories */}
        {Object.entries(otherCategories).map(([category, comps]) => (
          <CategorySection
            key={category}
            category={category}
            components={comps}
            expanded={expandedCategories.has(category)}
            onToggle={() => toggleCategory(category)}
          />
        ))}
      </div>
    </div>
  );
}

function CategorySection({
  category,
  components,
  expanded,
  onToggle,
}: {
  category: string;
  components: ComponentDefinition[];
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="space-y-2">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between text-left text-xs font-semibold text-slate-500 uppercase tracking-[0.2em]"
      >
        {category}
        {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>
      {expanded && (
        <div className="space-y-2">
          {components.map((component) => (
            <ComponentCard key={component.name} component={component} />
          ))}
        </div>
      )}
    </div>
  );
}

function ComponentCard({ component }: { component: ComponentDefinition }) {
  const Icon = iconMap[component.icon as keyof typeof iconMap] ?? CircuitBoard;

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData('application/feedbax-component', component.name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className={clsx(
        'rounded-xl border border-slate-100 bg-white/90 p-3 shadow-soft cursor-grab transition',
        'hover:border-slate-200 hover:-translate-y-0.5'
      )}
    >
      <div className="flex items-center gap-2">
        <div className="w-9 h-9 rounded-lg bg-slate-100 flex items-center justify-center shrink-0">
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
