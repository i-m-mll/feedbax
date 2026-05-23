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
  Hexagon,
  Cog,
  Info,
} from 'lucide-react';
import { useComponents } from '@/hooks/useComponents';
import { useGraphStore } from '@/stores/graphStore';
import type { ComponentDefinition } from '@/types/components';
import { groupComponentsByCategory } from '@/utils/components';
import clsx from 'clsx';

const CONTEXT_SUGGESTED_CATEGORIES: Record<string, string[]> = {
  'top-level': [],  // no filtering at top level
  'network': ['Neural Networks', 'Math', 'Signal Processing'],
  'penzai': [],  // penzai models cannot be edited — show nothing
  'muscle': ['Muscles', 'Math', 'Signal Processing'],
  'acausal': ['Mechanics', 'Control', 'Math', 'Signal Processing'],
  'generic': [],
};

/** Contexts where only the suggested categories should appear (exclusive filtering). */
const CONTEXT_EXCLUSIVE_FILTER = new Set(['penzai', 'acausal', 'muscle', 'network']);

/** Blank subgraph type containers — shown at top under Structure with a purple "Type" badge. */
const SUBGRAPH_TYPES = new Set(['Subgraph', 'AcausalSystem']);

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

type ComponentLibraryMode = 'components' | 'templates';

export function ComponentLibrary({ mode = 'components' }: { mode?: ComponentLibraryMode }) {
  const [search, setSearch] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['Neural Networks', 'CDE Controllers', 'Sensorimotor'])
  );
  const { components, isLoading } = useComponents();
  const currentContext = useGraphStore((state) => state.currentContext);
  const isExclusiveContext = CONTEXT_EXCLUSIVE_FILTER.has(currentContext);
  const coreComponents = useMemo(
    () => components.filter((component) => SUBGRAPH_TYPES.has(component.name)),
    [components]
  );

  const templateComponents = useMemo(
    () => components.filter((component) => Boolean(component.template_graph)),
    [components]
  );

  const { suggestedCategories, otherCategories } = useMemo<{
    suggestedCategories: Record<string, ComponentDefinition[]>;
    otherCategories: Record<string, ComponentDefinition[]>;
  }>(() => {
    const modelComponents = components.filter(
      (component) => component.category !== 'Tasks' && !component.template_graph
    );
    const filtered = search
      ? modelComponents.filter((component) =>
          component.name.toLowerCase().includes(search.toLowerCase()) ||
          component.description.toLowerCase().includes(search.toLowerCase())
        )
      : modelComponents;

    const withoutPinned = filtered.filter((component) => !SUBGRAPH_TYPES.has(component.name));
    const all = groupComponentsByCategory(withoutPinned);

    const suggested = CONTEXT_SUGGESTED_CATEGORIES[currentContext] ?? [];

    // For exclusive contexts with no suggested categories (e.g. penzai), show nothing.
    if (suggested.length === 0 && isExclusiveContext) {
      return {
        suggestedCategories: {},
        otherCategories: {},
      };
    }

    // For non-exclusive contexts with no suggestions (top-level, generic), show everything.
    if (suggested.length === 0) {
      return { suggestedCategories: {}, otherCategories: all };
    }

    const suggestedCategories: Record<string, ComponentDefinition[]> = {};
    const otherCategories: Record<string, ComponentDefinition[]> = {};

    for (const [category, comps] of Object.entries(all)) {
      if (suggested.includes(category)) {
        suggestedCategories[category] = comps;
      } else if (!isExclusiveContext) {
        // Only include non-suggested categories when filtering is not exclusive.
        otherCategories[category] = comps;
      }
    }

    return { suggestedCategories, otherCategories };
  }, [components, search, currentContext, isExclusiveContext]);

  const templateCategories = useMemo(() => {
    if (currentContext === 'penzai') {
      return {};
    }
    const filtered = search
      ? templateComponents.filter((component) =>
          component.name.toLowerCase().includes(search.toLowerCase()) ||
          component.description.toLowerCase().includes(search.toLowerCase())
        )
      : templateComponents;
    return groupComponentsByCategory(filtered);
  }, [currentContext, search, templateComponents]);

  const hasSuggestedCategories = Object.keys(suggestedCategories).length > 0;
  const hasOtherCategories = Object.keys(otherCategories).length > 0;
  const suggestedHeaderLabel = isExclusiveContext ? 'Available' : 'Suggested';

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
          placeholder={mode === 'templates' ? 'Search templates...' : 'Search components...'}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/40"
        />
      </div>
      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-3">
        {isLoading && components.length === 0 && (
          <div className="text-xs text-slate-400">Loading components...</div>
        )}
        {currentContext === 'penzai' && (
          <div className="flex items-start gap-2 rounded-lg border border-blue-100 bg-blue-50/60 p-3">
            <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
            <p className="text-xs text-blue-600">
              Penzai models cannot be edited in the graph editor. Navigate back to add or modify components.
            </p>
          </div>
        )}
        {mode === 'components' && coreComponents.length > 0 && currentContext !== 'penzai' && (
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
        {mode === 'templates' && Object.keys(templateCategories).length === 0 && !isLoading && (
          <div className="rounded-lg border border-slate-100 bg-slate-50/80 p-3 text-xs text-slate-500">
            No graph templates are available from the component registry.
          </div>
        )}
        {mode === 'templates' && Object.entries(templateCategories).map(([category, comps]) => (
          <CategorySection
            key={category}
            category={category}
            components={comps}
            expanded={expandedCategories.has(category)}
            onToggle={() => toggleCategory(category)}
          />
        ))}
        {mode === 'components' && hasSuggestedCategories && (
          <>
            <div className="text-[10px] text-brand-500 uppercase tracking-widest">
              {suggestedHeaderLabel}
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
            {hasOtherCategories && <div className="border-t border-slate-100 my-1" />}
          </>
        )}
        {mode === 'components' &&
          Object.entries(otherCategories).map(([category, comps]) => (
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
  // Blank subgraph type containers (Subgraph, AcausalSystem) get purple "Type" badge.
  const isSubgraphType = SUBGRAPH_TYPES.has(component.name);
  const isTemplate = Boolean(component.template_graph);
  const isDisplayTemplate = component.template_kind === 'display';
  const isExecutableTemplate = component.template_kind === 'executable';
  const templateBadgeLabel = isDisplayTemplate
    ? 'Display'
    : isExecutableTemplate
      ? 'Executable'
      : 'Template';
  const templateSummary = component.template_graph
    ? `${Object.keys(component.template_graph.nodes).length} nodes, ${component.template_graph.wires.length} wires`
    : null;

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData('application/feedbax-component', component.name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className={clsx(
        'rounded-xl bg-white/90 p-3 shadow-soft cursor-grab transition',
        isSubgraphType
          ? 'border border-violet-200 hover:border-violet-400 hover:-translate-y-0.5 hover:shadow'
          : isDisplayTemplate
            ? 'border border-amber-100 hover:border-amber-300 hover:-translate-y-0.5 hover:shadow'
            : isTemplate
              ? 'border border-teal-100 hover:border-teal-300 hover:-translate-y-0.5 hover:shadow'
            : 'border border-slate-100 hover:border-slate-200 hover:-translate-y-0.5'
      )}
    >
      <div className="flex items-center gap-2">
        <div
          className={clsx(
            'w-9 h-9 rounded-lg flex items-center justify-center shrink-0',
            isSubgraphType
              ? 'bg-violet-100'
              : isDisplayTemplate
                ? 'bg-amber-50'
                : isTemplate
                  ? 'bg-teal-50'
                  : 'bg-slate-100'
          )}
        >
          <Icon
            className={clsx(
              'w-5 h-5',
              isSubgraphType
                ? 'text-violet-600'
                : isDisplayTemplate
                  ? 'text-amber-500'
                  : isTemplate
                    ? 'text-teal-600'
                    : 'text-slate-600'
            )}
          />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            <div className="text-sm font-semibold text-slate-800 truncate">{component.name}</div>
            {isSubgraphType && (
              <span className="shrink-0 rounded-full bg-violet-100 border border-violet-200 px-2 py-0.5 text-[10px] uppercase tracking-wide text-violet-600">
                Type
              </span>
            )}
            {!isSubgraphType && isTemplate && (
              <span
                className={clsx(
                  'shrink-0 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wide',
                  isDisplayTemplate
                    ? 'border-amber-100 bg-amber-50 text-amber-600'
                    : 'border-teal-100 bg-teal-50 text-teal-600'
                )}
              >
                {templateBadgeLabel}
              </span>
            )}
            {!isSubgraphType && !isTemplate && component.is_composite && (
              <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-500">
                Composite
              </span>
            )}
          </div>
          <div className="text-xs text-slate-500 line-clamp-2">{component.description}</div>
          {templateSummary && (
            <div className="mt-1 text-[11px] uppercase tracking-wide text-slate-400">
              {templateSummary}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
