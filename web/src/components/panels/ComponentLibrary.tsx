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
  Info,
} from 'lucide-react';
import { useComponents } from '@/hooks/useComponents';
import { useGraphStore } from '@/stores/graphStore';
import type { ComponentDefinition } from '@/types/components';
import { groupComponentsByCategory } from '@/utils/components';
import { CDE_TEMPLATES, type CdeSubgraphTemplate } from '@/lib/cdeTemplates';
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
  const addSubgraphNode = useGraphStore((state) => state.addSubgraphNode);
  const uiViewport = useGraphStore((state) => state.uiState.viewport);
  const isExclusiveContext = CONTEXT_EXCLUSIVE_FILTER.has(currentContext);
  const coreComponents = useMemo(
    () => components.filter((component) => component.name === 'Subgraph'),
    [components]
  );

  const { suggestedCategories, otherCategories } = useMemo<{
    suggestedCategories: Record<string, ComponentDefinition[]>;
    otherCategories: Record<string, ComponentDefinition[]>;
  }>(() => {
    const filtered = search
      ? components.filter((component) =>
          component.name.toLowerCase().includes(search.toLowerCase()) ||
          component.description.toLowerCase().includes(search.toLowerCase())
        )
      : components;

    const withoutPinned = filtered.filter((component) => component.name !== 'Subgraph');
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

  /** Compute a flow-space position near the current viewport center for click-to-insert. */
  const insertPosition = useMemo(() => {
    // Convert viewport center to flow coordinates.
    // Flow position = (screen - pan) / zoom. We use a nominal 800×600 canvas estimate.
    const CANVAS_W = 800;
    const CANVAS_H = 600;
    return {
      x: (CANVAS_W / 2 - uiViewport.x) / uiViewport.zoom,
      y: (CANVAS_H / 2 - uiViewport.y) / uiViewport.zoom,
    };
  }, [uiViewport]);

  const [cdeSectionExpanded, setCdeSectionExpanded] = useState(true);

  const filteredCdeTemplates = useMemo(() => {
    if (!search) return CDE_TEMPLATES;
    const lower = search.toLowerCase();
    return CDE_TEMPLATES.filter(
      (t) =>
        t.name.toLowerCase().includes(lower) || t.description.toLowerCase().includes(lower)
    );
  }, [search]);

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
        {isLoading && components.length === 0 && (
          <div className="text-xs text-slate-400">Loading components...</div>
        )}
        {error && (
          <div className="text-xs text-amber-500">Using local component catalog.</div>
        )}
        {currentContext === 'penzai' && (
          <div className="flex items-start gap-2 rounded-lg border border-blue-100 bg-blue-50/60 p-3">
            <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
            <p className="text-xs text-blue-600">
              Penzai models cannot be edited in the graph editor. Navigate back to add or modify components.
            </p>
          </div>
        )}
        {coreComponents.length > 0 && currentContext !== 'penzai' && (
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
        {/* CDE Controllers — template-driven subgraph nodes */}
        {filteredCdeTemplates.length > 0 && currentContext !== 'penzai' && (
          <div className="space-y-2">
            <button
              onClick={() => setCdeSectionExpanded((v) => !v)}
              className="w-full flex items-center justify-between text-left text-xs font-semibold text-slate-500 uppercase tracking-[0.2em]"
            >
              CDE Controllers
              {cdeSectionExpanded ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
            </button>
            {cdeSectionExpanded && (
              <div className="space-y-2">
                {filteredCdeTemplates.map((template) => (
                  <CdeTemplateCard
                    key={template.name}
                    template={template}
                    onInsert={() => addSubgraphNode(template, insertPosition)}
                  />
                ))}
              </div>
            )}
          </div>
        )}
        {/* Suggested for context */}
        {hasSuggestedCategories && (
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

function CdeTemplateCard({
  template,
  onInsert,
}: {
  template: CdeSubgraphTemplate;
  onInsert: () => void;
}) {
  const Icon = iconMap[template.icon as keyof typeof iconMap] ?? BrainCircuit;

  return (
    <button
      type="button"
      onClick={onInsert}
      className={clsx(
        'w-full text-left rounded-xl border border-violet-100 bg-white/90 p-3 shadow-soft',
        'cursor-pointer transition hover:border-violet-300 hover:-translate-y-0.5 hover:shadow'
      )}
    >
      <div className="flex items-center gap-2">
        <div className="w-9 h-9 rounded-lg bg-violet-50 flex items-center justify-center shrink-0">
          <Icon className="w-5 h-5 text-violet-500" />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            <div className="text-sm font-semibold text-slate-800 truncate">{template.name}</div>
            <span className="shrink-0 rounded-full bg-violet-50 border border-violet-100 px-2 py-0.5 text-[10px] uppercase tracking-wide text-violet-500">
              Subgraph
            </span>
          </div>
          <div className="text-xs text-slate-500 line-clamp-2">{template.description}</div>
        </div>
      </div>
    </button>
  );
}
