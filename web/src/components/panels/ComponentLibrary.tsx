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
import { toast } from 'sonner';
import { useComponents } from '@/hooks/useComponents';
import { useDomains } from '@/hooks/useDomains';
import type { DomainContext } from '@/features/domains/context';
import { useGraphStore } from '@/stores/graphStore';
import { useLayoutStore } from '@/stores/layoutStore';
import type { ComponentDefinition } from '@/types/components';
import { groupComponentsByCategory } from '@/utils/components';
import clsx from 'clsx';

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

function nextInsertPosition(
  nodes: ReturnType<typeof useGraphStore.getState>['nodes'],
  viewport: ReturnType<typeof useGraphStore.getState>['uiState']['viewport']
) {
  const zoom = viewport.zoom || 1;
  const index = nodes.filter((node) => node.type !== 'tap').length;
  return {
    x: (320 - viewport.x) / zoom + (index % 5) * 32,
    y: (180 - viewport.y) / zoom + (index % 5) * 24,
  };
}

export function ComponentLibrary({ mode = 'components' }: { mode?: ComponentLibraryMode }) {
  const [search, setSearch] = useState('');
  const expandedCategoryList = useLayoutStore(
    (state) => state.componentLibraryExpandedCategories
  );
  const setExpandedCategories = useLayoutStore(
    (state) => state.setComponentLibraryExpandedCategories
  );
  const expandedCategories = useMemo(
    () => new Set(expandedCategoryList),
    [expandedCategoryList]
  );
  const { components, isLoading, error: componentsError } = useComponents();
  const domainsQuery = useDomains();
  const currentContext = useGraphStore((state) => state.currentContext);
  const domainContext = domainsQuery.domainContextFor(currentContext);
  const isInspectorDomain = domainContext?.domain.editor.kind === 'inspector';
  const registryError = componentsError ?? domainsQuery.error;
  const isRegistryLoading = (isLoading || domainsQuery.isLoading) && components.length === 0;

  const structureComponents = useMemo(
    () => domainContext
      ? components.filter(
          (component) => Boolean(component.interior_domain) && domainContext.canPlace(component).allowed
        )
      : [],
    [components, domainContext]
  );

  const templateComponents = useMemo(
    () => components.filter((component) => Boolean(component.template_graph)),
    [components]
  );

  const componentCategories = useMemo<Record<string, ComponentDefinition[]>>(() => {
    if (!domainContext) return {};
    const modelComponents = components.filter(
      (component) => component.category !== 'Tasks' && !component.template_graph
    );
    const filtered = search
      ? modelComponents.filter((component) =>
          component.name.toLowerCase().includes(search.toLowerCase()) ||
          component.description.toLowerCase().includes(search.toLowerCase())
        )
      : modelComponents;

    const unpinned = filtered.filter((component) => !component.interior_domain);
    return groupComponentsByCategory(unpinned.filter(domainContext.paletteFilter));
  }, [components, search, domainContext]);

  const templateCategories = useMemo(() => {
    if (!domainContext || isInspectorDomain) return {};
    const filtered = search
      ? templateComponents.filter((component) =>
          component.name.toLowerCase().includes(search.toLowerCase()) ||
          component.description.toLowerCase().includes(search.toLowerCase())
        )
      : templateComponents;
    return groupComponentsByCategory(filtered.filter((component) => domainContext.canPlace(component).allowed));
  }, [domainContext, isInspectorDomain, search, templateComponents]);

  const hasComponentCategories = Object.keys(componentCategories).length > 0;

  const toggleCategory = (category: string) => {
    const next = new Set(expandedCategories);
    if (next.has(category)) {
      next.delete(category);
    } else {
      next.add(category);
    }
    setExpandedCategories([...next]);
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
        {isRegistryLoading && (
          <div className="text-xs text-slate-400">Loading component registry...</div>
        )}
        {registryError && (
          <div className="rounded-lg border border-red-100 bg-red-50/80 p-3 text-xs text-red-600">
            Component domain registry is unavailable. Start the Studio backend to load the palette.
          </div>
        )}
        {isInspectorDomain && !hasComponentCategories && (
          <div className="flex items-start gap-2 rounded-lg border border-blue-100 bg-blue-50/60 p-3">
            <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
            <p className="text-xs text-blue-600">
              {domainContext.domain.display_name} models cannot be edited in the graph editor.
              Navigate back to add or modify components.
            </p>
          </div>
        )}
        {domainContext && mode === 'components' && structureComponents.length > 0 && !isInspectorDomain && (
          <div className="space-y-2">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-[0.2em]">
              Structure
            </div>
            <div className="space-y-2">
              {structureComponents.map((component) => (
                <ComponentCard key={component.name} component={component} domainContext={domainContext} />
              ))}
            </div>
          </div>
        )}
        {mode === 'templates' && Object.keys(templateCategories).length === 0 && !isRegistryLoading && (
          <div className="rounded-lg border border-slate-100 bg-slate-50/80 p-3 text-xs text-slate-500">
            No graph templates are available from the component registry.
          </div>
        )}
        {domainContext && mode === 'templates' && Object.entries(templateCategories).map(([category, comps]) => (
          <CategorySection
            key={category}
            category={category}
            components={comps}
            domainContext={domainContext}
            expanded={expandedCategories.has(category)}
            onToggle={() => toggleCategory(category)}
          />
        ))}
        {domainContext && mode === 'components' &&
          Object.entries(componentCategories).map(([category, comps]) => (
            <CategorySection
              key={category}
              category={category}
              components={comps}
              domainContext={domainContext}
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
  domainContext,
  expanded,
  onToggle,
}: {
  category: string;
  components: ComponentDefinition[];
  domainContext: DomainContext;
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
            <ComponentCard key={component.name} component={component} domainContext={domainContext} />
          ))}
        </div>
      )}
    </div>
  );
}

function ComponentCard({
  component,
  domainContext,
}: {
  component: ComponentDefinition;
  domainContext: DomainContext;
}) {
  const Icon = iconMap[component.icon as keyof typeof iconMap] ?? CircuitBoard;
  const addNodeFromComponent = useGraphStore((state) => state.addNodeFromComponent);
  const nodes = useGraphStore((state) => state.nodes);
  const viewport = useGraphStore((state) => state.uiState.viewport);
  const isStructure = Boolean(component.interior_domain);
  const isTemplate = Boolean(component.template_graph);
  const isDisplayTemplate = component.template_kind === 'display';
  const templateBadgeLabel = isDisplayTemplate ? 'Preview only' : null;
  const templateSummary = component.template_graph
    ? `${Object.keys(component.template_graph.nodes).length} nodes, ${component.template_graph.wires.length} wires`
    : null;

  const onDragStart = (event: DragEvent<HTMLButtonElement>) => {
    event.dataTransfer.setData('application/feedbax-component', component.name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <button
      type="button"
      draggable
      onDragStart={onDragStart}
      onClick={() => {
        const verdict = domainContext.canPlace(component);
        if (!verdict.allowed) {
          toast.error('reason' in verdict ? verdict.reason : 'Component is not available here.', {
            id: 'domain-placement-error',
          });
          return;
        }
        addNodeFromComponent(component, nextInsertPosition(nodes, viewport));
      }}
      title={`Add ${component.name}`}
      className={clsx(
        'w-full min-h-[92px] rounded-xl bg-white/90 p-3 text-left shadow-soft cursor-grab transition',
        'focus:outline-none focus:ring-2 focus:ring-brand-500/40',
        isStructure
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
            isStructure
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
              isStructure
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
            {isStructure && (
              <span className="shrink-0 rounded-full bg-violet-100 border border-violet-200 px-2 py-0.5 text-[10px] uppercase tracking-wide text-violet-600">
                Type
              </span>
            )}
            {!isStructure && templateBadgeLabel && (
              <span
                className={clsx(
                  'shrink-0 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wide',
                  'border-amber-100 bg-amber-50 text-amber-600'
                )}
              >
                {templateBadgeLabel}
              </span>
            )}
            {!isStructure && !isTemplate && component.is_composite && (
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
    </button>
  );
}
