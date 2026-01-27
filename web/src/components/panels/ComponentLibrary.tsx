import { useMemo, useState, type DragEvent } from 'react';
import {
  Activity,
  CircuitBoard,
  Minus,
  Sigma,
  Signal,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { useComponents } from '@/hooks/useComponents';
import type { ComponentDefinition } from '@/types/components';
import { groupComponentsByCategory } from '@/utils/components';
import clsx from 'clsx';

const iconMap = {
  CircuitBoard,
  Activity,
  Signal,
  Minus,
  Sigma,
};

export function ComponentLibrary() {
  const [search, setSearch] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['Neural Networks'])
  );
  const { components, isLoading, error } = useComponents();

  const byCategory = useMemo(() => {
    const filtered = search
      ? components.filter((component) =>
          component.name.toLowerCase().includes(search.toLowerCase()) ||
          component.description.toLowerCase().includes(search.toLowerCase())
        )
      : components;

    return groupComponentsByCategory(filtered);
  }, [search]);

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
        {Object.entries(byCategory).map(([category, components]) => (
          <CategorySection
            key={category}
            category={category}
            components={components}
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
        <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center">
          <Icon className="w-4 h-4 text-slate-600" />
        </div>
        <div>
          <div className="text-sm font-semibold text-slate-800">{component.name}</div>
          <div className="text-xs text-slate-500">{component.description}</div>
        </div>
      </div>
    </div>
  );
}
