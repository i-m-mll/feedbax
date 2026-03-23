/**
 * AnalysisLibrary — sidebar panel listing available analysis types.
 *
 * Mirrors the ComponentLibrary pattern: search/filter, categorized sections,
 * draggable cards. Uses the 'application/feedbax-analysis' drag data type
 * (parallel to 'application/feedbax-component' in the model library).
 */

import { useEffect, useMemo, useState, type DragEvent } from 'react';
import {
  BarChart3,
  TrendingUp,
  Grid3x3,
  Calculator,
  GitCompare,
  Filter,
  SlidersHorizontal,
  ChevronDown,
  ChevronRight,
  CircuitBoard,
} from 'lucide-react';
import { useAnalysisStore } from '@/stores/analysisStore';
import { fetchAnalysisClasses } from '@/api/analysisAPI';
import type { AnalysisClassDef } from '@/types/analysis';

// Icon mapping — matches icons from the stub data
const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  BarChart3,
  TrendingUp,
  Grid3x3,
  Calculator,
  GitCompare,
  Filter,
  SlidersHorizontal,
  // Fallbacks for icon names that don't have exact matches in lucide
  Axis3d: TrendingUp,
  Scatter: GitCompare,
};

function groupByCategory(classes: AnalysisClassDef[]): Record<string, AnalysisClassDef[]> {
  const groups: Record<string, AnalysisClassDef[]> = {};
  for (const cls of classes) {
    const cat = cls.category || 'Other';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(cls);
  }
  return groups;
}

export function AnalysisLibrary() {
  const [search, setSearch] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['Visualization'])
  );
  const { analysisClasses, setAnalysisClasses } = useAnalysisStore();

  // Load classes on mount if not already loaded
  useEffect(() => {
    if (analysisClasses.length > 0) return;
    fetchAnalysisClasses().then(setAnalysisClasses).catch(() => {});
  }, [analysisClasses.length, setAnalysisClasses]);

  const filtered = useMemo(() => {
    if (!search) return analysisClasses;
    const lower = search.toLowerCase();
    return analysisClasses.filter(
      (cls) =>
        cls.name.toLowerCase().includes(lower) ||
        cls.description.toLowerCase().includes(lower) ||
        cls.category.toLowerCase().includes(lower)
    );
  }, [analysisClasses, search]);

  const categories = useMemo<Record<string, AnalysisClassDef[]>>(
    () => groupByCategory(filtered),
    [filtered]
  );

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
          placeholder="Search analyses..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
        />
      </div>
      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-3">
        {analysisClasses.length === 0 && (
          <div className="text-xs text-slate-400">Loading analyses...</div>
        )}
        {(Object.entries(categories) as [string, AnalysisClassDef[]][]).map(([category, classes]) => (
          <div key={category} className="space-y-2">
            <button
              onClick={() => toggleCategory(category)}
              className="w-full flex items-center justify-between text-left text-xs font-semibold text-slate-500 uppercase tracking-[0.2em]"
            >
              {category}
              {expandedCategories.has(category) ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
            </button>
            {expandedCategories.has(category) && (
              <div className="space-y-2">
                {classes.map((cls) => (
                  <AnalysisCard key={cls.name} classDef={cls} />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function AnalysisCard({ classDef }: { classDef: AnalysisClassDef }) {
  const Icon = iconMap[classDef.icon] ?? CircuitBoard;
  const isPreprocessing = classDef.category === 'Preprocessing';

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData('application/feedbax-analysis', classDef.name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className={`rounded-xl bg-white/90 p-3 shadow-soft cursor-grab transition border ${
        isPreprocessing
          ? 'border-slate-100 hover:border-slate-200 hover:-translate-y-0.5'
          : 'border-emerald-100 hover:border-emerald-300 hover:-translate-y-0.5 hover:shadow'
      }`}
    >
      <div className="flex items-center gap-2">
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
            isPreprocessing ? 'bg-slate-100' : 'bg-emerald-50'
          }`}
        >
          <Icon
            className={`w-4 h-4 ${
              isPreprocessing ? 'text-slate-500' : 'text-emerald-600'
            }`}
          />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            <div className="text-sm font-semibold text-slate-800 truncate">
              {classDef.name}
            </div>
            {isPreprocessing && (
              <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-500">
                Prep
              </span>
            )}
          </div>
          <div className="text-xs text-slate-500 line-clamp-2">{classDef.description}</div>
        </div>
      </div>
    </div>
  );
}
