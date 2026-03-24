/**
 * AnalysisLibrary — sidebar panel listing available analysis types.
 *
 * Mirrors the ComponentLibrary pattern: search/filter, categorized sections,
 * draggable cards. Uses the 'application/feedbax-analysis' drag data type
 * (parallel to 'application/feedbax-component' in the model library).
 */

import { useEffect, useMemo, useState, type DragEvent } from 'react';
import {
  // Preprocessing icons
  Trophy,
  ListFilter,
  ArrowDownToLine,
  ArrowUpToLine,
  ListOrdered,
  Ungroup,
  FolderOpen,
  ArrowUpDown,
  Group,
  SplitSquareHorizontal,
  Compass,
  // Computation icons
  AlignHorizontalDistributeCenter,
  FunctionSquare,
  Equal,
  Workflow,
  // Decomposition icons
  Grid3x3,
  Columns3,
  Orbit,
  // Dynamics icons
  Crosshair,
  Waves,
  AudioWaveform,
  Radar,
  // Visualization icons
  BarChart3,
  TrendingUp,
  Route,
  ScatterChart,
  CircleDot,
  BrainCircuit,
  Box,
  // UI chrome
  ChevronDown,
  ChevronRight,
  CircuitBoard,
  // Category header icons
  SlidersHorizontal,
  Calculator,
  Atom,
  Activity,
  BarChart,
} from 'lucide-react';
import { useAnalysisStore } from '@/stores/analysisStore';
import { fetchAnalysisClasses } from '@/api/analysisAPI';
import type { AnalysisClassDef } from '@/types/analysis';

// ---------------------------------------------------------------------------
// Icon mapping — keyed by the `icon` string in AnalysisClassDef
// ---------------------------------------------------------------------------

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  // Preprocessing
  Trophy,
  ListFilter,
  ArrowDownToLine,
  ArrowUpToLine,
  ListOrdered,
  Ungroup,
  FolderOpen,
  ArrowUpDown,
  Group,
  SplitSquareHorizontal,
  Compass,
  // Computation
  AlignHorizontalDistributeCenter,
  FunctionSquare,
  Equal,
  Workflow,
  // Decomposition
  Grid3x3,
  Columns3,
  Orbit,
  Axis3d: Orbit, // alias
  // Dynamics
  Crosshair,
  Waves,
  AudioWaveform,
  Radar,
  // Visualization
  BarChart3,
  TrendingUp,
  Route,
  ScatterChart,
  CircleDot,
  BrainCircuit,
  Box,
};

// ---------------------------------------------------------------------------
// Per-category styling — colors, header icons, badge labels
// ---------------------------------------------------------------------------

interface CategoryStyle {
  headerIcon: React.ComponentType<{ className?: string }>;
  iconBg: string;
  iconText: string;
  borderDefault: string;
  borderHover: string;
  badge?: { label: string; bg: string; text: string };
}

const CATEGORY_STYLES: Record<string, CategoryStyle> = {
  Preprocessing: {
    headerIcon: SlidersHorizontal,
    iconBg: 'bg-slate-100',
    iconText: 'text-slate-500',
    borderDefault: 'border-slate-100',
    borderHover: 'hover:border-slate-200',
    badge: { label: 'Prep', bg: 'bg-slate-100', text: 'text-slate-500' },
  },
  Computation: {
    headerIcon: Calculator,
    iconBg: 'bg-blue-50',
    iconText: 'text-blue-600',
    borderDefault: 'border-blue-100',
    borderHover: 'hover:border-blue-300',
  },
  Decomposition: {
    headerIcon: Atom,
    iconBg: 'bg-violet-50',
    iconText: 'text-violet-600',
    borderDefault: 'border-violet-100',
    borderHover: 'hover:border-violet-300',
  },
  Dynamics: {
    headerIcon: Activity,
    iconBg: 'bg-amber-50',
    iconText: 'text-amber-600',
    borderDefault: 'border-amber-100',
    borderHover: 'hover:border-amber-300',
  },
  Visualization: {
    headerIcon: BarChart,
    iconBg: 'bg-emerald-50',
    iconText: 'text-emerald-600',
    borderDefault: 'border-emerald-100',
    borderHover: 'hover:border-emerald-300',
  },
};

const DEFAULT_STYLE: CategoryStyle = {
  headerIcon: CircuitBoard,
  iconBg: 'bg-slate-50',
  iconText: 'text-slate-500',
  borderDefault: 'border-slate-100',
  borderHover: 'hover:border-slate-200',
};

// Canonical display order for categories
const CATEGORY_ORDER = [
  'Preprocessing',
  'Computation',
  'Decomposition',
  'Dynamics',
  'Visualization',
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function groupByCategory(classes: AnalysisClassDef[]): Record<string, AnalysisClassDef[]> {
  const groups: Record<string, AnalysisClassDef[]> = {};
  for (const cls of classes) {
    const cat = cls.category || 'Other';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(cls);
  }
  return groups;
}

function sortedCategoryEntries(
  groups: Record<string, AnalysisClassDef[]>,
): [string, AnalysisClassDef[]][] {
  const ordered: [string, AnalysisClassDef[]][] = [];
  for (const cat of CATEGORY_ORDER) {
    if (groups[cat]) ordered.push([cat, groups[cat]]);
  }
  // Append any categories not in the canonical order
  for (const [cat, classes] of Object.entries(groups)) {
    if (!CATEGORY_ORDER.includes(cat)) ordered.push([cat, classes]);
  }
  return ordered;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

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

  // When searching, auto-expand all categories that contain matches
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

  const sortedCategories = useMemo(
    () => sortedCategoryEntries(categories),
    [categories]
  );

  // Auto-expand matching categories while searching
  useEffect(() => {
    if (search) {
      setExpandedCategories(new Set(Object.keys(categories)));
    }
  }, [search, categories]);

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

  const expandAll = () => setExpandedCategories(new Set(Object.keys(categories)));
  const collapseAll = () => setExpandedCategories(new Set());

  return (
    <div className="flex flex-col h-full overflow-x-hidden">
      <div className="px-4 pb-3 space-y-2">
        <input
          type="text"
          placeholder="Search analyses..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
        />
        <div className="flex gap-2">
          <button
            onClick={expandAll}
            className="text-[10px] text-slate-400 hover:text-slate-600 uppercase tracking-wide"
          >
            Expand all
          </button>
          <span className="text-[10px] text-slate-300">|</span>
          <button
            onClick={collapseAll}
            className="text-[10px] text-slate-400 hover:text-slate-600 uppercase tracking-wide"
          >
            Collapse all
          </button>
          <span className="ml-auto text-[10px] text-slate-400">
            {filtered.length} {filtered.length === 1 ? 'type' : 'types'}
          </span>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-3">
        {analysisClasses.length === 0 && (
          <div className="text-xs text-slate-400">Loading analyses...</div>
        )}
        {sortedCategories.map(([category, classes]) => {
          const style = CATEGORY_STYLES[category] ?? DEFAULT_STYLE;
          const HeaderIcon = style.headerIcon;
          return (
            <div key={category} className="space-y-2">
              <button
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center gap-1.5 text-left text-xs font-semibold text-slate-500 uppercase tracking-[0.2em]"
              >
                <HeaderIcon className="w-3.5 h-3.5 shrink-0" />
                <span className="flex-1">{category}</span>
                <span className="text-[10px] font-normal normal-case tracking-normal text-slate-400">
                  {classes.length}
                </span>
                {expandedCategories.has(category) ? (
                  <ChevronDown className="w-3 h-3 shrink-0" />
                ) : (
                  <ChevronRight className="w-3 h-3 shrink-0" />
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
          );
        })}
      </div>
    </div>
  );
}

function AnalysisCard({ classDef }: { classDef: AnalysisClassDef }) {
  const Icon = iconMap[classDef.icon] ?? CircuitBoard;
  const style = CATEGORY_STYLES[classDef.category] ?? DEFAULT_STYLE;

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData('application/feedbax-analysis', classDef.name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className={`rounded-xl bg-white/90 p-3 shadow-soft cursor-grab transition border hover:-translate-y-0.5 hover:shadow ${style.borderDefault} ${style.borderHover}`}
    >
      <div className="flex items-center gap-2">
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${style.iconBg}`}
        >
          <Icon className={`w-4 h-4 ${style.iconText}`} />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            <div className="text-sm font-semibold text-slate-800 truncate">
              {classDef.name}
            </div>
            {style.badge && (
              <span
                className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wide ${style.badge.bg} ${style.badge.text}`}
              >
                {style.badge.label}
              </span>
            )}
          </div>
          <div className="text-xs text-slate-500 line-clamp-2">{classDef.description}</div>
        </div>
      </div>
    </div>
  );
}
