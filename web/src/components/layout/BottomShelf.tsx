import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import { useLayoutStore, SHELF_HEADER_HEIGHT } from '@/stores/layoutStore';
import { TrainingPanel } from '@/components/panels/TrainingPanel';
import { AnalysisPanel } from '@/components/panels/AnalysisPanel';
import { TrajectoryPanel } from '@/components/panels/TrajectoryPanel';
import { StatisticsPanel } from '@/components/panels/StatisticsPanel';
import { ConsolePanel } from '@/components/panels/ConsolePanel';

const tabs = [
  { id: 'training', label: 'Training' },
  { id: 'console', label: 'Console' },
  { id: 'analysis', label: 'Analysis' },
  { id: 'trajectories', label: 'Trajectories' },
  { id: 'statistics', label: 'Statistics' },
] as const;

type TabId = (typeof tabs)[number]['id'];

export function BottomShelf({
  height,
  availableHeight,
}: {
  height: number;
  availableHeight: number;
}) {
  const [activeTab, setActiveTab] = useState<TabId>('training');
  const { bottomCollapsed, toggleBottom } = useLayoutStore();
  const tabsRef = useRef<HTMLDivElement | null>(null);
  const [fadeState, setFadeState] = useState({ left: false, right: false });

  const activeContent = useMemo(() => {
    if (activeTab === 'training') return <TrainingPanel />;
    if (activeTab === 'console') return <ConsolePanel />;
    if (activeTab === 'analysis') return <AnalysisPanel />;
    if (activeTab === 'trajectories') return <TrajectoryPanel />;
    return <StatisticsPanel />;
  }, [activeTab]);

  const updateFades = useCallback(() => {
    const el = tabsRef.current;
    if (!el) return;
    const left = el.scrollLeft > 4;
    const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 4;
    setFadeState({ left, right });
  }, []);

  useEffect(() => {
    updateFades();
    const el = tabsRef.current;
    if (!el) return;
    const handle = () => updateFades();
    el.addEventListener('scroll', handle);
    window.addEventListener('resize', handle);
    return () => {
      el.removeEventListener('scroll', handle);
      window.removeEventListener('resize', handle);
    };
  }, [bottomCollapsed, updateFades]);

  return (
    <section
      className="relative bg-white/90 backdrop-blur-sm border-t border-slate-100"
      style={{ height }}
    >
      <div
        className="flex items-center px-4 gap-4"
        style={{ height: SHELF_HEADER_HEIGHT }}
      >
        <div className="relative flex-1 min-w-0">
          <div ref={tabsRef} className="flex items-center gap-2 overflow-x-auto pr-6">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => {
                  if (bottomCollapsed) toggleBottom(availableHeight);
                  setActiveTab(tab.id);
                }}
                className={clsx(
                  'text-xs font-semibold uppercase tracking-[0.2em] px-3 py-1.5 rounded-full border whitespace-nowrap',
                  activeTab === tab.id
                    ? 'border-brand-500 text-brand-600 bg-brand-500/10'
                    : 'border-transparent text-slate-400 hover:text-slate-600 hover:bg-slate-100'
                )}
              >
                {tab.label}
              </button>
            ))}
          </div>
          {fadeState.left && (
            <div className="pointer-events-none absolute left-0 top-0 h-full w-6 bg-gradient-to-r from-white/90 to-transparent" />
          )}
          {fadeState.right && (
            <div className="pointer-events-none absolute right-0 top-0 h-full w-8 bg-gradient-to-l from-white/90 to-transparent" />
          )}
        </div>
      </div>
      {!bottomCollapsed && (
        <div
          style={{ height: Math.max(0, height - SHELF_HEADER_HEIGHT) }}
          className={activeTab === 'trajectories' || activeTab === 'statistics' || activeTab === 'console' || activeTab === 'analysis' ? 'overflow-hidden' : 'overflow-y-auto'}
        >
          {activeContent}
        </div>
      )}
    </section>
  );
}
