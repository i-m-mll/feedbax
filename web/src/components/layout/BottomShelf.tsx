import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import { useLayoutStore, SHELF_HEADER_HEIGHT } from '@/stores/layoutStore';
import { TrainingPanel } from '@/components/panels/TrainingPanel';
import { ValidationPanel } from '@/components/panels/ValidationPanel';
import { AnalysisPanel } from '@/components/panels/AnalysisPanel';
import { ChevronDown } from 'lucide-react';

const tabs = [
  { id: 'validation', label: 'Validation' },
  { id: 'training', label: 'Training' },
  { id: 'analysis', label: 'Analysis' },
] as const;

type TabId = (typeof tabs)[number]['id'];

export function BottomShelf({
  height,
  availableHeight,
}: {
  height: number;
  availableHeight: number;
}) {
  const [activeTab, setActiveTab] = useState<TabId>('validation');
  const { bottomCollapsed, bottomHeight, toggleBottom, setBottomHeight, topCollapsed } =
    useLayoutStore();
  const tabsRef = useRef<HTMLDivElement | null>(null);
  const [fadeState, setFadeState] = useState({ left: false, right: false });

  const activeContent = useMemo(() => {
    if (activeTab === 'training') return <TrainingPanel />;
    if (activeTab === 'analysis') return <AnalysisPanel />;
    return <ValidationPanel />;
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
      {!bottomCollapsed && !topCollapsed && (
        <div
          className="absolute -top-2 left-1/2 h-4 w-12 -translate-x-1/2 cursor-row-resize rounded-full bg-slate-200/70"
          onPointerDown={(event) => {
            const startY = event.clientY;
            const startHeight = bottomHeight;
            const onMove = (moveEvent: PointerEvent) => {
              const delta = startY - moveEvent.clientY;
              setBottomHeight(startHeight + delta, availableHeight);
            };
            const onUp = () => {
              window.removeEventListener('pointermove', onMove);
              window.removeEventListener('pointerup', onUp);
            };
            window.addEventListener('pointermove', onMove);
            window.addEventListener('pointerup', onUp);
          }}
        />
      )}
      <div
        className="flex items-center justify-between px-4 gap-4"
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
        <button
          className="p-1 rounded-full text-slate-500 hover:text-slate-700 hover:bg-slate-100"
          title={bottomCollapsed ? 'Expand workbench shelf' : 'Collapse workbench shelf'}
          onClick={() => toggleBottom(availableHeight)}
        >
          <ChevronDown
            className={`w-4 h-4 transition-transform ${bottomCollapsed ? 'rotate-180' : ''}`}
          />
        </button>
      </div>
      {!bottomCollapsed && (
        <div style={{ height: Math.max(0, height - SHELF_HEADER_HEIGHT) }} className="overflow-y-auto">
          {activeContent}
        </div>
      )}
    </section>
  );
}
