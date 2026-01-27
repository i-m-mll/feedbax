import { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import { PropertiesPanel } from '@/components/panels/PropertiesPanel';
import { TrainingPanel } from '@/components/panels/TrainingPanel';
import { InspectorPanel } from '@/components/panels/InspectorPanel';

const tabs = [
  { id: 'properties', label: 'Properties' },
  { id: 'training', label: 'Training' },
  { id: 'inspector', label: 'Inspector' },
] as const;

type TabId = (typeof tabs)[number]['id'];

export function RightPanel() {
  const [activeTab, setActiveTab] = useState<TabId>('properties');
  const tabsRef = useRef<HTMLDivElement | null>(null);
  const [showLeftFade, setShowLeftFade] = useState(false);
  const [showRightFade, setShowRightFade] = useState(false);

  useEffect(() => {
    const checkOverflow = () => {
      const el = tabsRef.current;
      if (!el) return;
      const canScroll = el.scrollWidth > el.clientWidth + 1;
      const left = el.scrollLeft > 0;
      const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
      setShowLeftFade(canScroll && left);
      setShowRightFade(canScroll && right);
    };

    checkOverflow();
    window.addEventListener('resize', checkOverflow);
    return () => window.removeEventListener('resize', checkOverflow);
  }, [activeTab]);

  return (
    <aside className="w-80 max-w-full border-l border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col overflow-x-hidden">
      <div className="relative">
        <div
          ref={tabsRef}
          className="flex gap-2 px-4 pt-4 pb-2 overflow-x-auto whitespace-nowrap"
          onScroll={() => {
            const el = tabsRef.current;
            if (!el) return;
            const canScroll = el.scrollWidth > el.clientWidth + 1;
            const left = el.scrollLeft > 0;
            const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
            setShowLeftFade(canScroll && left);
            setShowRightFade(canScroll && right);
          }}
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex-shrink-0 text-xs font-semibold uppercase tracking-[0.2em] px-3 py-2 rounded-full border',
                activeTab === tab.id
                  ? 'border-brand-500 text-brand-600 bg-brand-500/10'
                  : 'border-transparent text-slate-400 hover:text-slate-600 hover:bg-slate-100'
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>
        {showLeftFade && (
          <div className="pointer-events-none absolute left-0 top-4 h-8 w-10 bg-gradient-to-r from-white via-white/70 to-transparent" />
        )}
        {showRightFade && (
          <div className="pointer-events-none absolute right-0 top-4 h-8 w-10 bg-gradient-to-l from-white via-white/70 to-transparent" />
        )}
      </div>
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'properties' && <PropertiesPanel />}
        {activeTab === 'training' && <TrainingPanel />}
        {activeTab === 'inspector' && <InspectorPanel />}
      </div>
    </aside>
  );
}
