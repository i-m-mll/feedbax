import { useState } from 'react';
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

  return (
    <aside className="w-80 border-l border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col">
      <div className="flex gap-2 px-4 pt-4">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={clsx(
              'text-xs font-semibold uppercase tracking-[0.2em] px-3 py-2 rounded-full border',
              activeTab === tab.id
                ? 'border-brand-500 text-brand-600 bg-brand-500/10'
                : 'border-transparent text-slate-400 hover:text-slate-600 hover:bg-slate-100'
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'properties' && <PropertiesPanel />}
        {activeTab === 'training' && <TrainingPanel />}
        {activeTab === 'inspector' && <InspectorPanel />}
      </div>
    </aside>
  );
}
