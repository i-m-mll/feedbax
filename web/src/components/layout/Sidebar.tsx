import { useState } from 'react';
import { ComponentLibrary } from '@/components/panels/ComponentLibrary';
import { TaskLibrary } from '@/components/panels/TaskLibrary';
import { AnalysisLibrary } from '@/components/panels/AnalysisLibrary';
import { useLayoutStore } from '@/stores/layoutStore';
import { PanelLeftOpen, PanelLeftClose } from 'lucide-react';
import clsx from 'clsx';

type ActiveTab = 'components' | 'tasks' | 'analyses';

export function Sidebar() {
  const { leftSidebarWidth, leftSidebarVisible, toggleLeftSidebar, setLeftSidebarWidth } =
    useLayoutStore();
  const [activeTab, setActiveTab] = useState<ActiveTab>('components');

  if (!leftSidebarVisible) {
    return (
      <div className="relative flex items-center">
        <button
          onClick={toggleLeftSidebar}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-10 p-1 rounded-r bg-slate-100 hover:bg-slate-200 text-slate-400 hover:text-slate-600"
          title="Show component library"
        >
          <PanelLeftOpen className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <aside
      style={{ width: leftSidebarWidth }}
      className="border-r border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col relative shrink-0"
    >
      <div className="px-4 pt-4 pb-2 flex items-center justify-between">
        <div className="flex gap-1">
          <button
            onClick={() => setActiveTab('components')}
            className={clsx(
              'text-xs uppercase tracking-[0.2em] px-2 py-1 rounded transition-colors',
              activeTab === 'components'
                ? 'bg-slate-100 text-slate-700 font-semibold'
                : 'text-slate-400 hover:text-slate-600'
            )}
          >
            Components
          </button>
          <button
            onClick={() => setActiveTab('tasks')}
            className={clsx(
              'text-xs uppercase tracking-[0.2em] px-2 py-1 rounded transition-colors',
              activeTab === 'tasks'
                ? 'bg-slate-100 text-slate-700 font-semibold'
                : 'text-slate-400 hover:text-slate-600'
            )}
          >
            Tasks
          </button>
          <button
            onClick={() => setActiveTab('analyses')}
            className={clsx(
              'text-xs uppercase tracking-[0.2em] px-2 py-1 rounded transition-colors',
              activeTab === 'analyses'
                ? 'bg-emerald-50 text-emerald-700 font-semibold'
                : 'text-slate-400 hover:text-slate-600'
            )}
          >
            Analyses
          </button>
        </div>
        <button
          onClick={toggleLeftSidebar}
          className="p-1 rounded text-slate-400 hover:text-slate-600"
          title="Hide sidebar"
        >
          <PanelLeftClose className="w-3.5 h-3.5" />
        </button>
      </div>
      {activeTab === 'components' && <ComponentLibrary />}
      {activeTab === 'tasks' && <TaskLibrary />}
      {activeTab === 'analyses' && <AnalysisLibrary />}
      <div
        className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-brand-300/50 active:bg-brand-400/50"
        onPointerDown={(e) => {
          e.preventDefault();
          const startX = e.clientX;
          const startWidth = leftSidebarWidth;
          const onMove = (me: PointerEvent) => {
            setLeftSidebarWidth(startWidth + (me.clientX - startX));
          };
          const onUp = () => {
            window.removeEventListener('pointermove', onMove);
            window.removeEventListener('pointerup', onUp);
          };
          window.addEventListener('pointermove', onMove);
          window.addEventListener('pointerup', onUp);
        }}
      />
    </aside>
  );
}
