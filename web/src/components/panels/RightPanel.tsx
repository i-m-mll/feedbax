import { PropertiesPanel } from '@/components/panels/PropertiesPanel';
import { TreescopePanel } from '@/components/panels/TreescopePanel';
import { useLayoutStore } from '@/stores/layoutStore';
import { PanelRightOpen, PanelRightClose } from 'lucide-react';

export function RightPanel() {
  const { rightSidebarWidth, rightSidebarVisible, toggleRightSidebar, setRightSidebarWidth } =
    useLayoutStore();

  if (!rightSidebarVisible) {
    return (
      <div className="relative flex items-center">
        <button
          onClick={toggleRightSidebar}
          className="absolute right-0 top-1/2 -translate-y-1/2 z-10 p-1 rounded-l bg-slate-100 hover:bg-slate-200 text-slate-400 hover:text-slate-600"
          title="Show properties panel"
        >
          <PanelRightOpen className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <aside
      style={{ width: rightSidebarWidth }}
      className="max-w-full border-l border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col overflow-x-hidden relative shrink-0"
    >
      <div className="px-4 pt-4 flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">
          Properties
        </div>
        <button
          onClick={toggleRightSidebar}
          className="p-1 rounded text-slate-400 hover:text-slate-600"
          title="Hide panel"
        >
          <PanelRightClose className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        <PropertiesPanel />
        <TreescopePanel />
      </div>
      <div
        className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-brand-300/50 active:bg-brand-400/50"
        onPointerDown={(e) => {
          e.preventDefault();
          const startX = e.clientX;
          const startWidth = rightSidebarWidth;
          const onMove = (me: PointerEvent) => {
            setRightSidebarWidth(startWidth - (me.clientX - startX));
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
