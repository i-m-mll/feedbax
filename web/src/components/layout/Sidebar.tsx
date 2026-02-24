import { ComponentLibrary } from '@/components/panels/ComponentLibrary';
import { useLayoutStore } from '@/stores/layoutStore';
import { PanelLeftOpen, PanelLeftClose } from 'lucide-react';

export function Sidebar() {
  const { leftSidebarWidth, leftSidebarVisible, toggleLeftSidebar, setLeftSidebarWidth } =
    useLayoutStore();

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
        <h2 className="font-display text-xs uppercase tracking-[0.3em] text-slate-400">
          Components
        </h2>
        <button
          onClick={toggleLeftSidebar}
          className="p-1 rounded text-slate-400 hover:text-slate-600"
          title="Hide sidebar"
        >
          <PanelLeftClose className="w-3.5 h-3.5" />
        </button>
      </div>
      <ComponentLibrary />
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
