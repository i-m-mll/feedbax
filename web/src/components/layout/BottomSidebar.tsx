/**
 * BottomSidebar — left sidebar within the bottom pane content area.
 *
 * Renders the AnalysisLibrary when the Analysis tab is active.
 * Visual language matches the top Sidebar exactly: border style, background,
 * blur, resize handle, and collapse behavior.
 */

import { AnalysisLibrary } from '@/components/panels/AnalysisLibrary';
import { useLayoutStore } from '@/stores/layoutStore';
import { PanelLeftOpen, PanelLeftClose } from 'lucide-react';

export function BottomSidebar() {
  const {
    bottomSidebarWidth,
    bottomSidebarCollapsed,
    toggleBottomSidebar,
    setBottomSidebarWidth,
  } = useLayoutStore();

  if (bottomSidebarCollapsed) {
    return (
      <div className="relative flex items-center">
        <button
          onClick={toggleBottomSidebar}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-10 p-1 rounded-r bg-slate-100 hover:bg-slate-200 text-slate-400 hover:text-slate-600"
          title="Show analysis library"
        >
          <PanelLeftOpen className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <aside
      style={{ width: bottomSidebarWidth }}
      className="border-r border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col relative shrink-0"
    >
      <div className="px-4 pt-4 pb-2 flex items-center justify-between">
        <span className="text-xs uppercase tracking-[0.2em] bg-emerald-50 text-emerald-700 font-semibold px-2 py-1 rounded">
          Analyses
        </span>
        <button
          onClick={toggleBottomSidebar}
          className="p-1 rounded text-slate-400 hover:text-slate-600"
          title="Hide sidebar"
        >
          <PanelLeftClose className="w-3.5 h-3.5" />
        </button>
      </div>
      <AnalysisLibrary />
      {/* Resize handle — right edge */}
      <div
        className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-brand-300/50 active:bg-brand-400/50"
        onPointerDown={(e) => {
          e.preventDefault();
          const startX = e.clientX;
          const startWidth = bottomSidebarWidth;
          const onMove = (me: PointerEvent) => {
            setBottomSidebarWidth(startWidth + (me.clientX - startX));
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
