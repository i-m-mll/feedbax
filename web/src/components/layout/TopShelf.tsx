import { Sidebar } from '@/components/layout/Sidebar';
import { RightPanel } from '@/components/panels/RightPanel';
import { Canvas } from '@/components/canvas/Canvas';
import { useLayoutStore, SHELF_HEADER_HEIGHT } from '@/stores/layoutStore';
import { ChevronUp } from 'lucide-react';

export function TopShelf({
  height,
  availableHeight,
}: {
  height: number;
  availableHeight: number;
}) {
  const { topCollapsed, toggleTop } = useLayoutStore();
  return (
    <section
      className="flex flex-col h-full min-h-0 border-b border-slate-100 bg-white/80 backdrop-blur-sm"
      style={{ height }}
    >
      <div
        className="flex items-center justify-between px-4 border-b border-slate-100"
        style={{ height: SHELF_HEADER_HEIGHT }}
      >
        <div className="flex items-center gap-3 text-xs uppercase tracking-[0.28em] text-slate-400">
          <span className="text-slate-500 font-semibold tracking-[0.32em]">Model</span>
          <span className="text-slate-300">Canvas</span>
        </div>
        <button
          className="p-1 rounded-full text-slate-500 hover:text-slate-700 hover:bg-slate-100"
          title={topCollapsed ? 'Expand model shelf' : 'Collapse model shelf'}
          onClick={() => toggleTop(availableHeight)}
        >
          <ChevronUp
            className={`w-4 h-4 transition-transform ${topCollapsed ? 'rotate-180' : ''}`}
          />
        </button>
      </div>
      {!topCollapsed && (
        <div className="flex flex-1 min-h-0">
          <Sidebar />
          <main className="relative flex-1 min-h-0">
            <div className="absolute inset-0">
              <Canvas />
            </div>
          </main>
          <RightPanel />
        </div>
      )}
    </section>
  );
}
