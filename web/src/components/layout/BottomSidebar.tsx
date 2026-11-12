/**
 * BottomSidebar — left sidebar within the bottom pane content area.
 *
 * Renders the AnalysisLibrary when the Analysis tab is active.
 * Visual language matches the top Sidebar exactly: border style, background,
 * blur, and resize handle.
 */

import { AnalysisLibrary } from '@/components/panels/AnalysisLibrary';
import type { PaneLayout } from '@window-shell/browser/layout';
import { useLayoutStore } from '@/stores/layoutStore';
import { ResizeSeparator } from './ResizeSeparator';

export function BottomSidebar({ layout, onPreview }: {
  layout: PaneLayout;
  onPreview: (size: number | null) => void;
}) {
  const { bottomSidebarCollapsed, setBottomSidebarWidth } = useLayoutStore();
  if (bottomSidebarCollapsed) return null;

  return (
    <aside
      id="studio-analysis-library"
      aria-label="Analysis library"
      hidden={!layout.open}
      style={{ width: layout.size, display: layout.open ? undefined : 'none' }}
      className="border-r border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col relative shrink-0"
    >
      <div className="px-4 pt-4 pb-2 flex items-center justify-between">
        <span className="text-xs uppercase tracking-[0.2em] bg-emerald-50 text-emerald-700 font-semibold px-2 py-1 rounded">
          Analyses
        </span>
      </div>
      <AnalysisLibrary />
      {layout.open && <ResizeSeparator className="absolute inset-y-0 right-0 w-1 cursor-col-resize hover:bg-shell-hover"
        label="Resize analysis library" controls="studio-analysis-library" axis="x" direction={1}
        layout={layout} onPreview={onPreview} onResize={setBottomSidebarWidth} />}
    </aside>
  );
}
