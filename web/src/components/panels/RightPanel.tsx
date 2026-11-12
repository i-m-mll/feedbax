import type { PaneLayout } from '@window-shell/browser/layout';
import { ResizeSeparator } from '@/components/layout/ResizeSeparator';
import { ScenarioInspectorPanel } from '@/components/panels/ScenarioInspectorPanel';
import { TreescopePanel } from '@/components/panels/TreescopePanel';
import { ValidationPanel } from '@/components/panels/ValidationPanel';
import { useLayoutStore } from '@/stores/layoutStore';

export function RightPanel({ layout, onPreview }: {
  layout: PaneLayout;
  onPreview: (size: number | null) => void;
}) {
  const { rightSidebarVisible, setRightSidebarWidth } = useLayoutStore();
  if (!rightSidebarVisible) return null;

  return (
    <aside
      id="studio-properties"
      aria-label="Properties"
      hidden={!layout.open}
      style={{ width: layout.size, display: layout.open ? undefined : 'none' }}
      className="max-w-full border-l border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col overflow-x-hidden relative shrink-0"
    >
      <div className="px-4 pt-4">
        <div className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">
          Properties
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        <ScenarioInspectorPanel />
        <TreescopePanel />
        <ValidationPanel />
      </div>
      {layout.open && <ResizeSeparator className="absolute inset-y-0 left-0 w-1 cursor-col-resize hover:bg-shell-hover"
        label="Resize properties" controls="studio-properties" axis="x" direction={-1}
        layout={layout} onPreview={onPreview} onResize={setRightSidebarWidth} />}
    </aside>
  );
}
