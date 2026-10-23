import { ScenarioInspectorPanel } from '@/components/panels/ScenarioInspectorPanel';
import { TreescopePanel } from '@/components/panels/TreescopePanel';
import { ValidationPanel } from '@/components/panels/ValidationPanel';
import { useLayoutStore } from '@/stores/layoutStore';

export function RightPanel() {
  const { rightSidebarWidth, rightSidebarVisible, setRightSidebarWidth } =
    useLayoutStore();

  if (!rightSidebarVisible) {
    return null;
  }

  return (
    <aside
      style={{ width: rightSidebarWidth }}
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
