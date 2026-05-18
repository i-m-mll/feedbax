import { Sidebar } from '@/components/layout/Sidebar';
import { RightPanel } from '@/components/panels/RightPanel';
import { Canvas } from '@/components/canvas/Canvas';
import { useLayoutStore } from '@/stores/layoutStore';
import { getActiveStage, getScenario, useWorkspaceStore } from '@/stores/workspaceStore';

export function TopShelf({ height }: { height: number }) {
  const { topCollapsed } = useLayoutStore();
  const workspace = useWorkspaceStore((state) => state.workspace);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);

  return (
    <section
      className="flex flex-col h-full min-h-0 bg-white/80 backdrop-blur-sm"
      style={{ height }}
    >
      {!topCollapsed && (
        <div className="flex flex-1 min-h-0">
          <Sidebar />
          <main className="relative flex-1 min-h-0">
            <div className="absolute inset-0">
              <Canvas />
            </div>
            {activeStage && (
              <div className="pointer-events-none absolute left-4 top-4 z-10 max-w-[min(28rem,calc(100%-2rem))] rounded border border-slate-200 bg-white/90 px-3 py-2 shadow-sm backdrop-blur">
                <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.22em] text-slate-400">
                  <span>{activeStage.kind}</span>
                  <span className="h-1 w-1 rounded-full bg-slate-300" />
                  <span>{activeStage.status}</span>
                </div>
                <div className="mt-0.5 truncate text-sm font-semibold text-slate-800">
                  {activeStage.label}
                </div>
                {activeScenario && (
                  <div className="truncate text-xs text-slate-500">
                    {activeScenario.label}
                  </div>
                )}
              </div>
            )}
          </main>
          <RightPanel />
        </div>
      )}
    </section>
  );
}
