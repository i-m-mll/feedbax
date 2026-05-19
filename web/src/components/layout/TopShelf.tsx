import { Sidebar } from '@/components/layout/Sidebar';
import { RightPanel } from '@/components/panels/RightPanel';
import {
  ScenarioProjectionToolbar,
  ScenarioProjectionWorkspace,
} from '@/components/scenario/ScenarioProjectionWorkspace';
import { useLayoutStore } from '@/stores/layoutStore';

export function TopShelf({ height }: { height: number }) {
  const { topCollapsed } = useLayoutStore();

  return (
    <section
      className="flex flex-col h-full min-h-0 bg-white/80 backdrop-blur-sm"
      style={{ height }}
    >
      {!topCollapsed && (
        <>
          <ScenarioProjectionToolbar />
          <div className="flex flex-1 min-h-0">
            <Sidebar />
            <main className="relative flex-1 min-h-0">
              <ScenarioProjectionWorkspace />
            </main>
            <RightPanel />
          </div>
        </>
      )}
    </section>
  );
}
