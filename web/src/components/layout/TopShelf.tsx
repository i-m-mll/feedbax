import { Sidebar } from '@/components/layout/Sidebar';
import { RightPanel } from '@/components/panels/RightPanel';
import { TaskScenarioPanel } from '@/components/panels/TaskScenarioPanel';
import {
  ScenarioProjectionToolbar,
  ScenarioProjectionWorkspace,
} from '@/components/scenario/ScenarioProjectionWorkspace';
import { useLayoutStore } from '@/stores/layoutStore';

export function TopShelf({
  height,
  availableHeight,
}: {
  height: number;
  availableHeight: number;
}) {
  const { topCollapsed } = useLayoutStore();

  return (
    <section
      className="flex h-full w-full max-w-full flex-col overflow-hidden min-h-0 bg-white/80 backdrop-blur-sm"
      style={{ height }}
    >
      <ScenarioProjectionToolbar availableHeight={availableHeight} />
      {!topCollapsed && (
        <div className="flex min-w-0 flex-1 min-h-0">
          <Sidebar />
          <TaskScenarioPanel />
          <main className="relative min-w-0 flex-1 min-h-0">
            <ScenarioProjectionWorkspace />
          </main>
          <RightPanel />
        </div>
      )}
    </section>
  );
}
