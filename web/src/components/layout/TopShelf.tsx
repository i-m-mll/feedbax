import { lazy, Suspense } from 'react';
import { Sidebar } from '@/components/layout/Sidebar';
import { useLayoutStore } from '@/stores/layoutStore';

const RightPanel = lazy(() =>
  import('@/components/panels/RightPanel').then((module) => ({ default: module.RightPanel }))
);
const TaskScenarioPanel = lazy(() =>
  import('@/components/panels/TaskScenarioPanel').then((module) => ({
    default: module.TaskScenarioPanel,
  }))
);
const ScenarioProjectionToolbar = lazy(() =>
  import('@/components/scenario/ScenarioProjectionWorkspace').then((module) => ({
    default: module.ScenarioProjectionToolbar,
  }))
);
const ScenarioProjectionWorkspace = lazy(() =>
  import('@/components/scenario/ScenarioProjectionWorkspace').then((module) => ({
    default: module.ScenarioProjectionWorkspace,
  }))
);

function TopShelfFallback() {
  return <div className="h-full min-w-0 flex-1 bg-slate-50/60" />;
}

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
      <Suspense fallback={<div style={{ height: 44 }} className="border-b border-slate-100" />}>
        <ScenarioProjectionToolbar availableHeight={availableHeight} />
      </Suspense>
      {!topCollapsed && (
        <div className="flex min-w-0 flex-1 min-h-0">
          <Sidebar />
          <Suspense fallback={<TopShelfFallback />}>
            <TaskScenarioPanel />
          </Suspense>
          <main className="relative min-w-0 flex-1 min-h-0">
            <Suspense fallback={<TopShelfFallback />}>
              <ScenarioProjectionWorkspace />
            </Suspense>
          </main>
          <Suspense fallback={<TopShelfFallback />}>
            <RightPanel />
          </Suspense>
        </div>
      )}
    </section>
  );
}
