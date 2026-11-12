import { lazy, Suspense, useState } from 'react';
import { fitPanes } from '@window-shell/browser/layout';
import { Sidebar } from '@/components/layout/Sidebar';
import { StudioErrorBoundary } from '@/components/layout/StudioErrorBoundary';
import { MAX_LEFT_WIDTH, MAX_RIGHT_WIDTH, MIN_LEFT_WIDTH, MIN_RIGHT_WIDTH, useLayoutStore } from '@/stores/layoutStore';
import { getTopPaneState, useWorkspaceStore } from '@/stores/workspaceStore';
import { useDrawerSpace, WORKSPACE_MIN_WIDTH } from './useDrawerSpace';

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
}: {
  height?: number;
}) {
  const {
    topCollapsed, taskSidebarWidth, leftSidebarVisible, leftSidebarWidth, rightSidebarVisible, rightSidebarWidth,
  } = useLayoutStore();
  const workspace = useWorkspaceStore((state) => state.workspace);
  const taskWidth = getTopPaneState(workspace).active_projection === 'task' ? taskSidebarWidth : 0;
  const { ref, width } = useDrawerSpace();
  const [leftPreview, setLeftPreview] = useState<number | null>(null);
  const [rightPreview, setRightPreview] = useState<number | null>(null);
  const { panes: [left, right] } = fitPanes(width, WORKSPACE_MIN_WIDTH + taskWidth, [
    { open: leftSidebarVisible, preferredSize: leftPreview ?? leftSidebarWidth, minSize: MIN_LEFT_WIDTH, maxSize: MAX_LEFT_WIDTH },
    { open: rightSidebarVisible, preferredSize: rightPreview ?? rightSidebarWidth, minSize: MIN_RIGHT_WIDTH, maxSize: MAX_RIGHT_WIDTH },
  ]);

  return (
    <section
      ref={ref}
      className="flex h-full w-full max-w-full flex-col overflow-hidden min-h-0 bg-white/80 backdrop-blur-sm"
      style={{ height }}
    >
      <StudioErrorBoundary region="Toolbar">
        <Suspense fallback={<div style={{ height: 44 }} className="border-b border-slate-100" />}>
          <ScenarioProjectionToolbar />
        </Suspense>
      </StudioErrorBoundary>
      {!topCollapsed && (
        <div className="flex min-w-0 flex-1 min-h-0">
          <StudioErrorBoundary region="Library">
            <Sidebar layout={left!} onPreview={setLeftPreview} />
          </StudioErrorBoundary>
          <StudioErrorBoundary region="Task">
            <Suspense fallback={<TopShelfFallback />}>
              <TaskScenarioPanel />
            </Suspense>
          </StudioErrorBoundary>
          <main className="relative min-w-0 flex-1 min-h-0">
            <StudioErrorBoundary region="Editor">
              <Suspense fallback={<TopShelfFallback />}>
                <ScenarioProjectionWorkspace />
              </Suspense>
            </StudioErrorBoundary>
          </main>
          <StudioErrorBoundary region="Inspector">
            <Suspense fallback={<TopShelfFallback />}>
              <RightPanel layout={right!} onPreview={setRightPreview} />
            </Suspense>
          </StudioErrorBoundary>
        </div>
      )}
    </section>
  );
}
