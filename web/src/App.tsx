import { lazy, Suspense, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { fitPanes } from '@window-shell/browser/layout';
import { installTooltips } from '@window-shell/browser/tooltip';
import { toast, Toaster } from 'sonner';
import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { TopShelf } from '@/components/layout/TopShelf';
import { BottomShelf } from '@/components/layout/BottomShelf';
import { Divider } from '@/components/layout/Divider';
import { StudioErrorBoundary } from '@/components/layout/StudioErrorBoundary';
import { useAppShortcuts } from '@/hooks/useShortcuts';
import { useGraphStore } from '@/stores/graphStore';
import { persistLocalProjectTabs } from '@/stores/projectsStore';
import { synchronizeDesktopPreferences } from '@/stores/settingsStore';
import { startStudioPersistence } from '@/services/studioPersistence';
import {
  useLayoutStore,
  BOTTOM_COLLAPSED_HEIGHT,
  MIN_BOTTOM_HEIGHT,
  MIN_TOP_HEIGHT,
  TOP_COLLAPSED_HEIGHT,
  DIVIDER_HEIGHT,
} from '@/stores/layoutStore';

const PROJECT_CHANNEL_NAME = 'feedbax:studio-project-presence';
const StudioProjectWorkspace = lazy(() => import('@/features/projectFolder/SavedTrainWorkspace'));

export default function App({ projectId }: { projectId: string | null }) {
  return projectId ? (
    <Suspense fallback={<main aria-busy="true" />}>
      <StudioProjectWorkspace projectId={projectId} />
    </Suspense>
  ) : <Workspace />;
}

function Workspace() {
  useAppShortcuts();

  const graphId = useGraphStore((s) => s.graphId);

  useEffect(() => startStudioPersistence(), []);

  useEffect(() => {
    if (import.meta.env.VITE_FEEDBAX_DESKTOP === '1') {
      return synchronizeDesktopPreferences((message) => toast.error(message, { id: 'desktop-preferences' }));
    }
  }, []);

  useEffect(() => {
    if (!graphId || typeof BroadcastChannel === 'undefined') return;
    const instanceId =
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random()}`;
    const channel = new BroadcastChannel(PROJECT_CHANNEL_NAME);
    const announceOpen = (type: 'project-open' | 'project-present') => {
      channel.postMessage({ type, graphId, instanceId });
    };
    channel.onmessage = (event) => {
      const message = event.data as { type?: string; graphId?: string; instanceId?: string };
      if (
        (message.type !== 'project-open' && message.type !== 'project-present') ||
        message.graphId !== graphId ||
        message.instanceId === instanceId
      ) {
        return;
      }
      toast.warning('This project is open in another tab. Concurrent saves may conflict.', {
        id: `multi-tab-${graphId}`,
      });
      if (message.type === 'project-open') {
        announceOpen('project-present');
      }
    };
    announceOpen('project-open');
    return () => channel.close();
  }, [graphId]);

  // Local tab custody is synchronous. Network writes remain serialized through
  // the persistence coordinator and resume from this local draft after reload.
  useEffect(() => {
    const handlePageHide = (event: PageTransitionEvent) => {
      if (event.persisted) return;
      persistLocalProjectTabs();
    };
    window.addEventListener('pagehide', handlePageHide);
    return () => window.removeEventListener('pagehide', handlePageHide);
  }, []);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [availableHeight, setAvailableHeight] = useState<number | null>(null);
  const [previewHeight, setPreviewHeight] = useState<number | null>(null);
  const { topCollapsed, bottomCollapsed, bottomHeight, setBottomHeight } = useLayoutStore();

  useLayoutEffect(() => installTooltips(), []);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) return;
      const { height } = entries[0].contentRect;
      setAvailableHeight(height);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const splitSpace = Math.max(0, (availableHeight ?? 0) - DIVIDER_HEIGHT);
  const topMinimum = topCollapsed ? TOP_COLLAPSED_HEIGHT : MIN_TOP_HEIGHT;
  const split = fitPanes(splitSpace, topMinimum, [{
    open: !bottomCollapsed,
    preferredSize: topCollapsed
      ? splitSpace - topMinimum
      : previewHeight ?? bottomHeight ?? Math.round(splitSpace / 2),
    minSize: MIN_BOTTOM_HEIGHT + 1,
    collapsedSize: BOTTOM_COLLAPSED_HEIGHT,
  }]);
  const bottom = split.panes[0]!;

  return (
    <>
    <Toaster theme="dark" position="bottom-right" />
    <div className="shell-frame h-[100dvh] w-full max-w-full overflow-hidden">
      <StudioErrorBoundary region="Project controls">
        <Header />
      </StudioErrorBoundary>
      <div ref={containerRef} className="shell-frame__content overflow-hidden">
        <div
          className="grid h-full min-h-0 w-full max-w-full overflow-hidden"
          style={{
            gridTemplateRows:
              availableHeight === null
                ? '1fr auto auto'
                : `${split.mainSize}px ${DIVIDER_HEIGHT}px ${bottom.size}px`,
          }}
        >
          <div className="min-w-0 min-h-0 overflow-hidden">
            <StudioErrorBoundary region="Model editor">
              <TopShelf height={availableHeight === null ? undefined : split.mainSize} />
            </StudioErrorBoundary>
          </div>
          <Divider layout={bottom} resizable={!topCollapsed && bottom.open} onPreview={setPreviewHeight} onResize={setBottomHeight} />
          <StudioErrorBoundary region="Pipeline">
            <BottomShelf
              height={availableHeight === null ? BOTTOM_COLLAPSED_HEIGHT : bottom.size}
              availableHeight={availableHeight ?? 0}
              collapsed={availableHeight === null ? bottomCollapsed : !bottom.open}
            />
          </StudioErrorBoundary>
        </div>
      </div>
      <StudioErrorBoundary region="Status">
        <StatusBar />
      </StudioErrorBoundary>
    </div>
    </>
  );
}
