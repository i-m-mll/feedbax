import { useEffect, useMemo, useRef, useState } from 'react';
import { toast, Toaster } from 'sonner';
import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { TopShelf } from '@/components/layout/TopShelf';
import { BottomShelf } from '@/components/layout/BottomShelf';
import { Divider } from '@/components/layout/Divider';
import { useAppShortcuts } from '@/hooks/useShortcuts';
import { useGraphStore } from '@/stores/graphStore';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useTrainingStore } from '@/stores/trainingStore';
import { persistLocalProjectTabs } from '@/stores/projectsStore';
import { buildWorkspaceSnapshot, useWorkspaceStore } from '@/stores/workspaceStore';
import { fetchGraph, updateGraph } from '@/api/client';
import { isHttpConflict } from '@/api/request';
import {
  useLayoutStore,
  BOTTOM_COLLAPSED_HEIGHT,
  MIN_BOTTOM_HEIGHT,
  MIN_TOP_HEIGHT,
  TOP_COLLAPSED_HEIGHT,
  DIVIDER_HEIGHT,
} from '@/stores/layoutStore';

const AUTO_SAVE_DELAY_MS = 800;
const PROJECT_CHANNEL_NAME = 'feedbax:studio-project-presence';

/** Convert analysis snapshot into the snake_case wire format the backend expects. */
function getAnalysisForSave(): {
  pages: Array<Record<string, unknown>>;
  activePageId: string | null;
} | null {
  const snapshot = useAnalysisStore.getState().captureSnapshot();
  if (snapshot.pages.length === 0) return null;
  return {
    pages: snapshot.pages.map((page) => ({
      id: page.id,
      name: page.name,
      graph_spec: page.graphSpec,
      eval_params: page.evalParams,
      viewport: page.viewport,
      eval_run_id: page.evalRunId,
      expanded_field_paths: page.expandedFieldPaths ?? [],
    })),
    activePageId: snapshot.activePageId,
  };
}

export default function App() {
  useAppShortcuts();

  // Debounced auto-save: 800ms after the last dirty change, save to backend.
  // Only fires when a graphId exists (i.e., graph was already saved at least once).
  const isDirty = useGraphStore((s) => s.isDirty);
  const graphId = useGraphStore((s) => s.graphId);

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

  // Lifted timer ref so the pagehide handler can cancel a pending debounce.
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Guard against concurrent in-flight saves; re-arm after completion if still dirty.
  const savingRef = useRef(false);

  useEffect(() => {
    if (!isDirty || !graphId) return;

    const doSave = async () => {
      if (savingRef.current) return;
      savingRef.current = true;
      const graphStore = useGraphStore.getState();
      const { graph, uiState } = graphStore.capturePersistedGraph();
      const analysis = getAnalysisForSave();
      const workspace = buildWorkspaceSnapshot({
        workspace: useWorkspaceStore.getState().workspace,
        graph,
        uiState,
        trainingSpec: useTrainingStore.getState().trainingSpec,
        taskSpec: useTrainingStore.getState().taskSpec,
        analysisSnapshot: useAnalysisStore.getState().captureSnapshot(),
        graphStackPath: graphStore.captureGraphStackPath(),
      });
      useWorkspaceStore.getState().setWorkspace(workspace);
      let saveConflict = false;
      try {
        const response = await updateGraph(
          graphId,
          graph,
          uiState,
          analysis?.pages ?? null,
          analysis?.activePageId,
          workspace,
          graphStore.saveRevision,
        );
        graphStore.markSaved(graphId, response.metadata.save_revision);
      } catch (e) {
        persistLocalProjectTabs();
        if (isHttpConflict(e)) {
          saveConflict = true;
          await fetchGraph(graphId).catch(() => undefined);
          toast.error('Save conflict: project changed elsewhere. Review the server copy before saving again.', {
            id: 'autosave-conflict',
          });
        } else {
          toast.error('Auto-save failed — changes not saved', { id: 'autosave-error' });
        }
      } finally {
        savingRef.current = false;
        // If a new edit arrived while the PUT was in-flight, re-arm the timer.
        if (!saveConflict && useGraphStore.getState().isDirty) {
          timerRef.current = setTimeout(doSave, AUTO_SAVE_DELAY_MS);
        }
      }
    };

    timerRef.current = setTimeout(doSave, AUTO_SAVE_DELAY_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isDirty, graphId]);

  // Flush unsaved changes on page unload via sendBeacon (more reliable than beforeunload).
  useEffect(() => {
    const handlePageHide = (event: PageTransitionEvent) => {
      if (event.persisted) return; // page going into bfcache, not unloading
      persistLocalProjectTabs();
      const graphStore = useGraphStore.getState();
      const { isDirty: dirty, graphId: gid } = graphStore;
      if (!dirty || !gid) return;
      const { graph: rootGraph, uiState: rootUiState } = graphStore.capturePersistedGraph();
      const analysis = getAnalysisForSave();
      const workspace = buildWorkspaceSnapshot({
        workspace: useWorkspaceStore.getState().workspace,
        graph: rootGraph,
        uiState: rootUiState,
        trainingSpec: useTrainingStore.getState().trainingSpec,
        taskSpec: useTrainingStore.getState().taskSpec,
        analysisSnapshot: useAnalysisStore.getState().captureSnapshot(),
        graphStackPath: graphStore.captureGraphStackPath(),
      });
      useWorkspaceStore.getState().setWorkspace(workspace);
      // Cancel pending debounce timer
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      const beaconPayload: Record<string, unknown> = {
        graph: rootGraph,
        ui_state: rootUiState,
        expected_save_revision: graphStore.saveRevision,
      };
      if (analysis) {
        beaconPayload.analysis_pages = analysis.pages;
        beaconPayload.active_analysis_page_id = analysis.activePageId;
      }
      beaconPayload.workspace = workspace;
      const body = new Blob(
        [JSON.stringify(beaconPayload)],
        { type: 'application/json' }
      );
      const sent = navigator.sendBeacon(`/api/graphs/${gid}/beacon`, body);
      if (!sent) {
        // Fallback: keepalive fetch (fire-and-forget)
        fetch(`/api/graphs/${gid}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(beaconPayload),
          keepalive: true,
        }).catch(() => {});
      }
    };
    window.addEventListener('pagehide', handlePageHide);
    return () => window.removeEventListener('pagehide', handlePageHide);
  }, []); // empty deps — reads from store at event time

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [availableHeight, setAvailableHeight] = useState(0);
  const {
    topCollapsed,
    bottomCollapsed,
    bottomHeight,
    initializeBottomHeight,
  } = useLayoutStore();

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

  useEffect(() => {
    if (availableHeight > 0) {
      initializeBottomHeight(availableHeight);
    }
  }, [availableHeight, initializeBottomHeight]);

  const { topHeight, bottomEffectiveHeight } = useMemo(() => {
    if (availableHeight <= 0) {
      return {
        topHeight: undefined,
        bottomEffectiveHeight: undefined,
      };
    }
    const adjustedAvailable = availableHeight - DIVIDER_HEIGHT;
    if (topCollapsed) {
      const bottom = Math.max(adjustedAvailable - TOP_COLLAPSED_HEIGHT, BOTTOM_COLLAPSED_HEIGHT);
      return {
        topHeight: TOP_COLLAPSED_HEIGHT,
        bottomEffectiveHeight: bottom,
      };
    }
    if (bottomCollapsed) {
      const top = Math.max(adjustedAvailable - BOTTOM_COLLAPSED_HEIGHT, MIN_TOP_HEIGHT);
      return {
        topHeight: top,
        bottomEffectiveHeight: BOTTOM_COLLAPSED_HEIGHT,
      };
    }
    const clampedBottom = Math.max(
      MIN_BOTTOM_HEIGHT,
      Math.min(adjustedAvailable - MIN_TOP_HEIGHT, bottomHeight)
    );
    return {
      topHeight: Math.max(adjustedAvailable - clampedBottom, MIN_TOP_HEIGHT),
      bottomEffectiveHeight: clampedBottom,
    };
  }, [availableHeight, topCollapsed, bottomCollapsed, bottomHeight]);

  return (
    <>
    <Toaster theme="dark" position="bottom-right" />
    <div className="flex min-h-screen w-full max-w-full flex-col overflow-hidden">
      <Header />
      <div ref={containerRef} className="min-w-0 flex-1 min-h-0">
        <div
          className="grid h-full min-h-0 w-full max-w-full overflow-hidden"
          style={{
            gridTemplateRows:
              topHeight === undefined || bottomEffectiveHeight === undefined
                ? '1fr auto auto'
                : `${topHeight}px ${DIVIDER_HEIGHT}px ${bottomEffectiveHeight}px`,
          }}
        >
          <div className="min-w-0 min-h-0 overflow-hidden">
            <TopShelf
              height={topHeight ?? TOP_COLLAPSED_HEIGHT}
              availableHeight={availableHeight}
            />
          </div>
          <Divider availableHeight={availableHeight} />
          <BottomShelf
            height={bottomEffectiveHeight ?? BOTTOM_COLLAPSED_HEIGHT}
            availableHeight={availableHeight}
          />
        </div>
      </div>
      <StatusBar />
    </div>
    </>
  );
}
