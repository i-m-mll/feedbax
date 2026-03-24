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
import { updateGraph } from '@/api/client';
import {
  useLayoutStore,
  BOTTOM_COLLAPSED_HEIGHT,
  MIN_BOTTOM_HEIGHT,
  MIN_TOP_HEIGHT,
  TOP_COLLAPSED_HEIGHT,
  DIVIDER_HEIGHT,
} from '@/stores/layoutStore';

const AUTO_SAVE_DELAY_MS = 800;

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
  const graphStack = useGraphStore((s) => s.graphStack);
  const inSubgraph = graphStack.length > 0;

  // Lifted timer ref so the pagehide handler can cancel a pending debounce.
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Guard against concurrent in-flight saves; re-arm after completion if still dirty.
  const savingRef = useRef(false);

  useEffect(() => {
    if (!isDirty || !graphId || inSubgraph) return;

    const doSave = async () => {
      if (savingRef.current) return;
      savingRef.current = true;
      const { graph, uiState, markSaved } = useGraphStore.getState();
      const analysis = getAnalysisForSave();
      try {
        await updateGraph(graphId, graph, uiState, analysis?.pages ?? null, analysis?.activePageId);
        markSaved(graphId);
      } catch (e) {
        toast.error('Auto-save failed — changes not saved', { id: 'autosave-error' });
      } finally {
        savingRef.current = false;
        // If a new edit arrived while the PUT was in-flight, re-arm the timer.
        if (useGraphStore.getState().isDirty) {
          timerRef.current = setTimeout(doSave, AUTO_SAVE_DELAY_MS);
        }
      }
    };

    timerRef.current = setTimeout(doSave, AUTO_SAVE_DELAY_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isDirty, graphId, inSubgraph]);

  // Flush unsaved changes on page unload via sendBeacon (more reliable than beforeunload).
  useEffect(() => {
    const handlePageHide = (event: PageTransitionEvent) => {
      if (event.persisted) return; // page going into bfcache, not unloading
      const { isDirty: dirty, graphId: gid, graph, uiState, graphStack } = useGraphStore.getState();
      if (!dirty || !gid) return;
      // Always save the root graph — if inside a subgraph, graphStack[0] is the root context.
      // Saving the current (subgraph) view to the top-level ID would corrupt the project.
      const rootGraph = graphStack.length > 0 ? graphStack[0].graph : graph;
      const rootUiState = graphStack.length > 0 ? graphStack[0].uiState : uiState;
      const analysis = getAnalysisForSave();
      // Cancel pending debounce timer
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      const beaconPayload: Record<string, unknown> = {
        graph: rootGraph,
        ui_state: rootUiState,
      };
      if (analysis) {
        beaconPayload.analysis_pages = analysis.pages;
        beaconPayload.active_analysis_page_id = analysis.activePageId;
      }
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
    <div className="min-h-screen flex flex-col">
      <Header />
      <div ref={containerRef} className="flex-1 min-h-0">
        <div
          className="grid h-full min-h-0"
          style={{
            gridTemplateRows:
              topHeight === undefined || bottomEffectiveHeight === undefined
                ? '1fr auto auto'
                : `${topHeight}px ${DIVIDER_HEIGHT}px ${bottomEffectiveHeight}px`,
          }}
        >
          <div className="min-h-0">
            <TopShelf height={topHeight ?? TOP_COLLAPSED_HEIGHT} />
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
