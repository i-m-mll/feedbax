import { useEffect, useMemo, useRef, useState } from 'react';
import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { TopShelf } from '@/components/layout/TopShelf';
import { BottomShelf } from '@/components/layout/BottomShelf';
import { Divider } from '@/components/layout/Divider';
import { useAppShortcuts } from '@/hooks/useShortcuts';
import { useGraphStore } from '@/stores/graphStore';
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

export default function App() {
  useAppShortcuts();

  // Debounced auto-save: 800ms after the last dirty change, save to backend.
  // Only fires when a graphId exists (i.e., graph was already saved at least once).
  const isDirty = useGraphStore((s) => s.isDirty);
  const graphId = useGraphStore((s) => s.graphId);
  const graphStack = useGraphStore((s) => s.graphStack);
  const inSubgraph = graphStack.length > 0;

  useEffect(() => {
    if (!isDirty || !graphId || inSubgraph) return;
    const timer = setTimeout(async () => {
      const { graph, uiState, markSaved } = useGraphStore.getState();
      try {
        await updateGraph(graphId, graph, uiState);
        markSaved(graphId);
      } catch (e) {
        console.warn('[auto-save] failed:', e);
      }
    }, AUTO_SAVE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [isDirty, graphId, inSubgraph]);

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
  );
}
