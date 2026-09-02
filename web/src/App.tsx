import { useEffect, useMemo, useRef, useState } from 'react';
import { toast, Toaster } from 'sonner';
import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { TopShelf } from '@/components/layout/TopShelf';
import { BottomShelf } from '@/components/layout/BottomShelf';
import { Divider } from '@/components/layout/Divider';
import { useAppShortcuts } from '@/hooks/useShortcuts';
import { useGraphStore } from '@/stores/graphStore';
import { persistLocalProjectTabs } from '@/stores/projectsStore';
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

export default function App() {
  useAppShortcuts();

  const graphId = useGraphStore((s) => s.graphId);

  useEffect(() => startStudioPersistence(), []);

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
